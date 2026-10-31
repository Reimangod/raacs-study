"""Bounded no-FCI molecular LUCJ/SCI/RAACS comparisons, including Level-B systems.

One numerical actor; native ffsim circuits sampled via MPS, not full sectors.
No old result is rewritten. Counts commit before choosing the next proposal.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from io import BytesIO
import fcntl
import importlib.metadata
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import traceback
import zlib
import numpy as np
from qiskit import QuantumCircuit,qpy
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,rng_for,save_json,save_npz,save_binary
from .extended_molecular_problems import ExtendedMolecularSpec,extended_queue,build_extended_problem
from .molecular_mps_scaling import no_full_state_guards
from .molecular_lucj_primitives import initial_amplitudes,ucj_from_amplitudes,ucj_from_parameters,ucj_pairs
from .molecular_lucj_primitives import ucj_circuits
from .resource_adaptive_sampler import integral_problem,RAACSSpec,run_sampler
from .circuit_sampling_runtime import SamplingSpec,CircuitCountSampler
from .integral_connections import IntegralConnections
from .local_sampler_primitives import observed_ci,pack_counts64,unpack_counts64
from .ansatz import circuit_bytes,resources,measurement_equivalent_circuit
from .study.core import rss_peak

METHODS=('HF','RAACS-full-cost','LUCJ-MP2-2-fixed','LUCJ-CCSD-2-fixed','LUCJ-CCSD-2-QSCI4',
         'LUCJ-random-2-s101','LUCJ-random-2-s211','LUCJ-random-2-s307','classical-SCI64')


class CapacityPending(RuntimeError):
    """Unstarted next operation, not a molecular/algorithm failure."""


class SamplingCapped(RuntimeError):
    """Known partial draw retained; no replacement or budget top-up."""


def queue(mode):
    old=ROOT/'results/molecular-MPS-scaling-standard-v1';manifest=json.loads((old/'manifest.json').read_text())
    existing=manifest['queue'] if mode=='standard' else [x for x in manifest['queue'] if x['spec']['molecule']=='H10' and x['spec']['distance_A']==1.25]
    out=[{'id':x['id'],'kind':'inherited','problem':str(old/'cases'/x['id']/'physical-problem'),
          'source_case':str(old/'cases'/x['id']),'physical_spec':x['spec']} for x in existing]
    for spec in extended_queue(mode):
        out.append({'id':spec.molecule+'_'+spec.space+'_R'+str(spec.distance_A)+'_'+digest(asdict(spec))[:10],
                    'kind':'new-LevelB','physical_spec':asdict(spec)})
    return out


def configuration(mode):
    files=('molecular_strong_controls.py','extended_molecular_problems.py','molecular_lucj_primitives.py',
           'molecular_mps_scaling.py','resource_adaptive_sampler.py','circuit_sampling_runtime.py','mps_sampling.py',
           'integral_hamiltonian.py','integral_connections.py','local_sampler_primitives.py','ansatz.py','common.py')
    return {'schema':'molecular-strong-controls-v1','mode':mode,'queue':queue(mode),'methods':list(METHODS),
        'total_model_shots':2000,'pilot_shots':64,'QSCI_LUCJ_points':4,'R_grid':[8,32,64],
        'sampling':asdict(SamplingSpec(backend='mps',max_bond=64,cutoff=1e-12,cache_entries=2)),
        'free_floor_bytes':int(2.75*2**30),'output_cap_bytes':128*2**20,'RSS_cap_bytes':2*2**30,
        'case_slice_seconds':1800,'classical_SCI_seconds':60,'CCSD_seconds':180,'CCSD_max_cycles':150,
        'source_scaling_manifest_sha256':file_digest(ROOT/'results/molecular-MPS-scaling-standard-v1/manifest.json'),
        'implementation':{p:file_digest(ROOT/'benchmark'/p) for p in files},
        'spec_sha256':file_digest(ROOT/'study-v1/MOLECULAR_STRONG_CONTROLS_V1.md'),
        'packages':{p:importlib.metadata.version(p) for p in ('ffsim','pyscf','qiskit','quimb','numpy','scipy')},
        'FCI_used':False,'full_Fock_vector_allowed':False,'hardware_shots':0}


def boundary(root,cfg):
    if shutil.disk_usage(root).free<cfg['free_floor_bytes']:return 'PENDING_CAPACITY_BOUNDARY'
    if rss_peak()>cfg['RSS_cap_bytes']:return 'PENDING_RSS_BOUNDARY'
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>cfg['output_cap_bytes']:return 'PENDING_OUTPUT_BOUNDARY'


def database(folder):
    db=sqlite3.connect(folder/'events.sqlite');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS circuits(id TEXT PRIMARY KEY,preparation_qpy BLOB,sampling_qpy BLOB,payload TEXT);
       CREATE TABLE IF NOT EXISTS draws(id TEXT PRIMARY KEY,circuit TEXT,role TEXT,requested INTEGER,status TEXT,seed TEXT,counts BLOB,metadata TEXT);
       CREATE TABLE IF NOT EXISTS objectives(i INTEGER PRIMARY KEY,draw TEXT,loss REAL,parameter_sha TEXT);''')
    return db


def readouts(backend,info,counts,grid):
    out=[]
    for R in grid:
        ci=observed_ci(backend,counts,info['hf_id'],info['nalpha'],info['nbeta'],R)
        out.append({'R':R,'energy_Ha':ci['energy_Ha'],'actual_dimension':ci['actual_dimension'],
                    'selected_ids':list(map(int,ci['selected_ids'])),'internal_residual':ci['internal_Ritz_residual']})
    return out


def classical_control(backend,info,seconds=60):
    start=perf_counter();connections=IntegralConnections(backend,parent_cap=128);selected=[info['hf_id']];history=[]
    for iteration in range(8):
        hh=backend.matrix(selected,selected);w,c=np.linalg.eigh(hh);energy=float(w[0]);coeff=c[:,0]
        history.append({'iteration':iteration,'selected_ids':selected.copy(),'coefficients':coeff.real.tolist(),
                        'energy_Ha':energy,'internal_residual':float(np.linalg.norm(hh@coeff-energy*coeff))})
        if len(selected)>=64:return history,'DIMENSION_CAP'
        if perf_counter()-start>seconds:return history,'CLASSICAL_WALL_CAP'
        ext=connections.external(selected,coeff,energy)
        candidates=sorted([(int(x),float(v)) for x,v in zip(ext['ids'],ext['en_energy_importance']) if v>1e-16],key=lambda x:(-x[1],x[0]))
        if not candidates:return history,'NO_EXTERNAL_IMPORTANCE'
        selected=sorted(selected+[x for x,_ in candidates[:min(len(selected),64-len(selected))]])
    return history,'ITERATION_CAP'


def cell(problem,folder,method,cfg,root,dependencies):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);result_path=folder/'result.json'
    backend,info=integral_problem(problem);binding=digest([info['problem_id'],method,cfg])
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['binding']!=binding:raise RuntimeError('Frozen molecular task drift')
        return old
    before=perf_counter();result={'binding':binding,'method':method,'problem':info,'hardware_shots':0,'FCI_used':False,
        'ground_state_error_Ha':None,'chemical_accuracy':'UNKNOWN_NO_GROUND_STATE_REFERENCE','full_Fock_vector_used':False}
    db=None
    try:
        if method=='RAACS-full-cost':
            spec=RAACSSpec(score='full_cost',total_model_shots=cfg['total_model_shots'],max_steps=4,parent_connection_cap=64,
                target_cap=64,candidate_bank_cap=16,candidates_measured_per_step=2)
            actual=run_sampler(problem,folder/'native-RAACS',spec,SamplingSpec(**cfg['sampling']))
            result.update(status=actual['status'],native_result='native-RAACS/result.json',native_events_sha256=actual['event_database_sha256'],
                linked_primary_model_shots=actual['actual_total_model_shots'],new_primary_model_shots=actual['actual_total_model_shots'],
                final_energy_Ha=actual['final_QSCI_energy_Ha'],native_stop=actual['construction_stop_reason'])
        elif method=='classical-SCI64':
            history,stop=classical_control(backend,info,cfg['classical_SCI_seconds']);save_json(folder/'SCI-history.json',history)
            result.update(status='COMPLETED',classical_stop=stop,final_energy_Ha=history[-1]['energy_Ha'],
                final_selected_ids=history[-1]['selected_ids'],linked_primary_model_shots=0,new_primary_model_shots=0,
                named_H_requests=backend.matrix_entry_requests,
                scope='screened parent-cap128 EN-cardinality SCI, NOT converged SHCI/ground reference')
        else:
            db=database(folder);sampler=CircuitCountSampler(SamplingSpec(**cfg['sampling']));proposals=1
            n=backend.norb;balanced=info['nalpha']==info['nbeta'];parameters=None;op=None
            if method!='HF':
                if 'random' in method:
                    import ffsim
                    family=ffsim.UCJOpSpinBalanced if balanced else ffsim.UCJOpSpinUnbalanced
                    p=family.n_params(n,2,interaction_pairs=ucj_pairs(n,balanced),with_final_orbital_rotation=True)
                    rng,seed=rng_for('E-LUCJ-random-v1',info['problem_id'],int(method.rsplit('s',1)[1]))
                    parameters=rng.uniform(-np.pi,np.pi,p);op=ucj_from_parameters(parameters,n,balanced)
                    result['initialization']={'kind':'random','seed':str(seed),'dependency_quantum_shots':0}
                else:
                    kind='mp2' if 'MP2' in method else 'ccsd'
                    t1,t2,dependency=initial_amplitudes(backend,info,dependencies,kind,cfg['CCSD_max_cycles'],cfg['CCSD_seconds'])
                    tick=perf_counter();op,parameters,_=ucj_from_amplitudes(t1,t2,n,balanced)
                    result['initialization']={'kind':kind,'dependency':str(Path(dependencies)/(kind+'.json')),
                        'dependency_sha256':file_digest(Path(dependencies)/(kind+'.json')),'dependency_standalone_seconds':dependency['wall_seconds'],
                        'factorization_seconds':perf_counter()-tick,'dependency_quantum_shots':0,
                        'native_family':'spin balanced' if balanced else 'spin unbalanced'}
                save_npz(folder/'initial-parameters.npz',parameters=parameters)
                if 'QSCI4' in method:proposals=cfg['QSCI_LUCJ_points']
            best=None;best_parameters=None;pilot_total=0;changed_variables=set()
            rng,_=rng_for('E-QSCI-coordinate-proposals-v1',info['problem_id'],method)
            def measure(preparation,role,idx,shots,parameter_count):
                raw=circuit_bytes(preparation);cid=digest([info['problem_id'],method,hashlib.sha256(raw).hexdigest()])
                sample,_=measurement_equivalent_circuit(preparation)
                db.execute('INSERT OR IGNORE INTO circuits VALUES(?,?,?,?)',(cid,zlib.compress(raw),zlib.compress(circuit_bytes(sample)),
                    json.dumps({'resources':resources(sample,parameter_count,0 if method=='HF' else 2),'role':role})))
                did=digest([binding,role,idx,cid,shots]);row=db.execute('SELECT status,counts,metadata FROM draws WHERE id=?',(did,)).fetchone()
                if row:
                    if row[0] not in ('COMPLETED','RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED'):raise RuntimeError('Partial in-flight model work preserved, no silent redraw')
                    ii,cc=unpack_counts64(row[1]);return dict(zip(map(int,ii),map(int,cc))),did,json.loads(row[2]),cid
                if why:=boundary(root,cfg):raise CapacityPending(why)
                seed=rng_for('E-MPS-physical-draw-v1',did)[1]
                db.execute('INSERT INTO draws VALUES(?,?,?,?,?,?,NULL,NULL)',(did,cid,role,shots,'STARTED',str(seed)));db.commit()
                ii,cc,meta=sampler.draw(preparation,shots,seed,parameter_count,0 if method=='HF' else 2)
                status=meta['sample_metadata']['status']
                db.execute('UPDATE draws SET status=?,counts=?,metadata=? WHERE id=?',(status,pack_counts64(ii,cc),json.dumps(meta),did));db.commit()
                return dict(zip(map(int,ii),map(int,cc))),did,meta,cid
            for i in range(proposals):
                if method=='HF':
                    prep=QuantumCircuit(info['n_qubits'])
                    for q in range(info['n_qubits']):
                        if info['hf_id']>>q&1:prep.x(q)
                    trial=None
                else:
                    trial=parameters.copy() if i==0 else best_parameters.copy()
                    if i:
                        variable=int(rng.integers(len(trial)));changed_variables.add(variable)
                        trial[variable]+=float(rng.normal(0,.15))
                    op=ucj_from_parameters(trial,n,balanced);prep,_,_=ucj_circuits(op,n,(info['nalpha'],info['nbeta']))
                if proposals==1:
                    best=(0.,prep);best_parameters=trial;break
                counts,did,meta,cid=measure(prep,'QSCI-pilot',i,cfg['pilot_shots'],len(trial))
                if meta['sample_metadata']['status']!='COMPLETED':
                    raise SamplingCapped('Partial pilot; no unregistered top-up or final selection')
                ci=observed_ci(backend,counts,info['hf_id'],info['nalpha'],info['nbeta'],64)
                db.execute('INSERT OR REPLACE INTO objectives VALUES(?,?,?,?)',(i,did,ci['energy_Ha'],digest(trial.tolist())));db.commit()
                pilot_total+=meta['actual_raw_model_shots']
                if best is None or ci['energy_Ha']<best[0]:best=(ci['energy_Ha'],prep);best_parameters=trial.copy()
            remaining=cfg['total_model_shots']-(cfg['pilot_shots']*proposals if proposals>1 else 0)
            counts,did,meta,cid=measure(best[1],'independent-final',0,remaining,0 if best_parameters is None else len(best_parameters))
            output=readouts(backend,info,counts,cfg['R_grid']);save_json(folder/'readouts.json',output)
            if best_parameters is not None:save_npz(folder/'final-parameters.npz',parameters=best_parameters)
            primary_raw=db.execute('SELECT coalesce(sum(requested),0) FROM draws').fetchone()[0]
            actual_raw=sum(json.loads(x[0])['actual_raw_model_shots'] for x in db.execute('SELECT metadata FROM draws WHERE metadata IS NOT NULL'))
            result.update(status='COMPLETED' if actual_raw==cfg['total_model_shots'] else 'RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED',
                linked_primary_model_shots=actual_raw,new_primary_model_shots=actual_raw,requested_primary_model_shots=primary_raw,
                pilot_model_shots=pilot_total,independent_final_model_shots=meta['actual_raw_model_shots'],
                final_draw_id=did,final_circuit_id=cid,final_energy_Ha=output[-1]['energy_Ha'],
                n_independent_ansatz_parameters=0 if best_parameters is None else len(best_parameters),
                n_optimizer_objective_evaluations=proposals if proposals>1 else 0,
                n_optimized_variables_this_run=len(changed_variables),
                selection_scope='fixed initialization' if proposals==1 else 'minimum observed pilot QSCI; independent final draw',
                quantum_objective_budget_included=True)
        if db is not None:db.close();db=None;result['event_database_sha256']=file_digest(folder/'events.sqlite')
    except Exception as exc:
        if db is not None:
            rows=list(db.execute('SELECT status,requested,metadata FROM draws'))
            known=sum(json.loads(r[2])['actual_raw_model_shots'] for r in rows if r[2] is not None)
            inflight=any(r[0]=='STARTED' for r in rows)
            result.update(known_committed_new_model_shots=known,
                new_primary_model_shots=None if inflight else known,
                unreported_inflight_model_shots=None if inflight else 0,
                requested_primary_model_shots=sum(r[1] for r in rows))
            db.close();db=None;result['event_database_sha256']=file_digest(folder/'events.sqlite')
        if isinstance(exc,CapacityPending):
            result.update(status=str(exc),partial_evidence_preserved=True,terminal=False)
            save_json(folder/'pending-boundary.json',result);return result
        result.update(status='RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED' if isinstance(exc,SamplingCapped) else
                      'ENGINEERING_OR_INITIALIZATION_FAILURE_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),
                      partial_evidence_preserved=True)
    result.update(wall_seconds_this_invocation=perf_counter()-before,peak_RSS=rss_peak())
    save_json(result_path,result);return result


def run(root,mode):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);cfg=configuration(mode)
    with (root/'run.lock').open('a') as lock,threadpool_limits(limits=1),no_full_state_guards():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest=root/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=cfg:raise RuntimeError('Frozen strong molecular config/code drift')
        if mode=='standard':
            p=ROOT/'results/molecular-strong-controls-quick-v1/validation.json'
            if not p.exists() or json.loads(p.read_text())['status']!='MOLECULAR_STRONG_QUICK_INDEPENDENTLY_VALIDATED':raise RuntimeError('Actual quick qualification required')
        save_json(manifest,cfg);summary=[]
        for item in cfg['queue']:
            before=perf_counter();case=root/'cases'/item['id'];case.mkdir(parents=True,exist_ok=True)
            if why:=boundary(root,cfg):save_json(root/'execution-status.json',{'status':why,'rows':summary});return
            if item['kind']=='new-LevelB':
                problem=case/'physical-problem'
                try:build_extended_problem(ExtendedMolecularSpec(**item['physical_spec']),problem)
                except Exception as exc:
                    save_json(case/'physical-failure.json',{'error':repr(exc),'spec':item['physical_spec']})
                    summary.extend({'case':item['id'],'method':m,'status':'PHYSICAL_INPUT_FAILURE_PRESERVED'} for m in METHODS);continue
            else:problem=Path(item['problem'])
            physical=json.loads((problem/'problem.json').read_text())
            if physical['FCI_computed'] or physical['full_Fock_H_constructed']:raise RuntimeError('Practical physical input boundary')
            save_json(case/'physical-source.json',{'path':str(problem),'meta_sha256':file_digest(problem/'problem.json'),
                       'integrals_sha256':file_digest(problem/'integrals.npz'),'inherited':item['kind']=='inherited'})
            for method in cfg['methods']:
                if why:=boundary(root,cfg):save_json(root/'execution-status.json',{'status':why,'rows':summary});return
                if perf_counter()-before>cfg['case_slice_seconds']:save_json(root/'execution-status.json',{'status':'PENDING_CASE_SLICE','rows':summary});return
                folder=case/method;folder.mkdir(exist_ok=True)
                if method=='RAACS-full-cost' and item['kind']=='inherited':
                    original=Path(item['source_case'])/'full_cost';r=json.loads((original/'result.json').read_text())
                    if file_digest(original/'events.sqlite')!=r['event_database_sha256']:raise RuntimeError('Inherited RAACS event drift')
                    out={'status':r['status'],'method':method,'linked_primary_model_shots':r['actual_total_model_shots'],
                         'new_primary_model_shots':0,'native_result':str(original/'result.json'),
                         'native_result_sha256':file_digest(original/'result.json'),'native_events_sha256':r['event_database_sha256'],
                         'final_energy_Ha':r['final_QSCI_energy_Ha'],'inherited_original_work_not_free':True,
                         'historical_construction_seconds':r['actual_wall_seconds'],'hardware_shots':0}
                    if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text())!=out:raise RuntimeError('Inherited adapter drift')
                    save_json(folder/'result.json',out)
                else:out=cell(problem,folder,method,cfg,root,case/'amplitude-dependencies')
                if out.get('terminal') is False:
                    save_json(root/'execution-status.json',{'status':out['status'],'rows':summary,
                        'pending_case':item['id'],'pending_method':method});return
                summary.append({'case':item['id'],'method':method,'status':out['status']})
                save_json(root/'execution-status.json',{'status':'RUNNING','terminal':len(summary),'planned':len(cfg['queue'])*len(METHODS),'rows':summary})
                print(item['id'],method,out['status'],flush=True)
        result={'status':'MOLECULAR_STRONG_CONTROLS_EXECUTED_PENDING_AUDIT','terminal':len(summary),'planned':len(cfg['queue'])*len(METHODS),
                'rows':summary,'hardware_shots':0,'FCI_used':False,'whole_goal_complete':False}
        save_json(root/'execution-status.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','standard'],default='quick');p.add_argument('--output')
    a=p.parse_args();print(json.dumps(run(a.output or ROOT/f'results/molecular-strong-controls-{a.mode}-v1',a.mode)))
