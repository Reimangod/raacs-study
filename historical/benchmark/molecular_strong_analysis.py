"""Independent named-space/cost audit, native routing and MPS qualification.

Read-only on execution evidence. Diagnostic approximate states never enter the
sampler's selection or add primary shots. No full Fock states or ground labels.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,save_json
from .resource_adaptive_sampler import integral_problem
from .local_sampler_primitives import unpack_counts64,valid_sector
from .ansatz import resources
from .mps_sampling import construct_mps
from .molecular_mps_scaling import no_full_state_guards
from .gate_noise_runtime import compile_sampler


def selected(backend,info,counts,R):
    order=[info['hf_id']]+[x for x,n in sorted(counts.items(),key=lambda a:(-a[1],a[0]))
        if x!=info['hf_id'] and valid_sector(x,info['n_qubits'],info['nalpha'],info['nbeta'])][:R-1]
    return named_ritz(backend,order,R)


def named_ritz(backend,order,R):
    ids=sorted(order[:R]);h=backend.matrix(ids,ids);e,c=np.linalg.eigh(h)
    return {'R':R,'actual_dimension':len(ids),'energy_Ha':float(e[0]),'selected_ids':ids,
            'internal_residual':float(np.linalg.norm(h@c[:,0]-e[0]*c[:,0]))}


def final_evidence(root,item,method):
    """Return all paid draws and independent-final counts, never pilot union."""
    folder=root/'cases'/item['id']/method;r=json.loads((folder/'result.json').read_text())
    if method=='classical-SCI64':return r,None,[],None,None
    if method=='RAACS-full-cost':
        p=Path(r['native_result']);p=p if p.is_absolute() else folder/p
        native=json.loads(p.read_text());dbpath=p.parent/'events.sqlite'
        assert file_digest(dbpath)==r['native_events_sha256']
        db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
        rows=[];final={};finalcid=None
        for did,cid,role,requested,raw,blob,metadata in db.execute('SELECT id,circuit,role,requested,raw_shots,counts64,metadata FROM draws'):
            ii,nn=unpack_counts64(blob);counts=dict(zip(map(int,ii),map(int,nn)));meta=json.loads(metadata)
            assert sum(counts.values())==raw==meta['actual_raw_model_shots']
            rows.append({'id':did,'role':role,'requested':requested,'raw':raw,'counts':counts,'metadata':meta,'status':meta['sample_metadata']['status']})
            if did in native['final_draw_ids']:
                assert role=='independent_final'
                for x,n in counts.items():final[x]=final.get(x,0)+n
                if finalcid is not None:assert finalcid==cid
                finalcid=cid
        qrow=db.execute('SELECT preparation_qpy,sampling_qpy,resources FROM circuits WHERE id=?',(finalcid,)).fetchone()
        db.close()
        assert sum(x['raw'] for x in rows)==native['actual_total_model_shots']
        assert sum(final.values())==native['independent_final_raw_model_shots']
        prep=qpy.load(BytesIO(zlib.decompress(qrow[0])))[0];sample=qpy.load(BytesIO(zlib.decompress(qrow[1])))[0]
        res=json.loads(qrow[2]);assert int(sample.count_ops().get('cx',0))==res['cnot_count']
        return r,final,rows,sample,{'preparation':prep,'sampling_resources':res,'source_database':str(dbpath),
            'source_database_sha256':file_digest(dbpath),'final_circuit_id':finalcid}
    dbpath=folder/'events.sqlite';db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
    if r.get('event_database_sha256'):assert file_digest(dbpath)==r['event_database_sha256']
    rows=[];final=None;finalcid=None
    for did,cid,role,requested,status,blob,metadata in db.execute('SELECT id,circuit,role,requested,status,counts,metadata FROM draws'):
        counts=None;meta=None
        if blob is not None:
            ii,nn=unpack_counts64(blob);counts=dict(zip(map(int,ii),map(int,nn)));meta=json.loads(metadata)
            assert sum(counts.values())==meta['actual_raw_model_shots']<=requested
            assert meta['sample_metadata']['status']==status
            if role=='independent-final':final=counts;finalcid=cid
        rows.append({'id':did,'role':role,'requested':requested,'raw':None if counts is None else sum(counts.values()),
                     'counts':counts,'metadata':meta,'status':status})
    if finalcid is None:db.close();return r,final,rows,None,None
    assert r['final_draw_id']==next(x['id'] for x in rows if x['role']=='independent-final')
    qrow=db.execute('SELECT preparation_qpy,sampling_qpy,payload FROM circuits WHERE id=?',(finalcid,)).fetchone();db.close()
    prepraw=zlib.decompress(qrow[0]);prep=qpy.load(BytesIO(prepraw))[0];sample=qpy.load(BytesIO(zlib.decompress(qrow[1])))[0]
    assert finalcid==digest([r['problem']['problem_id'],method,hashlib.sha256(prepraw).hexdigest()])
    res=json.loads(qrow[2])['resources'];rebuilt=resources(sample,r['n_independent_ansatz_parameters'],0 if method=='HF' else 2)
    assert res==rebuilt
    return r,final,rows,sample,{'preparation':prep,'sampling_resources':res,'source_database':str(dbpath),
        'source_database_sha256':file_digest(dbpath),'final_circuit_id':finalcid}


def audit(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text())
    for p,sha in cfg['implementation'].items():assert file_digest(ROOT/'benchmark'/p)==sha,p
    assert file_digest(ROOT/'study-v1/MOLECULAR_STRONG_CONTROLS_V1.md')==cfg['spec_sha256']
    assert file_digest(ROOT/'results/molecular-MPS-scaling-standard-v1/manifest.json')==cfg['source_scaling_manifest_sha256']
    planned=len(cfg['queue'])*len(cfg['methods']);rows=[];comparison=[];checks=0;known=0;new=0;unknown=0;maximum=0.;amplitudes=[]
    with threadpool_limits(limits=1),no_full_state_guards():
        for item in cfg['queue']:
            case=root/'cases'/item['id'];source=case/'physical-source.json'
            if not source.exists():
                assert (case/'physical-failure.json').exists()
                rows.extend({'case':item['id'],'method':m,'status':'PHYSICAL_INPUT_FAILURE_PRESERVED'} for m in cfg['methods']);continue
            src=json.loads(source.read_text());problem=Path(src['path'])
            assert file_digest(problem/'problem.json')==src['meta_sha256']
            assert file_digest(problem/'integrals.npz')==src['integrals_sha256']
            backend,info=integral_problem(problem);finals={}
            # The v1 emitter marks MP2 finite algebraic output converged=True.
            # For noncanonical ROHF→UHF embedding that alone is insufficient:
            # PySCF actually enters iterative MP2. Reproduce and inspect its
            # native convergence flag; do not mutate the old dependency JSON.
            dep=case/'amplitude-dependencies'/'mp2.json'
            if dep.exists():
                dependency=json.loads(dep.read_text())
                if dependency['status']=='COMPLETED' and dependency['qualification'].get('UHF_stationary') is False:
                    from pyscf import mp
                    from .molecular_lucj_primitives import fixed_integral_reference
                    tick=perf_counter();mf,_=fixed_integral_reference(backend,info);solver=mp.MP2(mf);solver.max_memory=768;solver.kernel()
                    with np.load(dep.with_suffix('.npz')) as archive:
                        delta=max(float(np.max(abs(solver.t2[i]-archive[k]))) for i,k in enumerate(('t2aa','t2ab','t2bb')))
                    check={'case':item['id'],'native_iterative_MP2_converged':bool(solver.converged),
                           'maximum_t2_replay_difference':delta,'standalone_audit_seconds':perf_counter()-tick,
                           'source_dependency_sha256':file_digest(dep),'new_quantum_shots':0,
                           'scope':'independent post-outcome initialization qualification, no changed circuit/selection'}
                    amplitudes.append(check);assert check['native_iterative_MP2_converged'] and delta<1e-7
            for method in cfg['methods']:
                r,counts,paid,qc,physical=final_evidence(root,item,method)
                classical_values=None;classical_order=None
                if method=='classical-SCI64':
                    ids=r['final_selected_ids'];energy=float(np.linalg.eigvalsh(backend.matrix(ids,ids))[0])
                    assert abs(energy-r['final_energy_Ha'])<1e-8;checks+=1
                    history=json.loads((case/method/'SCI-history.json').read_text());classical_order=[]
                    for entry in history:
                        classical_order.extend(sorted(set(entry['selected_ids'])-set(classical_order)))
                    # Cardinality-level first-appearance order; numeric ID ties
                    # inside a batch. Post-hoc d diagnostic, not paid samples.
                    classical_values=[named_ritz(backend,classical_order,R) for R in cfg['R_grid']]
                    assert abs(classical_values[-1]['energy_Ha']-r['final_energy_Ha'])<1e-8;checks+=len(classical_values)
                raw=sum(x['raw'] for x in paid if x['raw'] is not None)
                inflight=sum(x['raw'] is None for x in paid);unknown+=inflight
                if paid:assert sum(x['requested'] for x in paid)<=cfg['total_model_shots']
                if r['status']=='COMPLETED' and method!='classical-SCI64':
                    assert raw==cfg['total_model_shots'] and not inflight and counts is not None
                known+=raw
                new+=raw if not r.get('inherited_original_work_not_free') else 0
                row={'case':item['id'],'method':method,'status':r['status'],'n_qubits':info['n_qubits'],
                    'known_committed_model_shots':raw,'unknown_inflight_draws':inflight,
                    'new_model_shots':raw if not r.get('inherited_original_work_not_free') else 0,
                    'wall_seconds_this_invocation':r.get('wall_seconds_this_invocation'),
                    'inherited_construction_seconds':r.get('historical_construction_seconds'),
                    'peak_RSS':r.get('peak_RSS'),'result_sha256':file_digest(case/method/'result.json'),
                    'ground_state_error_Ha':None,'chemical_accuracy':None,
                    'HF_energy_Ha':info['HF_energy_Ha'],
                    'MPS_build_seconds_actual_calls':sum(x['metadata']['construction_seconds_this_call'] for x in paid if x['metadata']),
                    'MPS_sampling_seconds_actual_calls':sum(x['metadata']['sample_metadata']['sampling_seconds'] for x in paid if x['metadata']),
                    'initialization':r.get('initialization')}
                if counts is not None:
                    assert all(0<=x<1<<info['n_qubits'] and n>0 for x,n in counts.items())
                    values=[selected(backend,info,counts,R) for R in cfg['R_grid']]
                    stored=(json.loads((case/method/'readouts.json').read_text()) if method!='RAACS-full-cost' else [values[-1]])
                    for v,s in zip(values if method!='RAACS-full-cost' else [values[-1]],stored):
                        error=abs(v['energy_Ha']-s['energy_Ha']);maximum=max(maximum,error)
                        assert error<1e-8 and v['selected_ids']==s['selected_ids'];checks+=1
                    assert abs(values[-1]['energy_Ha']-r['final_energy_Ha'])<1e-8
                    row['readouts']=values;row['sampling_resources']=physical['sampling_resources']
                    row['valid_sector_fraction']=sum(n for x,n in counts.items() if valid_sector(x,info['n_qubits'],info['nalpha'],info['nbeta']))/max(1,sum(counts.values()))
                    norms=[x['metadata']['original_build_metadata']['retained_norm_squared'] for x in paid if x['metadata']]
                    row['minimum_retained_MPS_norm_squared']=min(norms)
                    row['approximation_qualified']=None
                    finals[method]=(counts,values,None)
                if classical_values is not None:
                    row['readouts']=classical_values
                    row['sampling_resources']=None
                    row['minimum_retained_MPS_norm_squared']=None
                    row['scope']='classical screened SCI; no quantum circuit/resources'
                    finals[method]=(None,classical_values,classical_order)
                rows.append(row)
            if 'RAACS-full-cost' in finals:
                ca,va,_=finals['RAACS-full-cost']
                for method,(cb,vb,orderb) in finals.items():
                    if method=='RAACS-full-cost':continue
                    for a,b in zip(va,vb):
                        d=min(a['actual_dimension'],b['actual_dimension'])
                        ea=selected(backend,info,ca,d)['energy_Ha']
                        eb=(selected(backend,info,cb,d) if orderb is None else named_ritz(backend,orderb,d))['energy_Ha'];checks+=2
                        comparison.append({'case':item['id'],'baseline':method,'R':a['R'],'RAACS_energy_Ha':a['energy_Ha'],
                            'baseline_energy_Ha':b['energy_Ha'],'RAACS_minus_baseline_mHa':1000*(a['energy_Ha']-b['energy_Ha']),
                            'matched_d':d,'RAACS_minus_baseline_matched_d_mHa':1000*(ea-eb),
                            'scope':('bounded classical screened SCI; post-hoc first-appearance d diagnostic' if orderb is not None else
                                'one independent final readout; approximate MPS; no ground-state reference')})
    assert len(rows)==planned
    failure=[r for r in rows if r['status']=='ENGINEERING_OR_INITIALIZATION_FAILURE_PRESERVED']
    # Initialization nonconvergence/resource caps are visible conditions, not
    # simulator-free algorithm losses. Unknown programming exceptions block GO.
    unexpected=[]
    for r in failure:
        record=json.loads((root/'cases'/r['case']/r['method']/'result.json').read_text())
        if not any(t in record['error'] for t in ('CCSD','Preserved amplitude initialization failure','resource cap','cap at gate boundary')):unexpected.append(r)
    status='MOLECULAR_STRONG_QUICK_INDEPENDENTLY_VALIDATED' if cfg['mode']=='quick' and not unexpected else 'MOLECULAR_STRONG_ALL_RECORDS_AUDITED'
    if unexpected:status='MOLECULAR_STRONG_AUDIT_UNEXPECTED_FAILURE'
    out={'status':status,'planned':planned,'records':len(rows),'independent_projected_solves':checks,
         'maximum_energy_parity_error_Ha':maximum,'known_linked_model_shots':known,'known_new_model_shots':new,
         'unknown_inflight_draws':unknown,'unexpected_failures':unexpected,'rows':rows,'comparisons':comparison,
         'noncanonical_MP2_independent_checks':amplitudes,
         'hardware_shots':0,'FCI_used':False,'whole_goal_complete':False,
         'analysis_code_sha256':file_digest(__file__),'manifest_sha256':file_digest(root/'manifest.json')}
    save_json(root/'validation.json',out);return out


def diagnostics(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text());folder=root/'same-circuit-bond-and-routing-v1'
    folder.mkdir(exist_ok=True);planned=[]
    binding={'execution_manifest_sha256':file_digest(root/'manifest.json'),'analysis_code_sha256':file_digest(__file__),
             'bonds':[32,64,128],'cutoff':1e-12,'construction_seconds_per_bond':120,'RSS_cap_bytes':2*2**30,
             'new_model_shots':0,'diagnostic_used_in_policy':False,'hardware_shots':0}
    manifest=folder/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==binding
    else:save_json(manifest,binding)
    with threadpool_limits(limits=1),no_full_state_guards():
        for item in cfg['queue']:
            if not (root/'cases'/item['id']/'physical-source.json').exists():continue
            for method in cfg['methods']:
                if method=='classical-SCI64':continue
                result_path=root/'cases'/item['id']/method/'result.json'
                if not result_path.exists():continue
                r,counts,paid,qc,physical=final_evidence(root,item,method)
                if qc is None:continue
                key=digest([item['id'],method,physical['source_database_sha256'],physical['final_circuit_id']])
                path=folder/(key+'.json');planned.append(key)
                if path.exists():continue
                from .molecular_strong_controls import boundary
                if why:=boundary(root,cfg):save_json(folder/'progress.json',{'status':why,'keys':planned});return
                start=perf_counter();rows=[];states=[];native=[]
                for target in ('all_to_all_cx','line_cx'):
                    try:
                        compiled,resource=compile_sampler(qc,target)
                        native.append({'status':'COMPLETED','resource':resource})
                    except Exception as exc:native.append({'target':target,'status':'COMPILE_FAILURE_PRESERVED','error':repr(exc)})
                for bond in (32,64,128):
                    try:
                        state,metadata=construct_mps(qc,bond,1e-12,120,2*2**30)
                        psi=state.psi.copy();psi.normalize();states.append((bond,psi))
                        rows.append({'bond':bond,'status':'COMPLETED','metadata':metadata})
                    except Exception as exc:rows.append({'bond':bond,'status':'RESOURCE_OR_ENGINEERING_FAILURE_PRESERVED','error':repr(exc)})
                    save_json(folder/'active-progress.json',{'case':item['id'],'method':method,'rows':rows})
                if states:
                    refbond,ref=states[-1]
                    for row in rows:
                        state=next((s for b,s in states if b==row['bond']),None)
                        if state is not None:
                            fidelity=float(abs(state.H@ref)**2)
                            assert -1e-8<=fidelity<=1+1e-8
                            row.update(normalized_overlap_squared_to_highest_completed_bond=fidelity,reference_bond=refbond)
                record={'case':item['id'],'method':method,'physical':{k:v for k,v in physical.items() if k!='preparation'},
                        'identical_circuit':True,'rows':rows,'native':native,'wall_seconds':perf_counter()-start,
                        'is_exact_error_certificate':False,'source_result_sha256':file_digest(result_path),**binding}
                save_json(path,record)
                print('same-circuit diagnostic',item['id'],method,flush=True)
    save_json(folder/'completion.json',{'status':'SAME_CIRCUIT_BOND_AND_ROUTING_EXECUTED','keys':planned,'unique':len(set(planned)),**binding})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--component',choices=['audit','diagnostics'],default='audit')
    a=p.parse_args()
    if a.component=='audit':
        r=audit(a.root);print(json.dumps({k:v for k,v in r.items() if k not in ('rows','comparisons')}))
    else:diagnostics(a.root)
