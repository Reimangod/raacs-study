"""Actual fixed-source multi-orbital-basis acquisitions and bounded NOCI.

Counts are committed before projection; raw draws are shared by reference, not
recharged for anchors/R controls. Quantum simulator is count-only exact-small.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import asdict
import fcntl
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import traceback
import zlib
import numpy as np
from qiskit import qpy
from .common import ROOT,digest,file_digest,rng_for,save_json
from .resource_adaptive_sampler import integral_problem,allocation_counts
from .circuit_sampling_runtime import CircuitCountSampler,SamplingSpec
from .nonorthogonal_ci import NonorthogonalHamiltonian,occupied_columns,generalized_ground
from .multibasis_primitives import METHODS,construct_bases,measurement_rotation,source_preparation
from .ansatz import circuit_bytes,measurement_equivalent_circuit,resources
from .study.core import rss_peak

ARMS=('canonical','canonical_plus_anchors','multibasis','uniform_multibasis')
NAMESPACE='multiple-orbital-measurement-NOCI-v1'


def pack(x):return zlib.compress(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),6)
def unpack(x):return json.loads(zlib.decompress(x))


def configuration(mode):
    source=ROOT/'results/standard-v1'
    cases=['H2_R0.7','H4_R1.1'] if mode=='quick' else sorted(p.name for p in (source/'references').iterdir() if (p/'reference.json').exists())
    dependencies=['nonorthogonal_ci.py','multibasis_primitives.py','multibasis_study.py','circuit_sampling_runtime.py',
                  'mps_sampling.py','ansatz.py','resource_adaptive_sampler.py','integral_hamiltonian.py','common.py']
    return {'schema':NAMESPACE,'mode':mode,'source':str(source),'cases':cases,'methods':list(METHODS),'arms':list(ARMS),
            'M':[250,1000] if mode=='quick' else [250,1000,10000],
            'R':[8,16] if mode=='quick' else [8,32,64],'seeds':[101,211] if mode=='quick' else [101,211,307],
            'basis_objective_cap_per_stage':48,'rank_relative_cutoff':1e-9,
            'simulation':asdict(SamplingSpec(backend='exact_small',cache_entries=4)),
            'bounds':{'free_floor_bytes':int(2.75*2**30),'output_bytes':128*2**20,'rss_bytes':2*2**30,'case_seconds':1800},
            'implementation':{p:file_digest(ROOT/'benchmark'/p) for p in dependencies},
            'spec_sha256':file_digest(ROOT/'study-v1/MULTIBASIS_NOCI_V1.md'),
            'runtime_versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','qiskit','ffsim')},
            'FCI_used_for_selection':False,'hardware_shots':0,'paper_CDF_NOCI_reproduction':False}


def connect(root):
    db=sqlite3.connect(root/'events.sqlite');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS bases(case_id TEXT PRIMARY KEY,payload BLOB);
      CREATE TABLE IF NOT EXISTS circuits(case_id TEXT,method TEXT,basis INTEGER,qpy BLOB,payload BLOB,PRIMARY KEY(case_id,method,basis));
      CREATE TABLE IF NOT EXISTS acquisitions(id TEXT PRIMARY KEY,case_id TEXT,method TEXT,strategy TEXT,M INTEGER,rep INTEGER,status TEXT);
      CREATE TABLE IF NOT EXISTS draws(id TEXT PRIMARY KEY,acquisition TEXT,basis INTEGER,status TEXT,payload BLOB);
      CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,case_id TEXT,method TEXT,arm TEXT,M INTEGER,rep INTEGER,status TEXT,payload BLOB);
      CREATE TABLE IF NOT EXISTS readouts(task TEXT,R INTEGER,payload BLOB,PRIMARY KEY(task,R));
    ''');return db


def capacity(root,cfg):
    if shutil.disk_usage(root).free<cfg['bounds']['free_floor_bytes']:return 'PENDING_DISK_BOUNDARY'
    if rss_peak()>cfg['bounds']['rss_bytes']:return 'PENDING_RSS_BOUNDARY'
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>cfg['bounds']['output_bytes']:return 'PENDING_OUTPUT_BOUNDARY'


def valid(identifier,info):
    n=info['n_qubits']//2
    return all(sum(int(identifier)>>(2*n-1-2*p-spin)&1 for p in range(n))==info['nalpha' if spin==0 else 'nbeta'] for spin in (0,1))


def uniform_counts(norb,info,M,rng):
    out={}
    for _ in range(M):
        idx=sum(1<<(2*norb-1-2*int(p)-spin) for spin,k in ((0,info['nalpha']),(1,info['nbeta'])) for p in rng.choice(norb,k,replace=False))
        out[idx]=out.get(idx,0)+1
    return sorted(out),[out[x] for x in sorted(out)]


def acquisition(db,root,cfg,case,method,strategy,M,rep,info,sampler):
    effective_method='CLASSICAL_UNIFORM_NO_SAMPLER' if strategy=='uniform_multibasis' else method
    aid=digest([NAMESPACE,info['problem_id'],effective_method,strategy,M,rep])
    old=db.execute('SELECT status FROM acquisitions WHERE id=?',(aid,)).fetchone()
    if old and old[0]=='COMPLETED':return aid
    db.execute('INSERT OR IGNORE INTO acquisitions VALUES(?,?,?,?,?,?,?)',(aid,case,effective_method,strategy,M,rep,'RUNNING'));db.commit()
    ns=[M,0,0] if strategy=='canonical' else allocation_counts(M,[1,1,1])
    for basis,shots in enumerate(ns):
        if not shots:continue
        did=digest([aid,basis]);row=db.execute('SELECT status FROM draws WHERE id=?',(did,)).fetchone()
        if row:
            if row[0]=='COMPLETED':continue
            raise RuntimeError('Interrupted in-flight acquisition: preserve unknown work, no automatic redraw: '+did)
        if boundary:=capacity(root,cfg):raise RuntimeError(boundary)
        db.execute('INSERT INTO draws VALUES(?,?,?,?,?)',(did,aid,basis,'STARTED',None));db.commit()
        rng,seed=rng_for(NAMESPACE,'physical-draw',aid,basis);t=perf_counter()
        if strategy=='uniform_multibasis':
            ids,counts=uniform_counts(info['n_qubits']//2,info,shots,rng)
            meta={'raw_simulated_quantum_shots':0,'classical_uniform_samples':shots,'hardware_shots':0,
                  'gate_shot_proxy':0,'basis_switch_hardware_latency_seconds':None}
        else:
            raw=bytes(db.execute('SELECT qpy FROM circuits WHERE case_id=? AND method=? AND basis=?',(case,method,basis)).fetchone()[0])
            qc=qpy.load(BytesIO(raw))[0];ids,counts,meta=sampler.draw(qc,shots,seed)
            ids=[int(x) for x in ids];counts=[int(x) for x in counts]
            meta.update(raw_simulated_quantum_shots=int(sum(counts)),classical_uniform_samples=0,
                gate_shot_proxy=shots*meta['sampling_resources']['cnot_count'],basis_switch_hardware_latency_seconds=None)
        out={'basis':basis,'ids':ids,'counts':counts,'seed':str(seed),'requested':shots,'actual_raw':int(sum(counts)),
             'wall_seconds':perf_counter()-t,'metadata':meta,'valid_sector_raw':sum(c for i,c in zip(ids,counts) if valid(i,info))}
        db.execute('UPDATE draws SET status=?,payload=? WHERE id=?',('COMPLETED',pack(out),did));db.commit()
    db.execute('UPDATE acquisitions SET status=? WHERE id=?',('COMPLETED',aid));db.commit();return aid


def selected_nodes(draws,arm,hf,R):
    anchors=[(0,hf)] if arm=='canonical' else [(b,hf) for b in range(3)]
    frequencies={}
    for draw in draws:
        for i,c in zip(draw['ids'],draw['counts']):
            node=(draw['basis'],int(i));frequencies[node]=frequencies.get(node,0)+c
    rest=sorted((x for x in frequencies if x not in anchors),key=lambda x:(-frequencies[x],x))
    return (anchors+rest)[:R]


def case_run(db,root,cfg,case):
    start=perf_counter();backend,info=integral_problem(Path(cfg['source'])/'references'/case)
    h=NonorthogonalHamiltonian(backend.h1,backend.eri,backend.core);row=db.execute('SELECT payload FROM bases WHERE case_id=?',(case,)).fetchone()
    if row:
        basis=unpack(row[0]);rotations=[np.array(x) for x in basis['rotations']]
    else:
        rotations,record=construct_bases(h,info,cfg['basis_objective_cap_per_stage'])
        basis={'problem':info,'rotations':[u.tolist() for u in rotations],'construction':record,
               'factorization_max_error':h.factorization_error,'basis_pair_requests':h.pair_requests}
        db.execute('INSERT INTO bases VALUES(?,?)',(case,pack(basis)));db.commit()
    cache=OrderedDict();columns={};cache_hits=0
    def element(a,b):
        nonlocal cache_hits
        reverse=a<b;key=(b,a) if reverse else (a,b)
        if key in cache:
            cache_hits+=1;value=cache.pop(key);cache[key]=value
        else:
            for node in key:
                if node not in columns:columns[node]=occupied_columns(rotations[node[0]],node[1])
            value=h.pair(columns[key[0]],columns[key[1]]);cache[key]=value
            if len(cache)>50000:cache.popitem(last=False)
        return tuple(x.conjugate() for x in value) if reverse else value
    sampler=CircuitCountSampler(SamplingSpec(**cfg['simulation']))
    for method in cfg['methods']:
        qc,origin=source_preparation(case,method)
        if qc is not None:
            for b,u in enumerate(rotations):
                if db.execute('SELECT 1 FROM circuits WHERE case_id=? AND method=? AND basis=?',(case,method,b)).fetchone():continue
                prep=measurement_rotation(qc,u);sample,_=measurement_equivalent_circuit(prep)
                record={'origin':origin,'resources':resources(sample,0,0),'basis':b,'original_preparation_sha256':digest(list(circuit_bytes(qc))),
                        'resource_parameter_count_scope':'unknown in this combined numeric circuit; see source and orbital basis metadata',
                        'measurement_after_full_preparation':True}
                db.execute('INSERT INTO circuits VALUES(?,?,?,?,?)',(case,method,b,circuit_bytes(prep),pack(record)));db.commit()
        for M in cfg['M']:
            for rep in cfg['seeds']:
                for arm in cfg['arms']:
                    tid=digest([NAMESPACE,case,method,arm,M,rep]);old=db.execute('SELECT status FROM tasks WHERE id=?',(tid,)).fetchone()
                    if old and old[0] in ('COMPLETED','MISSING_SOURCE_METHOD','FAILED_ENGINEERING_PRESERVED'):continue
                    if boundary:=capacity(root,cfg):return boundary
                    if perf_counter()-start>cfg['bounds']['case_seconds']:return 'PENDING_CASE_SLICE'
                    if qc is None:
                        db.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?)',(tid,case,method,arm,M,rep,'MISSING_SOURCE_METHOD',pack(origin)));db.commit();continue
                    try:
                        strategy='canonical' if arm.startswith('canonical') else arm
                        aid=acquisition(db,root,cfg,case,method,strategy,M,rep,info,sampler)
                        draws=[unpack(x[0]) for x in db.execute('SELECT payload FROM draws WHERE acquisition=? ORDER BY basis',(aid,))]
                        # Preserve raw invalid samples in the draw ledger; selection
                        # rejects invalid sectors without replacement or free repair.
                        selected=[]
                        for draw in draws:
                            good=[(i,c) for i,c in zip(draw['ids'],draw['counts']) if valid(i,info)]
                            selected.append({**draw,'ids':[x[0] for x in good],'counts':[x[1] for x in good]})
                        nodes=selected_nodes(selected,arm,info['hf_id'],max(cfg['R']))
                        n=len(nodes);hh=np.empty((n,n),complex);ss=np.empty_like(hh);before=h.pair_requests;tick=perf_counter()
                        for i,a in enumerate(nodes):
                            for j,b in enumerate(nodes[:i+1]):
                                hij,sij=element(a,b);hh[i,j],ss[i,j]=hij,sij;hh[j,i],ss[j,i]=hij.conjugate(),sij.conjugate()
                        for R in cfg['R']:
                            cap=min(R,n);out,_=generalized_ground(hh[:cap,:cap],ss[:cap,:cap],cfg['rank_relative_cutoff'])
                            out.update(selected_nodes=nodes[:cap],new_pair_evaluations_for_shared_maxR_matrix=h.pair_requests-before,
                                       projection_wall_seconds_shared_maxR=perf_counter()-tick)
                            db.execute('INSERT OR REPLACE INTO readouts VALUES(?,?,?)',(tid,R,pack(out)))
                        payload={'acquisition_id':aid,'raw_M':sum(x['actual_raw'] for x in draws),
                            'valid_raw':sum(x['valid_sector_raw'] for x in draws),'classical_anchor_energy_Ha':basis['construction']['classical_anchor_ground']['energy_Ha'],
                            'raw_draws_reused_across_R_and_anchor_control':True,'hardware_shots':0}
                        db.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?)',(tid,case,method,arm,M,rep,'COMPLETED',pack(payload)));db.commit()
                    except Exception as exc:
                        db.rollback()
                        payload={'error':repr(exc),'traceback':traceback.format_exc(),'inflight_quantum_work_may_be_unknown':True}
                        db.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?)',(tid,case,method,arm,M,rep,'FAILED_ENGINEERING_PRESERVED',pack(payload)));db.commit()
                        raise
    save_json(root/'case-status'/f'{case}.json',{'status':'COMPLETED','case_wall_seconds_this_invocation':perf_counter()-start,
              'new_pair_evaluations_this_invocation':h.pair_requests,'pair_cache_hits_this_invocation':cache_hits,'peak_RSS':rss_peak()})
    return 'COMPLETED'


def run(root,mode):
    root=Path(root).resolve();cfg=configuration(mode);source=Path(cfg['source'])
    if root==source or source in root.parents or root in source.parents:raise ValueError('Protect immutable source')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=root/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=cfg:raise RuntimeError('Frozen multibasis code/spec drift')
        if mode=='standard':
            p=ROOT/'results/multibasis-NOCI-quick-v1/validation.json'
            if not p.exists() or json.loads(p.read_text()).get('status')!='ALL_QUICK_SPACES_INDEPENDENTLY_VALIDATED':raise RuntimeError('Actual independent quick qualification required')
        save_json(manifest,cfg)
        if not (root/'source-audit.json').exists():
            from .full_sqd_durable_cohort import verify_source
            verify_source(source,root)
        db=connect(root)
        for case in cfg['cases']:
            done=case_run(db,root,cfg,case)
            counts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())
            record={'status':'RUNNING' if done=='COMPLETED' else done,'tasks_by_status':counts,
                    'planned':len(cfg['cases'])*len(cfg['methods'])*len(cfg['arms'])*len(cfg['M'])*len(cfg['seeds']),
                    'hardware_shots':0,'whole_goal_complete':False}
            save_json(root/'execution-status.json',record);print(case,done,counts,flush=True)
            if done!='COMPLETED':db.close();return record
        record.update(status='ALL_TASKS_TERMINAL_PENDING_AUDIT');db.close();save_json(root/'execution-status.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','standard'],default='quick');p.add_argument('--output')
    a=p.parse_args();print(json.dumps(run(a.output or ROOT/f'results/multibasis-NOCI-{a.mode}-v1',a.mode)))
