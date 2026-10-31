"""N30 same-low-shot integration of existing tuned control; N29 stays immutable."""
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter
import os
import sqlite3
import xml.etree.ElementTree as ET
import numpy as np
import ffsim
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from . import optimized_control_sampling as source
from . import finite_source_study as n29
from . import finite_source_readout as parent
from .joint_ci_sparse import solve
from .prefix_parameter_bindings import binding_count
from .slater_prefix_compression import parse_prefix
from paperstudy.lucj import interaction_pairs

LOCAL=source.LOCAL/'readout-v1'


def inventory(p,sub):
    oldp=load(n29.LOCAL/'protocol.json');oldsub=load(n29.LUCJ/'protocol.json')
    rows=[dict(r,origin='N29_REUSED_DRAW_NOT_NEW_SHOT') for r in parent.inventory(oldp,oldsub)]
    for cell in sub['cells']:
        rows.append(dict(case=cell['case'],method=source.METHOD,replicate=cell['replicate'],
            draw_id=cell.get('draw_id'),cell_id=cell['cell_id'],circuit_id=cell.get('circuit_id'),
            available=cell['status']=='PLANNED',origin='N30_NEW_TUNED_MODEL_DRAW'))
    if len(rows)!=72 or sum(r['available'] for r in rows)!=63:raise RuntimeError('72sourcecells/63available9missing required')
    return rows


def tuned_counts(p,sub,row):
    if not row['available']:return dict(status='MISSING_SAME_PROBLEM_SOURCE')
    with sqlite3.connect(source.SAMPLING/'draw-ledger.sqlite') as db:
        ledger=db.execute('SELECT status,raw_sha,decoded_sha FROM draws WHERE draw_id=?',(row['draw_id'],)).fetchone()
    if ledger is None:raise RuntimeError('Unregistered tuned draw')
    if ledger[0]!='COMPLETED':return dict(status='WAITING_FOR_FROZEN_INPUTS',ledger_status=ledger[0])
    c=next(c for c in sub['circuits'] if c['circuit_id']==row['circuit_id'])
    request=next(r for r in c['sampling_requests'] if r['id']==row['draw_id'])
    folder=source.SAMPLING/'collected/circuits'/c['circuit_id']/'output'
    f=folder/'result.json';result=load(f)
    actual=next(r for r in result['sampling_results'] if r['id']==request['id'])
    decoded=source.SAMPLING/'decoded'/(request['id']+'.npz');evidence=load(decoded.with_suffix('.json'))
    raw=folder/(request['id']+'.npz');d=p['physical'][row['case']]
    if result['status']!='EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED' or result['input_id']!=c['input_id'] or result['physical_problem_id']!=d['problem_id'] or result['source_QPY_SHA256']!=c['original_QPY_SHA256'] or abs(result['norm_squared']-1)>1e-8 or result['sector_dimension']!=c['sector_dimension'] or result['actual_QPU_shots']!=0:
        raise RuntimeError('Actual tuned physical source differs')
    if result['cupy']!=p['expected_CUDA_cupy'] or result['primitive_semantics']!='HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP':raise RuntimeError('Unqualified actual runtime')
    job=next(load(f) for f in source.SAMPLING.glob('lane*-job.json') if c['circuit_id'] in load(f)['circuit_ids'])
    if result['job_id']!=job['job_id'] or load(folder/'started.json')['job_id']!=job['job_id']:raise RuntimeError('Job owner mismatch')
    if any(actual[k]!=v for k,v in request.items()) or actual['actual_model_shots']!=1000 or load(folder/(request['id']+'-draw-started.json'))!=request:raise RuntimeError('Seed/shot binding mismatch')
    if sha256(raw.read_bytes())!=actual['counts_SHA256'] or ledger[1]!=actual['counts_SHA256'] or evidence['raw_counts_SHA256']!=ledger[1] or sha256(decoded.read_bytes())!=ledger[2] or ledger[2]!=evidence['decoded_SHA256'] or sha256(f.read_bytes())!=evidence['result_SHA256']:raise RuntimeError('Count/ledger/context digest mismatch')
    with np.load(raw,allow_pickle=False) as z:
        logical=source.legacy.decode(z['blocked_ids'],c['decoder'],c['norb'],c['nelec']);order=np.argsort(logical)
        expected_ids,expected_counts=logical[order],z['counts'][order]
    with np.load(decoded,allow_pickle=False) as z:ids,counts=z['blocked_ids'],z['counts']
    if not np.array_equal(ids,expected_ids) or not np.array_equal(counts,expected_counts):raise RuntimeError('Independent tuned decoder mismatch')
    n29.validate_counts(ids,counts,d['norb'],d['nelec'])
    return dict(status='COMPLETED',**parent.rank_record(ids,counts,d['norb'],d['nelec'][0]),
        payload_SHA256=evidence['decoded_SHA256'],seed=request['seed'],result_SHA256=evidence['result_SHA256'],
        sampling_role='NEW_TUNED_FULL_CIRCUIT_MODEL_EQUIVALENT',backend='A100')


def parameter_count(s,c):
    with np.load(s['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
    n=s['norb'];pairs=interaction_pairs(n);full=len(op.to_parameters(interaction_pairs=pairs))
    if full!=s['old_surrogate_coordinate_slots']:raise RuntimeError('Historical optimizer coordinate family differs')
    q=qpy.load(BytesIO(Path(c['original_QPY']).read_bytes()))[0]
    prefix=parse_prefix(q.data[0].operation.definition,n)
    suffix=len(op.orbital_rotations)*(n*n+len(pairs[0])+len(pairs[1]))
    declared,child,bindings=binding_count(prefix,{},suffix)
    if declared!=child or bindings:raise RuntimeError('No pruning in tuned control')
    return dict(optimizer_coordinate_slots=full,matched_prefix_suffix_declared_parameters=declared,
        known_initial_Slater_chart_reduction=full-declared,physical_Jacobian_rank=None)


def prepare():
    p=source.verified();parent.verified();space()
    if LOCAL.exists():raise RuntimeError('Existing N30 readout; no overwrite')
    test=ROOT/'qualification-N30-readout-preexecution-v1-tests.xml';suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=3 or any(int(s.get('failures',0)) or int(s.get('errors',0)) for s in suites):raise RuntimeError('All3 readout qualifications required')
    sub=load(source.SAMPLING/'protocol.json');rows=inventory(p,sub)
    tasks=[dict(r,arm=arm) for r in rows for arm in parent.ARMS]
    parameters={s['case']:parameter_count(s,next(c for c in sub['circuits'] if c['case']==s['case'])) for s in p['sources'] if s['status']=='AVAILABLE'}
    save(LOCAL/'protocol.json',dict(version='N30_ALL6_METHODS_COMMON_LOWSHOT_READOUT_V1',tasks=tasks,
        source_protocol_SHA256=sha256((source.LOCAL/'protocol.json').read_bytes()),code_SHA256=sha256(Path(__file__).read_bytes()),
        helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('finite_source_readout.py','prefix_parameter_bindings.py','joint_ci_sparse.py','joint_ci_sparse.c','slater_prefix_compression.py')},
        parameters=parameters,library_SHA256=p['library_SHA256'],max_R=32,max_D=1024,root_seconds=20,global_seconds=180,
        all_n29_old_costs_and_TN_training_retained=True,new_optimizer_calls=0,new_QPU_shots=0,goal_complete=False))
    print('N30_READOUT_FROZEN144_ENDPOINTS126AVAILABLE18MISSING',flush=True)


def verified():
    p=source.verified();parent.verified();q=load(LOCAL/'protocol.json')
    if sha256(Path(__file__).read_bytes())!=q['code_SHA256'] or sha256((source.LOCAL/'protocol.json').read_bytes())!=q['source_protocol_SHA256'] or sha256(Path(p['library']).read_bytes())!=q['library_SHA256']:raise RuntimeError('Frozen N30 readout drift')
    for name,h in q['helper_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen readout helper drift')
    return p,q,load(source.SAMPLING/'protocol.json')


def run():
    p,q,sub=verified();space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing readout intent; reconcile, never blind restart')
    cpu,wall=process_time(),perf_counter();rows=inventory(p,sub);records={}
    oldp,_=n29.verified();oldsub=load(n29.LUCJ/'protocol.json')
    for row in rows:
        key=row['draw_id'] if row['available'] else row['cell_id']
        records[key]=parent.audited_counts(oldp,oldsub,row) if row['origin'].startswith('N29_') else tuned_counts(p,sub,row)
    waiting=[r for r in records.values() if r['status']=='WAITING_FOR_FROZEN_INPUTS']
    if waiting:
        print('N30_WAITING_FOR_INPUTS_NO_EXECUTION_INTENT',len(waiting),flush=True);return
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    states=[];attempts=[]
    try:
        for task in q['tasks']:
            space()
            if perf_counter()-wall>q['global_seconds']:raise TimeoutError('N30 global CI cap')
            if sum(f.stat().st_size for f in source.LOCAL.rglob('*') if f.is_file())>p['new_run_cap_bytes']:raise OSError('Bounded new N30 output cap')
            row=dict(task)
            if not task['available']:row.update(status='MISSING_SAME_PROBLEM_SOURCE',energy_Ha=None)
            else:
                record=records[task['draw_id']]
                R=record['R32'] if task['arm']=='R32_CAP' else min(records[r['draw_id']]['R32'] for r in rows if r['available'] and r['case']==task['case'] and r['replicate']==task['replicate'])
                strings=sorted(record['rank'][:R]);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);d=p['physical'][task['case']]
                key=parent.basis_key(d,pairs,q['library_SHA256']);root=LOCAL/'roots'/key;meta=root.with_suffix('.json')
                oldmeta=parent.LOCAL/'roots'/(key+'.json')
                reused=meta if meta.exists() else oldmeta if oldmeta.exists() else None
                if reused:
                    cache=load(reused);new=False;cache_origin='N30_SAME_BASIS' if reused==meta else 'N29_SAME_PHYSICAL_BASIS'
                    if cache['basis_id']!=key or cache['status'] not in ('COMPLETED','FAILED_PRESERVED'):raise RuntimeError('Invalid root-cache record')
                else:
                    if root.with_suffix('.started.json').exists():raise RuntimeError('Partial root; no blind restart')
                    f=Path(d['local_integrals'])
                    if sha256(f.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals drift')
                    save(root.with_suffix('.started.json'),dict(basis_id=key,case=task['case'],pid=os.getpid()));tick=process_time()
                    try:
                        with np.load(f,allow_pickle=False) as z:result,v=solve(z['h1'],z['eri'],d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=q['root_seconds'],triangle_cap=8000000)
                        vector=root.with_suffix('.npz');write(vector,arrays(joint_pairs=pairs,coefficients=v))
                        cache=dict(status='COMPLETED',basis_id=key,result=result,root_path=str(vector),root_SHA256=sha256(vector.read_bytes()))
                    except Exception as exc:cache=dict(status='FAILED_PRESERVED',basis_id=key,result=None,error=repr(exc))
                    cache['attempt_CPU_seconds']=process_time()-tick;save(meta,cache)
                    attempts.append(dict(basis_id=key,status=cache['status'],CPU_seconds=cache['attempt_CPU_seconds']));save(LOCAL/'attempts.json',attempts)
                    new=True;cache_origin='N30_NEW_ROOT'
                if cache['status']=='COMPLETED' and sha256(Path(cache['root_path']).read_bytes())!=cache['root_SHA256']:raise RuntimeError('Actual root hash mismatch')
                row.update(cache,R=R,actual_D=R*R,half_strings=strings,source_evidence=record,new_CI_attempt=new,cache_origin=cache_origin)
            states.append(row);save(LOCAL/'progress.json',dict(states=states))
        save(LOCAL/'terminal.json',dict(status='N30_ALL144_ENDPOINTS_TERMINAL',states=states,attempts=attempts,
            actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),failed_CI_attempts=sum(a['status']!='COMPLETED' for a in attempts),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
            new_QPU_shots=0,new_optimizers=0,goal_complete=False))
        print('N30_READOUT_TERMINAL144',len(attempts),'newCIattempts',flush=True)
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),completed_endpoints=len(states),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
