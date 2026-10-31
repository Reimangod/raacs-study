"""N33 common CI across all seven sources; old N29/N30 evidence immutable."""
from pathlib import Path
from time import process_time,perf_counter
import os,sqlite3
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from . import analytical_source_sampling as source
from . import optimized_control_sampling as oldsource
from . import optimized_control_readout as oldread
from . import finite_source_study as n29
from . import finite_source_readout as parent
from .joint_ci_sparse import solve

LOCAL=source.LOCAL/'readout-v1'


def inventory(p,sub):
    op=oldsource.verified();oldsub=load(oldsource.SAMPLING/'protocol.json')
    rows=oldread.inventory(op,oldsub)
    for cell in sub['cells']:
        rows.append(dict(case=cell['case'],method=source.METHOD,replicate=cell['replicate'],draw_id=cell['draw_id'],
            cell_id=cell['cell_id'],circuit_id=cell['circuit_id'],available=True,origin='N33_NEW_ANALYTICAL_OPTIMIZED_L1_MODEL_DRAW'))
    if len(rows)!=84 or sum(r['available'] for r in rows)!=75:raise RuntimeError('All84 source cells/75available9missing required')
    return rows


def new_counts(p,sub,row):
    with sqlite3.connect(source.SAMPLING/'draw-ledger.sqlite') as db:
        ledger=db.execute('SELECT status,raw_sha,decoded_sha FROM draws WHERE draw_id=?',(row['draw_id'],)).fetchone()
    if ledger is None:raise RuntimeError('Unregistered actual new draw')
    if ledger[0]!='COMPLETED':return dict(status='WAITING_FOR_FROZEN_INPUTS')
    c=next(c for c in sub['circuits'] if c['circuit_id']==row['circuit_id']);request=next(r for r in c['sampling_requests'] if r['id']==row['draw_id'])
    prefix='collected/circuits' if c['backend']=='A100' else 'circuits'
    folder=source.SAMPLING/prefix/c['circuit_id']/'output';f=folder/'result.json';result=load(f)
    raw=folder/(request['id']+'.npz');decoded=source.SAMPLING/'decoded'/(request['id']+'.npz');evidence=load(decoded.with_suffix('.json'))
    expected='EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED' if c['backend']=='A100' else 'EXACT_MATERIALIZED_CIRCUIT_M4_COMPLETED'
    if result['status']!=expected or result['input_id']!=c['input_id'] or result['physical_problem_id']!=c['problem_id'] or result['source_QPY_SHA256']!=c['original_QPY_SHA256'] or result['sector_dimension']!=c['sector_dimension'] or abs(result['norm_squared']-1)>1e-8 or result['actual_QPU_shots']!=0:raise RuntimeError('New actual source binding mismatch')
    if c['backend']=='A100':
        job=next(load(f) for f in source.SAMPLING.glob('lane*-job.json') if c['circuit_id'] in load(f)['circuit_ids'])
        if result['cupy']!=p['expected_CUDA_cupy'] or result['job_id']!=job['job_id'] or load(folder/'started.json')['job_id']!=job['job_id']:raise RuntimeError('Actual CUDA runtime/job ownership differs')
        if result['primitive_semantics']!='HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP':raise RuntimeError('Wrong quantum executor semantics')
    actual=next(r for r in result['sampling_results'] if r['id']==request['id'])
    if any(actual[k]!=v for k,v in request.items()) or actual['actual_model_shots']!=1000 or load(folder/(request['id']+'-draw-started.json'))!=request:raise RuntimeError('RNG/shot context mismatch')
    if sha256(raw.read_bytes())!=ledger[1] or ledger[1]!=actual['counts_SHA256'] or evidence['raw_counts_SHA256']!=ledger[1] or evidence['decoded_SHA256']!=ledger[2] or sha256(decoded.read_bytes())!=ledger[2] or evidence['result_SHA256']!=sha256(f.read_bytes()):raise RuntimeError('Raw/decoded/result/ledger hash mismatch')
    with np.load(raw,allow_pickle=False) as z:
        ids=source.legacy.decode(z['blocked_ids'],c['decoder'],c['norb'],c['nelec']);order=np.argsort(ids);rawids,rawcounts=ids[order],z['counts'][order]
    with np.load(decoded,allow_pickle=False) as z:ids,counts=z['blocked_ids'],z['counts']
    if not np.array_equal(ids,rawids) or not np.array_equal(counts,rawcounts):raise RuntimeError('Independent decoder mismatch')
    n29.validate_counts(ids,counts,c['norb'],c['nelec'])
    return dict(status='COMPLETED',**parent.rank_record(ids,counts,c['norb'],c['nelec'][0]),payload_SHA256=ledger[2],seed=request['seed'],result_SHA256=sha256(f.read_bytes()),sampling_role='NEW_ANALYTICAL_OPT_L1_MODEL_EQUIVALENT',backend=c['backend'])


def records(p,sub,rows):
    oldp,_=n29.verified();oldsub=load(n29.LUCJ/'protocol.json');op=oldsource.verified();osub=load(oldsource.SAMPLING/'protocol.json');out={}
    for row in rows:
        if row['origin'].startswith('N33_'):r=new_counts(p,sub,row)
        elif row['origin'].startswith('N29_'):r=parent.audited_counts(oldp,oldsub,row)
        else:r=oldread.tuned_counts(op,osub,row)
        out[row['draw_id'] if row['available'] else row['cell_id']]=r
    return out


def prepare():
    p=source.verified();oldread.verified();space();sub=load(source.SAMPLING/'protocol.json');rows=inventory(p,sub)
    if LOCAL.exists():raise RuntimeError('Existing N33 readout; never overwrite')
    save(LOCAL/'protocol.json',dict(version='N33_ALL7_METHOD_COMMON_CI_V1',tasks=[dict(r,arm=arm) for r in rows for arm in parent.ARMS],
        source_protocol_SHA256=sha256((source.LOCAL/'protocol.json').read_bytes()),code_SHA256=sha256(Path(__file__).read_bytes()),
        old_terminal_SHA256=sha256((oldread.LOCAL/'terminal.json').read_bytes()),
        helpers={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('optimized_control_readout.py','finite_source_readout.py','joint_ci_sparse.py','joint_ci_sparse.c')},
        library_SHA256=p['library_SHA256'],max_R=32,max_D=1024,root_seconds=20,partition_seconds=180,
        no_new_optimizers_or_QPU=True,old_initialization_and_N32_optimization_and_qualification_costs_retained=True,goal_complete=False))
    print('N33_READOUT_FROZEN168_ENDPOINTS_M4_42_A100INPUT_126',flush=True)


def verified():
    p=source.verified();q=load(LOCAL/'protocol.json')
    if q['code_SHA256']!=sha256(Path(__file__).read_bytes()) or q['source_protocol_SHA256']!=sha256((source.LOCAL/'protocol.json').read_bytes()) or q['old_terminal_SHA256']!=sha256((oldread.LOCAL/'terminal.json').read_bytes()):raise RuntimeError('Frozen readout/context drift')
    if sha256(Path(p['library']).read_bytes())!=q['library_SHA256']:raise RuntimeError('Qualified CI library drift')
    for name,h in q['helpers'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen readout helper drift')
    return p,q,load(source.SAMPLING/'protocol.json')


def run(partition):
    if partition not in ('M4','A100_INPUTS'):raise ValueError('Explicit partition required')
    p,q,sub=verified();space();cpu,wall=process_time(),perf_counter()
    tasks=[t for t in q['tasks'] if (t['case']=='NAPH20-BLOCK-BOYS')==(partition=='M4')]
    cases={t['case'] for t in tasks};rows=[r for r in inventory(p,sub) if r['case'] in cases];counts=records(p,sub,rows)
    if any(r['status']=='WAITING_FOR_FROZEN_INPUTS' for r in counts.values()):
        print('N33_WAITING_NO_CI_INTENT',partition,flush=True);return
    marker=LOCAL/(partition+'-started.json')
    if marker.exists():raise RuntimeError('Partial/terminal partition; no blind restart')
    save(marker,dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    states=[];attempts=[]
    try:
        for task in tasks:
            space()
            if perf_counter()-wall>q['partition_seconds']:raise TimeoutError('Partition wall cap')
            if sum(f.stat().st_size for f in source.LOCAL.rglob('*') if f.is_file())>p['new_run_cap_bytes']:raise OSError('New N33 storage cap')
            row=dict(task)
            if not task['available']:row.update(status='MISSING_SAME_PROBLEM_SOURCE',energy_Ha=None)
            else:
                record=counts[task['draw_id']];d=p['physical'][task['case']]
                R=record['R32'] if task['arm']=='R32_CAP' else min(counts[r['draw_id']]['R32'] for r in rows if r['available'] and r['case']==task['case'] and r['replicate']==task['replicate'])
                strings=sorted(record['rank'][:R]);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                key=parent.basis_key(d,pairs,q['library_SHA256']);root=LOCAL/'roots'/key;meta=root.with_suffix('.json')
                candidates=[meta,oldread.LOCAL/'roots'/(key+'.json'),parent.LOCAL/'roots'/(key+'.json')]
                existing=next((f for f in candidates if f.exists()),None)
                if existing:
                    cache=load(existing);new=False;origin='N33_SAME_BASIS' if existing==meta else 'OLD_SAME_PHYSICAL_BASIS'
                    if cache['basis_id']!=key or cache['status'] not in ('COMPLETED','FAILED_PRESERVED'):raise RuntimeError('Cached root binding differs')
                else:
                    if root.with_suffix('.started.json').exists():raise RuntimeError('Partial root; no blind rerun')
                    f=Path(d['local_integrals'])
                    if sha256(f.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Integral drift')
                    save(root.with_suffix('.started.json'),dict(basis_id=key,case=task['case'],pid=os.getpid()));tick=process_time()
                    try:
                        with np.load(f,allow_pickle=False) as z:res,v=solve(z['h1'],z['eri'],d['core_energy'],tuple(d['nelec']),pairs,p['library'],seconds=q['root_seconds'],triangle_cap=8000000)
                        path=root.with_suffix('.npz');write(path,arrays(joint_pairs=pairs,coefficients=v))
                        cache=dict(status='COMPLETED',basis_id=key,result=res,root_path=str(path),root_SHA256=sha256(path.read_bytes()))
                    except Exception as exc:cache=dict(status='FAILED_PRESERVED',basis_id=key,result=None,error=repr(exc))
                    cache['attempt_CPU_seconds']=process_time()-tick;save(meta,cache)
                    attempts.append(dict(basis_id=key,status=cache['status'],CPU_seconds=cache['attempt_CPU_seconds']));new=True;origin='N33_NEW_ROOT'
                if cache['status']=='COMPLETED' and sha256(Path(cache['root_path']).read_bytes())!=cache['root_SHA256']:raise RuntimeError('Root bytes differ')
                row.update(cache,R=R,actual_D=R*R,half_strings=strings,source_evidence=record,new_CI_attempt=new,cache_origin=origin)
            states.append(row);save(LOCAL/(partition+'-progress.json'),dict(states=states,attempts=attempts))
        save(LOCAL/(partition+'-terminal.json'),dict(status='PARTITION_TERMINAL',partition=partition,states=states,attempts=attempts,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),goal_complete=False))
        print('N33_PARTITION_TERMINAL',partition,len(states),len(attempts),'newroots',flush=True)
    except Exception as exc:
        save(LOCAL/(partition+'-failed.json'),dict(error=repr(exc),completed=len(states),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
