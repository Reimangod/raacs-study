"""N29 fresh finite draws; reuses qualified N17 exact materialized executor.

Only namespace, requests and seed/Shot configuration change for LUCJ. No gate,
decoder, CUDA kernel, quantum method or initialization is substituted.
"""
from pathlib import Path
from time import process_time,perf_counter
import json,xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from . import prefix_performance_sampling as legacy

LOCAL=ROOT/'runs/N29-FRESH-FINITE-SOURCE-v1'
LUCJ=LOCAL/'lucj'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N29-fresh-finite-source-v1'
N17=ROOT/'runs/N17-INDEPENDENT-PREFIX-PERFORMANCE-v1'
N28=ROOT/'runs/N28-PHASED-PAIR-STRESS-v1'
REPLICATES=(290101,290102,290103)
SHOTS=1000


def rng_seed(identity,replica):return int(digest(['N29_NEW_INDEPENDENT_RNG_V1',identity,replica])[:8],16)


def rebind_config(old,circuit_id):
    cfg=json.loads(json.dumps(old));cfg.pop('input_id',None)
    cfg['sampling_requests']=[dict(id=digest(['N29_NEW_DRAW',circuit_id,rep]),seed=rng_seed(circuit_id,rep),replicate_label=rep,shots=SHOTS) for rep in REPLICATES]
    cfg['input_id']=digest(cfg)
    return cfg


def markov_bits(n,k,angles,shots,rng):
    words=np.full(shots,(1<<k)-1,np.uint64)
    for e in angles:
        i,a=e['i'],e['a'];theta=e['theta']
        if not 0<=i<n or not 0<=a<n or i==a or not np.isfinite(theta):raise ValueError('Valid fixed pair gate required')
        active=((words>>np.uint64(i))&1)!=((words>>np.uint64(a))&1)
        chosen=active&(rng.random(shots)<np.sin(theta)**2)
        words[chosen]^=np.uint64((1<<i)|(1<<a))
    return words


def validate_counts(ids,counts,n,nelec,shots=SHOTS):
    ids=np.asarray(ids);counts=np.asarray(counts)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or counts.ndim!=1 or len(ids)!=len(counts) or len(set(map(int,ids)))!=len(ids) or counts.dtype.kind!='u' or np.any(counts==0) or int(counts.sum())!=shots:raise ValueError('Actual unique integer IDs/unsigned positive counts required')
    mask=(1<<n)-1
    if any(int(x)<0 or int(x)>>2*n or (int(x)&mask).bit_count()!=nelec[0] or (int(x)>>n).bit_count()!=nelec[1] for x in ids):raise ValueError('Wrong actual particle sector/width')


def configured_legacy():
    legacy.LOCAL=LUCJ;legacy.REMOTE=REMOTE;legacy.SHOTS=SHOTS
    return legacy


def verified():
    p=load(LOCAL/'protocol.json')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256((LUCJ/'protocol.json').read_bytes())!=p['lucj_protocol_SHA256']:raise RuntimeError('Frozen N29 adapter/context changed')
    if sha256((ROOT/'frontierloop/prefix_performance_sampling.py').read_bytes())!=p['legacy_adapter_SHA256'] or sha256((ROOT/'frontierloop/collect_prefix_performance.py').read_bytes())!=p['legacy_collector_SHA256']:raise RuntimeError('Qualified reuse driver changed')
    if sha256((N28/'terminal.json').read_bytes())!=p['N28_terminal_SHA256']:raise RuntimeError('Selected source context changed')
    m=configured_legacy();m.verified_protocol()
    return p,m


def prepare():
    storage=legacy.capacity()
    if LOCAL.exists():raise RuntimeError('Existing N29; preserve and reconcile')
    q=ROOT/'qualification-N29-preexecution-v1-tests.xml';suites=list(ET.parse(q).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=4 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise RuntimeError('All4N29 tests required')
    # Verify the entire original qualified execution inventory before reuse.
    old=legacy.verified_protocol();selected=[c for c in old['cells'] if c['method']=='ORIGINAL'];lookup={c['circuit_id']:c for c in old['circuits']}
    if len(selected)!=36 or sum(c['status']=='PLANNED' for c in selected)!=30:raise RuntimeError('All original L1/L2/L4 cells incl6missing required')
    stress=load(N28/'terminal.json');audit=load(ROOT/'analysis/N28-source-stress-summary.json');stressp=load(N28/'protocol.json')
    if audit['terminal_SHA256']!=sha256((N28/'terminal.json').read_bytes()) or audit['status']!='N28_ALL36_READOUTS24_SOURCES96_COMPILER_POINTS_AUDITED':raise RuntimeError('Audited N28 required')
    cpu_sources=[s for s in stress['sources'] if (s['scale'],s['pattern'])==(2,'REAL')]
    if len(cpu_sources)!=4:raise RuntimeError('Same fixed policy all4 required')
    physical={s['case']:dict(s['definition'],local_integrals=s['integrals']) for s in stressp['sources']}
    LOCAL.mkdir();LUCJ.mkdir()
    for name in ('worker.py','primitive_plan.py','base_kernel.py'):write(LUCJ/name,(N17/name).read_bytes())
    circuits=[];cells=[];seen={}
    for cell in selected:
        rep=REPLICATES[cell['replicate']-160101]
        if cell['status']!='PLANNED':
            row={k:v for k,v in cell.items() if k not in ('cell_id','replicate')};row.update(replicate=rep);row['cell_id']=digest(['N29',row]);cells.append(row);continue
        cid=cell['circuit_id'];original=lookup[cid]
        if original['problem_id']!=physical[original['case']]['problem_id']:raise RuntimeError('Same Hamiltonian/problem source required')
        if cid not in seen:
            for filename,hfield in [('config.json','config_SHA256'),('arrays.npz','arrays_SHA256')]:
                if sha256((N17/'circuits'/cid/filename).read_bytes())!=original[hfield]:raise RuntimeError('Actual primitive input drift')
            if sha256(Path(original['original_QPY']).read_bytes())!=original['original_QPY_SHA256']:raise RuntimeError('Actual logical QPY changed')
            folder=LUCJ/'circuits'/cid;folder.mkdir(parents=True)
            write(folder/'arrays.npz',(N17/'circuits'/cid/'arrays.npz').read_bytes())
            cfg=rebind_config(load(N17/'circuits'/cid/'config.json'),cid);save(folder/'config.json',cfg)
            copied=dict(original,config_SHA256=sha256((folder/'config.json').read_bytes()),input_id=cfg['input_id'],sampling_requests=cfg['sampling_requests'])
            circuits.append(copied);seen[cid]=copied
        row=dict(cell,replicate=rep,draw_id=next(r['id'] for r in seen[cid]['sampling_requests'] if r['replicate_label']==rep))
        row['cell_id']=digest(['N29',cell['cell_id'],rep]);cells.append(row)
    lanes=[[],[]];loads=[0,0]
    for c in sorted((c for c in circuits if c['backend']=='A100'),key=lambda c:(-c['estimated_work'],c['circuit_id'])):
        lane=min(range(2),key=lambda i:loads[i]);lanes[lane].append(c['circuit_id']);loads[lane]+=c['estimated_work']
    if len(circuits)!=10 or sum(c['backend']=='M4' for c in circuits)!=1:raise RuntimeError('Tenoriginalcircuits/oneMacnineGPU required')
    sub=dict(old,version='N29_FULL_LUCJ_NEW1000_REQUESTS_QUALIFIED_N17_EXECUTOR_V1',cells=cells,circuits=circuits,GPU_lanes=lanes,
        replicate_labels=list(REPLICATES),shots_per_draw=SHOTS,maximum_new_model_shots=30000,maximum_logical_model_shots=30000,
        RNG_rule='N29_PREFIX_SHA256_CANONICAL_CIRCUIT_AND_NEW_REPLICATE_FIRST32',runtime_SHOTS_override=SHOTS,
        namespace_override=str(LUCJ),remote_override=REMOTE,selection='ORIGINAL_FULL_LUCJ_ONLY_NO_COMPRESSION_NO_OLD_COUNT_REUSE',
        N29_adapter_SHA256=sha256(Path(__file__).read_bytes()),readouts=['NEW_N29_SOURCE_FREQUENCY_R32','NEW_N29_COMMON_ACTUAL_R'])
    legacy.init_ledger(LUCJ/'draw-ledger.sqlite',cells,circuits);save(LUCJ/'protocol.json',sub)
    cpu_requests=[];seeds=[]
    for s in cpu_sources:
        if sha256(Path(s['source_file']).read_bytes())!=s['source_SHA256']:raise RuntimeError('Fixed candidate source changed')
        for method in ('PAIR_SCALE2_REAL','MARKOV_SCALE2'):
            identity=digest(['N29',s['case'],method,s['source_SHA256']])
            for rep in REPLICATES:
                r=dict(case=s['case'],method=method,replicate=rep,draw_id=digest([identity,rep]),seed=rng_seed(identity,rep),shots=SHOTS)
                seeds.append(r['seed']);cpu_requests.append(r)
    seeds+=[r['seed'] for c in circuits for r in c['sampling_requests']]
    if len(seeds)!=54 or len(set(seeds))!=54:raise RuntimeError('Newseed collision')
    previous={r['seed'] for c in old['circuits'] for r in c['sampling_requests']}
    if previous&set(seeds):raise RuntimeError('Seed reused from previous N17')
    save(LOCAL/'protocol.json',dict(version='N29_FIXED_SCALE2_REAL_FRESH_FINITE_DRAWS_V1',cpu_sources=cpu_sources,cpu_requests=cpu_requests,physical=physical,
        replicates=list(REPLICATES),shots_per_draw=SHOTS,expected_cells=60,expected_available=54,expected_missing=6,
        maximum_pair_model_shots=12000,maximum_LUCJ_model_shots=30000,maximum_Markov_classical_draws=12000,
        code_SHA256=sha256(Path(__file__).read_bytes()),legacy_adapter_SHA256=old['code_SHA256'],
        legacy_collector_SHA256=sha256((ROOT/'frontierloop/collect_prefix_performance.py').read_bytes()),
        lucj_protocol_SHA256=sha256((LUCJ/'protocol.json').read_bytes()),N28_terminal_SHA256=sha256((N28/'terminal.json').read_bytes()),
        plan_SHA256=sha256((ROOT/'analysis/N29_FINITE_SHOT_NEXT_JA.md').read_bytes()),qualification_SHA256=sha256(q.read_bytes()),
        selection='GLOBAL_SCALE2_REAL_DEVELOPMENT_SELECTED_AFTER_N28_BEFORE_FRESH_DRAWS_NOT_GEOMETRY_SPECIFIC',
        expected_GPU_cupy='13.6.0',maximum_GPU_workers=2,maximum_nominal_GPU_hours=2,
        unchanged_legacy_submit_job_name_cosmetic='frontier-N17-prefix-LANE_ACTUAL_NAMESPACE_IS_N29',
        library=stressp['library'],library_SHA256=stressp['library_SHA256'],
        initialization_cost_context_SHA256=sha256((ROOT/'analysis/N25-lowshot-pair-summary.json').read_bytes()),
        old_initialization_and_N28_search_costs_retained=True,storage_before=storage,
        CPU_stage_wall_cap_seconds=120,new_run_cap_bytes=32*2**20,new_QPU_shots=0,goal_complete=False))
    print('N29_FROZEN60CELLS54AVAILABLE6MISSING_42KMODEL12KCLASSICAL_MAXIMUM',flush=True)


def m4():
    p,m=verified();m.capacity()
    marker=LOCAL/'cpu-started.json'
    if marker.exists():raise RuntimeError('Existing CPU attempt; no blind restart')
    save(marker,dict(pid=__import__('os').getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[]
    try:
        for s in p['cpu_sources']:
            case=s['case'];definition=p['physical'][case];n=definition['norb'];k=definition['nelec'][0]
            if sha256(Path(s['source_file']).read_bytes())!=s['source_SHA256']:raise RuntimeError('Fixed selected source changed')
            with np.load(s['source_file'],allow_pickle=False) as z:ids=z['half_strings'];v=z['amplitudes'];prob=z['probabilities']
            if any(int(x).bit_count()!=k for x in ids) or not np.array_equal(abs(v)**2,prob) or abs(float(prob.sum())-1)>1e-10:raise RuntimeError('Actual pair probability/sector context differs')
            for request in [r for r in p['cpu_requests'] if r['case']==case]:
                space()
                if perf_counter()-wall>p['CPU_stage_wall_cap_seconds']:raise TimeoutError('CPU source-stage cap')
                if sum(x.stat().st_size for x in LOCAL.rglob('*') if x.is_file())>p['new_run_cap_bytes']:raise OSError('Bounded new N29 run cap')
                out=LOCAL/'cpu-draws'/request['draw_id'];out.mkdir(parents=True);save(out/'started.json',request)
                rng=np.random.default_rng(request['seed'])
                half=rng.choice(ids,size=SHOTS,p=prob) if request['method']=='PAIR_SCALE2_REAL' else markov_bits(n,k,s['angles'],SHOTS,rng)
                full=half|(half<<np.uint64(n));values,counts=np.unique(full,return_counts=True);counts=counts.astype(np.uint64)
                validate_counts(values,counts,n,(k,k));path=out/'counts.npz';write(path,arrays(blocked_ids=values,counts=counts))
                record=dict(**request,status='COMPLETED',payload=str(path),payload_SHA256=sha256(path.read_bytes()),norb=n,nelec=[k,k],problem_id=definition['problem_id'],
                    sample_role='EXACT_PAIR_MODEL_EQUIVALENT' if request['method']=='PAIR_SCALE2_REAL' else 'POLYNOMIAL_CLASSICAL_PER_GATE_MARKOV',
                    new_model_shots=SHOTS if request['method']=='PAIR_SCALE2_REAL' else 0,
                    new_classical_draws=SHOTS if request['method']=='MARKOV_SCALE2' else 0,new_QPU_shots=0)
                save(out/'result.json',record);rows.append(record);save(LOCAL/'cpu-draws.json',rows)
            print('N29_CPU_PAIR_MARKOV',case,'6newdraws',flush=True)
        # This is the exact old qualified numpy primitive executor, fresh requests.
        m.m4()
        save(LOCAL/'cpu-terminal.json',dict(status='N29_ALL24_PAIR_MARKOV_AND3_NAPH_LUCJ_DRAWS_TERMINAL',draws=rows,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_pair_model_shots=12000,new_Markov_classical_draws=12000,
            Naph_LUCJ_model_shots=3000,new_QPU_shots=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'cpu-failed.json',dict(error=repr(exc),completed_pair_Markov_draws=len(rows),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


def submit():
    p,m=verified();m.submit()


def status():
    p,m=verified();m.status()
    rows=load(LOCAL/'cpu-draws.json') if (LOCAL/'cpu-draws.json').exists() else []
    print('N29_CPU_PROGRESS',len(rows),'of24','M4CPUterminal',(LOCAL/'cpu-terminal.json').exists(),flush=True)


def collect():
    p,m=verified()
    from . import collect_prefix_performance as c
    c.LOCAL=LUCJ;c.REMOTE=REMOTE;c.collect()
    for circuit in load(LUCJ/'protocol.json')['circuits']:
        if circuit['backend']!='A100':continue
        path=LUCJ/'collected/circuits'/circuit['circuit_id']/'output/result.json'
        if path.exists() and load(path)['cupy']!=p['expected_GPU_cupy']:raise RuntimeError('Runtime differs from actual CUDA qualification; preserve and audit before use')


if __name__=='__main__':
    import sys
    {'prepare':prepare,'m4':m4,'submit':submit,'status':status,'collect':collect}[sys.argv[1]]()
