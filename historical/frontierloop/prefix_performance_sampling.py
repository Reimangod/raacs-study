"""N17 independent, deduplicated model draws for the frozen N15 outputs.

No new compression policy or energy-guided selection. A draw's physical bytes,
logical decoder, problem, RNG and qualified executable are bound before draw.
"""
import argparse
from io import BytesIO
import math
from pathlib import Path
import shlex
import sqlite3
import subprocess
import tarfile
import types
from time import process_time, perf_counter

import numpy as np
from qiskit import qpy
from .paths import ROOT, OLD
from .io import load, save, write, sha256, digest, arrays, space

LOCAL = ROOT / 'runs/N17-INDEPENDENT-PREFIX-PERFORMANCE-v1'
N15 = ROOT / 'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1'
QUAL = ROOT / 'qualification/N16-qubit-primitive-GPU-v1'
REMOTE = '/share/reiwatanabe/raacs-frontier-loop-20261025/N17-independent-prefix-v1'
METHODS = ('ORIGINAL', 'STATE_OVERLAP_ROUND', 'ANGLE_ONLY_ROUND')
REPLICATES = (160101, 160102, 160103)
SHOTS = 100_000


def capacity():
    free=space()
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if used>1024**3:raise OSError('Research root cap exceeded; no deletion')
    return dict(free_bytes=free,root_bytes=used)


def validate_decoder(mapping, n):
    if len(mapping) != 2*n or sorted(mapping) != list(range(2*n)):
        raise ValueError('Decoder must be a wire permutation')
    if any(i//n != j//n for i,j in enumerate(mapping)):
        raise ValueError('Registered decoder must preserve each spin block')


def decode(ids, mapping, n, nelec):
    validate_decoder(mapping, n)
    values = np.asarray(ids, dtype=np.uint64)
    if np.any(values >> np.uint64(2*n)):
        raise ValueError('Physical IDs exceed circuit width')
    out = np.zeros_like(values)
    for logical, physical in enumerate(mapping):
        out |= ((values >> np.uint64(physical)) & np.uint64(1)) << np.uint64(logical)
    mask = (1 << n)-1
    if any((int(v)&mask).bit_count() != nelec[0] or (int(v)>>n).bit_count() != nelec[1] for v in out):
        raise ValueError('Decoded observation outside registered particle sector')
    if len(np.unique(out)) != len(out):
        raise ValueError('Physical-to-logical decoder must be bijective')
    return out


def rng_seed(circuit_id, replicate):
    # Labels, not shared common random uniforms across distinct circuits.
    return int(digest(['N17_INDEPENDENT_CIRCUIT_RNG_V1', circuit_id, replicate])[:8], 16)


def qualified_module():
    prepared = load(QUAL/'prepared.json')
    for name, key in [('worker.py','worker_SHA256'), ('primitive_plan.py','generated_plan_SHA256')]:
        if sha256((QUAL/name).read_bytes()) != prepared[key]:
            raise RuntimeError('Qualified executable changed')
    if load(QUAL/'collected.json')['status'] != 'N16_ALL3_NONADJACENT_QUBIT_GPU_PARITIES_PASSED':
        raise RuntimeError('Actual CUDA qualification missing')
    module = types.ModuleType('n17_qualified_plan')
    exec(compile((QUAL/'primitive_plan.py').read_text(), str(QUAL/'primitive_plan.py'), 'exec'), module.__dict__)
    return module


def db(path=None):
    conn = sqlite3.connect(path or LOCAL/'draw-ledger.sqlite', timeout=30)
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def init_ledger(path, cells, circuits):
    if Path(path).exists():
        raise RuntimeError('Existing ledger must not be replaced')
    with db(path) as c:
        c.executescript('''
        CREATE TABLE draws(draw_id TEXT PRIMARY KEY,circuit_id TEXT NOT NULL,
          replicate INTEGER NOT NULL,rng_seed INTEGER NOT NULL,authorized_shots INTEGER NOT NULL,
          status TEXT NOT NULL,actual_shots INTEGER,raw_sha TEXT,decoded_sha TEXT,
          UNIQUE(circuit_id,replicate));
        CREATE TABLE cells(cell_id TEXT PRIMARY KEY,case_id TEXT,source TEXT,method TEXT,
          replicate INTEGER,status TEXT NOT NULL,draw_id TEXT REFERENCES draws(draw_id));
        CREATE TABLE events(seq INTEGER PRIMARY KEY,event TEXT NOT NULL,draw_id TEXT,
          evidence_sha TEXT NOT NULL);
        ''')
        for circuit in circuits:
            for request in circuit['sampling_requests']:
                c.execute('INSERT INTO draws VALUES(?,?,?,?,?,?,NULL,NULL,NULL)',
                          (request['id'],circuit['circuit_id'],request['replicate_label'],request['seed'],request['shots'],'REGISTERED'))
        for cell in cells:
            c.execute('INSERT INTO cells VALUES(?,?,?,?,?,?,?)',
                      tuple(cell.get(k) for k in ('cell_id','case','source','method','replicate','status','draw_id')))


def ledger_transition(draw_id, status, evidence_sha):
    with db() as c:
        old = c.execute('SELECT status FROM draws WHERE draw_id=?',(draw_id,)).fetchone()
        if old is None or old[0] != 'REGISTERED' or status != 'STARTED':
            raise RuntimeError('Partial or terminal draw cannot be blindly restarted')
        c.execute('UPDATE draws SET status=? WHERE draw_id=?',(status,draw_id))
        c.execute('INSERT INTO events(event,draw_id,evidence_sha) VALUES(?,?,?)',(status,draw_id,evidence_sha))


def prepare():
    if LOCAL.exists():
        raise RuntimeError('Existing N17 preparation; inspect, never overwrite')
    storage=capacity(); cpu,tick=process_time(),perf_counter(); module=qualified_module()
    spec=load(N15/'protocol.json'); raw=load(N15/'result.json')
    if len(spec['sources']) != 12 or len(raw['rows']) != 72:
        raise RuntimeError('Incomplete frozen source inventory')
    LOCAL.mkdir(); write(LOCAL/'worker.py',(QUAL/'worker.py').read_bytes())
    write(LOCAL/'primitive_plan.py',(QUAL/'primitive_plan.py').read_bytes())
    base=OLD/'paperstudy/gpu_lucj_engine.py'
    if sha256(base.read_bytes()) != load(QUAL/'prepared.json')['base_kernel_SHA256']:
        raise RuntimeError('Qualified CUDA source drift')
    write(LOCAL/'base_kernel.py',base.read_bytes())
    cells=[]; unique={}; seeds=set()
    for source in spec['sources']:
        case,method=source['case'],source['method']
        for policy in METHODS:
            cellbase=dict(case=case,source=method,method=policy,problem_id=source['problem_id'])
            if source['status'] != 'AVAILABLE':
                for rep in REPLICATES:
                    cell=dict(**cellbase,replicate=rep,status='MISSING_SAME_PROBLEM_SOURCE')
                    cell['cell_id']=digest(cell); cells.append(cell)
                continue
            folder=N15/case/method
            row=next(r for r in raw['rows'] if r['case']==case and r['method']==method and r['tier']==.01 and r['policy']==('STATE_OVERLAP_ROUND' if policy=='ORIGINAL' else policy))
            original=policy=='ORIGINAL' or not row['adopted']
            path=folder/'source-logical.qpy' if original else folder/row['point_identity']/'actual-virtual-swap.qpy'
            mapping=list(range(2*source['norb'])) if original else row['decoder_logical_to_physical']
            validate_decoder(mapping,source['norb'])
            circuit=qpy.load(BytesIO(path.read_bytes()))[0]
            config,values=module.plan(circuit,source['norb'],tuple(source['nelec']))
            circuit_id=digest(dict(problem_id=source['problem_id'],QPY_SHA256=sha256(path.read_bytes()),
                decoder=mapping,qualified_plan_SHA256=sha256((LOCAL/'primitive_plan.py').read_bytes())))
            output=row['exact_control'] if policy=='ORIGINAL' else row['selected']
            for role in ('preparation','sampling'):
                record=output[role]
                if sha256((N15/record['path']).read_bytes())!=record['SHA256']:
                    raise RuntimeError('Registered compiled resource evidence changed')
            if circuit_id not in unique:
                d=LOCAL/'circuits'/circuit_id;d.mkdir(parents=True)
                payload=arrays(**values);write(d/'arrays.npz',payload)
                requests=[]
                for rep in REPLICATES:
                    seed=rng_seed(circuit_id,rep)
                    if seed in seeds:raise RuntimeError('32-bit independent RNG seed collision')
                    seeds.add(seed);requests.append(dict(id=digest([circuit_id,rep]),seed=seed,replicate_label=rep,shots=SHOTS))
                config.update(arrays_SHA256=sha256(payload),worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),
                    base_kernel_SHA256=sha256(base.read_bytes()),source_QPY_SHA256=sha256(path.read_bytes()),
                    physical_problem_id=source['problem_id'],sampling_requests=requests)
                config['input_id']=digest(config);save(d/'config.json',config)
                dimension=math.prod(math.comb(source['norb'],e) for e in source['nelec'])
                unique[circuit_id]=dict(circuit_id=circuit_id,case=case,source=method,problem_id=source['problem_id'],
                    norb=source['norb'],nelec=source['nelec'],physical_SHA256=source['physical_SHA256'],
                    original_QPY=str(path),original_QPY_SHA256=sha256(path.read_bytes()),decoder=mapping,
                    config_SHA256=sha256((d/'config.json').read_bytes()),input_id=config['input_id'],arrays_SHA256=sha256(payload),
                    sampling_requests=requests,sector_dimension=dimension,primitive_count=len(config['ops']),
                    estimated_work=dimension*len(config['ops']),backend='M4' if source['norb']==10 else 'A100')
            for rep in REPLICATES:
                cell=dict(**cellbase,replicate=rep,status='PLANNED',circuit_id=circuit_id,
                    draw_id=next(r['id'] for r in unique[circuit_id]['sampling_requests'] if r['replicate_label']==rep),
                    selected_resources=output['resources'],compiled_evidence=output,
                    source_compiler_frontier=row['candidate_compiler_outputs'] if not row['actions'] else None,
                    original_fallback=original,preparation_infidelity=0. if original else row['adopted_output_many_body_infidelity'])
                cell['cell_id']=digest([case,method,policy,rep]);cells.append(cell)
    circuits=list(unique.values());lanes=[[],[]];loads=[0,0]
    for c in sorted((c for c in circuits if c['backend']=='A100'),key=lambda c:(-c['estimated_work'],c['circuit_id'])):
        lane=min(range(2),key=lambda i:loads[i]);lanes[lane].append(c['circuit_id']);loads[lane]+=c['estimated_work']
    if len(cells)!=108 or sum(c['status']=='PLANNED' for c in cells)!=90:
        raise RuntimeError('All 108 logical cells required')
    protocol=dict(version='N17_INDEPENDENT_PREFIX_PERFORMANCE_V1',cells=cells,circuits=circuits,GPU_lanes=lanes,
        replicate_labels=list(REPLICATES),RNG_rule='SHA256_CIRCUIT_ID_AND_REPLICATE_LABEL_FIRST32_BITS',
        independent_seed_per_distinct_circuit=True,shots_per_draw=SHOTS,maximum_new_model_shots=len(circuits)*len(REPLICATES)*SHOTS,
        maximum_logical_model_shots=9_000_000,new_QPU_shots=0,maximum_GPU_workers=2,GPU_wall_cap_seconds_per_lane=3600,
        readouts=['SOURCE_FREQUENCY_R128_CARTESIAN','OBSERVED_POOL_CONSTRAINED_HCI_JOINT_D16384'],
        CI=dict(spin_penalty_Ha=.1,root_wall_cap=75,threads=1,orbital_optimization=False,
                native_cutoff=5e-4,native_iterations=10,native_wall_cap=60,unrestricted_HF_controls_required=True),
        selection='N15_ALREADY_SELECTED_TIER_0p01_NO_NEW_ENERGY_OR_COUNTS_INPUT',
        full_source_compiler_frontier_required_in_analysis=True,parameters_binding_SHA256=sha256((ROOT/'analysis/N15-declared-coordinate-bindings.json').read_bytes()),
        N15_protocol_SHA256=sha256((N15/'protocol.json').read_bytes()),N15_result_SHA256=sha256((N15/'result.json').read_bytes()),
        qualification_SHA256=sha256((QUAL/'collected.json').read_bytes()),
        code_SHA256=sha256(Path(__file__).read_bytes()),worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),
        plan_SHA256=sha256((LOCAL/'primitive_plan.py').read_bytes()),base_SHA256=sha256(base.read_bytes()),
        tests_SHA256=sha256((ROOT/'qualification-N17-sampling-v1-tests.xml').read_bytes()),
        storage_before=storage,prepare_CPU_seconds=process_time()-cpu,prepare_wall_seconds=perf_counter()-tick,goal_complete=False)
    init_ledger(LOCAL/'draw-ledger.sqlite',cells,circuits)
    save(LOCAL/'protocol.json',protocol)
    print('N17_FROZEN',len(circuits),'unique circuits',len(circuits)*3,'unique draws',protocol['maximum_new_model_shots'],'authorized model shots',flush=True)


def verified_protocol():
    spec=load(LOCAL/'protocol.json')
    if sha256(Path(__file__).read_bytes())!=spec['code_SHA256']:
        raise RuntimeError('Frozen N17 implementation changed')
    for name,key in [('worker.py','worker_SHA256'),('primitive_plan.py','plan_SHA256'),('base_kernel.py','base_SHA256')]:
        if sha256((LOCAL/name).read_bytes())!=spec[key]:raise RuntimeError('Frozen N17 executable changed')
    for c in spec['circuits']:
        d=LOCAL/'circuits'/c['circuit_id']
        if sha256((d/'config.json').read_bytes())!=c['config_SHA256'] or sha256((d/'arrays.npz').read_bytes())!=c['arrays_SHA256']:
            raise RuntimeError('Frozen N17 input changed')
    return spec


def accept_result(circuit, output):
    result=load(output/'result.json')
    expected_status='EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED' if circuit['backend']=='A100' else 'EXACT_MATERIALIZED_CIRCUIT_M4_COMPLETED'
    if result['status']!=expected_status or result['input_id']!=circuit['input_id'] or result['physical_problem_id']!=circuit['problem_id'] or result['source_QPY_SHA256']!=circuit['original_QPY_SHA256'] or abs(result['norm_squared']-1)>1e-8:
        raise RuntimeError('Actual draw result binding failed')
    if result['actual_QPU_shots']!=0 or result['sector_dimension']!=circuit['sector_dimension']:
        raise RuntimeError('Unexpected physical execution context')
    if len(result['sampling_results'])!=3:raise RuntimeError('Incomplete replicate set')
    for request in circuit['sampling_requests']:
        actual=next(r for r in result['sampling_results'] if r['id']==request['id'])
        if any(actual[k]!=request[k] for k in request) or actual['actual_model_shots']!=SHOTS:
            raise RuntimeError('RNG/shot request mismatch')
        raw=output/(request['id']+'.npz')
        if sha256(raw.read_bytes())!=actual['counts_SHA256']:
            raise RuntimeError('Actual raw physical count hash mismatch')
        with np.load(raw,allow_pickle=False) as z:ids=z['blocked_ids'];counts=z['counts']
        if counts.dtype.kind!='u' or np.any(counts==0) or len(counts)!=len(ids) or len(np.unique(ids))!=len(ids) or int(counts.sum())!=SHOTS:
            raise RuntimeError('Invalid raw count distribution')
        logical=decode(ids,circuit['decoder'],circuit['norb'],circuit['nelec']);order=np.argsort(logical)
        blob=arrays(blocked_ids=logical[order],counts=counts[order])
        decoded=LOCAL/'decoded'/request['id'];path=decoded.with_suffix('.npz')
        if path.exists() and path.read_bytes()!=blob:raise RuntimeError('Decoded evidence collision')
        write(path,blob)
        evidence=dict(draw_id=request['id'],circuit_id=circuit['circuit_id'],raw_counts_SHA256=sha256(raw.read_bytes()),
            decoded_SHA256=sha256(blob),decoder=circuit['decoder'],actual_shots=SHOTS,new_quantum_shots_for_decoding=0,
            result_SHA256=sha256((output/'result.json').read_bytes()))
        save(decoded.with_suffix('.json'),evidence)
        with db() as c:
            old=c.execute('SELECT status,raw_sha,decoded_sha FROM draws WHERE draw_id=?',(request['id'],)).fetchone()
            if old[0]=='COMPLETED':
                if old[1:]!=(evidence['raw_counts_SHA256'],evidence['decoded_SHA256']):raise RuntimeError('Terminal ledger evidence changed')
                continue
            c.execute('UPDATE draws SET status=?,actual_shots=?,raw_sha=?,decoded_sha=? WHERE draw_id=?',
                ('COMPLETED',SHOTS,evidence['raw_counts_SHA256'],evidence['decoded_SHA256'],request['id']))
            c.execute('UPDATE cells SET status=? WHERE draw_id=?',('COMPLETED',request['id']))
            c.execute('INSERT INTO events(event,draw_id,evidence_sha) VALUES(?,?,?)',('COMPLETED',request['id'],digest(evidence)))


def m4():
    spec=verified_protocol();capacity();module=qualified_module()
    for circuit in spec['circuits']:
        if circuit['backend']!='M4':continue
        d=LOCAL/'circuits'/circuit['circuit_id'];output=d/'output';output.mkdir(exist_ok=True)
        if (output/'result.json').exists():accept_result(circuit,output);continue
        if (output/'started.json').exists():raise RuntimeError('Partial local state work; reconcile, no blind rerun')
        save(output/'started.json',dict(input_id=circuit['input_id'],backend='M4',PID=__import__('os').getpid()))
        for request in circuit['sampling_requests']:ledger_transition(request['id'],'STARTED',sha256((output/'started.json').read_bytes()))
        config=load(d/'config.json');cpu,tick=process_time(),perf_counter()
        with np.load(d/'arrays.npz',allow_pickle=False) as z:values={k:z[k] for k in z.files}
        state=module.numpy_state(config,values);norm=float(np.vdot(state,state).real)
        if abs(norm-1)>1e-8:raise RuntimeError('M4 norm failed')
        cdf=np.cumsum(abs(state)**2);cdf/=cdf[-1];del state
        sa,sb=[module.strings(circuit['norb'],e) for e in circuit['nelec']];records=[]
        for request in circuit['sampling_requests']:
            save(output/(request['id']+'-draw-started.json'),request)
            sampled=np.searchsorted(cdf,np.random.RandomState(request['seed']).random_sample(request['shots']))
            index,counts=np.unique(sampled,return_counts=True)
            ids=sa[index//len(sb)].astype(np.uint64)|(sb[index%len(sb)].astype(np.uint64)<<np.uint64(circuit['norb']))
            path=output/(request['id']+'.npz');write(path,arrays(blocked_ids=ids,counts=counts.astype(np.uint64)))
            record=dict(**request,actual_model_shots=int(counts.sum()),counts_SHA256=sha256(path.read_bytes()))
            save(output/(request['id']+'-draw-completed.json'),record);records.append(record)
        save(output/'result.json',dict(status='EXACT_MATERIALIZED_CIRCUIT_M4_COMPLETED',input_id=circuit['input_id'],
            physical_problem_id=circuit['problem_id'],source_QPY_SHA256=circuit['original_QPY_SHA256'],
            norm_squared=norm,sector_dimension=circuit['sector_dimension'],sampling_results=records,
            whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
            actual_QPU_shots=0,statevector_saved=False,backend='M4_NUMPY_EXACT_SECTOR'))
        accept_result(circuit,output);print('N17_M4_COMPLETED',circuit['circuit_id'],flush=True)


def submit():
    from benchmark.gpu_remote import ssh, send
    from paperstudy.gpu_qualification import PYTHON, ENV
    spec=verified_protocol();capacity()
    if (LOCAL/'staging-intent.json').exists():raise RuntimeError('Existing staging/submission; reconcile only')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=30).strip()!='NEW':raise RuntimeError('Existing remote work')
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
    save(LOCAL/'pre-dispatch-queue.json',dict(raw=queue,other_jobs_not_modified=True))
    files=['worker.py','base_kernel.py']
    for circuit in spec['circuits']:
        if circuit['backend']=='A100':
            files.extend('circuits/'+circuit['circuit_id']+'/'+f for f in ['config.json','arrays.npz'])
    for lane,ids in enumerate(spec['GPU_lanes']):
        commands=[shlex.join([PYTHON,'-u',REMOTE+'/worker.py',REMOTE+'/circuits/'+cid+'/config.json',REMOTE+'/circuits/'+cid+'/arrays.npz',REMOTE+'/circuits/'+cid+'/output']) for cid in ids]
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
            'export CUDA_PATH='+ENV+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
            'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
            'export CUPY_ACCELERATORS=""',*commands])+'\n'
        write(LOCAL/f'lane{lane}.sbatch',script.encode());files.append(f'lane{lane}.sbatch')
    manifest={name:sha256((LOCAL/name).read_bytes()) for name in files}
    save(LOCAL/'staging-manifest.json',manifest);files.append('staging-manifest.json')
    archive=LOCAL/'staging.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in files:tar.add(LOCAL/name,arcname=name,recursive=False)
    if archive.stat().st_size>64*2**20:raise RuntimeError('Bounded staging archive exceeded')
    save(LOCAL/'staging-intent.json',dict(remote=REMOTE,archive_SHA256=sha256(archive.read_bytes()),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    ssh('mkdir '+shlex.quote(REMOTE),timeout=30);send(archive,REMOTE+'/staging.tar.gz')
    extractor="import tarfile,pathlib,hashlib,json; r=pathlib.Path("+repr(REMOTE)+"); p=r/'staging.tar.gz'; assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(sha256(archive.read_bytes()))+"; t=tarfile.open(p); ms=t.getmembers(); assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in ms); t.extractall(r); expected=json.loads((r/'staging-manifest.json').read_text()); assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v for k,v in expected.items()); print('STAGED_HASHES_VERIFIED')"
    print(ssh(shlex.join([PYTHON,'-c',extractor]),timeout=30),flush=True)
    jobs=[]
    for lane,ids in enumerate(spec['GPU_lanes']):
        if not ids:continue
        cmd=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1','--cpus-per-task=1','--mem=8G','--time=01:00:00',
            '--job-name=frontier-N17-prefix-'+str(lane),'--output='+REMOTE+f'/lane{lane}.out','--error='+REMOTE+f'/lane{lane}.err',REMOTE+f'/lane{lane}.sbatch'])
        save(LOCAL/f'lane{lane}-submission-intent.json',dict(command=cmd,circuit_ids=ids))
        answer=ssh(cmd,timeout=30).strip();job=answer.split(';')[0]
        if not job.isdigit():raise RuntimeError('Uncertain sbatch response; no blind resubmit')
        record=dict(job_id=job,lane=lane,circuit_ids=ids,remote=REMOTE)
        save(LOCAL/f'lane{lane}-job.json',record);jobs.append(record);print('N17_ONE_A100_LANE_SUBMITTED',lane,job,flush=True)
    save(LOCAL/'jobs.json',jobs)


def status():
    from benchmark.gpu_remote import ssh
    jobs=[load(p) for p in sorted(LOCAL.glob('lane*-job.json'))]
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
    accounts=ssh('sacct -X -n -j '+','.join(j['job_id'] for j in jobs)+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30) if jobs else ''
    with db() as c:
        states=dict(c.execute('SELECT status,count(*) FROM cells GROUP BY status').fetchall())
        actual=c.execute('SELECT sum(actual_shots) FROM draws').fetchone()[0] or 0
    result=dict(jobs=jobs,queue=queue,accounting=accounts,logical_cells=states,actual_completed_model_shots=actual,new_QPU_shots=0)
    save(LOCAL/'status.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','m4','submit','status']);a=p.parse_args();globals()[a.action]()
