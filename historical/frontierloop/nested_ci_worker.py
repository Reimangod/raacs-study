"""N20 bounded CPU-only saved-root CI scaling; existing counts, no sampling."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter,process_time
import traceback


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def one_task(root,task_id,warm_task):
    import numpy as np
    import pyscf
    sys.path.insert(0,str(root))
    from pipeline.nested_ci_kernel import evaluate
    spec=json.loads((root/'protocol.json').read_text());p=next(p for p in spec['tasks'] if p['task_id']==task_id)
    folder=root/'cells'/task_id;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'started.json').exists():raise RuntimeError('No duplicate/partial task rerun')
    save(folder/'started.json',dict(task_id=task_id,case=p['case'],problem_id=p['problem_id'],pid=os.getpid(),warm_task_id=warm_task))
    start,cpu=perf_counter(),process_time(); pyscf.lib.num_threads(1)
    resource.setrlimit(resource.RLIMIT_AS,(spec['child_virtual_memory_bytes'],spec['child_virtual_memory_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(32*2**20,32*2**20))
    try:
        warm=None
        if warm_task is not None:
            prior=root/'cells'/warm_task;old=json.loads((prior/'result.json').read_text())
            if old['status']!='COMPLETED' or old['problem_id']!=p['problem_id'] or sha(prior/'root.npz')!=old['root_SHA256']:
                raise RuntimeError('Warm root problem/provenance mismatch')
            with np.load(prior/'root.npz',allow_pickle=False) as a:warm={k:a[k] for k in a.files}
        definition=spec['physical'][p['case']]
        if sha(definition['integrals_path'])!=definition['integrals_SHA256']:raise RuntimeError('Physical integral bytes changed')
        with np.load(definition['integrals_path'],allow_pickle=False) as a:
            row,saved=evaluate(a['h1'],a['eri'],definition['core_energy'],tuple(definition['nelec']),p['strings'],seconds=spec['root_wall_seconds'],warm=warm)
        if p['requested_R']==128:
            errors={k:abs(row[k]-p['expected_R128'][k]) for k in ('energy_Ha','penalized_energy_Ha','spin_square')}
            if max(errors.values())>1e-8:raise RuntimeError('R128 prior-source/equation parity failed: '+str(errors))
        else:errors=None
        stream=BytesIO();np.savez_compressed(stream,**saved,problem_id=np.array(p['problem_id']),task_id=np.array(task_id))
        tmp=folder/'root.npz.tmp'
        with tmp.open('wb') as f:f.write(stream.getvalue());f.flush();os.fsync(f.fileno())
        os.replace(tmp,folder/'root.npz')
        out=dict(status='COMPLETED',task_id=task_id,case=p['case'],problem_id=p['problem_id'],requested_R=p['requested_R'],row=row,
            warm_task_id=warm_task,R128_parity_errors=errors,root_SHA256=sha(folder/'root.npz'),
            root_basis_representation='SORTED_HOLE_STRINGS_CARTESIAN_PYSCF_ORDER' if row['particle_hole_used'] else 'SORTED_ELECTRON_STRINGS_CARTESIAN_PYSCF_ORDER',
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception as exc:
        out=dict(status='FAILED_PRESERVED',task_id=task_id,case=p['case'],problem_id=p['problem_id'],requested_R=p['requested_R'],
            error=repr(exc),traceback=traceback.format_exc(),warm_task_id=warm_task,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(folder/'result.json',out)


def run(root):
    import importlib.metadata as metadata
    import numpy as np
    import pyscf,scipy
    spec=json.loads((root/'protocol.json').read_text());start,cpu=perf_counter(),process_time()
    if (root/'started.json').exists():raise RuntimeError('Existing N20 campaign; no blind restart')
    for name,expected in spec['input_SHA256'].items():
        if sha(root/name)!=expected:raise RuntimeError('Frozen N20 code/input changed: '+name)
    versions={k:metadata.version(k) for k in spec['packages']}
    if versions!=spec['packages']:raise RuntimeError('Qualified numerical versions differ')
    site=Path(spec['common_runtime'])/'site'
    if any(not Path(module.__file__).resolve().is_relative_to(site) for module in (np,pyscf,scipy)):
        raise RuntimeError('Numerical kernel escaped private overlay')
    pyscf.lib.num_threads(1)
    save(root/'started.json',dict(protocol_SHA256=sha(root/'protocol.json'),job_id=os.environ.get('SLURM_JOB_ID'),
        versions=versions,pyscf_threads=pyscf.lib.num_threads(),thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0))
    results={};points=[];latest={};case_stopped=set();baseline_ok=True
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    tasks={t['task_id']:t for t in spec['tasks']}
    for point in spec['points']:
        task_id=point['task_id'];case=point['case'];task=tasks[task_id]
        if task_id in results:
            result=dict(point,status=results[task_id]['status'],execution_origin_task_id=task_id,new_CI_solve=False)
        elif point['requested_R']>128 and (not baseline_ok or case in case_stopped):
            status='NOT_RUN_BASELINE_PARITY_INCOMPLETE' if not baseline_ok else 'NOT_RUN_PREVIOUS_SCALING_LIMIT'
            result=dict(point,status=status,new_CI_solve=False,execution_origin_task_id=None)
        else:
            folder=root/'cells'/task_id;folder.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(root/'worker.py'),str(root),'--task',task_id]
            warm=latest.get(case)
            if warm:command+=['--warm',warm]
            tick=perf_counter()
            with (folder/'stdout.txt').open('wb') as out,(folder/'stderr.txt').open('wb') as err:
                try:
                    completed=subprocess.run(command,stdout=out,stderr=err,timeout=spec['child_hard_wall_seconds'],check=False)
                    code=completed.returncode;timeout=False
                except subprocess.TimeoutExpired:code=None;timeout=True
            if (folder/'result.json').exists() and code==0:
                actual=json.loads((folder/'result.json').read_text())
            else:
                actual=dict(status='HARD_TIME_LIMIT_PRESERVED' if timeout else 'PROCESS_FAILED_PRESERVED',task_id=task_id,case=case,
                    requested_R=point['requested_R'],exit_code=code,wall_seconds=perf_counter()-tick)
                save(folder/'process-failure.json',actual)
            results[task_id]=actual
            if actual['status']=='COMPLETED':latest[case]=task_id
            elif point['requested_R']==128:baseline_ok=False
            else:case_stopped.add(case)
            result=dict(point,status=actual['status'],execution_origin_task_id=task_id,new_CI_solve=True)
        points.append(result)
        save(root/'progress.json',dict(points=points,actual_tasks=list(results.values()),all_R128_parity_passed_so_far=baseline_ok))
    final=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'terminal.json',dict(status='N20_ALL16_PLANNED_POINTS_TERMINAL',points=points,actual_tasks=list(results.values()),
        attempted_unique_tasks=len(results),completed_new_roots=sum(r['status']=='COMPLETED' for r in results.values()),
        CPU_seconds_children=final.ru_utime+final.ru_stime-usage.ru_utime-usage.ru_stime,
        CPU_seconds_supervisor=process_time()-cpu,wall_seconds=perf_counter()-start,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False))


if __name__=='__main__':
    root=Path(sys.argv[1])
    if '--task' in sys.argv:
        task=sys.argv[sys.argv.index('--task')+1]
        warm=sys.argv[sys.argv.index('--warm')+1] if '--warm' in sys.argv else None
        one_task(root,task,warm)
    else:run(root)
