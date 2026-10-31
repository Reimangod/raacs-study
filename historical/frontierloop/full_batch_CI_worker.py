"""N51 fixed full-batch roots, versioned private pipeline, no new sampling."""
import hashlib,json,os,resource,signal,subprocess,sys,traceback
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('w') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def verify(root):
    p=json.loads((root/'protocol.json').read_text())
    for name,expected in p['input_SHA256'].items():
        if sha(root/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
    return p


def qualify(root,label):
    import numpy as np,importlib.metadata as metadata
    sys.path.insert(0,str(root));from pipeline.full_batch_CI_kernel import evaluate
    p=verify(root);versions={k:metadata.version(k) for k in p['numerical_packages']}
    if versions!=p['numerical_packages']:raise RuntimeError('Numerical runtime differs')
    cpu,wall=process_time(),perf_counter();rows=[]
    with np.load(root/'qualification-inputs.npz',allow_pickle=False) as a:
        for i,oracle in enumerate(p['qualification_oracles']):
            row,state=evaluate(a[f'h{i}'],a[f'g{i}'],oracle['core'],tuple(oracle['nelec']),a[f's{i}'],seconds=30)
            errors={k:abs(row[k]-oracle[k]) for k in ['energy_Ha','penalized_energy_Ha','spin_square']}
            if max(errors.values())>1e-7:raise RuntimeError('Independent small-matrix parity failed')
            rows.append(dict(errors=errors,residual=row['residual_norm']))
    result=dict(status='QUALIFIED',label=label,versions=versions,rows=rows,
         CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
         protocol_SHA256=sha(root/'protocol.json'),job_id=os.environ.get('SLURM_JOB_ID'))
    save(root/'out'/('qualification-'+label+'.json'),result)
    return result


def cell(root,index):
    import numpy as np
    import pyscf
    sys.path.insert(0,str(root));from pipeline.full_batch_CI_kernel import evaluate
    p=verify(root);task=p['tasks'][index];folder=root/'out'/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'started.json').exists():raise RuntimeError('Existing cell; no duplicate computation')
    save(folder/'started.json',dict(pid=os.getpid(),task=task,job_id=os.environ.get('SLURM_JOB_ID'),protocol_SHA256=sha(root/'protocol.json')))
    cpu,wall=process_time(),perf_counter();pyscf.lib.num_threads(1)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(p['CPU_cap_seconds'],p['CPU_cap_seconds']+1))
    resource.setrlimit(resource.RLIMIT_FSIZE,(32*2**20,32*2**20))
    if sys.platform!='darwin':resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30))
    try:
        with np.load(root/'integrals.npz',allow_pickle=False) as a:
            row,state=evaluate(a['h1'],a['eri'],p['core'],tuple(p['nelec']),task['strings'],seconds=p['wall_cap_seconds'],
                   progress=lambda z:save(folder/'progress.json',dict(task_id=task['task_id'],**z)))
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if rss>p['RSS_cap_bytes']:raise MemoryError('Measured RSS exceeds fixed pilot cap')
        stream=BytesIO();np.savez_compressed(stream,**state,problem_id=np.array(p['problem_id']),task_id=np.array(task['task_id']))
        tmp=folder/'root.npz.tmp'
        with tmp.open('wb') as f:f.write(stream.getvalue());f.flush();os.fsync(f.fileno())
        os.replace(tmp,folder/'root.npz')
        out=dict(status='COMPLETED',task_id=task['task_id'],method=task['method'],row=row,
                 root_SHA256=sha(folder/'root.npz'),peak_RSS_bytes=rss)
    except Exception as e:
        out=dict(status='LIMIT_OR_ENGINEERING_FAILURE_PRESERVED',task_id=task['task_id'],method=task['method'],error=repr(e),traceback=traceback.format_exc())
    out.update(CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,job_id=os.environ.get('SLURM_JOB_ID'),protocol_SHA256=sha(root/'protocol.json'))
    save(folder/'result.json',out)
    print('N51_CELL',index,out['status'],flush=True)


def supervise(root,label,indices):
    p=verify(root);marker=root/'out'/('supervisor-'+label+'-started.json')
    if marker.exists():raise RuntimeError('Supervisor already started; no blind retry')
    save(marker,dict(pid=os.getpid(),indices=indices,job_id=os.environ.get('SLURM_JOB_ID')))
    q=qualify(root,label);usage=resource.getrusage(resource.RUSAGE_CHILDREN);wall=perf_counter();rows=[]
    for i in indices:
        folder=root/'out'/p['tasks'][i]['task_id'];folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(root/'worker.py'),str(root),'cell',str(i)]
        with (folder/'stdout.txt').open('wb') as stdout,(folder/'stderr.txt').open('wb') as stderr:
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
            save(folder/'child-intent.json',dict(pid=child.pid,PGID=child.pid,command=command))
            limited=False
            try:code=child.wait(timeout=p['wall_cap_seconds']+30)
            except subprocess.TimeoutExpired:
                limited=True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        if (folder/'result.json').exists() and code==0:
            r=json.loads((folder/'result.json').read_text())
        else:
            r=dict(status='HARD_LIMIT_OR_PROCESS_FAILURE_PRESERVED',task_id=p['tasks'][i]['task_id'],method=p['tasks'][i]['method'],returncode=code,wall_limit=limited,CPU_seconds=None)
            save(folder/'process-failure.json',r)
        rows.append(r);save(root/'out'/('supervisor-'+label+'-progress.json'),dict(rows=rows))
    final=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'out'/('supervisor-'+label+'-terminal.json'),dict(status='PARTITION_TERMINAL',rows=rows,
         children_CPU_seconds=final.ru_utime+final.ru_stime-usage.ru_utime-usage.ru_stime,
         wall_seconds=perf_counter()-wall,qualification=q,new_model_samples=0,new_QPU_shots=0,
         GPU_allocations=0,protocol_SHA256=sha(root/'protocol.json')))


if __name__=='__main__':
    root=Path(sys.argv[1]);mode=sys.argv[2]
    if mode=='cell':cell(root,int(sys.argv[3]))
    elif mode=='qualify':qualify(root,sys.argv[3])
    else:supervise(root,sys.argv[3],[int(s) for s in sys.argv[4:]])
