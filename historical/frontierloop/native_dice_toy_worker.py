"""Execute only the seven frozen tiny native DICE qualifications on Slurm."""
import hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)


def run(path):
    path=Path(path).resolve();spec=json.loads(path.read_text());root=path.parent
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:
        raise RuntimeError('Only the registered Slurm qualification job may execute')
    if sha(Path(__file__))!=spec['worker_SHA256']:raise RuntimeError('Tiny qualifier worker changed')
    binary=Path(spec['binary'])
    if sha(binary)!=spec['binary_SHA256']:raise RuntimeError('Native binary changed')
    if sha(root/'fixtures.json')!=spec['fixtures_SHA256']:raise RuntimeError('Tiny fixtures changed')
    cases=json.loads((root/'fixtures.json').read_text())['fixtures'];rows=[];tick=time.perf_counter();cpu=time.process_time()
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(path)))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in cases:
        folder=root/'inputs'/case['name'];row=dict(name=case['name']);start=time.perf_counter()
        try:
            for filename,value in case['input_SHA256'].items():
                if sha(folder/filename)!=value:raise RuntimeError('Native tiny input changed')
            with (folder/'native.log').open('wb') as log:
                proc=subprocess.Popen([str(binary)],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:rc=proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            row.update(status='NATIVE_PROCESS_RETURNED_NOT_YET_NUMERICALLY_VERIFIED',returncode=rc,
                outputs={name:sha(folder/name) for name in ['shci.e','dets.bin','spin1RDM.0.0.txt'] if (folder/name).exists()})
            for filename,value in case['input_SHA256'].items():
                if sha(folder/filename)!=value:raise RuntimeError('Native modified a frozen tiny input')
        except Exception as exc:row.update(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
        row['wall_seconds']=time.perf_counter()-start;rows.append(row);save(root/'progress.json',dict(rows=rows))
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'terminal.json',dict(status='ALL7_TINY_NATIVE_ATTEMPTS_TERMINAL_NOT_PRODUCTION_QUALIFICATION',rows=rows,
        wall_seconds=time.perf_counter()-tick,CPU_seconds_including_children=time.process_time()-cpu+usage.ru_utime+usage.ru_stime,
        output_bytes=used,new_model_shots=0,new_QPU_shots=0,new_molecular_energy_evaluations=0,
        binary_SHA256=spec['binary_SHA256'],goal_complete=False))
    print('TINY_NATIVE_ATTEMPTS_TERMINAL',[(r['name'],r['status']) for r in rows],flush=True)


if __name__=='__main__':run(sys.argv[1])
