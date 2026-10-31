"""Finite HF-initialized native HCI source only. No new solver or final CI."""
from pathlib import Path
import fcntl,hashlib,json,os,resource,signal,struct,subprocess,sys,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(p)+'.pending')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def run(path):
    path=Path(path).resolve();root=path.parent;p=json.loads(path.read_text())
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=p['remote']:raise RuntimeError('Private registered Slurm source job only')
    for f,h in p['runtime_bindings'].items():
        file=Path(f) if Path(f).is_absolute() else root/f
        if sha(file)!=h:raise RuntimeError('HCI source binding drift '+str(file))
    if (root/'started.json').exists():raise RuntimeError('Existing HCI intent, no retry')
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(path)))
    from native_dice_binary import decode
    from native_hci_plan import marginal_strings
    lock=Path(p['native_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def others():
        live=[]
        for d in Path('/proc').iterdir():
            if not d.name.isdigit():continue
            try:
                if d.stat().st_uid==os.getuid() and (d/'exe').readlink().name=='Dice':live.append(int(d.name))
            except (PermissionError,FileNotFoundError,ProcessLookupError):pass
        return live
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30));resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,64*2**20))
    start=time.perf_counter();cpu=time.process_time();rows=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in p['cases']:
        row=dict(case=case['case'],status='STARTED',new_model_samples=0,new_QPU_shots=0,final_CI_state_prepared_as_zero_CNOT_state=False)
        d=root/'cases'/case['case'];d.mkdir(exist_ok=False,parents=True)
        tick=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            if others():raise RuntimeError('Unrelated same-user Dice live; not launching conflicting IPC')
            time.sleep(1.01-(time.time()%1))
            for spin in ['Alpha','Beta']:(d/(spin+'Dets.bin')).write_bytes(int(case['HF_string']).to_bytes(16,'big'))
            fcidump=root/'physical'/case['case']/'fcidump.txt'
            if sha(fcidump)!=case['fcidump_SHA256']:raise RuntimeError('Hamiltonian drift')
            text=['nroots 1','spin 0','orbitals '+str(fcidump),'schedule','0 0.0005','end','davidsonTol 1e-8','dE 1e-10','maxiter 10','noio','writeBestDeterminants 2147483647','nPTiter 0','nocc '+str(sum(case['nelec'])),' '.join(map(str,range(sum(case['nelec'])))),'end']
            (d/'input.dat').write_text('\n'.join(text)+'\n')
            with (d/'native.log').open('wb') as log:
                child=subprocess.Popen([p['binary']],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
                save(d/'child-started.json',dict(pid=child.pid,job_id=os.environ['SLURM_JOB_ID']))
                try:rc=child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            if rc:raise RuntimeError('Native HCI return '+str(rc))
            raw=(d/'dets.bin').read_bytes();n,pairs=decode(raw)
            if n!=case['norb']:raise RuntimeError('Native orbital mismatch')
            selected,norm=marginal_strings(pairs,n,tuple(case['nelec']),case['HF_string'])
            scores={}
            for a,b,v in pairs:
                scores[a]=scores.get(a,0.)+.5*v*v;scores[b]=scores.get(b,0.)+.5*v*v
            rank=[case['HF_string']]+sorted([s for s in scores if s!=case['HF_string']],key=lambda s:(-scores[s],s))[:127]
            assert sorted(rank)==selected
            energy=(d/'shci.e').read_bytes()
            if len(energy)!=8:raise RuntimeError('Native energy absent')
            import gzip
            with (d/'dets.bin.gz').open('wb') as out:
                with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as gz:gz.write(raw)
            row.update(status='COMPLETED',rank=rank,actual_native_joint_dimension=len(pairs),coefficient_norm=norm,native_energy_Ha=struct.unpack('<d',energy)[0],
                native_spin0_not_S0=True,native_energy_not_common_penalized_endpoint=True,raw_determinants_SHA256=hashlib.sha256(raw).hexdigest())
        except Exception as exc:row.update(status='FAILED_NATIVE_HCI_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        row.update(native_attempt_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,wall_seconds=time.perf_counter()-tick)
        save(d/'result.json',row);rows.append(row);save(root/'progress.json',dict(rows=rows));print(case['case'],row['status'],flush=True)
    save(root/'terminal.json',dict(status='FOUR_BOUNDED_HCI_SOURCE_STATUSES_TERMINAL',rows=rows,new_model_samples=0,new_QPU_shots=0,worker_CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start,job_id=os.environ['SLURM_JOB_ID']))

if __name__=='__main__':run(sys.argv[1])
