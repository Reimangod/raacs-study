"""N4: native HCI proposals, COMMON finite-penalty final evaluator, all20 cells.

No old energy/FCI labels enter initialization or ranking. Frequency controls
run independently BEFORE native proposals, so native failure cannot hide them.
Native DICE uses timestamp-based IPC names: SERIAL only, one owned process at
a time, check other same-user Dice processes and begin in a fresh clock tick.
"""
import fcntl,gzip,hashlib,json,os,resource,shutil,signal,struct,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)


def run(path):
    path=Path(path).resolve();root=path.parent;spec=json.loads(path.read_text());runtime=Path(spec['common_runtime'])
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:
        raise RuntimeError('Registered private Slurm molecular pilot only')
    if sha(Path(__file__))!=spec['worker_SHA256'] or sha(Path(spec['binary']))!=spec['binary_SHA256']:
        raise RuntimeError('Native kernel or pipeline worker drift')
    for name,value in spec['kernel_SHA256'].items():
        if sha(runtime/name)!=value:raise RuntimeError('Qualified common evaluator changed')
    if sha(runtime/'result.json')!=spec['common_qualification_SHA256']:
        raise RuntimeError('Qualified four-case common runtime changed')
    for name,value in spec['local_module_SHA256'].items():
        if sha(root/'pipeline'/name)!=value:raise RuntimeError('Proposal/decoder rule changed')
    with (root/'owner.claim').open('x') as f:f.write(str(os.getpid()))
    tick=time.perf_counter();cpu=time.process_time();sys.path[:0]=[str(runtime/'site'),str(runtime),str(root)]
    import importlib.metadata as metadata
    import numpy as np
    from kernels.portable_penalized_ci import evaluate
    from pipeline.native_hci_plan import marginal_strings
    from pipeline.native_dice_binary import decode
    if {name:metadata.version(name) for name in spec['packages']}!=spec['packages']:
        raise RuntimeError('Pinned common numerical versions changed')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(path)))
    cache={};cache_rows=[];states=[];physical={};attempts=[]
    for case,definition in spec['physical'].items():
        p=Path(definition['integrals_path'])
        if sha(p)!=definition['integrals_SHA256'] or sha(root/'physical'/case/'fcidump.txt')!=definition['fcidump_SHA256']:
            raise RuntimeError('Physical Hamiltonian changed')
        with np.load(p,allow_pickle=False) as a:physical[case]=(a['h1'],a['eri'])
    def ci(case,strings,origin):
        key=(case,tuple(strings));first=cache.get(key)
        if first:return dict(row=first['row'],evaluation_origin=first['origin'],new_CI_solve=False)
        definition=spec['physical'][case];h,g=physical[case]
        start,wall=time.process_time(),time.perf_counter()
        attempt=dict(case=case,strings=strings,origin=origin,status='STARTED');attempts.append(attempt)
        save(root/'common-ci-attempts.json',attempts)
        try:row=evaluate(h,g,definition['core_energy'],tuple(definition['nelec']),strings,seconds=60)
        except Exception as exc:
            attempt.update(status='FAILED_PRESERVED',error=repr(exc),CPU_seconds=time.process_time()-start,
                wall_seconds=time.perf_counter()-wall,matvec_calls=None)
            save(root/'common-ci-attempts.json',attempts);raise
        attempt.update(status='COMPLETED',CPU_seconds=time.process_time()-start,
            wall_seconds=time.perf_counter()-wall,matvec_calls=row['matvec_calls'])
        save(root/'common-ci-attempts.json',attempts)
        cache[key]=dict(row=row,origin=origin);cache_rows.append(dict(case=case,strings=strings,row=row,origin=origin))
        save(root/'common-ci-observations.json',cache_rows)
        return dict(row=row,evaluation_origin=origin,new_CI_solve=True)
    def native_limits():
        resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
    def other_dice():
        found=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if p.stat().st_uid==os.getuid() and (p/'exe').readlink().name=='Dice':found.append(int(p.name))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        return found
    # One pilot owns the native serial lane. Never stop another user's/chat's job.
    lock=(Path(spec['native_lock'])).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Check the NEW no-RDM production export path before molecular outcomes.
    # This is a frozen synthetic control, not a production reference in ranking.
    q=spec['minimal_qualification'];folder=root/'qualification'
    if sha(folder/'fcidump.txt')!=q['fcidump_SHA256']:raise RuntimeError('Minimal synthetic Hamiltonian changed')
    if other_dice():raise RuntimeError('Another same-user native Dice process is live')
    time.sleep(1.01-(time.time()%1))
    for spin in ['Alpha','Beta']:(folder/(spin+'Dets.bin')).write_bytes(int(1).to_bytes(16,'big'))
    lines=['nroots 1','spin 0','orbitals fcidump.txt','schedule','0 0.0005','end','davidsonTol 1e-8','dE 1e-10',
        'maxiter 10','noio','writeBestDeterminants 2147483647','nPTiter 0','nocc 2','0 1','end']
    (folder/'input.dat').write_text('\n'.join(lines)+'\n')
    with (folder/'native.log').open('wb') as log:
        proc=subprocess.Popen([spec['binary']],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=native_limits)
        try:rc=proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    n,pairs=decode((folder/'dets.bin').read_bytes());v=np.zeros(4)
    for a,b,value in pairs:v[(a-1)*2+(b-1)]=value
    raw=(folder/'shci.e').read_bytes()
    if len(raw)!=8:raise RuntimeError('Minimal native export energy missing')
    value=struct.unpack('<d',raw)[0];H=np.asarray(q['H_full'])
    if rc or n!=2 or len(pairs)<=1 or abs(v@v-1)>1e-8 or abs(value-q['expected_energy_Ha'])>1e-8 or np.linalg.norm(H@v-value*v)>1e-8:
        raise RuntimeError('No-RDM native HCI export path not physically qualified')
    save(folder/'result.json',dict(status='NO_RDM_NATIVE_HCI_EXPORT_AND_PHYSICAL_PARITY_PASSED',energy_Ha=value,
        records=len(pairs),new_molecular_candidate_outcomes=0))
    for plan in spec['plans']:
        start,wall= time.process_time(),time.perf_counter();cell_before=resource.getrusage(resource.RUSAGE_CHILDREN)
        folder=root/'cells'/plan['task_id'];folder.mkdir(exist_ok=False)
        row=dict(case=plan['case'],method=plan['method'],task_id=plan['task_id'],source=plan['source'],
            baseline=None,proposal=None,final=None,new_model_shots=0,new_QPU_shots=0)
        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if time.perf_counter()-tick>3660 or used>512*1024**2 or shutil.disk_usage(root).free<3*1024**3:
            row['status']='NOT_EXECUTED_GLOBAL_RESOURCE_CAP';states.append(row);save(folder/'result.json',row);continue
        case=plan['case'];definition=spec['physical'][case]
        # Baseline has no HCI dependency and does not pay native proposal work.
        try:row['baseline']=ci(case,plan['initial_strings'],plan['task_id']+':FREQUENCY')
        except Exception as exc:row['baseline_error']=dict(error=repr(exc),traceback=traceback.format_exc())
        save(folder/'progress.json',row)
        try:
            if other_dice():raise RuntimeError('Unrelated same-user native Dice process live; not starting a conflicting process')
            # Prevent reusing upstream's second-resolution IPC namespace of the
            # immediately previous serial child; count this wait in wall time.
            time.sleep(1.01-(time.time()%1))
            strings=plan['initial_strings']
            for spin in ['Alpha','Beta']:(folder/(spin+'Dets.bin')).write_bytes(b''.join(int(s).to_bytes(16,'big') for s in strings))
            lines=['nroots 1','spin 0','orbitals '+str(root/'physical'/case/'fcidump.txt'),'schedule','0 0.0005','end',
                'davidsonTol 1e-8','dE 1e-10','maxiter 10','noio','writeBestDeterminants 2147483647',
                'nPTiter 0','nocc '+str(sum(definition['nelec'])),' '.join(map(str,range(sum(definition['nelec'])))),'end']
            (folder/'input.dat').write_text('\n'.join(lines)+'\n')
            before=resource.getrusage(resource.RUSAGE_CHILDREN);nt=time.perf_counter()
            with (folder/'native.log').open('wb') as log:
                proc=subprocess.Popen([spec['binary']],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=native_limits)
                row['native_pid']=proc.pid;save(folder/'progress.json',row)
                try:
                    try:rc=proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        raise
                finally:
                    after=resource.getrusage(resource.RUSAGE_CHILDREN)
                    row['native_attempt_CPU_seconds']=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
                    row['native_attempt_wall_seconds']=time.perf_counter()-nt
            native_cpu=row['native_attempt_CPU_seconds']
            row['native_returncode']=rc
            if rc:raise RuntimeError('Native proposal process returned '+str(rc))
            n,pairs=decode((folder/'dets.bin').read_bytes())
            if n!=definition['norb']:raise RuntimeError('Native proposal orbital/basis mismatch')
            selected,norm=marginal_strings(pairs,n,tuple(definition['nelec']),plan['HF_string'])
            energy_data=(folder/'shci.e').read_bytes()
            if len(energy_data)!=8:raise RuntimeError('One native diagnostic energy required')
            row['proposal']=dict(status='NATIVE_HCI_PROPOSAL_AVAILABLE',actual_joint_determinants=len(pairs),
                initial_joint_determinants=len(strings)**2,coefficient_norm=norm,selected_strings=selected,
                native_physical_energy_Ha=struct.unpack('<d',energy_data)[0],native_S2=None,
                native_CPU_seconds=native_cpu,native_wall_seconds=time.perf_counter()-nt,
                native_spin0_is_not_S0_or_common_finite_penalty=True,raw_determinants_SHA256=sha(folder/'dets.bin'))
            # Lossless archive, raw file retained remotely; never delete evidence.
            with (folder/'dets.bin').open('rb') as inp,(folder/'dets.bin.gz').open('wb') as out:
                with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as zipped:
                    for block in iter(lambda:inp.read(1048576),b''):zipped.write(block)
            save(folder/'progress.json',row)
            row['final']=ci(case,selected,plan['task_id']+':NATIVE_HCI_COMMON_FINAL')
            row['standalone_native_plus_common_numeric_CPU_proxy']=native_cpu+row['final']['row']['CPU_seconds']
            row['status']='COMPLETED_NATIVE_PROPOSAL_AND_COMMON_FINAL'
        except Exception as exc:
            row['status']='NATIVE_OR_FINAL_UNAVAILABLE_PRESERVED';row['error']=repr(exc);row['traceback']=traceback.format_exc()
        cell_after=resource.getrusage(resource.RUSAGE_CHILDREN)
        row.update(actual_cell_CPU_seconds_including_children=time.process_time()-start+cell_after.ru_utime+cell_after.ru_stime-cell_before.ru_utime-cell_before.ru_stime,
            actual_cell_wall_seconds=time.perf_counter()-wall)
        save(folder/'result.json',row);states.append(row);save(root/'progress.json',dict(states=states,planned=20,terminal=len(states)))
        print('N4',case,plan['method'],row['status'],flush=True)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'terminal.json',dict(status='ALL20_PLANNED_NATIVE_HCI_SOURCE_CELLS_TERMINAL_WITH_MISSINGNESS',states=states,
        CPU_seconds_including_children=time.process_time()-cpu+usage.ru_utime+usage.ru_stime,wall_seconds=time.perf_counter()-tick,
        new_unique_common_CI_solves=len(cache_rows),common_CI_attempts=attempts,new_model_shots=0,new_QPU_shots=0,
        fixed_policy_independent_quantum_confirmation=False,paper_highest_HCI_or_matched_total_work_reproduction=False,goal_complete=False))


if __name__=='__main__':run(sys.argv[1])
