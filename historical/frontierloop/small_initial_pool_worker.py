"""N7: same kernels, uniformly smaller source-only initial spaces."""
import fcntl,gzip,hashlib,json,os,resource,shutil,signal,struct,subprocess,sys,time,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(temp,p)


def run(path):
    path=Path(path).resolve();root=path.parent;spec=json.loads(path.read_text())
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:raise RuntimeError('Registered private Slurm N7 only')
    if sha(__file__)!=spec['worker_SHA256']:raise RuntimeError('Frozen worker changed')
    if any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise RuntimeError('Explicit numerical1thread required')
    for name,value in spec['pipeline_SHA256'].items():
        if sha(root/'pipeline'/name)!=value:raise RuntimeError('Frozen helper/kernel drift')
    for name,value in spec['qualified_external_SHA256'].items():
        if sha(name)!=value:raise RuntimeError('Qualified native/numerical dependency changed')
    (root/'owner.claim').open('x').write(str(os.getpid()))
    start,cpu=time.perf_counter(),time.process_time();children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime=Path(spec['common_runtime']);sys.path[:0]=[str(runtime/'site'),str(root)]
    import importlib.metadata as metadata
    import numpy as np
    from pyscf import lib
    from pipeline.native_dice_binary import decode
    from pipeline.native_hci_plan import marginal_strings
    from pipeline.joint_ci_plan import joint_pairs
    from pipeline.joint_ci_sparse import solve
    lib.num_threads(1)
    if {name:metadata.version(name) for name in spec['packages']}!=spec['packages']:raise RuntimeError('Numerical package version drift')
    physical={}
    for case,definition in spec['physical'].items():
        for filename,key in [('integrals_path','integrals_SHA256'),('fcidump_path','fcidump_SHA256')]:
            if sha(definition[filename])!=definition[key]:raise RuntimeError('Physical integrals changed')
        with np.load(definition['integrals_path'],allow_pickle=False) as a:physical[case]=(a['h1'],a['eri'])
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(path),pyscf_threads=lib.num_threads(),
        thread_env={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    def other_native():
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                if p.stat().st_uid==os.getuid() and (p/'exe').readlink().name=='Dice':return True
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        return False
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
    def native(folder,fcidump,nelec,strings):
        folder.mkdir(parents=True,exist_ok=True)
        if other_native():raise RuntimeError('Another same-user Dice live; do not conflict or stop it')
        time.sleep(1.01-(time.time()%1))
        for spin in ['Alpha','Beta']:(folder/(spin+'Dets.bin')).write_bytes(b''.join(int(x).to_bytes(16,'big') for x in strings))
        lines=['nroots 1','spin 0','orbitals '+str(fcidump),'schedule','0 0.0005','end','davidsonTol 1e-8',
            'dE 1e-10','maxiter 10','noio','writeBestDeterminants 2147483647','nPTiter 0','nocc '+str(sum(nelec)),
            ' '.join(map(str,range(sum(nelec)))),'end']
        (folder/'input.dat').write_text('\n'.join(lines)+'\n')
        before=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.perf_counter()
        attempt=dict(status='STARTED');save(folder/'native-attempt.json',attempt)
        try:
            with (folder/'native.log').open('wb') as log:
                p=subprocess.Popen([spec['native_binary']],cwd=folder,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=limits)
                attempt['pid']=p.pid;save(folder/'native-attempt.json',attempt)
                try:rc=p.wait(timeout=spec['policy']['native_wall_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise
            attempt['returncode']=rc
            if rc:raise RuntimeError('Native DICE returned '+str(rc))
            attempt['status']='RETURNED_OUTPUTS_NOT_YET_NUMERICALLY_CHECKED'
        except Exception as exc:attempt.update(status='FAILED_PRESERVED',error=repr(exc));raise
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            attempt.update(CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,wall_seconds=time.perf_counter()-wall)
            save(folder/'native-attempt.json',attempt)
        raw=(folder/'dets.bin').read_bytes();n,pairs=decode(raw);energy=(folder/'shci.e').read_bytes()
        if len(energy)!=8:raise RuntimeError('One diagnostic native energy required')
        with (folder/'dets.bin.gz').open('wb') as output:
            with gzip.GzipFile(fileobj=output,mode='wb',mtime=0) as f:f.write(raw)
        return n,pairs,dict(**attempt,native_diagnostic_spinfree_energy_Ha=struct.unpack('<d',energy)[0],raw_determinants_SHA256=sha(folder/'dets.bin'))
    lane=Path(spec['native_lock']).open('a');fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=spec['minimal_qualification']
    if sha(q['fcidump_path'])!=q['fcidump_SHA256']:raise RuntimeError('Tiny native control changed')
    n,pairs,attempt=native(root/'qualification',q['fcidump_path'],(1,1),[1]);v=np.zeros(4)
    for a,b,value in pairs:v[(a-1)*2+(b-1)]=value
    H=np.array(q['H_full']);energy=attempt['native_diagnostic_spinfree_energy_Ha']
    if n!=2 or len(pairs)<=1 or abs(v@v-1)>1e-8 or abs(energy-q['expected_energy_Ha'])>1e-8 or np.linalg.norm(H@v-energy*v)>1e-8:
        raise RuntimeError('Native export-path tiny physical parity failed')
    save(root/'qualification/result.json',dict(status='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED',native_attempt=attempt))
    cache={};observations=[];ci_attempts=[];states=[]
    def evaluate(case,pairs,origin):
        pairs=np.ascontiguousarray(pairs,dtype=np.uint64);key=(case,hashlib.sha256(pairs.tobytes()).hexdigest())
        if key in cache:
            o=cache[key];return dict(row=o['row'],observation_id=o['id'],evaluation_origin=o['origin'],new_CI_solve=False)
        at=dict(case=case,basis_digest=key[1],origin=origin,status='STARTED');ci_attempts.append(at);save(root/'ci-attempts.json',ci_attempts)
        c,w=time.process_time(),time.perf_counter();definition=spec['physical'][case];h,g=physical[case]
        try:
            row,vector=solve(h,g,definition['core_energy'],tuple(definition['nelec']),pairs,spec['sparse_library'],seconds=75,triangle_cap=8000000)
            oid=hashlib.sha256((case+key[1]+sha(path)).encode()).hexdigest();file=root/'roots'/(oid+'.npz');file.parent.mkdir(exist_ok=True)
            np.savez_compressed(file,coefficients=vector,joint_pairs=pairs)
            o=dict(id=oid,case=case,basis_digest=key[1],origin=origin,row=row,vector_file=str(file.relative_to(root)),vector_SHA256=sha(file))
            cache[key]=o;observations.append(o);save(root/'observations.json',observations)
            at.update(status='COMPLETED',observation_id=oid)
            return dict(row=row,observation_id=oid,evaluation_origin=origin,new_CI_solve=True)
        except Exception as exc:at.update(status='FAILED_PRESERVED',error=repr(exc),matvec_calls=None);raise
        finally:at.update(CPU_seconds=time.process_time()-c,wall_seconds=time.perf_counter()-w);save(root/'ci-attempts.json',ci_attempts)
    for plan in spec['plans']:
        folder=root/'cells'/plan['task_id'];folder.mkdir(parents=True,exist_ok=False)
        row=dict(case=plan['case'],method=plan['method'],task_id=plan['task_id'],source=plan['source'],arms={},errors={},
            new_model_shots=0,new_QPU_shots=0)
        if plan['source']['status']!='AVAILABLE':row['status']='MISSING_SAME_PROBLEM_SOURCE'
        elif time.perf_counter()-start>spec['policy']['global_wall_seconds'] or shutil.disk_usage(root).free<3*1024**3 or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>128*1024**2:
            row['status']='NOT_EXECUTED_GLOBAL_RESOURCE_CAP'
        else:
            try:
                definition=spec['physical'][plan['case']]
                n,pairs,native_record=native(folder,definition['fcidump_path'],tuple(definition['nelec']),plan['initial_strings'])
                if n!=definition['norb']:raise RuntimeError('Native orbital identity mismatch')
                strings,norm=marginal_strings(pairs,n,tuple(definition['nelec']),plan['HF_string'])
                joint,selection=joint_pairs(pairs,n,tuple(definition['nelec']),plan['HF_string'])
                row['native']=native_record;row['selection']=selection;row['product_strings']=strings
                save(folder/'progress.json',row)
                product=[(a,b) for a in strings for b in strings]
                spaces={'PRODUCT_R128':product,'JOINT_TOP_D16384':joint}
                for arm in spec['policy']['readout_arms']:
                    basis=spaces[arm]
                    try:row['arms'][arm]=evaluate(plan['case'],basis,plan['task_id']+':'+arm)
                    except Exception as exc:row['errors'][arm]=dict(error=repr(exc),traceback=traceback.format_exc())
                    save(folder/'progress.json',row)
                row['status']='REQUESTED_READOUTS_COMPLETED' if len(row['arms'])==len(spec['policy']['readout_arms']) else 'PARTIAL_CI_PRESERVED'
            except Exception as exc:row.update(status='NATIVE_OR_DECODER_FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
        if (folder/'native-attempt.json').exists():row['native_attempt']=json.loads((folder/'native-attempt.json').read_text())
        save(folder/'result.json',row);states.append(row);save(root/'progress.json',dict(planned=len(spec['plans']),terminal=len(states),states=states))
        print('N7',plan['case'],plan['method'],row['status'],flush=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'terminal.json',dict(status='ALL_SMALL_INITIAL_POOL_SOURCES_TERMINAL_WITH_MISSINGNESS',states=states,
        actual_new_CI_roots=len(observations),ci_attempts=len(ci_attempts),
        CPU_seconds=time.process_time()-cpu+after.ru_utime+after.ru_stime-children_before.ru_utime-children_before.ru_stime,
        wall_seconds=time.perf_counter()-start,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        independent_quantum_confirmation=False,full_strongest_TN_paper_reproduction=False,goal_complete=False))


if __name__=='__main__':run(sys.argv[1])
