"""Build a private input-only DICE adapter; execute only frozen synthetic tests."""
import difflib,fcntl,hashlib,json,os,resource,shutil,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(t,p)


def run(path):
    path=Path(path).resolve();root=path.parent;spec=json.loads(path.read_text())
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:raise RuntimeError('Private registered Slurm only')
    if sha(__file__)!=spec['worker_SHA256'] or sha(root/'paired_dice_patch.py')!=spec['patch_SHA256'] or sha(root/'fixtures.json')!=spec['fixtures_SHA256']:raise RuntimeError('Frozen input/code drift')
    if any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise RuntimeError('Single-thread numerical environment required')
    (root/'owner.claim').open('x').write(str(os.getpid()))
    original=Path(spec['original_source']);main=original/'SHCI/SHCI.cpp';oldbin=Path(spec['original_binary'])
    if sha(main)!=spec['original_main_SHA256'] or sha(oldbin)!=spec['original_binary_SHA256']:raise RuntimeError('Original qualified source/binary changed')
    tick,cpu=time.perf_counter(),time.process_time();usage=resource.getrusage(resource.RUSAGE_CHILDREN);rows=[]
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(path)))
    def limited(argv,cwd,log,timeout,env=None):
        before=resource.getrusage(resource.RUSAGE_CHILDREN);w=time.perf_counter()
        with log.open('wb') as f:
            p=subprocess.Popen(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        return dict(returncode=rc,wall_seconds=time.perf_counter()-w,
            CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    def resource_floor():
        if shutil.disk_usage(root).free<spec['disk_floor_bytes']:raise OSError('Remote disk floor')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>spec['local_runtime_cap_bytes']:raise OSError('N9 private runtime storage cap')
    try:
        resource_floor();buildstart=time.perf_counter();source=root/'source';source.mkdir()
        # Git archive contains source+GPL license, not .git, old output, or objects.
        with (root/'source.tar').open('wb') as out:
            r=subprocess.run(['git','archive',spec['original_commit']],cwd=original,stdout=out,stderr=subprocess.PIPE,timeout=35)
        if r.returncode:raise RuntimeError('Pinned source archive failed')
        with tarfile.open(root/'source.tar') as t:t.extractall(source,filter='data')
        shutil.copytree(original/'obj',source/'obj');(source/'bin').mkdir(exist_ok=True)
        before=(source/'SHCI/SHCI.cpp').read_text()
        if sha(source/'SHCI/SHCI.cpp')!=spec['original_main_SHA256']:raise RuntimeError('Archived source mismatch')
        sys.path.insert(0,str(root));from paired_dice_patch import apply
        after=apply(before);(source/'SHCI/SHCI.cpp').write_text(after)
        (root/'source-diff.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/SHCI.cpp',tofile='paired-adapter/SHCI.cpp')))
        boost=Path(spec['boost_root']);flags='-std=c++14 -O3 -g0 -w -fPIC -fopenmp -fpermissive -DSERIAL -I. -I./eigen -I./SHCI -I'+str(boost)
        b=limited(['make','-j1','Dice','CXX=g++','HAS_AVX2=no','FLAGS_SHCI='+flags,
            'LFLAGS_SHCI='+str(boost/'stage/lib/libboost_serialization.a')+' -lrt',
            'GIT_HASH='+spec['original_commit'],'GIT_BRANCH=private-paired-input-v1','COMPILE_TIME=N9_FROZEN'],source,root/'build.log',spec['build_wall_cap_seconds'])
        binary=source/'bin/Dice';b.update(binary=str(binary),binary_SHA256=sha(binary) if binary.exists() else None,
            original_main_SHA256=sha(main),patched_main_SHA256=sha(source/'SHCI/SHCI.cpp'),
            patch_file_SHA256=sha(root/'source-diff.patch'),wall_including_copy_seconds=time.perf_counter()-buildstart)
        save(root/'build.json',b)
        if b['returncode'] or not binary.exists():raise RuntimeError('Private adapter build failed; do not substitute old kernel')
        resource_floor();lane=Path(spec['native_lock']).open('a');fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def other_native():
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    if p.stat().st_uid==os.getuid() and (p/'exe').readlink().name=='Dice':return True
                except (PermissionError,FileNotFoundError,ProcessLookupError):pass
            return False
        fixtures=json.loads((root/'fixtures.json').read_text())['fixtures']
        for f in fixtures:
            resource_floor();folder=root/'outputs'/f['name'];shutil.copytree(root/'inputs'/f['name'],folder)
            row=dict(name=f['name'],variant=f['variant'])
            try:
                if other_native():raise RuntimeError('Other same-user Dice live; do not stop/conflict')
                time.sleep(1.01-time.time()%1)
                for name,value in f['input_SHA256'].items():
                    if sha(folder/name)!=value:raise RuntimeError('Frozen tiny input changed')
                env=dict(os.environ);env.pop('CEO_DICE_PAIRED_INPUT_V1',None)
                if f['variant'] in ['CUSTOM_PAIRED','INVALID_PAIRED']:env['CEO_DICE_PAIRED_INPUT_V1']=f.get('env_value','1')
                exe=oldbin if f['variant']=='ORIGINAL_DEFAULT' else binary
                row.update(limited([str(exe)],folder,folder/'native.log',spec['fixture_wall_cap_seconds'],env))
                row['binary_SHA256']=sha(exe)
                row['status']='RETURNED_NOT_YET_PHYSICS_CHECKED'
                row['outputs']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in f['input_SHA256']}
                for name,value in f['input_SHA256'].items():
                    if sha(folder/name)!=value:raise RuntimeError('Native changed frozen input')
            except Exception as exc:row.update(status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
            rows.append(row);save(root/'progress.json',dict(planned=len(fixtures),terminal=len(rows),rows=rows));print(f['name'],row['status'],row.get('returncode'),flush=True)
        result=dict(status='ALL_TINY_ADAPTER_ATTEMPTS_TERMINAL_PENDING_INDEPENDENT_PHYSICS_AUDIT',rows=rows,build=b)
    except Exception as exc:result=dict(status='QUALIFICATION_ENGINEERING_FAILURE_PRESERVED',rows=rows,error=repr(exc),traceback=traceback.format_exc())
    end=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(CPU_seconds=time.process_time()-cpu+end.ru_utime+end.ru_stime-usage.ru_utime-usage.ru_stime,
        wall_seconds=time.perf_counter()-tick,original_source_unchanged=sha(main)==spec['original_main_SHA256'],
        original_binary_unchanged=sha(oldbin)==spec['original_binary_SHA256'],new_molecular_evaluations=0,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,production_molecular_execution_authorized=False,goal_complete=False)
    save(root/'terminal.json',result);print(result['status'],flush=True)
    if 'FAILURE' in result['status'] or not result['original_source_unchanged'] or not result['original_binary_unchanged']:raise SystemExit(2)


if __name__=='__main__':run(sys.argv[1])
