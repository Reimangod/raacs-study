"""Isolated serial native DICE build, no scientific outcomes or system install.

Run only inside the registered Linux Slurm job. Source and archive pins are
checked before compilation; original Makefiles and kernels are not edited.
"""
import hashlib,json,os,resource,shutil,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from urllib.request import urlopen


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)


def boost_member(name):
    parts=Path(name).parts
    if not parts or parts[0]!='boost_1_85_0' or '..' in parts:return False
    return len(parts)<=2 or parts[1]=='boost' or parts[1:3]==('tools','build') or (
        len(parts)>=3 and parts[1]=='libs' and parts[2] in {'serialization','headers','config','predef'})


def run(spec_path):
    spec_path=Path(spec_path).resolve();spec=json.loads(spec_path.read_text());root=spec_path.parent
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:
        raise RuntimeError('Only registered private Linux Slurm job may build')
    if sha(Path(__file__))!=spec['worker_SHA256']:raise RuntimeError('Worker changed')
    work=root/'runtime';work.mkdir(exist_ok=False);started=time.perf_counter();cpu=time.process_time();events=[]
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(spec_path),scientific_outcomes=0))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')

    def remaining():
        result=spec['max_wall_seconds']-(time.perf_counter()-started)
        if result<=0:raise TimeoutError('Registered whole build wall cap')
        return result

    def guard():
        remaining()
        if shutil.disk_usage(work).free<spec['disk_floor_bytes']:raise OSError('Private runtime free-space floor')
        used=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
        if used>spec['runtime_cap_bytes']:raise OSError('Private runtime storage cap')
        return used

    def command(name,argv,cwd):
        limit=remaining();before=resource.getrusage(resource.RUSAGE_CHILDREN);tick=time.perf_counter()
        with (root/(name+'.log')).open('wb') as log:
            proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            event=dict(name=name,argv=argv,cwd=str(cwd),pid=proc.pid,status='RUNNING');events.append(event)
            save(root/'progress.json',dict(events=events))
            try:rc=proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                event['status']='OWN_BUILD_GROUP_TIME_LIMIT';save(root/'progress.json',dict(events=events));raise
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        event.update(status='COMPLETED' if rc==0 else 'FAILED_PRESERVED',returncode=rc,
            wall_seconds=time.perf_counter()-tick,child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        save(root/'progress.json',dict(events=events))
        if rc:raise RuntimeError('Native build step failed: '+name)

    def download(url,path,expected,max_bytes):
        total=0;h=hashlib.sha256()
        with urlopen(url,timeout=30) as response,path.open('xb') as out:
            while True:
                remaining();block=response.read(1048576)
                if not block:break
                total+=len(block)
                if total>max_bytes:raise OSError('Bounded source download size exceeded')
                h.update(block);out.write(block)
        if h.hexdigest()!=expected:raise RuntimeError('Pinned source SHA256 mismatch')

    try:
        guard();source=work/'Dice'
        command('01-init',['git','init',str(source)],work)
        command('02-remote',['git','remote','add','origin','https://github.com/caleb-johnson/Dice.git'],source)
        command('03-partial',['git','config','remote.origin.promisor','true'],source)
        command('04-filter',['git','config','remote.origin.partialclonefilter','blob:none'],source)
        command('05-cone',['git','sparse-checkout','init','--cone'],source)
        command('06-sparse',['git','sparse-checkout','set','SHCI','eigen'],source)
        command('07-fetch',['git','fetch','--filter=blob:none','--depth=1','origin',spec['dice_commit']],source)
        command('08-checkout',['git','checkout','--detach','FETCH_HEAD'],source)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        if head!=spec['dice_commit']:raise RuntimeError('Native DICE commit changed')
        download(spec['wrapper_url'],work/'pinned_dice_solver.py',spec['wrapper_SHA256'],200000)
        archive=work/'boost_1_85_0.tar.gz'
        download(spec['boost_url'],archive,spec['boost_SHA256'],160*1024**2)
        guard()
        # The full archive is retained and verified. Extract only Boost headers,
        # serialization and its build system, not unrelated docs/tests/libraries.
        with tarfile.open(archive) as tar:
            for member in tar:
                if boost_member(member.name):tar.extract(member,work,filter='data')
        guard();boost=work/'boost_1_85_0'
        command('09-bootstrap',['./bootstrap.sh','--with-libraries=serialization','--with-toolset=gcc'],boost)
        command('10-serialization',['./b2','-j1','variant=release','link=static','threading=multi',
            'cxxstd=14','--with-serialization','stage'],boost)
        guard();(source/'obj/SHCI').mkdir(parents=True,exist_ok=True);(source/'bin').mkdir(exist_ok=True)
        flags='-std=c++14 -O3 -g0 -w -fPIC -fopenmp -fpermissive -DSERIAL -I. -I./eigen -I./SHCI -I'+str(boost)
        library=boost/'stage/lib/libboost_serialization.a'
        if not library.exists():raise RuntimeError('Pinned private Boost serialization missing')
        command('11-native-Dice',['make','-j1','Dice','CXX=g++','HAS_AVX2=no',
            'FLAGS_SHCI='+flags,'LFLAGS_SHCI='+str(library)+' -lrt'],source)
        binary=source/'bin/Dice'
        if not binary.is_file():raise RuntimeError('Native target missing')
        command('12-native-source-unmodified',['git','diff','--exit-code'],source)
        result=dict(status='NATIVE_DICE_SERIAL_BUILT_UNQUALIFIED_NO_SCIENTIFIC_OUTCOMES',
            binary=str(binary),binary_SHA256=sha(binary),dice_commit=head,runtime_bytes=guard())
    except Exception as exc:
        result=dict(status='BUILD_ONLY_FAILURE_PRESERVED_NOT_SCIENTIFIC_NEGATIVE',error=repr(exc),traceback=traceback.format_exc())
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(events=events,wall_seconds=time.perf_counter()-started,
        CPU_seconds_including_children=time.process_time()-cpu+usage.ru_utime+usage.ru_stime,
        new_model_shots=0,new_QPU_shots=0,new_molecular_energy_evaluations=0,goal_complete=False)
    save(root/'result.json',result);print(result['status'],flush=True)
    if 'FAILURE' in result['status']:raise SystemExit(2)


if __name__=='__main__':run(sys.argv[1])
