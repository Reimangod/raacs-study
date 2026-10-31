"""N1 additive build repair: retain v1 evidence, reuse its verified source cache.

Only missing Boost build helpers are extracted. Native scientific sources,
source versions, and all v1 logs/results remain unchanged.
"""
import hashlib,json,os,resource,shutil,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)


def run(spec_path):
    spec_path=Path(spec_path).resolve();spec=json.loads(spec_path.read_text());root=spec_path.parent
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:
        raise RuntimeError('Only registered private Linux Slurm job may build')
    if sha(Path(__file__))!=spec['worker_SHA256']:raise RuntimeError('Repair worker changed')
    prior=Path(spec['prior_root']);work=prior/'runtime';boost=work/'boost_1_85_0';source=work/'Dice'
    for name,value in spec['prior_evidence_SHA256'].items():
        if sha(prior/name)!=value:raise RuntimeError('Previous build evidence changed')
    original=json.loads((prior/'protocol.json').read_text());archive=work/'boost_1_85_0.tar.gz'
    if sha(archive)!=original['boost_SHA256']:raise RuntimeError('Verified archive changed')
    started=time.perf_counter();cpu=time.process_time();events=[]
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(spec_path)))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def command(name,argv,cwd):
        remaining=840-(time.perf_counter()-started)
        if remaining<=0:raise TimeoutError('Whole repair wall cap')
        before=resource.getrusage(resource.RUSAGE_CHILDREN);tick=time.perf_counter()
        with (root/(name+'.log')).open('wb') as log:
            proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            event=dict(name=name,argv=argv,pid=proc.pid,status='RUNNING');events.append(event);save(root/'progress.json',dict(events=events))
            try:rc=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        event.update(returncode=rc,status='COMPLETED' if rc==0 else 'FAILED_PRESERVED',wall_seconds=time.perf_counter()-tick,
            child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        save(root/'progress.json',dict(events=events))
        if rc:raise RuntimeError('Native build repair step failed: '+name)
    try:
        if shutil.disk_usage(work).free<original['disk_floor_bytes']:raise OSError('Private runtime free-space floor')
        added=[]
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.name.startswith('boost_1_85_0/tools/boost_install/'):continue
                target=work/member.name
                if target.exists():continue
                tar.extract(member,work,filter='data');added.append(member.name)
        save(root/'added-verified-build-helpers.json',dict(archive_SHA256=original['boost_SHA256'],paths=added))
        command('01-serialization',['./b2','-j1','variant=release','link=static','threading=multi','cxxstd=14',
            '--with-serialization','stage'],boost)
        (source/'obj/SHCI').mkdir(parents=True,exist_ok=True);(source/'bin').mkdir(exist_ok=True)
        library=boost/'stage/lib/libboost_serialization.a'
        if not library.exists():raise RuntimeError('Native private Boost archive missing')
        flags='-std=c++14 -O3 -g0 -w -fPIC -fopenmp -fpermissive -DSERIAL -I. -I./eigen -I./SHCI -I'+str(boost)
        command('02-native-Dice',['make','-j1','Dice','CXX=g++','HAS_AVX2=no','FLAGS_SHCI='+flags,
            'LFLAGS_SHCI='+str(library)+' -lrt'],source)
        command('03-native-source-unmodified',['git','diff','--exit-code'],source)
        binary=source/'bin/Dice';used=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
        if used>original['runtime_cap_bytes']:raise OSError('Private source/build cache storage cap')
        result=dict(status='NATIVE_DICE_SERIAL_BUILT_UNQUALIFIED_NO_SCIENTIFIC_OUTCOMES',binary=str(binary),
            binary_SHA256=sha(binary),dice_commit=original['dice_commit'],runtime_bytes=used)
    except Exception as exc:
        result=dict(status='BUILD_ONLY_REPAIR_FAILURE_PRESERVED_NOT_SCIENTIFIC_NEGATIVE',error=repr(exc),traceback=traceback.format_exc())
    unchanged=all(sha(prior/name)==value for name,value in spec['prior_evidence_SHA256'].items())
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(events=events,prior_evidence_unchanged=unchanged,wall_seconds=time.perf_counter()-started,
        CPU_seconds_including_children=time.process_time()-cpu+usage.ru_utime+usage.ru_stime,
        new_model_shots=0,new_QPU_shots=0,new_molecular_energy_evaluations=0,goal_complete=False)
    save(root/'result.json',result);print(result['status'],flush=True)
    if not unchanged or 'FAILURE' in result['status']:raise SystemExit(2)


if __name__=='__main__':run(sys.argv[1])
