"""Private pinned CPU dependencies and four source-bound C7 equation checks."""
import hashlib,json,os,resource,shutil,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)


def run(path):
    path=Path(path).resolve();root=path.parent;spec=json.loads(path.read_text())
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:
        raise RuntimeError('Registered private Slurm CPU runtime only')
    if sha(Path(__file__))!=spec['worker_SHA256']:raise RuntimeError('CPU setup worker changed')
    for name,value in spec['kernel_SHA256'].items():
        if sha(root/name)!=value:raise RuntimeError('Portable numerical kernel changed')
    tick=time.perf_counter();cpu=time.process_time();rows=[];site=root/'site'
    save(root/'started.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(path)))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        if shutil.disk_usage(root).free<3*1024**3:raise OSError('Private runtime free-space floor')
        site.mkdir(exist_ok=False)
        command=[sys.executable,'-m','pip','install','--target',str(site),'--ignore-installed','--only-binary=:all:',
            '--no-deps','--no-cache-dir','--report',str(root/'pip-report.json'),*spec['requirements']]
        save(root/'install-started.json',dict(command=command,scientific_candidate_outcomes=0))
        with (root/'pip-install.log').open('wb') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if used>512*1024**2:raise OSError('Private CPU runtime storage cap')
        sys.path[:0]=[str(site),str(root)]
        import importlib.metadata as metadata
        import numpy as np,pyscf,scipy
        pyscf.lib.num_threads(1)
        versions={name:metadata.version(name) for name in spec['packages']}
        if versions!=spec['packages']:raise RuntimeError('CPU runtime version mismatch')
        if any(not str(Path(module.__file__).resolve()).startswith(str(site)) for module in [np,pyscf,scipy]):
            raise RuntimeError('Numerical package escaped the private overlay')
        from kernels.portable_penalized_ci import evaluate
        save(root/'runtime.json',dict(versions=versions,python=sys.version,threads=pyscf.lib.num_threads(),bytes=used))
        for case in spec['cases']:
            folder=root/'inputs'/case['case'];start=time.perf_counter()
            try:
                if sha(folder/'integrals.npz')!=case['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
                with np.load(folder/'integrals.npz',allow_pickle=False) as a:
                    row=evaluate(a['h1'],a['eri'],case['core_energy'],tuple(case['nelec']),case['CI_strings'],seconds=60)
                errors={name:abs(row[name]-case['expected'][name]) for name in ['energy_Ha','penalized_energy_Ha','spin_square']}
                if max(errors.values())>1e-8:raise RuntimeError('Native-host common-CI numerical parity failure: '+str(errors))
                rows.append(dict(case=case['case'],status='C7_EQUATION_AND_SOURCE_ROOT_PARITY_PASSED',row=row,errors=errors))
            except Exception as exc:rows.append(dict(case=case['case'],status='FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc()))
            save(root/'progress.json',dict(rows=rows))
        status='PRIVATE_CPU_RUNTIME_FOUR_CASE_PARITY_PASSED' if len(rows)==4 and all(r['status']=='C7_EQUATION_AND_SOURCE_ROOT_PARITY_PASSED' for r in rows) else 'PRIVATE_CPU_RUNTIME_PARITY_INCOMPLETE'
        result=dict(status=status,rows=rows,versions=versions,runtime_bytes=used)
    except Exception as exc:
        result=dict(status='PRIVATE_CPU_SETUP_FAILURE_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),rows=rows)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(CPU_seconds_including_children=time.process_time()-cpu+usage.ru_utime+usage.ru_stime,
        wall_seconds=time.perf_counter()-tick,new_model_shots=0,new_QPU_shots=0,
        scientific_candidate_structures_created=0,CI_requalification_solves=len(rows),goal_complete=False)
    save(root/'result.json',result);print(result['status'],flush=True)


if __name__=='__main__':run(sys.argv[1])
