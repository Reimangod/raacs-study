"""N5: common explicit sparse kernel for product and arbitrary-joint readouts."""
import hashlib,json,os,shutil,sys,time,traceback,resource
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,p)


def run(path):
    path=Path(path).resolve();root=path.parent;spec=json.loads(path.read_text())
    if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or str(root)!=spec['remote_root']:
        raise RuntimeError('Registered private Slurm study only')
    if sha(__file__)!=spec['worker_SHA256']:raise RuntimeError('Frozen worker drift')
    for name,value in spec['pipeline_SHA256'].items():
        if sha(root/'pipeline'/name)!=value:raise RuntimeError('Frozen sparse evaluator changed')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        if os.environ.get(key)!='1':raise RuntimeError('Parent numerical threads must be set before process startup')
    (root/'owner.claim').open('x').write(str(os.getpid()))
    runtime=Path(spec['common_runtime']);sys.path[:0]=[str(runtime/'site'),str(runtime),str(root)]
    import importlib.metadata as metadata
    import numpy as np
    from pyscf import lib
    from pipeline.joint_ci_sparse import build_library,solve
    from kernels.portable_penalized_ci import evaluate as legacy_evaluate
    lib.num_threads(1)
    if {name:metadata.version(name) for name in spec['packages']}!=spec['packages']:raise RuntimeError('Private numerical versions changed')
    for name,value in spec['legacy_kernel_SHA256'].items():
        if sha(runtime/name)!=value:raise RuntimeError('Timing comparator kernel changed')
    start,cpu=time.perf_counter(),time.process_time();children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    library=build_library(root/'library');physical={}
    save(root/'started.json',dict(job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),protocol_SHA256=sha(path),
        thread_env={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},pyscf_threads=lib.num_threads()))
    for case,definition in spec['physical'].items():
        p=Path(definition['integrals_path'])
        if sha(p)!=definition['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
        with np.load(p,allow_pickle=False) as a:physical[case]=(a['h1'],a['eri'])
    for plan in spec['plans']:
        if sha(root/plan['basis_file'])!=plan['basis_SHA256']:raise RuntimeError('Frozen joint/product basis changed')
    cache={};observations=[];attempts=[];states=[]
    def evaluate(case,pairs,origin):
        key=(case,hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest())
        if key in cache:
            item=cache[key];return dict(row=item['row'],observation_id=item['id'],evaluation_origin=item['origin'],new_CI_solve=False)
        definition=spec['physical'][case];h,g=physical[case];at=dict(case=case,basis_digest=key[1],origin=origin,status='STARTED')
        attempts.append(at);save(root/'attempts.json',attempts);c,w=time.process_time(),time.perf_counter()
        try:
            result,vector=solve(h,g,definition['core_energy'],tuple(definition['nelec']),pairs,library,
                seconds=spec['policy']['root_wall_seconds'],triangle_cap=spec['policy']['triangle_entry_cap'])
            oid=hashlib.sha256((case+key[1]+sha(path)).encode()).hexdigest();vpath=root/'roots'/(oid+'.npz');vpath.parent.mkdir(exist_ok=True)
            np.savez_compressed(vpath,coefficients=vector,joint_pairs=pairs)
            item=dict(id=oid,case=case,basis_digest=key[1],origin=origin,row=result,
                vector_file=str(vpath.relative_to(root)),vector_SHA256=sha(vpath))
            cache[key]=item;observations.append(item)
            at.update(status='COMPLETED',CPU_seconds=time.process_time()-c,wall_seconds=time.perf_counter()-w,observation_id=oid)
            save(root/'observations.json',observations);save(root/'attempts.json',attempts)
            return dict(row=result,observation_id=oid,evaluation_origin=origin,new_CI_solve=True)
        except Exception as exc:
            at.update(status='FAILED_PRESERVED',CPU_seconds=time.process_time()-c,wall_seconds=time.perf_counter()-w,
                error=repr(exc),matvec_calls=None);save(root/'attempts.json',attempts);raise
    # Four real FULL product spaces must reproduce qualified C7 roots BEFORE
    # any newly selected arbitrary-joint molecular candidate is evaluated.
    qualified=[];legacy_rows=[];qualification_passed=False
    try:
        for q in spec['qualification']:
            with np.load(root/q['basis_file'],allow_pickle=False) as a:pairs=a['product']
            result=evaluate(q['case'],pairs,'QUALIFICATION:'+q['case']);r=result['row'];expected=q['expected']
            errors={k:abs(r[k]-expected[k]) for k in ['energy_Ha','penalized_energy_Ha','spin_square']}
            if max(errors.values())>1e-8:raise RuntimeError('Real full-product common-CI parity mismatch '+str(errors))
            qualified.append(dict(case=q['case'],errors=errors,result=result))
            save(root/'qualification.json',dict(status='IN_PROGRESS',rows=qualified))
            # Same-machine, explicitly one-thread product-space timing control.
            # Its per-root RDM probes differ from the new kernel's prequalified
            # matrix tests; record this validation-work difference honestly.
            old_start,old_cpu=time.perf_counter(),time.process_time();old=dict(case=q['case'])
            try:
                definition=spec['physical'][q['case']];h,g=physical[q['case']]
                strings=sorted(set(map(int,pairs[:,0])))
                prior=legacy_evaluate(h,g,definition['core_energy'],tuple(definition['nelec']),strings,
                    seconds=spec['policy']['root_wall_seconds'])
                differences={k:abs(r[k]-prior[k]) for k in ['energy_Ha','penalized_energy_Ha','spin_square']}
                old.update(status='COMPLETED',row=prior,errors_vs_explicit=differences)
                if max(differences.values())>1e-8:raise RuntimeError('Live same-thread product parity failed')
            except Exception as exc:
                old.update(status='FAILED_PRESERVED',error=repr(exc))
                if 'errors_vs_explicit' in old:raise
            finally:
                old.update(actual_CPU_seconds=time.process_time()-old_cpu,actual_wall_seconds=time.perf_counter()-old_start)
                legacy_rows.append(old);save(root/'legacy-comparisons.json',legacy_rows)
        qualification_passed=True
        save(root/'qualification.json',dict(status='ALL4_REAL_FULL_PRODUCT_C7_PARITY_PASSED',rows=qualified))
    except Exception as exc:
        save(root/'qualification.json',dict(status='FAILED_PRESERVED_NO_JOINT_CANDIDATES',rows=qualified,error=repr(exc),traceback=traceback.format_exc()))
    for plan in spec['plans']:
        record=dict(task_id=plan['task_id'],case=plan['case'],method=plan['method'],arms={},errors={},new_model_shots=0,new_QPU_shots=0)
        if not qualification_passed:record['status']='NOT_EXECUTED_FULL_PRODUCT_QUALIFICATION_FAILED'
        elif time.perf_counter()-start>3600 or shutil.disk_usage(root).free<3*1024**3 or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>128*1024**2:
            record['status']='NOT_EXECUTED_GLOBAL_RESOURCE_CAP'
        else:
            with np.load(root/plan['basis_file'],allow_pickle=False) as a:
                for arm,key in [('PRODUCT_R128','product'),('JOINT_TOP_D16384','joint')]:
                    try:record['arms'][arm]=evaluate(plan['case'],a[key],plan['task_id']+':'+arm)
                    except Exception as exc:record['errors'][arm]=dict(error=repr(exc),traceback=traceback.format_exc())
                    save(root/'progress-active.json',record)
            record['status']='BOTH_READOUTS_COMPLETED' if len(record['arms'])==2 else 'PARTIAL_OR_FAILED_PRESERVED'
        states.append(record);save(root/'cells'/(plan['task_id']+'.json'),record)
        save(root/'progress.json',dict(planned=20,terminal=len(states),states=states))
        print('N5',plan['case'],plan['method'],record['status'],flush=True)
    children_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(root/'terminal.json',dict(status='ALL20_SOURCE_CELLS_TERMINAL_WITH_QUALIFICATION_AND_MISSINGNESS',
        states=states,actual_unique_CI_roots=len(observations),actual_attempts=len(attempts),
        legacy_timing_attempts=len(legacy_rows),legacy_timing_successes=sum(r['status']=='COMPLETED' for r in legacy_rows),
        CPU_seconds=time.process_time()-cpu+children_after.ru_utime+children_after.ru_stime-children_before.ru_utime-children_before.ru_stime,wall_seconds=time.perf_counter()-start,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,qualification_passed=qualification_passed,
        same_final_dimension_cap_not_equal_actual_dimension=True,independent_quantum_confirmation=False,goal_complete=False))


if __name__=='__main__':run(sys.argv[1])
