"""N53 actual CUDA H/S2 qualification and fixed-large-space profiling only."""
from pathlib import Path
from time import process_time,perf_counter
import gc,hashlib,json,os,sys,traceback
import numpy as np
from structured_CI_device import StructuredAction,dense_probes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def one_case(cp,root,case,tile,tolerance):
    cpu,wall=process_time(),perf_counter();folder=root/'out'/case['id']
    save(folder/'started.json',dict(case_id=case['id'],job_id=os.environ['SLURM_JOB_ID']))
    file=root/case['file']
    if sha(file)!=case['SHA256']:raise ValueError('Frozen mathematical input changed')
    with np.load(file,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    dev=StructuredAction(cp,data,tile=tile);cp.cuda.Stream.null.synchronize()
    setup_wall=perf_counter()-wall;setup_CPU=process_time()-cpu
    peak=cp.get_default_memory_pool().total_bytes()
    def measured(func,v):
        nonlocal peak
        cp.cuda.Stream.null.synchronize();start=cp.cuda.Event();stop=cp.cuda.Event()
        start.record();t=perf_counter();out=func(v);stop.record();stop.synchronize()
        peak=max(peak,cp.get_default_memory_pool().total_bytes())
        if out.dtype!=cp.float64:raise ValueError('Non-float64 mathematical action')
        return out,dict(GPU_event_seconds=cp.cuda.get_elapsed_time(start,stop)*.001,
                        wall_seconds=perf_counter()-t)
    sparse=cp.asarray(data['sparse_vector']);hs,th=measured(dev.physical_H,sparse);ss,ts=measured(dev.spin_S2,sparse)
    errors=dict(sparse_H_full_max=float(cp.max(cp.abs(hs-cp.asarray(data['sparse_H']))).item()),
                sparse_S2_full_max=float(cp.max(cp.abs(ss-cp.asarray(data['sparse_S2']))).item()))
    # Constants and the registered spin penalty are also checked, not absorbed
    # into an energy offset inferred from a label.
    errors['sparse_penalized_with_core_max']=float(cp.max(cp.abs(hs+.1*ss+float(data['core_shift'])*sparse-cp.asarray(data['sparse_total']))).item())
    X=dense_probes(dev.R**2,case['dense_seed'])
    if hashlib.sha256(X.astype('<f8').tobytes()).hexdigest()!=case['dense_input_SHA256']:
        raise ValueError('CPU generated profiling-vector bytes differ')
    x,y=cp.asarray(X[0]),cp.asarray(X[1]);hx,tx=measured(dev.physical_H,x);hy,ty=measured(dev.physical_H,y);sx,tsx=measured(dev.spin_S2,x)
    observed={'x_H_x':float((x@hx).item()),'x_H_y':float((x@hy).item()),'y_H_x':float((y@hx).item()),
              'H_x_norm':float(cp.linalg.norm(hx).item()),'S_x_norm':float(cp.linalg.norm(sx).item()),'x_S_x':float((x@sx).item())}
    for key,value in observed.items():errors[key]=abs(value-case['dense_oracle'][key])
    indices=cp.asarray(data['probe_indices'])
    errors['dense_H_selected_components']=float(cp.max(cp.abs(hx[indices]-cp.asarray(data['dense_H_components']))).item())
    errors['dense_S2_selected_components']=float(cp.max(cp.abs(sx[indices]-cp.asarray(data['dense_S2_components']))).item())
    if 'dense_H_full' in data:
        errors['dense_H_full_max']=float(cp.max(cp.abs(hx-cp.asarray(data['dense_H_full']))).item())
        errors['dense_S2_full_max']=float(cp.max(cp.abs(sx-cp.asarray(data['dense_S2_full']))).item())
    if max(errors.values())>tolerance:raise ValueError('CUDA independent parity failed: '+str(errors))
    result=dict(status='QUALIFIED_ACTION_ONLY',case_id=case['id'],R=dev.R,D=dev.R**2,
        errors=errors,setup_wall_seconds=setup_wall,setup_CPU_seconds=setup_CPU,
        sparse_H_timing=th,sparse_S2_timing=ts,dense_H_timings=[tx,ty],dense_S2_timing=tsx,
        peak_observed_memory_pool_reserved_bytes=peak,retained_factor_count=len(data['factors']),
        ERI_factor_cutoff=None,CPU_reference=case['CPU_reference'],
        full_wall_seconds=perf_counter()-wall,full_CPU_seconds=process_time()-cpu,
        same_CUDA_context_shared_across_cases=True,source_problem=case['source_problem'],
        no_dense_D_by_D_H=True,new_CI_eigensolves=0)
    save(folder/'result.json',result)
    return result


def run(root):
    if (root/'out/started.json').exists():raise RuntimeError('Existing worker; do not replay')
    cpu,wall=process_time(),perf_counter();p=json.loads((root/'protocol.json').read_text())
    for name,h in p['input_SHA256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen worker/input changed: '+name)
    import cupy as cp
    if cp.__version__!=p['expected_cupy']:raise ValueError('Unqualified CuPy version')
    if cp.cuda.runtime.getDeviceCount()!=1:raise ValueError('Exactly one Slurm-assigned GPU required')
    props=cp.cuda.runtime.getDeviceProperties(0);name=props['name']
    if isinstance(name,bytes):name=name.decode()
    if 'A100' not in name:raise ValueError('Expected actual A100')
    cp.get_default_memory_pool().set_limit(size=p['memory_pool_cap_bytes'])
    context=dict(job_id=os.environ.get('SLURM_JOB_ID'),device_name=name,cupy=cp.__version__,numpy=np.__version__,
        device_total_memory_bytes=int(props['totalGlobalMem']),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        CUDA_runtime_version=cp.cuda.runtime.runtimeGetVersion(),CUDA_driver_version=cp.cuda.runtime.driverGetVersion(),
        protocol_SHA256=sha(root/'protocol.json'))
    save(root/'out/started.json',context);rows=[]
    try:
        for case in p['cases']:
            if perf_counter()-wall>p['body_wall_cap_seconds']:raise TimeoutError('N53 profiling body cap')
            rows.append(one_case(cp,root,case,p['tile'],p['action_tolerance']))
            save(root/'out/progress.json',dict(context,rows=rows))
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        out=dict(context,status='ALL_CUDA_ACTION_QUALIFICATIONS_COMPLETED',rows=rows,
            whole_CPU_seconds=process_time()-cpu,whole_wall_seconds=perf_counter()-wall,
            new_model_samples=0,new_QPU_shots=0,new_CI_eigensolves=0,
            whole_pipeline_speedup_certified=False,goal_complete=False)
        save(root/'out/terminal.json',out);print(out['status'],len(rows),flush=True)
    except Exception as exc:
        save(root/'out/failure.json',dict(context,status='CUDA_QUALIFICATION_STOPPED_PRESERVED',rows=rows,
            error=repr(exc),traceback=traceback.format_exc(),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            new_model_samples=0,new_CI_eigensolves=0))
        raise


if __name__=='__main__':run(Path(sys.argv[1]))
