"""N54 two independent one-A100 lanes; same N51 physical SCI spaces."""
from pathlib import Path
from time import process_time,perf_counter
import gc,hashlib,importlib.metadata as metadata,json,os,resource,sys,traceback
import numpy as np
from hybrid_CI_solver import solve
from structured_CI_device import StructuredAction


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def cell(cp,root,p,c,lane):
    cpu,wall=process_time(),perf_counter();folder=root/'out'/lane/c['id']
    if (folder/'started.json').exists():raise RuntimeError('Do not repeat an existing cell')
    save(folder/'started.json',dict(case_id=c['id'],job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(root/'protocol.json')))
    counts=dict(physical_H_calls=0,S2_calls=0,host_device_action_wall_seconds=0.)
    try:
        with np.load(root/c['file'],allow_pickle=False) as a:data={k:a[k] for k in a.files}
        dev=StructuredAction(cp,data,tile=8);cp.cuda.Stream.null.synchronize()
        setup=perf_counter()-wall
        def hop(x):
            t=perf_counter();v=cp.asarray(x,dtype=cp.float64)
            y=dev.penalized(v);counts['physical_H_calls']+=1;counts['S2_calls']+=1
            result=cp.asnumpy(y);counts['host_device_action_wall_seconds']+=perf_counter()-t
            return result
        row,v=solve(hop,data['exact_diagonal'],data['initial_vector'],max_actions=p['max_H_actions']-1,
            wall_seconds=p['case_wall_seconds'],CPU_seconds=p['case_CPU_seconds'],
            progress=lambda z:save(folder/'progress.json',dict(case_id=c['id'],**z,**counts)))
        cv=cp.asarray(v);hv=cp.asnumpy(dev.physical_H(cv));sv=cp.asnumpy(dev.spin_S2(cv))
        counts['physical_H_calls']+=1;counts['S2_calls']+=1
        physical=float(v@hv+float(data['core_shift']));spin=float(v@sv)
        objective=float(row['objective_without_core']+float(data['core_shift']))
        residual=float(np.linalg.norm(hv+.1*sv-row['objective_without_core']*v))
        if (abs(objective-physical-.1*spin)>1e-8 or residual>1e-8 or counts['physical_H_calls']>p['max_H_actions']):
            raise ValueError('Final physical energy/spin/residual or primitive cap differs')
        row.update(energy_Ha=physical,penalized_energy_Ha=objective,spin_square=spin,residual_norm=residual,
            common_equation=p['equation'],actual_CI_dimension=c['D'],actual_half_dimension=c['R'],
            particle_hole_used=c['molecular'],setup_wall_seconds=setup,**counts)
        if c.get('oracle'):
            errors={k:abs(row[k]-x) for k,x in c['oracle'].items()}
            if max(errors.values())>1e-7:raise ValueError('Independent qualification energy/spin differs '+str(errors))
            row['qualification_errors']=errors
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if rss>p['RSS_cap_bytes']:raise MemoryError('Registered host RSS cap exceeded')
        tmp=folder/'root.npz.tmp'
        with tmp.open('wb') as f:
            np.savez_compressed(f,coefficients=v,solver_strings=data['strings'],
                electron_strings=data['electron_strings'],solver_nelec=np.array([int(data['k'])]*2),
                electron_nelec=data['electron_nelec'],norb=data['n'],particle_hole_used=np.bool_(c['molecular']),
                problem_id=np.array(c.get('problem_id','SYNTHETIC')),task_id=np.array(c.get('N51_task_id',c['id'])),
                penalized_energy_Ha=np.float64(objective))
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,folder/'root.npz')
        result=dict(status='COMPLETED',case_id=c['id'],row=row,root_SHA256=sha(folder/'root.npz'),
            peak_host_RSS_bytes=rss,memory_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes())
    except Exception as exc:
        result=dict(status='LIMIT_OR_ENGINEERING_FAILURE_PRESERVED',case_id=c['id'],
            error=repr(exc),traceback=traceback.format_exc(),**counts)
    result.update(CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        job_id=os.environ['SLURM_JOB_ID'],protocol_SHA256=sha(root/'protocol.json'))
    save(folder/'result.json',result);print('N54_CELL',lane,c['id'],result['status'],flush=True)
    return result


def run(root,lane):
    from pyscf import lib
    import cupy as cp
    cpu,wall=process_time(),perf_counter();p=json.loads((root/'protocol.json').read_text())
    for file,h in p['input_SHA256'].items():
        if sha(root/file)!=h:raise ValueError('Frozen scientific packet/worker changed')
    versions={k:metadata.version(k) for k in p['numerical_packages']}
    if versions!=p['numerical_packages'] or cp.__version__!=p['cupy_version']:raise ValueError('Qualified numerical versions differ')
    if cp.cuda.runtime.getDeviceCount()!=1:raise ValueError('Exactly one assigned GPU required')
    props=cp.cuda.runtime.getDeviceProperties(0);device=props['name']
    if isinstance(device,bytes):device=device.decode()
    if 'A100' not in device:raise ValueError('Actual A100 required')
    lib.num_threads(1);cp.get_default_memory_pool().set_limit(size=p['GPU_pool_cap_bytes'])
    folder=root/'out'/lane
    if (folder/'started.json').exists():raise RuntimeError('Existing lane marker; no replay')
    context=dict(job_id=os.environ['SLURM_JOB_ID'],device=device,versions=versions,
        protocol_SHA256=sha(root/'protocol.json'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    save(folder/'started.json',context);rows=[]
    for c in p['qualifications']+[p['production'][lane]]:
        r=cell(cp,root,p,c,lane);rows.append(r)
        save(folder/'progress.json',dict(context,rows=rows))
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
        if r['status']!='COMPLETED':break
    save(folder/'terminal.json',dict(context,status='PARTITION_TERMINAL',rows=rows,
        all_cells_completed=len(rows)==len(p['qualifications'])+1 and all(r['status']=='COMPLETED' for r in rows),
        whole_CPU_seconds=process_time()-cpu,whole_wall_seconds=perf_counter()-wall,
        new_model_samples=0,new_QPU_shots=0,full_paper_reproduction=False,goal_complete=False))


if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2])
