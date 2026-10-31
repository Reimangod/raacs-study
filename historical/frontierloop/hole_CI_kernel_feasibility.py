"""N50 mechanical N49 timing-driver reuse with exact hole-coordinate H action.

Changed only namespace, helper binding and representation. No N49 file edited.

Not a CI eigensolve or source optimization. Each bounded child is owned by this
invocation; an observation timeout never causes an automatic retry.
"""
from pathlib import Path
from time import process_time,perf_counter
import json
import os
import resource
import signal
import subprocess
import sys
import numpy as np
from pyscf.fci import selected_ci, direct_spin1
from .paths import ROOT,OLD
from .io import load,save,sha256,space
from . import native_batch_contract as batch

LOCAL=ROOT/'runs/N50-SDK-CI-KERNEL-FEASIBILITY-v1'


from .hole_CI_action import action_inputs


def prepare():
    if (LOCAL/'protocol.json').exists(): raise RuntimeError('Already prepared; no overwrite')
    space(); source=batch.LOCAL/'terminal.json';z=load(source)
    if z['batch_counts']!=80 or z['new_CI_roots']!=0: raise ValueError('Dimension inventory differs')
    tasks=[]
    for method in batch.METHODS:
        row=next(r for r in z['rows'] if r['method']==method)
        b=next(b for b in row['batches'] if b['arm']==batch.ARMS[1] and b['batch']==0)
        tasks.append(dict(method=method,batch=b,source_payload_SHA256=row['source']['payload_SHA256']))
    f=OLD/'data/physical/FE2S2-40-P2/integrals.npz'
    save(LOCAL/'protocol.json',dict(tasks=tasks,N48_SHA256=sha256(source.read_bytes()),
          integrals=str(f),integrals_SHA256=sha256(f.read_bytes()),
          executor_SHA256=sha256(Path(__file__).read_bytes()),
          hole_helper_SHA256=sha256((ROOT/'frontierloop/hole_CI_action.py').read_bytes()),
          old_electron_kernel_SHA256=sha256((ROOT/'frontierloop/native_CI_kernel_feasibility.py').read_bytes()),
          integral_identity_helper_SHA256=sha256((ROOT/'frontierloop/energy_cache.py').read_bytes()),
          CPU_seconds_per_child=25,wall_seconds_per_child=35,
          measured_peak_RSS_limit_bytes=2*2**30,
          operation='TWO_REAL_H_VECTOR_PRODUCTS_AND_BILINEAR_HERMITICITY_CHECK',
          max_new_HVP=8,new_CI_eigensolves=0,new_model_samples=0,new_GPU_jobs=0))
    print('N50_KERNEL_ONLY_FROZEN4TASKS_NO_ROOTS')


def child(index):
    p=load(LOCAL/'protocol.json'); row=p['tasks'][index]
    if sha256(Path(__file__).read_bytes())!=p['executor_SHA256'] or sha256(Path(p['integrals']).read_bytes())!=p['integrals_SHA256']: raise ValueError('Frozen inputs changed')
    if sha256((ROOT/'frontierloop/hole_CI_action.py').read_bytes())!=p['hole_helper_SHA256'] or sha256((ROOT/'frontierloop/native_CI_kernel_feasibility.py').read_bytes())!=p['old_electron_kernel_SHA256'] or sha256((ROOT/'frontierloop/energy_cache.py').read_bytes())!=p['integral_identity_helper_SHA256']: raise ValueError('Exact representation helper drift')
    resource.setrlimit(resource.RLIMIT_CPU,(p['CPU_seconds_per_child'],p['CPU_seconds_per_child']+1))
    cpu,wall=process_time(),perf_counter()
    with np.load(p['integrals'],allow_pickle=False) as z:h,g=z['h1'],z['eri']
    b=row['batch']; hop,info=action_inputs(h,g,20,(15,15),b['half_strings'])
    rng=np.random.default_rng(4901);x=rng.normal(size=b['D']);y=rng.normal(size=b['D'])
    x/=np.linalg.norm(x); y/=np.linalg.norm(y)
    context=dict(method=row['method'],protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),**info)
    save(LOCAL/f'child-{index}-progress.json',dict(context,completed_HVP=0))
    t=process_time();hx=hop(x);first=process_time()-t
    save(LOCAL/f'child-{index}-progress.json',dict(context,completed_HVP=1,HVP_CPU_seconds=[first]))
    t=process_time();hy=hop(y);second=process_time()-t
    err=float(abs(x@hy-y@hx))
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    if not np.isfinite(hx).all() or not np.isfinite(hy).all() or err>1e-8: raise ValueError('Nonfinite H action or bilinear asymmetry')
    if rss>p['measured_peak_RSS_limit_bytes']: raise MemoryError('Measured peak memory over preflight bound')
    result=dict(context,status='COMPLETED',completed_HVP=2,HVP_CPU_seconds=[first,second],
          bilinear_hermiticity_error=err,peak_RSS_bytes=rss,
          CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
          output_hx_SHA256=sha256(hx.tobytes()),new_CI_eigensolves=0)
    save(LOCAL/f'child-{index}-result.json',result)
    print(json.dumps(result))


def run():
    if (LOCAL/'started.json').exists(): raise RuntimeError('Existing attempt; inspect child handles/results, do not repeat')
    space();p=load(LOCAL/'protocol.json');cpu,wall=process_time(),perf_counter()
    if sha256((batch.LOCAL/'terminal.json').read_bytes())!=p['N48_SHA256']: raise ValueError('Batch inventory changed')
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    results=[]
    for i,task in enumerate(p['tasks']):
        command=['rtk','proxy',sys.executable,'-m','frontierloop.hole_CI_kernel_feasibility','child',str(i)]
        started=perf_counter();proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        save(LOCAL/f'child-{i}-intent.json',dict(pid=proc.pid,command=command,method=task['method']))
        timed=False
        try:out,err=proc.communicate(timeout=p['wall_seconds_per_child'])
        except subprocess.TimeoutExpired:
            timed=True;os.killpg(proc.pid,signal.SIGTERM)
            try:out,err=proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);out,err=proc.communicate()
        if proc.returncode==0:
            row=load(LOCAL/f'child-{i}-result.json')
        else:
            row=dict(method=task['method'],status='FEASIBILITY_CAP_OR_ENGINEERING_STOP',
                     timeout=timed,returncode=proc.returncode,stderr=err[-8000:],
                     stdout=out[-4000:],CPU_seconds=None,
                     progress=load(LOCAL/f'child-{i}-progress.json') if (LOCAL/f'child-{i}-progress.json').exists() else None)
        row=dict(row,child_launch_wall_seconds=perf_counter()-started)
        save(LOCAL/f'cell-{i}.json',row);results.append(row)
        print('N50_CELL',i,row['status'],flush=True)
    save(LOCAL/'terminal.json',dict(status='N50_ALL4_KERNEL_FEASIBILITY_TERMINAL',rows=results,
         protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
         parent_CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
         child_CPU_seconds_known_subtotal=sum(r['CPU_seconds'] for r in results if r['CPU_seconds'] is not None),
         unknown_failed_child_CPU_not_zero=any(r['CPU_seconds'] is None for r in results),
         new_CI_eigensolves=0,new_model_samples=0,new_GPU_jobs=0,goal_complete=False))


if __name__=='__main__':
    if sys.argv[1]=='child':child(int(sys.argv[2]))
    else:{'prepare':prepare,'run':run}[sys.argv[1]]()
