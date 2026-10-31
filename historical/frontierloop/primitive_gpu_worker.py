"""Standalone CUDA execution of a bound materialized primitive circuit.

CUDA kernels are reused BYTE-IDENTICALLY from the previously full32q-qualified
Givens engine. This new driver supports actual initial configurations, distinct
spin streams and phases. It must independently qualify before production.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from time import process_time,perf_counter
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    tmp=Path(str(path)+'.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def execute(config_path,arrays_path,output):
    import cupy as cp
    config=json.loads(Path(config_path).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if sha(arrays_path)!=config['arrays_SHA256'] or sha(__file__)!=config['worker_SHA256']:
        raise RuntimeError('Driver/array binding mismatch')
    base_path=Path(__file__).with_name('base_kernel.py')
    if sha(base_path)!=config['base_kernel_SHA256']:raise RuntimeError('Prior CUDA kernel source drift')
    import base_kernel as base
    if (output/'result.json').exists():return
    if (output/'started.json').exists():raise RuntimeError('Partial GPU work preserved; no blind retry')
    save(output/'started.json',{'input_id':config['input_id'],'job_id':os.environ.get('SLURM_JOB_ID')})
    with np.load(arrays_path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    cpu,tick=process_time(),perf_counter();n=config['norb'];ne=config['nelec']
    sa,sb=[base.spin_strings(n,e) for e in ne];da,db=len(sa),len(sb)
    if 16*da*db>12*2**30:raise MemoryError('Qualified single-vector allocation bound')
    wa,wb=[((s[:,None]>>np.arange(n))&1).astype(float) for s in (sa,sb)]
    sga,sgb=cp.asarray(sa),cp.asarray(sb)
    state=cp.zeros((da,db),dtype=cp.complex128)
    ia,ib=[sum(1<<i for i in occ) for occ in config['occupied']]
    state[int(np.searchsorted(sa,ia)),int(np.searchsorted(sb,ib))]=1.
    kernels={name:cp.RawKernel(base.CUDA,name,options=('--std=c++11',)) for name in ('givens','phases','jastrow')}
    grid=((da*db+255)//256,);mapping={};onea=cp.ones(da,dtype=cp.complex128);oneb=cp.ones(db,dtype=cp.complex128)
    for op in config['ops']:
        if op['kind']=='givens':
            spin=op['spin'];i,j=op['i'],op['j']
            if abs(i-j)!=1 or spin not in (0,1):raise ValueError('Unsupported Givens')
            key=(spin,i,j)
            if key not in mapping:
                a,b=base.pairs((sa,sb)[spin],i,j);mapping[key]=cp.asarray(a),cp.asarray(b)
            a,b=mapping[key];width=db if spin==0 else da
            kernels['givens'](((len(a)*width+255)//256,),(256,),
                (state,a,b,np.int64(da),np.int64(db),np.int32(len(a)),np.int32(spin),
                 np.float64(op['c']),np.float64(op['sr']),np.float64(op['si'])))
        elif op['kind']=='phase':
            spin=op['spin'];values=wa if spin==0 else wb
            p=cp.asarray(np.exp(1j*op['theta']*values[:,op['i']]))
            kernels['phases'](grid,(256,),(state,p if spin==0 else onea,oneb if spin==0 else p,np.int64(da),np.int64(db)))
        elif op['kind']=='jastrow':
            aa,ab,bb=arrays['diagonal_matrices'][op['index']];scale=-op['time']
            ea=cp.asarray(.5*scale*np.einsum('bi,ij,bj->b',wa,aa,wa))
            eb=cp.asarray(.5*scale*np.einsum('bi,ij,bj->b',wb,bb,wb));ia,ib=np.nonzero(ab)
            kernels['jastrow'](grid,(256,),(state,sga,sgb,ea,eb,cp.asarray(ia,dtype=cp.int32),
                cp.asarray(ib,dtype=cp.int32),cp.asarray(scale*ab[ia,ib]),np.int32(len(ia)),np.int64(da),np.int64(db)))
        else:raise ValueError('Unregistered primitive')
    cp.cuda.Stream.null.synchronize();build=perf_counter()-tick
    norm=float(cp.vdot(state,state).real.item())
    if abs(norm-1)>1e-8:raise RuntimeError('State normalization failed')
    parity=None
    if 'expected_sector_vector' in arrays:
        expected=cp.asarray(arrays['expected_sector_vector']).reshape(state.shape)
        overlap=cp.vdot(expected,state);aligned=state*cp.exp(-1j*cp.angle(overlap))
        parity={'state_max_error':float(cp.max(cp.abs(aligned-expected)).item()),
            'probability_L1_error':float(cp.sum(cp.abs(cp.abs(state)**2-cp.abs(expected)**2)).item())}
        if max(parity.values())>1e-8:raise RuntimeError('Independent circuit/GPU parity failed')
        del expected,aligned
    records=[]
    requests=config.get('sampling_requests',[])
    if requests:
        probability=cp.abs(state.ravel())**2;cdf=cp.cumsum(probability);cdf/=cdf[-1];del probability
        for request in requests:
            name=request['id'];save(output/(name+'-draw-started.json'),request)
            sampled=cp.searchsorted(cdf,cp.random.RandomState(request['seed']).random_sample(request['shots']))
            unique,counts=cp.unique(sampled,return_counts=True);idx,counts=cp.asnumpy(unique),cp.asnumpy(counts)
            ids=sa[idx//db].astype(np.uint64)|(sb[idx%db].astype(np.uint64)<<n)
            path=output/(name+'.npz');temporary=Path(str(path)+'.pending')
            with temporary.open('wb') as f:np.savez_compressed(f,blocked_ids=ids,counts=counts.astype(np.uint64))
            temporary.replace(path)
            record={**request,'actual_model_shots':int(counts.sum()),'counts_SHA256':sha(path)}
            save(output/(name+'-draw-completed.json'),record);records.append(record)
    result={'status':'EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED','input_id':config['input_id'],
        'physical_problem_id':config.get('physical_problem_id'),'source_QPY_SHA256':config['source_QPY_SHA256'],
        'independent_ffsim_parity':parity,'norm_squared':norm,'sector_dimension':da*db,
        'state_build_wall_seconds_including_CUDA_start':build,'whole_body_CPU_seconds':process_time()-cpu,
        'whole_body_wall_seconds':perf_counter()-tick,'GPU_memory_pool_retained_bytes':int(cp.get_default_memory_pool().total_bytes()),
        'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'cupy':cp.__version__,
        'job_id':os.environ.get('SLURM_JOB_ID'),'actual_QPU_shots':0,'sampling_results':records,
        'statevector_saved':False,'global_phase_only_ignored':True}
    save(output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':execute(*sys.argv[1:])
