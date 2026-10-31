"""Standalone NumPy/CuPy component profile; no Qiskit/PySCF required remotely."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse

CUDA_SOURCE=r'''
extern "C" __global__ void pauli_expect(
 const double2* state,const double2* hstate,const long long* src,int ns,
 const unsigned long long* masks,int nt,double2* out){
 int term=blockIdx.x; if(term>=nt)return;
 unsigned long long flip=masks[3*term],phase=masks[3*term+1],ny=masks[3*term+2];
 double r=0,im=0;
 for(int k=threadIdx.x;k<ns;k+=blockDim.x){
  unsigned long long i=src[k],dest=i^flip; double2 v=state[i],w=hstate[dest];
  double a=v.x,b=v.y,t; int p=ny&3;
  if(p==1){t=a;a=-b;b=t;} else if(p==2){a=-a;b=-b;} else if(p==3){t=a;a=b;b=-t;}
  if(__popcll(i&phase)&1){a=-a;b=-b;}
  r+=w.x*a+w.y*b; im+=w.x*b-w.y*a;
 }
 __shared__ double real_part[256],imag_part[256];
 real_part[threadIdx.x]=r;imag_part[threadIdx.x]=im;__syncthreads();
 for(int width=128;width>0;width>>=1){if(threadIdx.x<width){
  real_part[threadIdx.x]+=real_part[threadIdx.x+width];
  imag_part[threadIdx.x]+=imag_part[threadIdx.x+width];}__syncthreads();}
 if(threadIdx.x==0)out[term]=make_double2(real_part[0],imag_part[0]);
}
'''


def profile(fixture,backend,repetitions=7):
    began=time.perf_counter();import_start=time.perf_counter()
    xp=np;sp=sparse
    device=None
    if backend=="cuda":
        import cupy as cp
        from cupyx.scipy import sparse as csp
        xp=cp;sp=csp
        device={"name":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "device_id":int(cp.cuda.runtime.getDevice()),"cuda_runtime":cp.cuda.runtime.runtimeGetVersion()}
        if "A100" not in device["name"]:
            raise RuntimeError("Allocated GPU is not A100")
    def sync():
        if backend=="cuda":xp.cuda.Stream.null.synchronize()
    import_seconds=time.perf_counter()-import_start
    load=time.perf_counter()
    with np.load(fixture) as z:
        data={k:z[k] for k in z.files}
    input_load_seconds=time.perf_counter()-load
    n=int(data["n_qubits"]);d=1<<n;theta=data["theta"]
    transfer=time.perf_counter()
    h=sp.csr_matrix((xp.asarray(data["h_data"]),xp.asarray(data["h_indices"]),xp.asarray(data["h_indptr"])),shape=(d,d))
    ops=[sp.csr_matrix((xp.asarray(data[f"a{j}_data"]),xp.asarray(data[f"a{j}_indices"]),xp.asarray(data[f"a{j}_indptr"])),shape=(d,d)) for j in range(len(theta))]
    coef=sp.csr_matrix((xp.asarray(data["coef_data"]),xp.asarray(data["coef_indices"]),xp.asarray(data["coef_indptr"])),shape=(int(data["pool_size"]),len(data["masks"])))
    ci_h=xp.asarray(data["ci_h"])
    masks=xp.asarray(data["masks"])
    sync();transfer_seconds=time.perf_counter()-transfer
    init=xp.zeros(d,dtype=xp.complex128);init[int(data["hf_id"])]=1
    def apply(a,t,v):
        av=a@v
        return v+np.sin(t)*av+(1-np.cos(t))*(a@av)
    def prepare(keep=False):
        states=[init];v=init
        for a,t in zip(ops,theta):
            v=apply(a,t,v);states.append(v)
        return states if keep else v
    state=prepare();hs=h@state
    def energy():return xp.vdot(state,h@state).real
    def gradient():
        states=prepare(True);adj=h@states[-1];g=xp.empty(len(theta))
        for j in reversed(range(len(theta))):
            g[j]=2*xp.vdot(adj,ops[j]@states[j+1]).real
            adj=apply(ops[j],-theta[j],adj)
        return g
    if backend=="cuda":
        kernel=xp.RawKernel(CUDA_SOURCE,"pauli_expect")
        src=xp.flatnonzero(state!=0).astype(xp.int64)
        def pool_gradient():
            expectations=xp.empty(len(masks),dtype=xp.complex128)
            kernel((len(masks),),(256,),(state,hs,src,np.int32(len(src)),masks,np.int32(len(masks)),expectations))
            return 2*(coef@expectations).real
    else:
        src=np.flatnonzero(state!=0);amps=state[src]
        parity=np.array([i.bit_count()&1 for i in range(d)],dtype=np.int8)
        def pool_gradient():
            values=[]
            for flip,phase,ny in data["masks"]:
                signs=(1-2*parity[src & int(phase)])*(1j**int(ny))
                values.append(np.vdot(hs[src ^ int(flip)],signs*amps))
            return 2*(coef@np.array(values)).real
    functions={"state_preparation":prepare,"energy":energy,"parameter_gradient":gradient,
               "full_pool_gradient":pool_gradient,"ci_diagonalization":lambda:xp.linalg.eigvalsh(ci_h)}
    cold={};timings={};values={}
    for name,fun in functions.items():
        t=time.perf_counter();v=fun();sync();cold[name]=time.perf_counter()-t
        samples=[]
        for _ in range(repetitions):
            t=time.perf_counter();v=fun();sync();samples.append(time.perf_counter()-t)
        timings[name]={"median_seconds":float(np.median(samples)),"samples_seconds":samples}
        values[name]=xp.asnumpy(v) if backend=="cuda" else np.asarray(v)
    # Include repeated objective work, not just a tiny hot matvec, in a mixed workload.
    t=time.perf_counter()
    for _ in range(30):
        prepare();energy();gradient()
    pool_gradient();xp.linalg.eigvalsh(ci_h);sync()
    workload_seconds=time.perf_counter()-t
    return {"backend_requested":backend,"backend_executed":"CUDA_CUPY_FP64" if backend=="cuda" else "CPU_NUMPY_SCIPY_FP64",
            "device":device,"n_qubits":n,"parameters":len(theta),"pool_size":int(data["pool_size"]),
            "import_seconds":import_seconds,"input_load_seconds":input_load_seconds,
            "host_to_device_seconds":transfer_seconds,"cold_component_seconds":cold,
            "warm_components":timings,"objective_workload_30_seconds":workload_seconds,
            "worker_measured_total_seconds":time.perf_counter()-began,
            "fixture_sha256":hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
            "numpy_version":np.__version__,"timings_are_classical_simulation_not_quantum_hardware_time":True},values


def main():
    p=argparse.ArgumentParser();p.add_argument("--fixture",required=True);p.add_argument("--backend",choices=["cpu","cuda"],required=True)
    p.add_argument("--output",required=True);p.add_argument("--repetitions",type=int,default=7)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result,values=profile(a.fixture,a.backend,a.repetitions)
    np.savez_compressed(out/"values.npz",**values)
    (out/"profile.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
