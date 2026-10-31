"""Same Pauli-defined excitation generators on NumPy or CuPy, complex128.

All terms of an excitation share a single computational-basis flip mask.
Its anti-Hermitian generator therefore consists of disjoint1x1/2x2 blocks.
Exponentiating these blocks is exact, not a Trotter or pruning approximation.
Unsupported multi-flip generators are explicitly rejected, not approximated.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .gpu_profile_worker import CUDA_SOURCE


class ExcitationBackend:
    def __init__(self,h,catalog,backend="numpy"):
        self.backend=backend;self.catalog=catalog
        self.n=catalog["n_qubits"];self.dim=1<<self.n
        self.xp=np;sp=sparse
        if backend=="cupy":
            import cupy as cp
            from cupyx.scipy import sparse as csp
            self.xp=cp;sp=csp
            self.device_name=cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            if "A100" not in self.device_name:
                raise RuntimeError("Production CUDA backend requires the qualified A100 allocation")
        elif backend!="numpy":
            raise ValueError(backend)
        xp=self.xp
        h=h.tocsr()
        self.h=sp.csr_matrix((xp.asarray(h.data),xp.asarray(h.indices),xp.asarray(h.indptr)),shape=h.shape)
        self.ids=xp.arange(self.dim,dtype=xp.uint64)
        self.parity=xp.asarray(np.array([i.bit_count()&1 for i in range(self.dim)],dtype=np.int8))
        terms={};rows=[];columns=[];coefficients=[];self.op_flips=[]
        for operator in catalog["operators"]:
            flips=set()
            for term in operator["terms"]:
                flip=phase=ny=0
                for q,p in term["pauli"]:
                    bit=1 << (self.n-1-q)
                    if p in ("X","Y"):flip|=bit
                    if p in ("Y","Z"):phase|=bit
                    ny+=p=="Y"
                key=(flip,phase,ny)
                position=terms.setdefault(key,len(terms))
                rows.append(operator["index"]);columns.append(position)
                coefficients.append(complex(*term["coefficient"]))
                flips.add(flip)
            if len(flips)!=1:
                raise NotImplementedError("Multi-flip generator: exact two-level backend not applicable")
            self.op_flips.append(next(iter(flips)))
        self.masks_host=np.array(list(terms),dtype=np.uint64)
        self.masks=xp.asarray(self.masks_host)
        self.coef_host=sparse.csr_matrix((coefficients,(rows,columns)),shape=(len(self.op_flips),len(terms)))
        self.coef=sp.csr_matrix((xp.asarray(self.coef_host.data),xp.asarray(self.coef_host.indices),xp.asarray(self.coef_host.indptr)),shape=self.coef_host.shape)
        self.generators={}
        if backend=="cupy":
            self.rotate_kernel=xp.ElementwiseKernel(
                "raw complex128 state, raw complex128 gen, uint64 flip, float64 angle",
                "complex128 output",
                "double w=abs(gen[i]); output=cos(angle*w)*state[i]; "
                "if(w>0) output+=(sin(angle*w)/w)*gen[i]*state[((unsigned long long)i)^flip];",
                "ceo_exact_two_level_rotation_fp64")
            self.expect_kernel=xp.RawKernel(CUDA_SOURCE,"pauli_expect")

    def host(self,value):
        return self.xp.asnumpy(value) if self.backend=="cupy" else np.asarray(value)

    def generator(self,index):
        if index not in self.generators:
            xp=self.xp;flip=self.op_flips[index]
            target=self.ids ^ np.uint64(flip)
            values=xp.zeros(self.dim,dtype=xp.complex128)
            row=self.coef_host.getrow(index)
            for term,c in zip(row.indices,row.data):
                _,phase,ny=self.masks_host[term]
                sign=1-2*self.parity[target & phase]
                values+=c*(1j**int(ny))*sign
            error=xp.max(xp.abs(values[target]+values.conj()))
            if float(self.host(error))>1e-10:
                raise RuntimeError("Generator embedding is not anti-Hermitian")
            self.generators[index]=(values,target)
        return self.generators[index]

    def apply(self,index,angle,state):
        values,target=self.generator(index)
        if self.backend=="cupy":
            return self.rotate_kernel(state,values,np.uint64(self.op_flips[index]),np.float64(angle),size=self.dim)
        norm=np.abs(values)
        return np.cos(angle*norm)*state+angle*np.sinc(angle*norm/np.pi)*values*state[target]

    def prepare(self,indices,theta,hf_id,keep=False):
        xp=self.xp
        state=xp.zeros(self.dim,dtype=xp.complex128);state[hf_id]=1
        states=[state]
        for index,angle in zip(indices,theta):
            state=self.apply(index,angle,state);states.append(state)
        return states if keep else state

    def energy_and_gradient(self,indices,theta,hf_id):
        xp=self.xp;states=self.prepare(indices,theta,hf_id,True)
        adj=self.h@states[-1]
        energy=xp.vdot(states[-1],adj).real
        gradient=xp.empty(len(theta),dtype=xp.float64)
        for j in reversed(range(len(theta))):
            values,target=self.generator(indices[j])
            gradient[j]=2*xp.vdot(adj,values*states[j+1][target]).real
            adj=self.apply(indices[j],-theta[j],adj)
        # SciPy BFGS consumes host scalar/gradient. These transfers are measured.
        return float(self.host(energy)),self.host(gradient)

    def selection_gradients(self,state):
        xp=self.xp;hstate=self.h@state
        source=xp.flatnonzero(state!=0).astype(xp.int64)
        if self.backend=="cupy":
            expectations=xp.empty(len(self.masks_host),dtype=xp.complex128)
            self.expect_kernel((len(self.masks_host),),(256,),
                 (state,hstate,source,np.int32(len(source)),self.masks,np.int32(len(self.masks_host)),expectations))
        else:
            expectations=[]
            for flip,phase,ny in self.masks_host:
                signs=(1-2*self.parity[source & int(phase)])*(1j**int(ny))
                expectations.append(np.vdot(hstate[source ^ int(flip)],signs*state[source]))
            expectations=np.array(expectations)
        return self.host(2*(self.coef@expectations).real)
