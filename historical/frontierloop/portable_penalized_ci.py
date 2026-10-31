"""Portable no-OO evaluator of the SAME finite-penalty equation as C7.

Only NumPy/SciPy/PySCF are needed on the isolated CPU runtime. This is a
numerical successor, not native DICE spin semantics or a replacement of old
results. Its independent RDM and particle-hole checks are retained.
"""
from time import perf_counter,process_time
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
from pyscf.fci import selected_ci


class CountedCI(selected_ci.SelectedCI):
    def __init__(self,deadline):
        super().__init__();self.deadline=deadline;self.matvec_calls=0;self.verbose=0
    def contract_2e(self,eri,civec_strs,norb,nelec,link_index=None,**kwargs):
        if perf_counter()>self.deadline:raise TimeoutError('Portable common-CI matvec wall cap')
        self.matvec_calls+=1
        return super().contract_2e(eri,civec_strs,norb,nelec,link_index=link_index,**kwargs)


def expectation(h,g,core,one,two):
    return float(core+np.einsum('pq,pq->',h,one)+.5*np.einsum('pqrs,pqrs->',g,two))


def evaluate(h,g,core,nelec,strings,penalty=.1,seconds=60):
    cpu,wall=process_time(),perf_counter();n=len(h);nelec=tuple(nelec);raw=np.asarray(strings)
    if penalty!=.1 or np.iscomplexobj(h) or np.iscomplexobj(g):
        raise ValueError('Only registered real-integral finite0.1Ha S2 equation qualified')
    if not np.issubdtype(raw.dtype,np.integer) or raw.dtype==np.bool_:
        raise ValueError('Do not truncate unknown/noninteger physical strings')
    strings=np.asarray(strings,np.int64)
    if nelec[0]!=nelec[1] or len(set(map(int,strings)))!=len(strings) or not np.array_equal(strings,np.sort(strings)):
        raise ValueError('Balanced unique sorted physical CI strings required')
    if not len(strings) or any(int(s)<0 or int(s)>=1<<n or int(s).bit_count()!=nelec[0] for s in strings):
        raise ValueError('Physical particle-sector/basis mismatch')
    original_h,original_g,original_core=h,g,core;holes=all(x>n/2 for x in nelec)
    if holes:
        # The qualified real-integral particle-hole identity used by C7.
        h=-h-2*np.einsum('pqrr->pq',g)+np.einsum('prrq->pq',g)
        core=float(core+2*np.trace(original_h)+2*np.einsum('pprr->',g)-np.einsum('prrp->',g))
        s=np.sort(((1<<n)-1)^strings);ne=(n-nelec[0],n-nelec[1])
    else:s=strings;ne=nelec
    shape=(len(s),)*2;d=len(s)**2;solver=CountedCI(wall+seconds)
    h2=solver.absorb_h1e(h,g,n,ne,.5);links=selected_ci._all_linkstr_index((s,s),n,ne)
    def sci(x):return selected_ci._as_SCIvector(np.array(x,dtype=np.float64,order='C',copy=True).reshape(shape),(s,s))
    def hop(x):
        v=sci(x)
        return (solver.contract_2e(h2,v,n,ne,link_index=links)+penalty*solver.contract_ss(v,n,ne)).ravel()
    rng=np.random.default_rng(20261025);symmetry=[];rdm_errors=[]
    for _ in range(3):
        a,b=rng.normal(size=(2,d));a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        symmetry.append(float(abs(np.vdot(a,hop(b))-np.vdot(hop(a),b))))
        v=sci(a);one,two=solver.make_rdm12(v,n,ne);s2=float(solver.spin_square(v,n,ne)[0])
        rdm_errors.append(abs(float(np.vdot(a,hop(a)))+core-expectation(h,g,core,one,two)-penalty*s2))
    if max(symmetry)>1e-10 or max(rdm_errors)>1e-8:raise RuntimeError('Portable Hamiltonian/RDM probe mismatch')
    v0=rng.normal(size=d);v0/=np.linalg.norm(v0)
    if d<=2:values,vectors=np.linalg.eigh(np.column_stack([hop(np.eye(d)[:,i]) for i in range(d)]))
    else:values,vectors=eigsh(LinearOperator((d,d),matvec=hop,dtype=np.float64),k=2,which='SA',v0=v0,tol=1e-12,maxiter=1000)
    order=np.argsort(values);values=values[order];vector=vectors[:,order[0]]
    residual=float(np.linalg.norm(hop(vector)-values[0]*vector));v=sci(vector)
    one,two=solver.make_rdm12(v,n,ne);physical=expectation(h,g,core,one,two)
    s2=float(solver.spin_square(v,n,ne)[0]);objective=float(values[0]+core)
    if holes:
        # Import only the separately qualified pure-NumPy conversion, not OO/JAX.
        from .hole_rdm import electron_rdms
        one,two=electron_rdms(one,two)
    converted=expectation(original_h,original_g,original_core,one,two)
    if residual>1e-8 or abs(objective-physical-penalty*s2)>1e-8 or abs(converted-physical)>1e-8:
        raise RuntimeError('Portable residual/spin/physical conversion mismatch')
    number=sum(nelec)
    if abs(np.trace(one)-number)>1e-8 or abs(np.einsum('pprr->',two)-number*(number-1))>1e-6:
        raise RuntimeError('Portable electron/pair number mismatch')
    return dict(energy_Ha=converted,penalized_energy_Ha=objective,spin_square=s2,
        lowest_two_eigenvalues_Ha=(values[:2]+core).tolist(),residual_norm=residual,
        particle_hole_used=holes,actual_CI_dimension=d,CI_strings=strings.tolist(),
        matvec_calls=solver.matvec_calls,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        random_bilinear_hermiticity_errors=symmetry,independent_RDM_expectation_errors=rdm_errors,
        common_equation='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
        not_global_optimality_certificate=True,new_model_shots=0)
