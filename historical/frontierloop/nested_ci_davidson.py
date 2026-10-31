"""N21 diagonal-preconditioned successor; N20 numerical operator unchanged.

The projected H+0.1 S² and particle-hole/RDM identities are unchanged.
Only one eigenpair is requested; a registered parent root can warm-start the
larger space. Every saved root states its actual electron/hole representation.
"""
from time import perf_counter, process_time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from pyscf.fci import selected_ci
from pyscf import lib
from .portable_penalized_ci import CountedCI, expectation


def embed_root(root, new_solver_strings):
    old=np.asarray(root['solver_strings'],np.int64); new=np.asarray(new_solver_strings,np.int64)
    coefficients=np.asarray(root['coefficients'],float).reshape((len(old),len(old)))
    positions=np.searchsorted(new,old)
    if np.any(positions>=len(new)) or not np.array_equal(new[positions],old):
        raise ValueError('Warm root basis must be a subset of the larger solver basis')
    if abs(float(np.vdot(coefficients,coefficients))-1)>1e-10:
        raise ValueError('Warm root normalization differs')
    out=np.zeros((len(new),len(new))); out[np.ix_(positions,positions)]=coefficients
    return out.ravel()


def evaluate(h,g,core,nelec,strings,seconds=75,warm=None):
    cpu,wall=process_time(),perf_counter(); n=len(h); nelec=tuple(nelec); raw=np.asarray(strings)
    if np.iscomplexobj(h) or np.iscomplexobj(g) or not np.issubdtype(raw.dtype,np.integer) or raw.dtype==np.bool_:
        raise ValueError('Registered real integrals and integer configurations required')
    strings=np.asarray(strings,np.int64)
    if nelec[0]!=nelec[1] or not len(strings) or len(set(map(int,strings)))!=len(strings) or not np.array_equal(strings,np.sort(strings)):
        raise ValueError('Balanced unique sorted physical configurations required')
    if len(strings)>1024 or any(int(s)<0 or int(s)>=1<<n or int(s).bit_count()!=nelec[0] for s in strings):
        raise ValueError('Registered R<=1024 particle sector required')
    oh,og,oc=h,g,core; holes=all(x>n/2 for x in nelec)
    if holes:
        h=-h-2*np.einsum('pqrr->pq',g)+np.einsum('prrq->pq',g)
        core=float(core+2*np.trace(oh)+2*np.einsum('pprr->',g)-np.einsum('prrp->',g))
        s=np.sort(((1<<n)-1)^strings); ne=(n-nelec[0],n-nelec[1])
    else: s=strings; ne=nelec
    shape=(len(s),)*2; d=len(s)**2; solver=CountedCI(wall+seconds)
    h2=solver.absorb_h1e(h,g,n,ne,.5); links=selected_ci._all_linkstr_index((s,s),n,ne)
    def sci(x): return selected_ci._as_SCIvector(np.array(x,dtype=float,order='C',copy=True).reshape(shape),(s,s))
    def hop(x):
        v=sci(x)
        return (solver.contract_2e(h2,v,n,ne,link_index=links)+.1*solver.contract_ss(v,n,ne)).ravel()
    rng=np.random.default_rng(20261025); symmetry=[]; rdm_errors=[]
    for _ in range(3):
        a,b=rng.normal(size=(2,d)); a/=np.linalg.norm(a); b/=np.linalg.norm(b)
        symmetry.append(float(abs(np.vdot(a,hop(b))-np.vdot(hop(a),b))))
        v=sci(a); one,two=solver.make_rdm12(v,n,ne); spin=float(solver.spin_square(v,n,ne)[0])
        rdm_errors.append(abs(float(np.vdot(a,hop(a)))+core-expectation(h,g,core,one,two)-.1*spin))
    if max(symmetry)>1e-10 or max(rdm_errors)>1e-8:
        raise RuntimeError('Numerical Hamiltonian/RDM qualification failed')
    v0=rng.normal(size=d); v0/=np.linalg.norm(v0); witness=None; witness_error=None
    if warm is not None:
        if int(warm['norb'])!=n or bool(warm['particle_hole_used'])!=holes or tuple(warm['electron_nelec'])!=nelec:
            raise ValueError('Warm root physical representation mismatch')
        embedded=embed_root(warm,s)
        witness=float(np.vdot(embedded,hop(embedded))+core)
        witness_error=abs(witness-float(warm['penalized_energy_Ha']))
        if witness_error>1e-8: raise RuntimeError('Nested root witness energy changed')
        v0=embedded+.001*v0; v0/=np.linalg.norm(v0)
    diagonal_error=None; davidson_converged=None
    if d<=2:
        values,vectors=np.linalg.eigh(np.column_stack([hop(np.eye(d)[:,i]) for i in range(d)]))
    else:
        diagonal=selected_ci.make_hdiag(h,g,(s,s),n,ne)
        overlap=np.zeros(shape,np.int16)
        for orbital in range(n):
            occupied=((s>>orbital)&1).astype(np.int16)
            overlap+=occupied[:,None]*occupied[None,:]
        diagonal+=.1*(ne[0]-overlap.ravel())
        diagonal_probe=np.zeros(d);diagonal_probe[int(np.argmin(diagonal))]=1.
        diagonal_error=abs(float(diagonal_probe@hop(diagonal_probe))-float(np.min(diagonal)))
        if diagonal_error>1e-8:raise RuntimeError('Finite-spin Hamiltonian diagonal mismatch')
        precondition=lambda x,e,*args:x/(diagonal-e+1e-4)
        converged,values,states=lib.davidson1(lambda xs:[hop(x) for x in xs],[v0],precondition,
            tol=1e-12,tol_residual=1e-10,max_cycle=500,max_space=24,nroots=1,
            max_memory=2048,lindep=1e-20,verbose=0)
        # PySCF can stop on linear dependence without its stricter internal
        # flag. Keep that flag, and accept only the unchanged explicit 1e-8
        # physical residual/energy/witness tests below, never the flag alone.
        davidson_converged=bool(converged[0])
        vectors=np.asarray(states).T
    i=int(np.argmin(values)); value=float(values[i]); vector=vectors[:,i]
    residual=float(np.linalg.norm(hop(vector)-value*vector)); v=sci(vector)
    one,two=solver.make_rdm12(v,n,ne); physical=expectation(h,g,core,one,two)
    spin=float(solver.spin_square(v,n,ne)[0]); objective=value+core
    if holes:
        from .hole_rdm import electron_rdms
        one,two=electron_rdms(one,two)
    converted=expectation(oh,og,oc,one,two)
    if (residual>1e-8 or abs(objective-physical-.1*spin)>1e-8 or abs(converted-physical)>1e-8
        or abs(float(vector@vector)-1)>1e-10):
        raise RuntimeError('Saved-root residual/spin/physical energy mismatch')
    number=sum(nelec)
    if abs(np.trace(one)-number)>1e-8 or abs(np.einsum('pprr->',two)-number*(number-1))>1e-6:
        raise RuntimeError('Electron/pair number mismatch')
    if witness is not None and objective>witness+1e-8:
        raise RuntimeError('New root exceeds feasible nested parent witness; preserve, do not accept')
    row=dict(energy_Ha=converted,penalized_energy_Ha=objective,spin_square=spin,residual_norm=residual,
        particle_hole_used=holes,actual_CI_dimension=d,actual_half_dimension=len(strings),
        matvec_calls=solver.matvec_calls,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        random_bilinear_hermiticity_errors=symmetry,independent_RDM_expectation_errors=rdm_errors,
        embedded_parent_witness_Ha=witness,embedded_parent_witness_error_Ha=witness_error,
        eigensolver='PYSCF_DAVIDSON_DIAGONAL_PRECONDITIONED',preconditioner_diagonal_probe_error_Ha=diagonal_error,
        eigensolver_internal_converged=davidson_converged,
        requested_eigenpairs=1,common_equation='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
        not_global_optimality_certificate=True)
    root=dict(coefficients=vector,solver_strings=s,electron_strings=strings,norb=np.int64(n),
        solver_nelec=np.array(ne,np.int64),electron_nelec=np.array(nelec,np.int64),
        particle_hole_used=np.bool_(holes),penalized_energy_Ha=np.float64(objective))
    return row,root
