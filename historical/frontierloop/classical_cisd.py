"""Direct CISD-only classical comparator from fixed physical integrals.

No CCSD-first fallback, no historical coefficients, no reference/FCI read.
The installed CISD kernel does not invoke an arbitrary solver.callback, so
limits are enforced inside each real Hamiltonian contraction instead.
"""
from time import process_time,perf_counter
import numpy as np
from pyscf import ao2mo,gto,scf
from pyscf.ci import cisd
from .cisd_sparse import coefficients


class BoundedCISD(cisd.CISD):
    def __init__(self,mf,cpu_seconds,wall_seconds):
        super().__init__(mf)
        self.cpu_deadline=process_time()+cpu_seconds
        self.wall_deadline=perf_counter()+wall_seconds
        self.matvec_calls=0

    def contract(self,civec,eris):
        if process_time()>self.cpu_deadline or perf_counter()>self.wall_deadline:
            raise TimeoutError('CISD contraction-boundary cost cap')
        self.matvec_calls+=1
        return super().contract(civec,eris)


def solve(meta,h,g,cpu_seconds=30,wall_seconds=60):
    cpu,tick=process_time(),perf_counter()
    n,na,nb=meta['norb'],meta['nalpha'],meta['nbeta']
    if na!=nb:
        raise ValueError('Registered restricted closed-shell CISD only')
    mol=gto.M(verbose=0);mol.nelectron=na+nb;mol.spin=0;mol.nao_nr=lambda *args:n
    mf=scf.RHF(mol)
    mf.get_hcore=lambda *args:h
    mf.get_ovlp=lambda *args:np.eye(n)
    mf.energy_nuc=lambda *args:meta['core_energy']
    mf._eri=ao2mo.restore(8,g,n)
    mf.mo_coeff=np.eye(n);mf.mo_occ=np.array([2.]*na+[0.]*(n-na))
    hf=float(meta['core_energy']+2*np.trace(h[:na,:na])
        +2*np.einsum('iijj->',g[:na,:na,:na,:na])-np.einsum('ijji->',g[:na,:na,:na,:na]))
    if abs(hf-meta['hf_energy_Ha'])>1e-8:
        raise RuntimeError('Classical CISD reference does not match fixed physical problem')
    mf.e_tot=hf;fock=mf.get_fock(dm=mf.make_rdm1());mf.mo_energy=np.diag(fock)
    fov=float(np.linalg.norm(fock[:na,na:]));mf.converged=fov<1e-6
    solver=BoundedCISD(mf,cpu_seconds,wall_seconds)
    solver.verbose=0;solver.max_memory=768;solver.max_cycle=150;solver.conv_tol=1e-10
    eris=solver.ao2mo();solver.kernel(eris=eris)
    if not solver.converged:
        raise RuntimeError('Direct CISD failed to converge')
    residual=solver.contract(solver.ci,eris)-solver.e_corr*solver.ci
    squared=float(cisd.dot(residual,residual,n,na))
    norm=float(np.sqrt(max(0.,squared)))
    ids,values=coefficients(solver.ci,n,na)
    state_norm=float(np.sum(abs(values)**2))
    if not np.all(np.isfinite(values)) or abs(state_norm-1)>1e-7 or norm>1e-5 or squared < -1e-12:
        raise RuntimeError('Direct CISD physical norm/residual validation failed')
    return {'energy_Ha':float(solver.e_tot),'correlation_energy_Ha':float(solver.e_corr),
        'converged':True,'residual_norm':norm,'residual_squared_metric':squared,
        'sparse_state_norm_squared':state_norm,'matvec_calls':solver.matvec_calls,
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
        'SCF_orbitals_reoptimized':False,'occupied_virtual_Fock_norm':fov,
        'full_Fock_vector_used':False,'CCSD_first_attempted':False,
        'old_coefficients_or_reference_read':False,'quantum_model_shots':0,'actual_QPU_shots':0,
        'ids':ids,'coefficients':values}
