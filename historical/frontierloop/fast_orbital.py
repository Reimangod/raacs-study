"""Qualified candidate backend: exact particle-hole CI with electron RDMs.

No changes to the frozen C5 or B4 executors. Use only after independent parity
and timing tests. Particle-hole conversion changes representation, not the
physical Hamiltonian, selected determinant space or spin constraint.
"""
from time import process_time,perf_counter
import ffsim
import jax
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator,eigsh
from pyscf.fci import selected_ci
from paperstudy.ci import CountedSelectedCI
from .energy_cache import hole_integrals
from .hole_rdm import electron_rdms
from .orbital import diagonalize_with_rdm
from .spin_orbital import diagonalize as penalized
from .singlet import basis,diagonalize as singlet


class Reader:
    def __init__(self,n,nelec,strings,mode,spec):
        if mode not in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO'):
            raise ValueError('Explicit spin rule required')
        self.n,self.nelec,self.strings,self.mode,self.spec=n,tuple(nelec),np.asarray(strings,np.int64),mode,spec
        self.holes=all(x>n/2 for x in nelec)
        self.ne=(n-nelec[0],n-nelec[1]) if self.holes else tuple(nelec)
        self.s=np.sort(((1<<n)-1)^self.strings) if self.holes else self.strings
        self.q,self.info=basis(self.s,n,self.ne) if mode=='C4-SINGLET-OO' else (None,None)
        if self.holes and mode=='C4-SINGLET-OO':
            # Preserve the old reduced coordinates and deterministic starting
            # vector, not an unrelated basis of the degenerate S^2 nullspace.
            qe,self.info=basis(self.strings,n,self.nelec)
            lookup={int(s):i for i,s in enumerate(self.s)}
            order=[lookup[((1<<n)-1)^int(s)] for s in self.strings]
            signs=[(-1)**sum(p for p in range(n) if int(s)&(1<<p)) for s in self.strings]
            rows=[];cols=[];data=[];width=len(self.strings)
            for i,a in enumerate(order):
                for j,b in enumerate(order):
                    rows.append(a*width+b);cols.append(i*width+j);data.append(signs[i]*signs[j])
            transform=sp.csr_matrix((data,(rows,cols)),shape=(width**2,width**2))
            self.q=transform@qe

    def _singlet_aligned(self,ham,physical_core):
        solver=CountedSelectedCI(perf_counter()+self.spec['per_CI_wall_cap_seconds'])
        solver.verbose=0;solver.max_memory=1024
        h2=solver.absorb_h1e(ham.one_body_tensor,ham.two_body_tensor,self.n,self.ne,.5)
        shape=(len(self.s),len(self.s))
        shift=ham.constant-physical_core
        def hop(c):
            vector=selected_ci._as_SCIvector(np.asarray(self.q@c).reshape(shape),(self.s,self.s))
            image=solver.contract_2e(h2,vector,self.n,self.ne).ravel()
            # ARPACK's relative tolerance depends on the eigenvalue scale.
            # Keep that scale identical to the electron implementation.
            return np.asarray(self.q.T@image)+shift*c
        d=self.q.shape[1]
        if d==1:
            reduced=np.ones(1);eigenvalue=float(hop(reduced)[0])
        else:
            v0=np.random.default_rng(4713).normal(size=d)
            eigenvalues,vectors=eigsh(LinearOperator((d,d),matvec=hop,dtype=np.float64),
                k=1,which='SA',tol=1e-10,maxiter=1000,v0=v0)
            eigenvalue=float(eigenvalues[0]);reduced=vectors[:,0]
        residual=float(np.linalg.norm(hop(reduced)-eigenvalue*reduced))
        vector=selected_ci._as_SCIvector(np.asarray(self.q@reduced).reshape(shape),(self.s,self.s))
        dm1,dm2=solver.make_rdm12(vector,self.n,self.ne)
        rdm=ffsim.ReducedDensityMatrix(dm1,dm2)
        value=float(rdm.expectation(ham).real);spin=float(solver.spin_square(vector,self.n,self.ne)[0])
        if (not np.all(np.isfinite([value,spin,residual])) or residual>self.spec['eigenvector_residual_tolerance']
            or abs(value-eigenvalue-physical_core)>1e-8 or abs(spin)>1e-8):
            raise RuntimeError('Aligned particle-hole singlet CI failed physical checks')
        return {'energy_Ha':value,'spin_square':spin,'residual_norm':residual,
            'matvec_calls':solver.matvec_calls,'CI_converged':True,'exact_singlet_projection':True,
            'spin_penalty_energy_Ha':0.,'reduced_Krylov_coordinates_and_energy_scale_aligned':True},rdm

    def solve(self,physical):
        if self.holes:
            h,g,core=hole_integrals(physical.one_body_tensor,physical.two_body_tensor,physical.constant)
            ham=ffsim.MolecularHamiltonian(h,g,constant=core)
        else:
            ham=physical
        if self.mode=='C2-OO':
            row,rdm=diagonalize_with_rdm(ham,self.ne,self.s,self.spec['per_CI_wall_cap_seconds'])
        elif self.mode=='C3-SPIN-OO':
            row,rdm=penalized(ham,self.ne,self.s,self.spec)
        elif self.holes:
            row,rdm=self._singlet_aligned(ham,physical.constant)
        else:
            row,rdm=singlet(ham,self.ne,self.s,self.q,self.spec)
        if self.holes:
            one,two=electron_rdms(rdm.one_rdm,rdm.two_rdm)
            rdm=ffsim.ReducedDensityMatrix(one,two)
        value=float(rdm.expectation(physical).real)
        error=value-row['energy_Ha']
        ne=sum(self.nelec)
        if abs(error)>1e-8 or abs(np.trace(rdm.one_rdm)-ne)>1e-8:
            raise RuntimeError('Particle-hole physical energy or particle number mismatch')
        if abs(np.einsum('pprr->',rdm.two_rdm)-ne*(ne-1))>1e-6:
            raise RuntimeError('Particle-hole 2-RDM normalization mismatch')
        row.update(energy_Ha=value,particle_hole_used=self.holes,
            physical_energy_conversion_error_Ha=float(error),
            RDM_representation='ELECTRONS_IN_PHYSICAL_ORBITAL_BASIS')
        return row,rdm


def optimize(h,g,core,nelec,strings,spec,mode,callback=None):
    jax.config.update('jax_enable_x64',True)
    cpu,tick=process_time(),perf_counter()
    ham=ffsim.MolecularHamiltonian(h,g,constant=core)
    reader=Reader(len(h),nelec,strings,mode,spec)
    initial,rdm=reader.solve(ham);history=[{'outer':0,**initial}];rotations=[]
    field='penalized_energy_Ha' if mode=='C3-SPIN-OO' else 'energy_Ha'
    stop='OUTER_ITERATION_CAP'
    for outer in range(1,spec['outer_iterations']+1):
        if process_time()-cpu>=spec['per_method_outer_boundary_CPU_cap_seconds']:
            stop='CPU_BOUNDARY_CAP';break
        c,t=process_time(),perf_counter()
        u,result=ffsim.optimize_orbitals(rdm,ham,real=True,method='L-BFGS-B',
            options={'maxiter':spec['optimizer_maxiter_per_outer'],
                     'maxfun':spec['optimizer_maxfun_per_outer']},return_optimize_result=True)
        if np.max(abs(u.T@u-np.eye(len(h))))>1e-9:
            raise RuntimeError('Nonorthogonal orbital transform')
        changed=ham.rotated(u.T.conj());fixed=float(rdm.expectation(changed).real)
        objective=fixed+(history[-1]['spin_penalty_energy_Ha'] if mode=='C3-SPIN-OO' else 0.)
        if objective>history[-1][field]+1e-8:
            stop='NON_IMPROVING_ORBITAL_STEP';break
        diag,new_rdm=reader.solve(changed)
        if diag[field]>objective+1e-8:
            raise RuntimeError('CI variational objective increased after orbital rotation')
        ham,rdm=changed,new_rdm;rotations.append(u)
        history.append({'outer':outer,**diag,'fixed_old_vector_energy_Ha':fixed,
            'fixed_old_vector_objective_Ha':objective,
            'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
            'optimizer_nfev':int(result.nfev),'optimizer_njev':int(result.njev),'optimizer_nit':int(result.nit),
            'optimizer_gradient_inf':float(np.max(abs(result.jac))),
            'CPU_seconds':process_time()-c,'wall_seconds':perf_counter()-t})
        if callback:callback(history,rotations)
    return {'history':history,'rotations':rotations,'final_h1':ham.one_body_tensor,
        'final_eri':ham.two_body_tensor,'core':ham.constant,'spin_basis':reader.info,
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'stop':stop,
        'mode':mode,'particle_hole_used':reader.holes,'new_quantum_model_shots':0,
        'classical_orbital_coordinates':len(h)*(len(h)-1)//2,
        'final_energy_Ha':history[-1]['energy_Ha'],'initial_energy_Ha':initial['energy_Ha'],
        'sampler_not_a_preparation_of_orbital_optimized_CI_state':True}
