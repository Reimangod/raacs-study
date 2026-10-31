"""Additive C3 numerical successor with lowest-root targeting and warm starts.

The physical objective, finite spin penalty and OO budget are unchanged.
Old frozen executors/results remain untouched. Particle-hole representation
is exact and is verified against electron-space energies and RDMs separately.
"""
from time import process_time,perf_counter
import ffsim
import jax
import numpy as np
from .lowest_penalized import solve
from .energy_cache import hole_integrals
from .hole_rdm import electron_rdms


class Reader:
    def __init__(self,n,nelec,strings,spec):
        self.n=n;self.nelec=tuple(nelec);self.spec=spec
        if nelec[0]!=nelec[1] or spec['spin_target_S_squared']!=0:raise ValueError('Registered M_S=0 target only')
        self.holes=all(x>n/2 for x in nelec)
        self.ne=(n-nelec[0],n-nelec[1]) if self.holes else tuple(nelec)
        self.s=np.sort(((1<<n)-1)^np.array(strings,np.int64)) if self.holes else np.array(strings,np.int64)
        self.previous=None

    def diagonalize(self,physical):
        if self.holes:
            h,g,core=hole_integrals(physical.one_body_tensor,physical.two_body_tensor,physical.constant)
            ham=ffsim.MolecularHamiltonian(h,g,constant=core)
        else:ham=physical
        row,vector,rdm=solve(ham,self.ne,self.s,penalty=self.spec['spin_penalty_Ha'],
            seconds=self.spec['per_CI_wall_cap_seconds'],seed=20261025,known_vector=self.previous)
        self.previous=vector.copy()
        if self.holes:
            one,two=electron_rdms(rdm.one_rdm,rdm.two_rdm);rdm=ffsim.ReducedDensityMatrix(one,two)
        energy=float(rdm.expectation(physical).real);number=sum(self.nelec)
        if abs(energy-row['physical_energy_Ha'])>1e-8 or abs(np.trace(rdm.one_rdm)-number)>1e-8:
            raise RuntimeError('Lowest-root particle-hole physical energy/number mismatch')
        if abs(np.einsum('pprr->',rdm.two_rdm)-number*(number-1))>1e-6:
            raise RuntimeError('Lowest-root particle-hole pair count mismatch')
        row.update(energy_Ha=energy,spin_penalty_energy_Ha=self.spec['spin_penalty_Ha']*row['spin_square'],
            particle_hole_used=self.holes,CI_converged=True,exact_singlet_projection=False,
            RDM_representation='ELECTRONS_IN_PHYSICAL_ORBITAL_BASIS',
            physical_energy_conversion_error_Ha=energy-row['physical_energy_Ha'])
        return row,rdm


def optimize(h,g,core,nelec,strings,spec,callback=None):
    jax.config.update('jax_enable_x64',True);cpu,tick=process_time(),perf_counter()
    ham=ffsim.MolecularHamiltonian(h,g,constant=core);reader=Reader(len(h),nelec,strings,spec)
    initial,rdm=reader.diagonalize(ham);history=[{'outer':0,**initial}];rotations=[];stop='OUTER_ITERATION_CAP'
    for outer in range(1,spec['outer_iterations']+1):
        if process_time()-cpu>=spec['per_method_outer_boundary_CPU_cap_seconds']:stop='CPU_BOUNDARY_CAP';break
        c,t=process_time(),perf_counter()
        u,opt=ffsim.optimize_orbitals(rdm,ham,real=True,method='L-BFGS-B',
            options={'maxiter':spec['optimizer_maxiter_per_outer'],'maxfun':spec['optimizer_maxfun_per_outer']},return_optimize_result=True)
        if np.max(abs(u.T@u-np.eye(len(h))))>1e-9:raise RuntimeError('Nonorthogonal orbital step')
        changed=ham.rotated(u.T.conj());fixed=float(rdm.expectation(changed).real)
        known=fixed+history[-1]['spin_penalty_energy_Ha'];tol=spec['monotonicity_tolerance_Ha']
        if known>history[-1]['penalized_energy_Ha']+tol:stop='NON_IMPROVING_ORBITAL_STEP';break
        diag,new_rdm=reader.diagonalize(changed)
        if diag['penalized_energy_Ha']>known+tol:raise RuntimeError('Lowest-root successor exceeded known-vector Rayleigh bound')
        rotations.append(u);ham,rdm=changed,new_rdm
        history.append({'outer':outer,**diag,'fixed_old_vector_energy_Ha':fixed,'fixed_old_vector_penalized_energy_Ha':known,
            'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message),
            'optimizer_nfev':int(opt.nfev),'optimizer_njev':int(opt.njev),'optimizer_nit':int(opt.nit),
            'optimizer_gradient_inf':float(np.max(abs(opt.jac))),'CPU_seconds':process_time()-c,'wall_seconds':perf_counter()-t})
        if callback:callback(history,rotations)
    return {'history':history,'rotations':rotations,'final_h1':ham.one_body_tensor,'final_eri':ham.two_body_tensor,
        'core':ham.constant,'initial_energy_Ha':initial['energy_Ha'],'final_energy_Ha':history[-1]['energy_Ha'],
        'final_penalized_energy_Ha':history[-1]['penalized_energy_Ha'],'CPU_seconds':process_time()-cpu,
        'wall_seconds':perf_counter()-tick,'stop':stop,'new_quantum_model_shots':0,
        'classical_orbital_coordinates':len(h)*(len(h)-1)//2,'particle_hole_used':reader.holes,
        'numerical_successor_not_old_frozen_result':True,'sampler_not_a_preparation_of_orbital_optimized_CI_state':True}
