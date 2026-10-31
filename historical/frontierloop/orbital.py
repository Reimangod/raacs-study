"""Paired fixed-subspace orbital optimization, source-aligned with official SQD.

No new measurement is fabricated in a rotated basis. Original samples select
indices once; subsequent orbital changes are explicit classical variational
parameters. This is not preparation of the resulting CI state by the sampler.
"""
from time import process_time,perf_counter
import ffsim
import jax
import numpy as np
from paperstudy.ci import CountedSelectedCI


def diagonalize_with_rdm(ham,nelec,strings,seconds=60):
    n=ham.norb;solver=CountedSelectedCI(perf_counter()+seconds);solver.verbose=0;solver.max_memory=1024
    energy,vector=solver.kernel_fixed_space(ham.one_body_tensor,ham.two_body_tensor,n,tuple(nelec),
        (strings,strings),tol=1e-10,max_cycle=200,max_space=20,davidson_only=True,ecore=ham.constant,verbose=0)
    if not solver.converged:raise RuntimeError('OO CI solver not converged')
    residual=solver.contract_2e(solver.absorb_h1e(ham.one_body_tensor,ham.two_body_tensor,n,tuple(nelec),.5),
        vector,n,tuple(nelec))-(float(energy)-ham.constant)*vector
    residual_norm=float(np.linalg.norm(residual))
    if not np.isfinite(energy)or residual_norm>1e-5:raise RuntimeError('OO CI residual failed')
    dm1,dm2=solver.make_rdm12(vector,n,tuple(nelec))
    rdm=ffsim.ReducedDensityMatrix(dm1,dm2);check=float(rdm.expectation(ham).real)
    if abs(check-float(energy))>1e-8:raise RuntimeError('PySCF/ffsim RDM energy or tensor convention mismatch')
    spin,_=solver.spin_square(vector,n,tuple(nelec))
    return {'energy_Ha':check,'spin_square':float(spin),'matvec_calls':solver.matvec_calls,'residual_norm':residual_norm,
        'CI_converged':True},rdm


def optimize(h,g,core,nelec,strings,spec,callback=None):
    jax.config.update('jax_enable_x64',True)
    ham=ffsim.MolecularHamiltonian(h,g,constant=core)
    cpu,tick=process_time(),perf_counter();history=[];rotations=[]
    initial,rdm=diagonalize_with_rdm(ham,nelec,strings,spec['per_CI_wall_cap_seconds'])
    history.append({'outer':0,**initial});stop='OUTER_ITERATION_CAP'
    for outer in range(1,spec['outer_iterations']+1):
        if process_time()-cpu>=spec['per_method_outer_boundary_CPU_cap_seconds']:
            stop='CPU_BOUNDARY_CAP';break
        c,t=process_time(),perf_counter()
        u,result=ffsim.optimize_orbitals(rdm,ham,real=True,method='L-BFGS-B',
            options={'maxiter':spec['optimizer_maxiter_per_outer'],'maxfun':spec['optimizer_maxfun_per_outer']},
            return_optimize_result=True)
        if np.max(abs(u.T@u-np.eye(len(h))))>1e-9:raise RuntimeError('Non-orthogonal orbital transform')
        changed=ham.rotated(u.T.conj())
        fixed_value=float(rdm.expectation(changed).real)
        if fixed_value>history[-1]['energy_Ha']+1e-8:stop='NON_IMPROVING_ORBITAL_STEP';break
        diag,new_rdm=diagonalize_with_rdm(changed,nelec,strings,spec['per_CI_wall_cap_seconds'])
        if diag['energy_Ha']>fixed_value+1e-8:raise RuntimeError('CI failed variational monotonicity after orbital step')
        rotations.append(u);ham,rdm=changed,new_rdm
        row={'outer':outer,**diag,'fixed_old_vector_energy_Ha':fixed_value,
            'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
            'optimizer_nfev':int(result.nfev),'optimizer_njev':int(result.njev),'optimizer_nit':int(result.nit),
            'optimizer_gradient_inf':float(np.max(abs(result.jac))),
            'CPU_seconds':process_time()-c,'wall_seconds':perf_counter()-t}
        history.append(row)
        if callback:callback(history,rotations)
    return {'history':history,'rotations':rotations,'final_h1':ham.one_body_tensor,'final_eri':ham.two_body_tensor,
        'core':ham.constant,'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
        'stop':stop,'new_quantum_model_shots':0,'classical_orbital_coordinates':len(h)*(len(h)-1)//2,
        'final_energy_Ha':history[-1]['energy_Ha'],'initial_energy_Ha':history[0]['energy_Ha'],
        'sampler_not_a_preparation_of_orbital_optimized_CI_state':True}
