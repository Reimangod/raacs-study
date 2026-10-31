"""C3: finite spin-penalty selected CI, with physical energy kept separate.

No claim of exact singlet projection. The selected determinant product space
need not be invariant under S^2. Tests compare the projected action against
independent full-space spin contractions on tiny fixtures.
"""
from time import process_time, perf_counter
import ffsim
import jax
import numpy as np
from pyscf.fci import addons
from paperstudy.ci import CountedSelectedCI


def diagonalize(ham, nelec, strings, spec):
    n = ham.norb
    if nelec[0] != nelec[1] or spec['spin_target_S_squared'] != 0:
        raise ValueError('C3 is registered only for M_S=0 and target S=0')
    solver = CountedSelectedCI(perf_counter()+spec['per_CI_wall_cap_seconds'])
    solver.verbose = 0
    solver.max_memory = 1024
    solver = addons.fix_spin_(solver, shift=spec['spin_penalty_Ha'], ss=0)
    eigenvalue, vector = solver.kernel_fixed_space(
        ham.one_body_tensor, ham.two_body_tensor, n, tuple(nelec), (strings, strings),
        tol=1e-10, max_cycle=200, max_space=20, davidson_only=True,
        ecore=ham.constant, verbose=0)
    if not solver.converged:
        raise RuntimeError('Spin-penalty CI did not converge')
    effective_h = solver.absorb_h1e(ham.one_body_tensor, ham.two_body_tensor, n, tuple(nelec), .5)
    effective_action = solver.contract_2e(effective_h, vector, n, tuple(nelec))
    residual = float(np.linalg.norm(effective_action-(eigenvalue-ham.constant)*vector))
    dm1, dm2 = solver.make_rdm12(vector, n, tuple(nelec))
    rdm = ffsim.ReducedDensityMatrix(dm1, dm2)
    physical = float(rdm.expectation(ham).real)
    spin = float(solver.spin_square(vector, n, tuple(nelec))[0])
    projected_spin = float(np.vdot(vector, solver.contract_ss(vector, n, tuple(nelec))).real)
    penalty = spec['spin_penalty_Ha']*spin
    gap = float(eigenvalue)-physical-penalty
    if not np.all(np.isfinite([physical, spin, eigenvalue, residual, gap])):
        raise RuntimeError('Nonfinite spin-penalty CI result')
    if abs(gap)>spec['eigenvalue_physical_penalty_consistency_tolerance_Ha']:
        raise RuntimeError('Physical energy plus spin penalty does not match eigensolver')
    if abs(projected_spin-spin)>1e-8 or spin < -1e-8:
        raise RuntimeError('Spin expectation mismatch')
    if residual>spec['eigenvector_residual_tolerance']:
        raise RuntimeError('Spin-penalty eigenvector residual failed')
    return {
        'energy_Ha':physical, 'penalized_energy_Ha':float(eigenvalue),
        'spin_square':spin, 'spin_penalty_energy_Ha':penalty,
        'eigenvalue_consistency_error_Ha':gap, 'residual_norm':residual,
        'matvec_calls':solver.matvec_calls, 'CI_converged':True,
        'exact_singlet_projection':False,
    }, rdm


def optimize(h, g, core, nelec, strings, spec, callback=None):
    jax.config.update('jax_enable_x64', True)
    ham = ffsim.MolecularHamiltonian(h, g, constant=core)
    cpu, tick = process_time(), perf_counter()
    history, rotations = [], []
    initial, rdm = diagonalize(ham, nelec, strings, spec)
    history.append({'outer':0, **initial})
    stop = 'OUTER_ITERATION_CAP'
    tol = spec['monotonicity_tolerance_Ha']
    for outer in range(1, spec['outer_iterations']+1):
        if process_time()-cpu >= spec['per_method_outer_boundary_CPU_cap_seconds']:
            stop = 'CPU_BOUNDARY_CAP'
            break
        c, t = process_time(), perf_counter()
        u, result = ffsim.optimize_orbitals(
            rdm, ham, real=True, method='L-BFGS-B',
            options={'maxiter':spec['optimizer_maxiter_per_outer'],
                     'maxfun':spec['optimizer_maxfun_per_outer']},
            return_optimize_result=True)
        if np.max(abs(u.T@u-np.eye(len(h))))>1e-9:
            raise RuntimeError('Nonorthogonal orbital transform')
        changed = ham.rotated(u.T.conj())
        fixed_physical = float(rdm.expectation(changed).real)
        # A common spin-independent orbital rotation leaves S^2 invariant.
        fixed_penalized = fixed_physical+history[-1]['spin_penalty_energy_Ha']
        if fixed_penalized>history[-1]['penalized_energy_Ha']+tol:
            stop = 'NON_IMPROVING_ORBITAL_STEP'
            break
        diag, new_rdm = diagonalize(changed, nelec, strings, spec)
        if diag['penalized_energy_Ha']>fixed_penalized+tol:
            raise RuntimeError('Penalized CI variational monotonicity failed')
        rotations.append(u)
        ham, rdm = changed, new_rdm
        history.append({
            'outer':outer, **diag, 'fixed_old_vector_energy_Ha':fixed_physical,
            'fixed_old_vector_penalized_energy_Ha':fixed_penalized,
            'optimizer_success':bool(result.success), 'optimizer_message':str(result.message),
            'optimizer_nfev':int(result.nfev), 'optimizer_njev':int(result.njev),
            'optimizer_nit':int(result.nit),
            'optimizer_gradient_inf':float(np.max(abs(result.jac))),
            'CPU_seconds':process_time()-c, 'wall_seconds':perf_counter()-t,
        })
        if callback:
            callback(history, rotations)
    return {
        'history':history, 'rotations':rotations, 'final_h1':ham.one_body_tensor,
        'final_eri':ham.two_body_tensor, 'core':ham.constant,
        'CPU_seconds':process_time()-cpu, 'wall_seconds':perf_counter()-tick,
        'stop':stop, 'new_quantum_model_shots':0,
        'classical_orbital_coordinates':len(h)*(len(h)-1)//2,
        'final_energy_Ha':history[-1]['energy_Ha'], 'initial_energy_Ha':initial['energy_Ha'],
        'final_penalized_energy_Ha':history[-1]['penalized_energy_Ha'],
        'sampler_not_a_preparation_of_orbital_optimized_CI_state':True,
    }
