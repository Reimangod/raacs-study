"""Exact S=0 subspace within the existing selected determinant space.

No determinants are added. P S^2 P is positive semidefinite, so its zero
eigenspace consists of exact singlets in the full space. Blocks are classified
by spatial occupations; small block eigensolves avoid a dense d-by-d matrix.
This is a known spin-adaptation control, not a novel quantum algorithm.
"""
from collections import defaultdict
from time import perf_counter, process_time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
import ffsim
import jax
from pyscf.fci import selected_ci
from paperstudy.ci import CountedSelectedCI


def act(bits, orbital, create):
    occupied = bool(bits & (1 << orbital))
    if occupied == create:
        return None
    phase = -1 if (bits & ((1 << orbital)-1)).bit_count() % 2 else 1
    return bits ^ (1 << orbital), phase


def spin_action(bits, n):
    """S+ S- action, equal to S^2 only for M_S=0."""
    output = defaultdict(float)
    for q in range(n):
        a = act(bits, q, False)
        if a is None:
            continue
        b = act(a[0], n+q, True)
        if b is None:
            continue
        for p in range(n):
            c = act(b[0], n+p, False)
            if c is None:
                continue
            d = act(c[0], p, True)
            if d is not None:
                output[d[0]] += a[1]*b[1]*c[1]*d[1]
    return dict(output)


def basis(strings, n, nelec, tol=1e-10):
    if nelec[0] != nelec[1]:
        raise ValueError('Only equal alpha/beta populations are supported')
    strings = tuple(map(int, strings))
    if any(s.bit_count()!=nelec[0] for s in strings):
        raise ValueError('Wrong particle sector')
    bits = [a | (b << n) for a in strings for b in strings]
    groups = defaultdict(list)
    mask = (1 << n)-1
    for i, x in enumerate(bits):
        a, b = x & mask, x >> n
        groups[(a | b, a & b)].append(i)
    rows, cols, values = [], [], []
    column = 0
    max_block = 0
    all_eigenvalues = []
    for indices in groups.values():
        lookup = {bits[i]:j for j, i in enumerate(indices)}
        matrix = np.zeros((len(indices), len(indices)))
        for j, i in enumerate(indices):
            for target, coefficient in spin_action(bits[i], n).items():
                if target in lookup:
                    matrix[lookup[target], j] += coefficient
        if np.max(abs(matrix-matrix.T))>1e-12:
            raise RuntimeError('Spin matrix is not Hermitian')
        eig, vec = np.linalg.eigh(matrix)
        if eig[0]<-tol:
            raise RuntimeError('Projected S^2 must be positive semidefinite')
        all_eigenvalues.extend(eig.tolist())
        for k in np.flatnonzero(abs(eig)<=tol):
            for i, v in zip(indices, vec[:, k]):
                if abs(v)>1e-14:
                    rows.append(i); cols.append(column); values.append(float(v))
            column += 1
        max_block = max(max_block, len(indices))
    transform = sp.csc_matrix((values,(rows,cols)),shape=(len(bits),column))
    if not column:
        raise RuntimeError('No exact singlet inside this selected CI space')
    return transform, {'determinant_dimension':len(bits),'singlet_dimension':column,
        'spatial_occupation_blocks':len(groups),'max_spin_block_dimension':max_block,
        'spin_nullspace_tolerance':tol, 'added_determinants':0,
        'smallest_projected_spin_eigenvalue':min(all_eigenvalues)}


def diagonalize(ham, nelec, strings, transform, spec):
    n = ham.norb
    solver = CountedSelectedCI(perf_counter()+spec['per_CI_wall_cap_seconds'])
    solver.verbose = 0
    solver.max_memory = 1024
    h2 = solver.absorb_h1e(ham.one_body_tensor,ham.two_body_tensor,n,tuple(nelec),.5)
    shape = (len(strings), len(strings))
    def hop(c):
        full = selected_ci._as_SCIvector(np.asarray(transform@c).reshape(shape),(strings,strings))
        image = solver.contract_2e(h2,full,n,tuple(nelec)).ravel()
        return np.asarray(transform.T@image)
    d = transform.shape[1]
    if d == 1:
        reduced = np.ones(1)
        eigenvalue = hop(reduced)[0]
    else:
        # Deterministic generic initial vector; no prior ground-state information.
        v0 = np.random.default_rng(4713).normal(size=d)
        eig, vec = eigsh(LinearOperator((d,d),matvec=hop,dtype=np.float64),
            k=1,which='SA',tol=1e-10,maxiter=1000,v0=v0)
        eigenvalue, reduced = float(eig[0]), vec[:,0]
    residual = float(np.linalg.norm(hop(reduced)-eigenvalue*reduced))
    vector = selected_ci._as_SCIvector(np.asarray(transform@reduced).reshape(shape),(strings,strings))
    dm1,dm2 = solver.make_rdm12(vector,n,tuple(nelec))
    rdm = ffsim.ReducedDensityMatrix(dm1,dm2)
    physical = float(rdm.expectation(ham).real)
    spin = float(solver.spin_square(vector,n,tuple(nelec))[0])
    if not np.all(np.isfinite([physical,spin,residual])) or residual>spec['eigenvector_residual_tolerance']:
        raise RuntimeError('Singlet CI residual failed')
    if abs(physical-eigenvalue-ham.constant)>1e-8 or abs(spin)>1e-8:
        raise RuntimeError('Singlet physical energy or full-state spin expectation failed')
    return {'energy_Ha':physical,'spin_square':spin,'residual_norm':residual,
        'matvec_calls':solver.matvec_calls,'CI_converged':True,
        'exact_singlet_projection':True,'spin_penalty_energy_Ha':0.0},rdm


def optimize(h,g,core,nelec,strings,spec,callback=None):
    jax.config.update('jax_enable_x64',True)
    cpu,tick = process_time(),perf_counter()
    q,info = basis(strings,len(h),nelec)
    ham = ffsim.MolecularHamiltonian(h,g,constant=core)
    initial,rdm = diagonalize(ham,nelec,strings,q,spec)
    history = [{'outer':0,**initial}]
    rotations = []
    stop = 'OUTER_ITERATION_CAP'
    for outer in range(1,spec['outer_iterations']+1):
        if process_time()-cpu>=spec['per_method_outer_boundary_CPU_cap_seconds']:
            stop = 'CPU_BOUNDARY_CAP'
            break
        c,t = process_time(),perf_counter()
        u,result = ffsim.optimize_orbitals(rdm,ham,real=True,method='L-BFGS-B',
            options={'maxiter':spec['optimizer_maxiter_per_outer'],
                     'maxfun':spec['optimizer_maxfun_per_outer']},return_optimize_result=True)
        if np.max(abs(u.T@u-np.eye(len(h))))>1e-9:
            raise RuntimeError('Nonorthogonal orbital transform')
        changed = ham.rotated(u.T.conj())
        fixed = float(rdm.expectation(changed).real)
        if fixed>history[-1]['energy_Ha']+1e-8:
            stop = 'NON_IMPROVING_ORBITAL_STEP'
            break
        diag,new_rdm = diagonalize(changed,nelec,strings,q,spec)
        if diag['energy_Ha']>fixed+1e-8:
            raise RuntimeError('Exact singlet CI energy increased after rediagonalization')
        ham,rdm = changed,new_rdm
        rotations.append(u)
        history.append({'outer':outer,**diag,'fixed_old_vector_energy_Ha':fixed,
            'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
            'optimizer_nfev':int(result.nfev),'optimizer_njev':int(result.njev),'optimizer_nit':int(result.nit),
            'optimizer_gradient_inf':float(np.max(abs(result.jac))),
            'CPU_seconds':process_time()-c,'wall_seconds':perf_counter()-t})
        if callback:
            callback(history,rotations)
    return {'history':history,'rotations':rotations,'final_h1':ham.one_body_tensor,
        'final_eri':ham.two_body_tensor,'core':ham.constant,'spin_basis':info,
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'stop':stop,
        'new_quantum_model_shots':0,'classical_orbital_coordinates':len(h)*(len(h)-1)//2,
        'final_energy_Ha':history[-1]['energy_Ha'],'initial_energy_Ha':initial['energy_Ha'],
        'sampler_not_a_preparation_of_orbital_optimized_CI_state':True}
