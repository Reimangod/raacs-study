"""N43 toy-only qualification of a known fixed-U, real symmetric J least-squares restriction.

The union-mask t2 objective is a heuristic for the final separate spin masks,
not a promise about physical expectation or sample-CI energies.
"""
from io import BytesIO
from pathlib import Path
from time import process_time, perf_counter
import argparse
import inspect
import os
import traceback
import numpy as np
import ffsim
from ffsim.linalg.double_factorized_decomposition import reconstruct_t2
from qiskit import qpy
from qiskit.quantum_info import Statevector
from paperstudy.lucj import interaction_pairs, make_circuit, circuit_views, sector_to_blocked_full
from .direct_sample_pilot import error_up_to_phase
from .paths import ROOT, OLD
from .io import load, save, sha256, space

RUN = ROOT/'runs/N43-FIXED-ROTATION-T2-LS-TOY-v1'


def symmetric_basis(n, indices):
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= i <= j < n for i, j in indices):
        raise ValueError('Unique real symmetric upper-triangle mask required')
    out = np.zeros((len(indices), n, n))
    for a, (i, j) in enumerate(indices):
        out[a, i, j] = out[a, j, i] = 1
    return out


def fit(target, rotations, indices):
    k, kk, v, vv = target.shape; n = k+v
    if kk != k or vv != v or rotations.shape != (1, n, n):
        raise ValueError('Exactly one fixed orbital rotation and restricted t2 required')
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(rotations)):
        raise ValueError('Finite target and rotations required')
    if not np.allclose(rotations[0].conj().T @ rotations[0], np.eye(n), atol=1e-10, rtol=0):
        raise ValueError('Fixed U must be unitary')
    basis = symmetric_basis(n, indices)
    a = np.stack([reconstruct_t2(b[None], rotations, k).ravel() for b in basis], axis=1)
    real_a = np.vstack([a.real, a.imag]); b = np.r_[target.ravel().real, target.ravel().imag]
    z, _, rank, singular = np.linalg.lstsq(real_a, b, rcond=None)
    j = np.einsum('m,mij->ij', z, basis)[None]
    reconstruction = reconstruct_t2(j, rotations, k)
    if not np.all(np.isfinite(z)) or np.max(abs(reconstruction.ravel()-a@z)) > 1e-10:
        raise ValueError('SDK complex reconstruction is not this real linear map')
    residual = real_a@z-b
    normal = float(np.linalg.norm(real_a.T@residual))
    return j, dict(loss_half_squared=float(.5*(residual@residual)), normal_equation_residual=normal,
                   linear_matrix_rank=int(rank), coefficient_count=len(z),
                   singular_values=singular.tolist(), coefficient_norm=float(np.linalg.norm(z)),
                   rcond='NUMPY_DEFAULT_EPS_TIMES_MAX_SHAPE', real_coefficient_constraint=True), (a, real_a, z)


def bindings():
    paths = [Path(__file__), Path(inspect.getfile(reconstruct_t2)), OLD/'paperstudy/lucj.py']
    return {str(p): sha256(p.read_bytes()) for p in paths}


def freeze():
    space()
    if (RUN/'protocol.json').exists(): raise RuntimeError('No freeze overwrite')
    save(RUN/'protocol.json', dict(version='N43_FIXED_U_UNION_J_REAL_LS_TOY_ONLY_V1', bindings=bindings(),
         fixtures=[[3, 1, 43003], [4, 2, 43004], [5, 2, 43005]],
         CPU_seconds=5, wall_seconds=30, new_run_bytes=2*2**20,
         U_origin='NATIVE_EXPLICIT_T2_FIRST1_NOT_EXISTING_COMPRESSED_FIT',
         J_origin='REAL_SYMMETRIC_UNION_MASK_LINEAR_LS_MIN_NORM',
         source_energy_or_SQD_improvement_guarantee=False, new_method_novelty=False,
         molecular_source_sampling_CI_FCI_optimizer_GPU_authorized=False,
         objective='KNOWN_CDF_RESTRICTION_NOT_FULLSPIN_PHYSICAL_ENERGY',
         plan_SHA256=sha256((ROOT/'analysis/N43_FIXED_ROTATION_LS_NEXT_JA.md').read_bytes())))
    print('N43 toy-only qualification frozen; no molecular calculations authorized')


def run():
    p = load(RUN/'protocol.json'); space()
    if p['bindings'] != bindings(): raise ValueError('Frozen code/SDK changed')
    if (RUN/'started.json').exists(): raise RuntimeError('No retry or overwrite')
    save(RUN/'started.json', dict(pid=os.getpid(), protocol_SHA256=sha256((RUN/'protocol.json').read_bytes())))
    cpu, wall = process_time(), perf_counter(); rows = []; failure = None
    try:
        for n, k, seed in p['fixtures']:
            rng = np.random.default_rng(seed); t = rng.normal(0, .04, (k, k, n-k, n-k)); t = (t+t.transpose(1, 0, 3, 2))/2
            j0, u = ffsim.linalg.double_factorized_t2(t, max_terms=1, optimize=False)
            aa, ab = interaction_pairs(n); indices = sorted(set(aa+ab)); mask = np.zeros((n, n))
            for i, j in indices: mask[i, j] = mask[j, i] = 1
            original = j0*mask; j, info, (a, real_a, z) = fit(t, u, indices)
            loss0 = float(.5*np.sum(abs(reconstruct_t2(original, u, k)-t)**2))
            if info['loss_half_squared'] > loss0+1e-10 or info['normal_equation_residual'] > 1e-9:
                raise ValueError('Least-squares monotonicity/optimality failure')
            trial = rng.normal(size=len(indices)); mats = np.einsum('m,mij->ij', trial, symmetric_basis(n, indices))[None]
            linear_error = float(np.max(abs(reconstruct_t2(mats, u, k).ravel()-a@trial)))
            # Separate spin masks preserve the same physical LUCJ family, but are
            # not asserted to retain the union-mask t2 reconstruction exactly.
            ja = np.zeros_like(j); jb = np.zeros_like(j)
            for i, l in aa: ja[0, i, l] = ja[0, l, i] = j[0, i, l]
            for i, l in ab: jb[0, i, l] = jb[0, l, i] = j[0, i, l]
            op = ffsim.UCJOpSpinBalanced(np.stack([ja, jb], axis=1), u, np.eye(n))
            prep, sampling, resources = circuit_views(make_circuit(op, n, (k, k)))
            stream = BytesIO(); qpy.dump(prep, stream); restored = qpy.load(BytesIO(stream.getvalue()))[0]
            state = ffsim.apply_unitary(ffsim.hartree_fock_state(n, (k, k)), op, n, (k, k))
            full = sector_to_blocked_full(state, n, (k, k))
            state_error = error_up_to_phase(full, Statevector(restored).data)
            probability_error = float(np.sum(abs(abs(full)**2-abs(Statevector(sampling).data)**2)))
            if max(linear_error, state_error, probability_error) > 1e-8:
                raise ValueError('Actual SDK linear map/state/QPY parity failed')
            rows.append(dict(norb=n, nelec=[k, k], naive_masked_loss=loss0, **info,
                             SDK_linear_map_error=linear_error, Qiskit_ffsim_QPY_state_error=state_error,
                             sampling_probability_L1_error=probability_error, resources=resources,
                             independent_parameter_rank=None, physical_energy_claim=False))
            if process_time()-cpu > p['CPU_seconds'] or perf_counter()-wall > p['wall_seconds']:
                raise TimeoutError('N43 tiny qualification cap')
    except Exception:
        failure = traceback.format_exc()
    result = dict(status='ALL3_TOY_LINEAR_MAP_AND_PHYSICAL_CIRCUIT_CHECKS_PASSED' if failure is None else 'FAILED_PRESERVED',
         rows=rows, failure=failure, CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
         package_import_cost_not_in_run_timer=True, new_molecular_sources_samples_CI_FCI_GPU=0,
         molecular_energy_Recovery_or_resource_advantage_proven=False, best_paper_superiority_proven=False)
    save(RUN/'terminal.json', result); print(result['status'], 'CPU', result['CPU_seconds'])
    for row in rows: print(row['norb'], 'loss before/after', row['naive_masked_loss'], row['loss_half_squared'], 'rank', row['linear_matrix_rank'])
    if failure: print(failure); raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[parser.parse_args().action]()
