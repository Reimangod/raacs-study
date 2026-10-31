"""N41: bounded semicanonical MP2 qualification, then one naive full-spin L1 per case.

No sampling, molecular CI/FCI diagonalization, optimizer, or GPU entrypoint.
The PySCF CISD vector conversion below is a synthetic linear-coordinate check,
not a molecular CISD/FCI calculation.
"""
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
from time import process_time, perf_counter
import argparse
import importlib.metadata
import inspect
import os
import traceback
import numpy as np
import ffsim
from pyscf import scf
from pyscf.mp import mp2
from pyscf.ci import cisd
from qiskit import qpy
from qiskit.quantum_info import Statevector
from paperstudy.lucj import naive, make_circuit, circuit_views, sector_to_blocked_full, interaction_pairs
from .paths import ROOT, OLD
from .io import load, save, write, arrays, sha256, space
from .direct_sample_pilot import catalog, error_up_to_phase

RUN = ROOT / 'runs/N41-CHEAP-MP2-L1-QUALIFICATION-v1'
CPU_CAP, WALL_CAP, BYTE_CAP = 120., 180., 32 * 2**20


def transform(h, g, c):
    return c.T @ h @ c, np.einsum('pqrs,pi,qj,rk,sl->ijkl', g, c, c, c, c, optimize=True)


def fock(h, g, k):
    return h + 2 * np.einsum('pqii->pq', g[:, :, :k, :k]) - np.einsum('piiq->pq', g[:, :k, :k, :])


def fixed_eigh(a):
    e, c = np.linalg.eigh(a)
    for j in range(c.shape[1]):
        if c[np.argmax(abs(c[:, j])), j] < 0:
            c[:, j] *= -1
    return e, c


def semicanonical_mp2(h, g, k, gradient_tolerance):
    n = len(h)
    if h.shape != (n, n) or g.shape != (n, n, n, n) or not 0 < k < n:
        raise ValueError('Real chemist-integral closed-shell shape required')
    if np.iscomplexobj(h) or np.iscomplexobj(g) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
        raise ValueError('Finite real registered integrals required')
    f = fock(h, g, k)
    herm = float(np.max(abs(f-f.T)))
    # Same unnormalized RHF orbital gradient and default threshold rule as
    # installed PySCF. This is numerical stationarity, not exact Brillouin zero.
    grad_norm = float(np.linalg.norm(2*f[k:, :k]))
    if herm > 1e-10 or grad_norm >= gradient_tolerance:
        raise ValueError(f'HF stationarity not qualified: herm={herm}, grad_norm={grad_norm}')
    eo, co = fixed_eigh(f[:k, :k]); ev, cv = fixed_eigh(f[k:, k:])
    c = np.zeros_like(f); c[:k, :k] = co; c[k:, k:] = cv
    _, gs = transform(h, g, c)
    den = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    if not np.all(np.isfinite(den)) or np.max(den) >= -1e-12:
        raise ValueError('Nonnegative/near-zero MP2 denominator; no clipping or regularization')
    ovov = gs[:k, k:, :k, k:]
    gijab = ovov.transpose(0, 2, 1, 3)
    ts = gijab / den
    energy = float(2*np.einsum('ijab,ijab', ts, gijab)-np.einsum('ijab,ijba', ts, gijab))
    # Independent actual installed MP2 loop, not a second call to our equation.
    pe, pt = mp2.kernel(SimpleNamespace(nocc=k, nmo=n),
                       eris=SimpleNamespace(ovov=ovov, mo_energy=np.r_[eo, ev]))
    err = float(np.max(abs(ts-pt)))
    if err > 1e-10 or abs(energy-pe) > 1e-10:
        raise ValueError('PySCF MP2 kernel index/energy parity failed')
    # Return amplitudes to original problem gauge BEFORE truncated LUCJ factory.
    t = np.einsum('iI,jJ,IJAB,aA,bB->ijab', co, co, ts, cv, cv, optimize=True)
    symmetry = float(np.max(abs(t-t.transpose(1, 0, 3, 2))))
    if symmetry > 1e-9 or not np.all(np.isfinite(t)):
        raise ValueError('Invalid restricted MP2 amplitudes')
    return t, c, dict(HF_RHF_orbital_gradient_norm=grad_norm,
                     HF_Fov_max=float(np.max(abs(f[:k, k:]))), Fock_hermiticity_error=herm,
                     gradient_tolerance=gradient_tolerance,
                     numerical_stationarity_only=True, semicanonical_gap=float(ev[0]-eo[-1]),
                     smallest_absolute_denominator=float(np.min(abs(den))),
                     MP2_correlation_energy_Ha=energy, MP2_energy_not_SQD_energy=True,
                     PySCF_t2_max_error=err, PySCF_energy_error_Ha=abs(energy-float(pe)),
                     restricted_pair_symmetry_error=symmetry,
                     occupied_virtual_mixing=False, denominator_clipping=False)


def synthetic_fixture(n, k, seed):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, .08, (3, n, n)); factors = (factors+factors.transpose(0, 2, 1))/2
    g = np.einsum('xpq,xrs->pqrs', factors, factors)
    e = np.r_[np.linspace(-1.4, -.7, k), np.linspace(.3, 1.3, n-k)]
    h = np.diag(e) - fock(np.zeros((n, n)), g, k)
    q = np.zeros((n, n)); q[:k, :k] = np.linalg.qr(rng.normal(size=(k, k)))[0]
    q[k:, k:] = np.linalg.qr(rng.normal(size=(n-k, n-k)))[0]
    return h, g, e, q


def synthetic_checks():
    rows = []
    for n, k in ((3, 1), (4, 2), (5, 2)):
        h, g, e, q = synthetic_fixture(n, k, 41000+n)
        hr, gr = transform(h, g, q)
        t, _, result = semicanonical_mp2(hr, gr, k, 1e-10)
        pe, pt = mp2.kernel(SimpleNamespace(nocc=k, nmo=n),
                           eris=SimpleNamespace(ovov=g[:k, k:, :k, k:], mo_energy=e))
        expected = np.einsum('Ii,Jj,IJAB,Aa,Bb->ijab', q[:k, :k], q[:k, :k], pt, q[k:, k:], q[k:, k:], optimize=True)
        amp_error = float(np.max(abs(t-expected)))
        def vec(t2):
            return cisd.to_fcivec(cisd.amplitudes_to_cisdvec(1., np.zeros((k, n-k)), t2), n, (k, k)).reshape(-1).astype(complex)
        rotated = ffsim.apply_orbital_rotation(vec(pt), q.T, n, (k, k))
        vector_error = error_up_to_phase(rotated, vec(t))
        # Actual full-spin LUCJ with original-gauge t2; do NOT assert that the
        # masked/truncated factory itself is invariant under orbital rotations.
        op = naive(np.zeros((k, n-k)), t, n, 1)
        state = ffsim.apply_unitary(ffsim.hartree_fock_state(n, (k, k)), op, n, (k, k))
        qs = Statevector(make_circuit(op, n, (k, k))).data
        state_error = error_up_to_phase(sector_to_blocked_full(state, n, (k, k)), qs)
        if max(amp_error, vector_error, state_error, abs(pe-result['MP2_correlation_energy_Ha'])) > 1e-8:
            raise ValueError('Synthetic orbital/amplitude/full-spin state parity failed')
        rows.append(dict(norb=n, nelec=[k, k], amplitude_covariance_error=amp_error,
                         synthetic_first_order_vector_covariance_error=vector_error,
                         MP2_energy_covariance_error=abs(float(pe)-result['MP2_correlation_energy_Ha']),
                         independent_Qiskit_ffsim_state_error=state_error))
    return rows


def bindings():
    paths = [Path(__file__), OLD/'paperstudy/lucj.py',
             Path(inspect.getfile(mp2)), Path(inspect.getfile(scf.hf)), Path(inspect.getfile(ffsim.UCJOpSpinBalanced))]
    return {str(p): sha256(p.read_bytes()) for p in paths}


def freeze():
    space()
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Existing freeze; no overwrite')
    save(RUN/'protocol.json', dict(version='N41_CHEAP_FULLSPIN_MP2_L1_QUALIFICATION_V1',
         catalog=catalog(), bindings=bindings(), cases=4, CPU_seconds=CPU_CAP, wall_seconds=WALL_CAP,
         new_run_bytes=BYTE_CAP, sampling_CI_FCI_optimizer_GPU_authorized=False,
         synthetic_fixtures=[[3, 1, 41003], [4, 2, 41004], [5, 2, 41005]],
         HF_gradient_tolerance=float(np.sqrt(scf.hf.SCF.conv_tol)),
         HF_threshold_origin='sqrt(installed_PySCF_SCF_default_conv_tol); norm(2Fvo); no fit to molecule outcomes',
         HF_default_conv_tol=float(scf.hf.SCF.conv_tol), PySCF_TIGHT_GRAD_CONV_TOL=bool(scf.hf.TIGHT_GRAD_CONV_TOL),
         method='BLOCK_SEMICANONICAL_MP2_BACKTRANSFORM_T2_TO_ORIGINAL_GAUGE_NAIVE_FULLSPIN_LUCJ_L1',
         t1='EXPLICIT_ZEROS_NUMERIC_BRILLOUIN_RESIDUAL_REPORTED', n_reps=1, factory_optimize=False,
         factory_tol=1e-8, mask='EXISTING_AUTHOR_FFSIM_HEAVY_HEX',
         source_variants_per_case=1, parameter_rank=None,
         exact_HF_stationarity_claim=False, new_method_novelty_claim=False,
         packages={name: importlib.metadata.version(name) for name in ('pyscf', 'ffsim', 'qiskit', 'numpy', 'scipy')}))
    print('N41 qualification freeze recorded; no new shots/CI/optimizer/GPU authorized')


def verified():
    p = load(RUN/'protocol.json')
    if p['bindings'] != bindings() or p['catalog'] != catalog():
        raise ValueError('Source/code drift')
    if p['packages'] != {name: importlib.metadata.version(name) for name in p['packages']}:
        raise ValueError('Runtime drift')
    return p


def run():
    p = verified(); space()
    if (RUN/'started.json').exists():
        raise RuntimeError('No retry or overwrite of partial/terminal qualification')
    save(RUN/'started.json', dict(pid=os.getpid(), protocol_SHA256=sha256((RUN/'protocol.json').read_bytes())))
    cpu, wall = process_time(), perf_counter(); rows = []; small = []; failure = None
    def bound():
        space()
        if process_time()-cpu > CPU_CAP or perf_counter()-wall > WALL_CAP:
            raise TimeoutError('N41 small qualification cap')
        if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > BYTE_CAP:
            raise OSError('N41 new-run storage cap')
    try:
        small = synthetic_checks(); save(RUN/'synthetic.json', dict(status='PASSED', rows=small))
        bound()
        # First qualify every physical gauge; preserve any rejection in the full roster.
        qualified = []
        for item in p['catalog']:
            bound(); d = item['physical']; s = item['source']; tick = process_time(); w = perf_counter()
            with np.load(d['local_integrals'], allow_pickle=False) as z:
                h, g = z['h1'], z['eri']
            n, ne = d['norb'], tuple(d['nelec'])
            if ne[0] != ne[1]:
                raise ValueError('Closed-shell alpha/beta equality required')
            t, c, result = semicanonical_mp2(h, g, ne[0], p['HF_gradient_tolerance'])
            folder = RUN/'sources'/s['case']
            write(folder/'amplitudes.npz', arrays(t1=np.zeros((ne[0], n-ne[0])), t2=t, semicanonical_to_original=c))
            qualified.append((item, t, result, process_time()-tick, perf_counter()-w))
        for item, t, result, mp2cpu, mp2wall in qualified:
            bound(); d, s = item['physical'], item['source']; n, ne = d['norb'], tuple(d['nelec'])
            folder = RUN/'sources'/s['case']; tick = process_time(); w = perf_counter()
            op = naive(np.zeros((ne[0], n-ne[0])), t, n, 1)
            factory_cpu, factory_wall = process_time()-tick, perf_counter()-w
            write(folder/'operator.npz', arrays(diag_coulomb_mats=op.diag_coulomb_mats,
                  orbital_rotations=op.orbital_rotations, final_orbital_rotation=op.final_orbital_rotation))
            tick = process_time(); w = perf_counter()
            qc = ffsim.qiskit.PRE_INIT.run(make_circuit(op, n, ne))
            stream = BytesIO(); qpy.dump(qc, stream); write(folder/'logical.qpy', stream.getvalue())
            restored = qpy.load(BytesIO(stream.getvalue()))[0]
            prep, sampling, resources = circuit_views(restored)
            for name, circuit in [('preparation.qpy', prep), ('sampling.qpy', sampling)]:
                stream = BytesIO(); qpy.dump(circuit, stream); write(folder/name, stream.getvalue())
            compile_cpu, compile_wall = process_time()-tick, perf_counter()-w
            err = None; check_cpu = 0.
            if s['case'] == 'NAPH20-BLOCK-BOYS':
                tick = process_time()
                from .prefix_performance_sampling import qualified_module
                cfg, vals = qualified_module().plan(restored, n, ne)
                actual = qualified_module().numpy_state(cfg, vals)
                expected = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne)
                err = error_up_to_phase(expected, actual)
                if err > 1e-8 or abs(np.vdot(actual, actual)-1) > 1e-8:
                    raise ValueError('Actual 20q primitive/QPY/native state mismatch')
                check_cpu = process_time()-tick
            files = {name: dict(path=str(folder/name), SHA256=sha256((folder/name).read_bytes()))
                     for name in ('amplitudes.npz', 'operator.npz', 'logical.qpy', 'preparation.qpy', 'sampling.qpy')}
            row = dict(case=s['case'], problem_id=d['problem_id'], norb=n, nelec=list(ne),
                       status='SOURCE_CONSTRUCTED_NO_SQD_OUTCOME', MP2_qualification=result,
                       MP2_CPU_seconds=mp2cpu, MP2_wall_seconds=mp2wall,
                       factory_CPU_seconds=factory_cpu, factory_wall_seconds=factory_wall,
                       compile_CPU_seconds=compile_cpu, compile_wall_seconds=compile_wall,
                       actual20q_state_check_CPU_seconds=check_cpu,
                       actual20q_state_error_up_to_phase=err,
                       other_case_full_molecular_state_check='NOT_RUN' if err is None else 'NOT_APPLICABLE',
                       resources=resources, coefficient_slots=len(op.to_parameters(interaction_pairs=interaction_pairs(n))),
                       independent_ansatz_parameter_rank=None, old_initialization_cost=item['old_initialization_cost'],
                       expensive_CCSD_CISD_amplitudes_used=False, compressed_fit_used=False, files=files)
            save(folder/'result.json', row); rows.append(row); save(RUN/'progress.json', dict(rows=rows)); bound()
    except Exception:
        failure = traceback.format_exc()
    result = dict(status='ALL4_SOURCES_QUALIFIED_AND_CONSTRUCTED' if failure is None else 'FAILED_PRESERVED',
                  failure=failure, rows=rows, small_fixtures=small, CPU_seconds=process_time()-cpu,
                  wall_seconds=perf_counter()-wall, new_model_samples=0, new_QPU_shots=0,
                  new_molecular_CI_FCI_roots=0, new_optimizer_calls=0, GPU_allocation_seconds=0,
                  SQD_energy_Recovery_advantage_proven=False, best_paper_superiority_proven=False,
                  package_import_cost_not_in_run_timer=True)
    save(RUN/'terminal.json', result)
    print(result['status'], 'rows', len(rows), 'CPU', result['CPU_seconds'])
    if failure:
        print(failure); raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run'))
    action = parser.parse_args().action
    {'freeze': freeze, 'run': run}[action]()
