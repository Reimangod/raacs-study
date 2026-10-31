"""N44: one fixed-U linear-J full-spin L1 per existing physical case; no sampling."""
from pathlib import Path
from io import BytesIO
from time import process_time, perf_counter
import argparse
import os
import traceback
import numpy as np
import ffsim
from qiskit import qpy
from ffsim.linalg.double_factorized_decomposition import reconstruct_t2
from paperstudy.lucj import amplitudes_from_saved, naive, interaction_pairs
from . import fixed_rotation_ls as math
from .cheap_mp2_qualification_v2 import live_views, serialize
from .direct_sample_pilot import catalog, error_up_to_phase
from .prefix_performance_sampling import qualified_module
from .paths import ROOT, OLD
from .io import load, save, write, arrays, sha256, space

RUN = ROOT/'runs/N44-FIXED-ROTATION-L1-SOURCES-v1'


def inputs():
    result = []
    for item in catalog():
        case = item['source']['case']; directory = OLD/'data/initialization-fresh'/case
        record = load(directory/'result.json'); path = directory/'amplitudes.npz'
        if record['problem_id'] != item['physical']['problem_id'] or sha256(path.read_bytes()) != record['amplitudes_sha256']:
            raise ValueError('Same physical gauge/amplitude identity required')
        entry = dict(item, amplitude_file=str(path), amplitude_record=str(directory/'result.json'),
                     amplitude_record_SHA256=sha256((directory/'result.json').read_bytes()),
                     amplitude_SHA256=record['amplitudes_sha256'], amplitude_provenance=record)
        if record.get('source_converged_CCSD'):
            upstream = OLD/record['source_converged_CCSD']
            entry['upstream_CCSD_record'] = dict(path=str(upstream), SHA256=sha256(upstream.read_bytes()), record=load(upstream))
        result.append(entry)
    return result


def bindings():
    files = [Path(__file__), Path(math.__file__), OLD/'paperstudy/lucj.py',
             ROOT/'frontierloop/cheap_mp2_qualification_v2.py']
    result = {str(f): sha256(f.read_bytes()) for f in files}
    result.update(math.bindings())
    return result


def freeze():
    space(); q = load(math.RUN/'terminal.json')
    if q['status'] != 'ALL3_TOY_LINEAR_MAP_AND_PHYSICAL_CIRCUIT_CHECKS_PASSED':
        raise ValueError('Actual N43 toy evidence required')
    if (RUN/'protocol.json').exists(): raise RuntimeError('No freeze overwrite')
    save(RUN/'protocol.json', dict(version='N44_FIXED_EXPLICIT_U_REAL_J_LS_SAME_AMPS_L1_V1',
         catalog=inputs(), bindings=bindings(), CPU_seconds=30, wall_seconds=60, new_run_bytes=8*2**20,
         source_variants_per_case=1, t1='SAME_SAVED_CCSD_OR_REGISTERED_CISD_TRANSFORMATION',
         U='FIRST1_NATIVE_EXPLICIT_DF_NOT_EXISTING_PAID_FIT',
         J='ONCE_REAL_SYMMETRIC_UNION_LS_THEN_SEPARATE_SPIN_MASKS',
         N43_terminal_SHA256=sha256((math.RUN/'terminal.json').read_bytes()),
         molecular_sampling_CI_FCI_optimizer_CCSD_GPU_authorized=False,
         source_loss_is_not_SQD_energy=True, independent_parameter_rank=None,
         plan_SHA256=sha256((ROOT/'analysis/N44_FIXED_ROTATION_SOURCE_NEXT_JA.md').read_bytes())))
    print('N44 all4 source construction frozen; no new molecular samples/CI/GPU')


def verified():
    p = load(RUN/'protocol.json')
    if p['catalog'] != inputs() or p['bindings'] != bindings(): raise ValueError('Frozen inputs/code changed')
    if p['N43_terminal_SHA256'] != sha256((math.RUN/'terminal.json').read_bytes()): raise ValueError('Toy evidence changed')
    return p


def run():
    p = verified(); space()
    if (RUN/'started.json').exists(): raise RuntimeError('No restart/overwrite')
    save(RUN/'started.json', dict(pid=os.getpid(), protocol_SHA256=sha256((RUN/'protocol.json').read_bytes())))
    cpu, wall = process_time(), perf_counter(); rows = []; failure = None
    def bound():
        space()
        if process_time()-cpu > p['CPU_seconds'] or perf_counter()-wall > p['wall_seconds']: raise TimeoutError('N44 construction cap')
        if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > p['new_run_bytes']: raise OSError('N44 storage cap')
    try:
        for item in p['catalog']:
            bound(); d, s = item['physical'], item['source']; n, ne = d['norb'], tuple(d['nelec']); k = ne[0]
            tick, w = process_time(), perf_counter()
            t1, t2, kind = amplitudes_from_saved(item['amplitude_file'], n, k)
            j0, u = ffsim.linalg.double_factorized_t2(t2, max_terms=1, optimize=False)
            base = naive(t1, t2, n, 1)
            if np.max(abs(base.orbital_rotations-u)) > 1e-12: raise ValueError('Explicit U differs from same-amp naive factory')
            aa, ab = interaction_pairs(n); indices = sorted(set(aa+ab)); mask = np.zeros((n, n))
            for i, j in indices: mask[i, j] = mask[j, i] = 1
            loss0 = float(.5*np.sum(abs(reconstruct_t2(j0*mask, u, k)-t2)**2))
            j, info, _ = math.fit(t2, u, indices)
            if info['loss_half_squared'] > loss0+1e-9 or info['normal_equation_residual'] > 1e-8:
                raise ValueError('Molecular algebraic fit failed; not a physical energy rejection')
            ja = np.zeros_like(j); jb = np.zeros_like(j)
            for i, l in aa: ja[0, i, l] = ja[0, l, i] = j[0, i, l]
            for i, l in ab: jb[0, i, l] = jb[0, l, i] = j[0, i, l]
            op = ffsim.UCJOpSpinBalanced(np.stack([ja, jb], axis=1), u, base.final_orbital_rotation.copy())
            initcpu, initwall = process_time()-tick, perf_counter()-w
            tick, w = process_time(), perf_counter(); logical, prep, sampling, resources = live_views(op, n, ne)
            compilecpu, compilewall = process_time()-tick, perf_counter()-w
            folder = RUN/'sources'/s['case']
            write(folder/'operator.npz', arrays(diag_coulomb_mats=op.diag_coulomb_mats, orbital_rotations=op.orbital_rotations, final_orbital_rotation=op.final_orbital_rotation))
            for name, qc in [('logical.qpy', logical), ('preparation.qpy', prep), ('sampling.qpy', sampling)]: write(folder/name, serialize(qc))
            checkcpu = 0.; error = None
            if n == 10:
                tick = process_time(); module = qualified_module()
                restored = qpy.load(BytesIO((folder/'logical.qpy').read_bytes()))[0]
                cfg, values = module.plan(restored, n, ne); actual = module.numpy_state(cfg, values)
                expected = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne)
                error = error_up_to_phase(expected, actual)
                if error > 1e-8 or abs(np.vdot(actual, actual)-1) > 1e-8: raise ValueError('Actual20q source state failed')
                checkcpu = process_time()-tick
            files = {name: dict(path=str(folder/name), SHA256=sha256((folder/name).read_bytes()))
                     for name in ('operator.npz', 'logical.qpy', 'preparation.qpy', 'sampling.qpy')}
            row = dict(case=s['case'], problem_id=d['problem_id'], norb=n, nelec=list(ne), status='CONSTRUCTED_NO_SQD_OUTCOME',
                       initializer='FIXED_EXPLICIT_U_REAL_J_LS', amplitude_kind=kind,
                       amplitude_file=item['amplitude_file'], amplitude_SHA256=item['amplitude_SHA256'],
                       source_amplitude_creation_provenance=item['amplitude_provenance'],
                       upstream_CCSD_record=item.get('upstream_CCSD_record'),
                       naive_masked_loss=loss0, linear_fit=info, resources=resources,
                       coefficient_slots=len(op.to_parameters(interaction_pairs=interaction_pairs(n))),
                       independent_parameter_rank=None, initialization_CPU_seconds=initcpu, initialization_wall_seconds=initwall,
                       compile_CPU_seconds=compilecpu, compile_wall_seconds=compilewall, actual20q_state_check_CPU_seconds=checkcpu,
                       actual20q_state_error=error, large_molecular_state_not_run=True,
                       old_initialization_cost=item['old_initialization_cost'], paid_CDF_UJ_used_as_input=False,
                       new_CCSD_CISD_FCI_solves=0, files=files)
            save(folder/'result.json', row); rows.append(row); save(RUN/'progress.json', dict(rows=rows)); bound()
    except Exception: failure = traceback.format_exc()
    result = dict(status='ALL4_FIXED_U_LINEAR_J_SOURCES_CONSTRUCTED' if failure is None else 'FAILED_PRESERVED',
         failure=failure, rows=rows, CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
         package_import_cost_not_in_run_timer=True, new_model_samples_QPU_CI_FCI_optimizer_GPU=0,
         molecular_energy_Recovery_advantage_proven=False, best_paper_superiority_proven=False)
    save(RUN/'terminal.json', result); print(result['status'], 'CPU', result['CPU_seconds'])
    for row in rows: print(row['case'], 'loss', row['naive_masked_loss'], row['linear_fit']['loss_half_squared'],
                          'rank', row['linear_fit']['linear_matrix_rank'], 'initCPU', row['initialization_CPU_seconds'])
    if failure: print(failure); raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[parser.parse_args().action]()
