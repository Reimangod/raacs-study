"""N41 additive QPY boundary repair: compile live SDK, audit ordinary-gate QPY.

V1 source, freeze, failure and spent CPU remain immutable. No scientific change.
"""
from io import BytesIO
from pathlib import Path
from time import process_time, perf_counter
import argparse
import os
import traceback
import numpy as np
import ffsim
from qiskit import qpy
from qiskit.quantum_info import Statevector
from paperstudy.lucj import make_circuit, circuit_views, naive, interaction_pairs, sector_to_blocked_full
from . import cheap_mp2_qualification as v1
from .io import load, save, write, arrays, sha256, space
from .paths import ROOT
from .direct_sample_pilot import catalog, error_up_to_phase

RUN = ROOT/'runs/N41-CHEAP-MP2-L1-QUALIFICATION-v2'


def serialize(qc):
    stream = BytesIO(); qpy.dump(qc, stream)
    return stream.getvalue()


def live_views(op, n, ne):
    # PRE_INIT must see SDK object types, not deserialized generic Gate wrappers.
    logical = ffsim.qiskit.PRE_INIT.run(make_circuit(op, n, ne))
    prep, sampling, views = circuit_views(logical)
    for name, qc in [('preparation', prep), ('Z_sampling_equivalent', sampling)]:
        restored = qpy.load(BytesIO(serialize(qc)))[0]
        actual = dict(cnot_count=int(restored.count_ops().get('cx', 0)),
                      cnot_depth=int(restored.depth(filter_function=lambda x: x.operation.name == 'cx')),
                      total_unitary_depth=int(restored.depth()))
        if any(actual[k] != views[name][k] for k in actual):
            raise ValueError('Serialized actual resources disagree')
    return logical, prep, sampling, views


def boundary_checks():
    rows = []
    for n, k in ((3, 1), (4, 2), (5, 2)):
        h, g, _, _ = v1.synthetic_fixture(n, k, 41000+n)
        t, _, _ = v1.semicanonical_mp2(h, g, k, 1e-10)
        op = naive(np.zeros((k, n-k)), t, n, 1)
        _, prep, sample, _ = live_views(op, n, (k, k))
        prep = qpy.load(BytesIO(serialize(prep)))[0]
        sample = qpy.load(BytesIO(serialize(sample)))[0]
        state = ffsim.apply_unitary(ffsim.hartree_fock_state(n, (k, k)), op, n, (k, k))
        full = sector_to_blocked_full(state, n, (k, k))
        err = error_up_to_phase(full, Statevector(prep).data)
        prob_err = float(np.sum(abs(abs(full)**2-abs(Statevector(sample).data)**2)))
        if max(err, prob_err) > 1e-8:
            raise ValueError('Live compile/ordinary QPY state or sampling mismatch')
        rows.append(dict(norb=n, nelec=[k, k], actual_preparation_state_error=err,
                         sampling_probability_L1_error=prob_err))
    return rows


def bindings():
    result = v1.bindings(); result[str(Path(__file__))] = sha256(Path(__file__).read_bytes())
    for name in ('protocol.json', 'terminal.json', 'synthetic.json'):
        result[str(v1.RUN/name)] = sha256((v1.RUN/name).read_bytes())
    for item in catalog():
        f = v1.RUN/'sources'/item['source']['case']/'amplitudes.npz'
        result[str(f)] = sha256(f.read_bytes())
    return result


def freeze():
    v1.verified(); old = load(v1.RUN/'terminal.json')
    if old['status'] != 'FAILED_PRESERVED' or old['rows'] or "'Gate' object has no attribute 'norb'" not in old['failure']:
        raise ValueError('Only exact pre-outcome boundary failure repair authorized')
    if (RUN/'protocol.json').exists():
        raise RuntimeError('No freeze overwrite')
    p = load(v1.RUN/'protocol.json'); p['version'] = 'N41_CHEAP_FULLSPIN_MP2_L1_QPY_BOUNDARY_REPAIR_V2'
    p['bindings'] = bindings(); p['predecessor_CPU_seconds'] = old['CPU_seconds']
    p['repair_only'] = 'Compile live SDK before QPY serialization; audit ordinary-gate QPY; no amplitude or method change'
    save(RUN/'protocol.json', p); print('N41 v2 narrow QPY boundary repair frozen')


def verified():
    p = load(RUN/'protocol.json')
    if p['bindings'] != bindings() or p['catalog'] != catalog():
        raise ValueError('Frozen code/source/predecessor drift')
    v1.verified()
    return p


def run():
    p = verified(); space()
    if (RUN/'started.json').exists():
        raise RuntimeError('No restart or overwrite')
    save(RUN/'started.json', dict(pid=os.getpid(), protocol_SHA256=sha256((RUN/'protocol.json').read_bytes())))
    cpu, wall = process_time(), perf_counter(); rows = []; checks = []; failure = None
    def bound():
        space()
        if process_time()-cpu+p['predecessor_CPU_seconds'] > v1.CPU_CAP or perf_counter()-wall > v1.WALL_CAP:
            raise TimeoutError('Cumulative N41 cap')
        if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) > v1.BYTE_CAP:
            raise OSError('New N41 storage cap')
    try:
        checks = boundary_checks(); save(RUN/'boundary-qualifier.json', dict(status='PASSED', rows=checks)); bound()
        for item in p['catalog']:
            bound(); s, d = item['source'], item['physical']; n, ne = d['norb'], tuple(d['nelec'])
            folder = RUN/'sources'/s['case']; tick = process_time(); w = perf_counter()
            with np.load(d['local_integrals'], allow_pickle=False) as z: h, g = z['h1'], z['eri']
            with np.load(v1.RUN/'sources'/s['case']/'amplitudes.npz', allow_pickle=False) as z:
                old_t, old_c = z['t2'], z['semicanonical_to_original']
            t, c, result = v1.semicanonical_mp2(h, g, ne[0], p['HF_gradient_tolerance'])
            if not np.array_equal(t, old_t) or not np.array_equal(c, old_c):
                raise ValueError('No scientific change permitted in QPY boundary repair')
            mp2cpu, mp2wall = process_time()-tick, perf_counter()-w
            write(folder/'amplitudes.npz', arrays(t1=np.zeros((ne[0], n-ne[0])), t2=t, semicanonical_to_original=c))
            tick = process_time(); w = perf_counter()
            op = naive(np.zeros((ne[0], n-ne[0])), t, n, 1)
            factorycpu, factorywall = process_time()-tick, perf_counter()-w
            write(folder/'operator.npz', arrays(diag_coulomb_mats=op.diag_coulomb_mats,
                  orbital_rotations=op.orbital_rotations, final_orbital_rotation=op.final_orbital_rotation))
            tick = process_time(); w = perf_counter()
            logical, prep, sampling, resources = live_views(op, n, ne)
            for name, qc in [('logical.qpy', logical), ('preparation.qpy', prep), ('sampling.qpy', sampling)]:
                write(folder/name, serialize(qc))
            compilecpu, compilewall = process_time()-tick, perf_counter()-w
            err = None; checkcpu = 0.
            if s['case'] == 'NAPH20-BLOCK-BOYS':
                tick = process_time()
                from .prefix_performance_sampling import qualified_module
                module = qualified_module(); restored = qpy.load(BytesIO((folder/'logical.qpy').read_bytes()))[0]
                cfg, vals = module.plan(restored, n, ne); actual = module.numpy_state(cfg, vals)
                expected = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne)
                err = error_up_to_phase(expected, actual)
                if err > 1e-8 or abs(np.vdot(actual, actual)-1) > 1e-8:
                    raise ValueError('Actual 20q primitive/QPY/native state mismatch')
                checkcpu = process_time()-tick
            files = {name: dict(path=str(folder/name), SHA256=sha256((folder/name).read_bytes()))
                     for name in ('amplitudes.npz', 'operator.npz', 'logical.qpy', 'preparation.qpy', 'sampling.qpy')}
            row = dict(case=s['case'], problem_id=d['problem_id'], norb=n, nelec=list(ne),
                       status='SOURCE_CONSTRUCTED_NO_SQD_OUTCOME', MP2_qualification=result,
                       MP2_CPU_seconds=mp2cpu, MP2_wall_seconds=mp2wall,
                       factory_CPU_seconds=factorycpu, factory_wall_seconds=factorywall,
                       compile_CPU_seconds=compilecpu, compile_wall_seconds=compilewall,
                       actual20q_state_check_CPU_seconds=checkcpu, actual20q_state_error_up_to_phase=err,
                       other_case_full_molecular_state_check='NOT_RUN' if err is None else 'NOT_APPLICABLE',
                       resources=resources, coefficient_slots=len(op.to_parameters(interaction_pairs=interaction_pairs(n))),
                       independent_ansatz_parameter_rank=None, old_initialization_cost=item['old_initialization_cost'],
                       expensive_CCSD_CISD_amplitudes_used=False, compressed_fit_used=False,
                       seed_or_scale_sweep=False, files=files)
            save(folder/'result.json', row); rows.append(row); save(RUN/'progress.json', dict(rows=rows)); bound()
    except Exception:
        failure = traceback.format_exc()
    result = dict(status='ALL4_SOURCES_QUALIFIED_AND_CONSTRUCTED' if failure is None else 'FAILED_PRESERVED',
                  failure=failure, rows=rows, boundary_checks=checks, CPU_seconds=process_time()-cpu,
                  wall_seconds=perf_counter()-wall, predecessor_CPU_seconds=p['predecessor_CPU_seconds'],
                  new_model_samples=0, new_QPU_shots=0, new_molecular_CI_FCI_roots=0,
                  new_optimizer_calls=0, GPU_allocation_seconds=0, package_import_cost_not_in_run_timer=True,
                  SQD_energy_Recovery_advantage_proven=False, best_paper_superiority_proven=False)
    save(RUN/'terminal.json', result)
    print(result['status'], 'rows', len(rows), 'CPU', result['CPU_seconds'])
    if failure:
        print(failure); raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[parser.parse_args().action]()
