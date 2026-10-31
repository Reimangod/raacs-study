"""Materialize the circuit actually implemented by native Givens compression.

Never simulate a capped UCJ/OrbitalRotation high-level gate directly: ffsim
0.0.84's high-level simulation dispatch uses its original operator, not the
approximate decomposition. Decompose the orbital gates before simulation.
No molecular outcomes or fitted sampling objective occur in this preflight.
"""
from io import BytesIO
from importlib.metadata import version
from time import process_time, perf_counter
import inspect
import numpy as np
import ffsim
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
from .paths import ROOT, OLD
from .io import load, save, write, sha256, space
from paperstudy.lucj import make_circuit, circuit_views


OPTIONS = {'method': 'L-BFGS-B', 'options': {'maxiter': 25, 'maxfun': 300}}


def materialize(op, nelec, max_layers=None):
    """Return HF + actual native orbital primitives + exact Jastrow gates."""
    circuit = QuantumCircuit(2 * op.norb)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(op.norb, nelec), circuit.qubits)
    kwargs = OPTIONS if max_layers is not None else {}
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(
        op, orb_rot_max_layers=max_layers, **kwargs), circuit.qubits)
    circuit = circuit.decompose(gates_to_decompose=['ucj_balanced_jw'])
    rotations = []
    for item in circuit:
        gate = item.operation
        if isinstance(gate, ffsim.qiskit.OrbitalRotationJW):
            # Materialize once. The simulation and resource counters share this
            # same definition, rather than independently rerunning the fitter.
            definition = gate.definition
            rotations.append((gate, definition.copy()))
    actual = circuit.decompose(gates_to_decompose=['orb_rot_jw'])
    if any(isinstance(i.operation, (ffsim.qiskit.OrbitalRotationJW,
            ffsim.qiskit.UCJOpSpinBalancedJW)) for i in actual):
        raise RuntimeError('Approximation-bearing high-level gate not materialized')
    return actual, rotations


def single_particle_matrix(circuit, norb):
    """Exact alpha 1-electron action, without a 2**n unitary allocation."""
    matrix = np.eye(norb, dtype=complex)
    for item in circuit:
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if all(q >= norb for q in wires):
            continue
        if any(q >= norb for q in wires):
            raise ValueError('Not a spin-separable orbital rotation')
        gate = item.operation
        if gate.name == 'p':
            matrix[wires[0]] *= np.exp(1j * float(gate.params[0]))
        elif gate.name == 'xx_plus_yy' and len(wires) == 2:
            # Little-endian 01 means an electron on the first supplied qubit.
            block = Operator(gate).data[np.ix_([1, 2], [1, 2])]
            matrix[wires] = block @ matrix[wires]
        else:
            raise ValueError(f'Unqualified orbital primitive {gate.name}')
    return matrix


def provenance():
    import ffsim.linalg.givens as givens
    import ffsim.qiskit.sim as simulation
    return {'ffsim': version('ffsim'), 'qiskit': version('qiskit'),
        'scipy': version('scipy'), 'jax': version('jax'),
        'installed_givens_source_SHA256': sha256(inspect.getsource(givens).encode()),
        'installed_simulator_source_SHA256': sha256(inspect.getsource(simulation).encode()),
        'common_resource_adapter_SHA256': sha256((OLD/'paperstudy/lucj.py').read_bytes()),
        'adapter_SHA256': sha256((ROOT/'frontierloop/native_compression.py').read_bytes())}


def binding():
    cases = load(ROOT/'protocol-B4.json')['cases']
    sources = {}
    for case in cases:
        directory = OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1'
        sources[case] = {str(p): sha256(p.read_bytes()) for p in
            (directory/'operator.npz', directory/'construction.json',
             OLD/'data/physical'/case/'problem.json')}
    return {'protocol_SHA256': sha256((ROOT/'protocol-D0-NATIVE-RESOURCE.json').read_bytes()),
        'dependencies': provenance(), 'sources': sources,
        'qualification_SHA256': sha256((ROOT/'qualification-native-compression-tests.xml').read_bytes()),
        'new_model_shots': 0, 'molecular_energy_evaluations': 0}


def run():
    if load(ROOT/'freeze-D0-NATIVE-RESOURCE.json') != binding():
        raise RuntimeError('D0 frozen preflight drift')
    spec = load(ROOT/'protocol-D0-NATIVE-RESOURCE.json')
    directory = ROOT/'runs/D0-NATIVE-RESOURCE'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'started.json').exists():
        raise RuntimeError('Preserve partial D0; no blind retry')
    space(); save(directory/'started.json', {'freeze': binding()})
    rows = []
    for case in spec['cases']:
        source = OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1'
        meta = load(OLD/'data/physical'/case/'problem.json')
        with np.load(source/'operator.npz', allow_pickle=False) as a:
            op = ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'], a['orbital_rotations'], a['final_orbital_rotation'])
        nelec = meta['nalpha'], meta['nbeta']
        tick, cpu = perf_counter(), process_time()
        _, _, optimized_exact = circuit_views(make_circuit(op, op.norb, nelec))
        for fraction in spec['layer_fractions']:
            space(); limit = max(1, int(np.ceil(fraction * op.norb)))
            trial_cpu, trial_wall = process_time(), perf_counter()
            circuit, rotations = materialize(op, nelec, limit)
            errors = []
            for gate, definition in rotations:
                target = gate.orbital_rotation_a
                effective = single_particle_matrix(definition, op.norb)
                errors.append(float(max(0, 1 - abs(np.vdot(target, effective) / op.norb)**2)))
                if not np.allclose(effective.conj().T @ effective, np.eye(op.norb), atol=1e-10):
                    raise RuntimeError('Compressed primitive circuit lost unitarity')
            preparation, sampling, resources = circuit_views(circuit)
            target_dir = directory/case/str(limit); target_dir.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for name, qc in [('native-materialized', circuit), ('preparation', preparation), ('sampling', sampling)]:
                stream = BytesIO(); qpy.dump(qc, stream); payload = stream.getvalue()
                write(target_dir/(name+'.qpy'), payload); artifacts[name] = sha256(payload)
            row = {'case': case, 'problem_id': meta['problem_id'], 'fraction': fraction,
                'max_layers_per_orbital': limit, 'native_approximation': True,
                'resources': resources, 'optimized_exact_source_resources': optimized_exact,
                'orbital_single_particle_HS_infidelities': errors,
                'same_circuit_saved_for_future_simulation': True,
                'compiled_numeric_gate_coefficient_slots': sum(len(i.operation.params) for i in circuit),
                'nonredundant_independent_parameters': None,
                'coefficient_slots_not_a_rank_certificate': True,
                'fit_and_compilation_CPU_seconds': process_time()-trial_cpu,
                'fit_and_compilation_wall_seconds': perf_counter()-trial_wall,
                'new_model_shots': 0, 'energy_Ha': None, 'recovery_pct': None,
                'whole_molecular_state_fidelity': None, 'QPY_SHA256': artifacts,
                'not_an_original_paper_performance_reproduction': True}
            save(target_dir/'result.json', row); rows.append(row)
            print(case, limit, resources['Z_sampling_equivalent'], flush=True)
        save(directory/case/'terminal.json', {'rows': 3, 'CPU_seconds': process_time()-cpu,
            'wall_seconds': perf_counter()-tick, 'new_model_shots': 0})
    save(directory/'result.json', {'status': 'RESOURCE_PREFLIGHT_COMPLETE_NOT_PERFORMANCE',
        'freeze': binding(), 'rows': rows, 'new_model_shots': 0, 'molecular_energy_evaluations': 0,
        'goal_complete': False})


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); args = p.parse_args()
    if args.freeze:
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(ROOT/'qualification-native-compression-tests.xml').getroot().iter('testsuite'))
        if sum(int(s.get('tests', '0')) for s in suites) != 5 or any(
            int(s.get(k, '0')) for s in suites for k in ('failures', 'errors')):
            raise RuntimeError('D0 primitive simulation qualification not passed')
        path = ROOT/'freeze-D0-NATIVE-RESOURCE.json'; value = binding()
        if path.exists() and load(path) != value: raise RuntimeError('Never replace a D0 freeze')
        save(path, value)
    else:
        run()
