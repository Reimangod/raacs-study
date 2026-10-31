"""N36 TKET adapter and bounded independent zero-input qualification."""
import argparse
import json
import resource
from importlib.metadata import version
from time import process_time, perf_counter
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator, Statevector
from pytket import Circuit
from pytket.circuit import OpType
from pytket.passes import FullPeepholeOptimise, SimplifyInitial
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
from .io import load, save, sha256, space
from .tket_runtime_preflight import RUN, RUNTIME

MODES = ('UNITARY', 'ZERO_INPUT')


def compile_tket(qc, mode):
    if mode not in MODES:
        raise ValueError('Unknown compiler mode')
    if qc.num_clbits or any(x.operation.name in ('measure', 'reset', 'initialize') for x in qc.data):
        raise ValueError('Only explicit unitary circuits, no reset/measurement accepted')
    tk = circuit_from_qasm_str(qasm2.dumps(qc))
    passes = []
    if mode == 'ZERO_INPUT':
        p = SimplifyInitial(allow_classical=False, create_all_qubits=True)
        p.apply(tk); passes.append(p.to_dict())
    p = FullPeepholeOptimise(allow_swaps=False, target_2qb_gate=OpType.CX)
    p.apply(tk); passes.append(p.to_dict())
    if mode == 'ZERO_INPUT':
        p = SimplifyInitial(allow_classical=False, create_all_qubits=True)
        p.apply(tk); passes.append(p.to_dict())
    implicit = tk.implicit_qubit_permutation()
    tk.replace_implicit_wire_swaps()
    # The exporter does not emit resets for created_qubits in pytket 2.18.5.
    # Preserve the input declaration in evidence; never strip operations/reset.
    declared = tk.to_dict()['created_qubits']
    output_qasm = circuit_to_qasm_str(tk)
    result = qasm2.loads(output_qasm)
    if result.num_qubits != qc.num_qubits or result.num_clbits or any(
            x.operation.name in ('reset', 'measure', 'initialize') for x in result.data):
        raise ValueError('QASM boundary changed width or introduced a nonunitary operation')
    return result, {'passes': passes, 'created_qubits': declared,
                    'implicit_permutation_before_explicit_swaps': [[str(k),str(v)] for k,v in implicit.items()],
                    'implicit_swaps_materialized': True, 'output_qasm_SHA256': sha256(output_qasm.encode()),
                    'input_global_phase_ignored_only_for_global_equivalence': float(qc.global_phase),
                    'global_unitary_preservation_claimed': mode == 'UNITARY',
                    'zero_state_preservation_claimed': True}


def state_error(a, b):
    overlap = np.vdot(a, b)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    return float(np.max(np.abs(b - phase*a)))


def fixtures():
    rows = []
    c = QuantumCircuit(3); c.cx(0,1); c.h(2); c.cx(2,1)
    rows.append(('zero_control', c))
    c = QuantumCircuit(3); c.x(0); c.cx(0,1); c.h(2); c.cx(2,1)
    rows.append(('one_control', c))
    c = QuantumCircuit(3); c.h(0); c.cx(0,1); c.ry(.31,2); c.cx(1,2)
    rows.append(('unknown_control_must_remain', c))
    c = QuantumCircuit(5); c.x(4); c.h(1); c.swap(4,0); c.cx(1,3); c.rz(.77,3)
    c.global_phase = .217
    rows.append(('asymmetric_wire_order_phase_swap', c))
    for n in (2,3,4,5):
        rng = np.random.default_rng(36000+n)
        c = QuantumCircuit(n)
        for _ in range(12):
            a,b = rng.choice(n,2,replace=False)
            c.rx(float(rng.normal()),int(a)); c.ry(float(rng.normal()),int(b))
            c.cx(int(a),int(b)); c.rz(float(rng.normal()),int(b))
        rows.append((f'complex_random_{n}q', c))
    return rows


def binding():
    installation = load(RUN/'installation-receipt.json')
    for rel, expected in installation['installed_files_SHA256'].items():
        if sha256((RUNTIME/rel).read_bytes()) != expected:
            raise ValueError('Private compiler runtime drift: '+rel)
    return {'implementation_SHA256': sha256(Path(__file__).read_bytes()),
            'installation_receipt_SHA256': sha256((RUN/'installation-receipt.json').read_bytes()),
            'versions': {k: version(k) for k in ('pytket','qiskit','numpy','scipy','sympy','networkx')},
            'fixtures': [n for n,_ in fixtures()], 'modes': list(MODES),
            'tolerance': 1e-8, 'max_CPU_seconds': 600, 'max_runtime_bytes': 32*2**20,
            'max_new_run_bytes': 4*2**20, 'new_model_shots': 0, 'new_CI': 0,
            'scope': 'SMALL_FIXTURES_ONLY_NOT_PCOAST_MQT_REPRODUCTION_OR_MOLECULAR_ADVANTAGE'}


def run():
    if (RUN/'qualification-terminal.json').exists() or (RUN/'qualification-started.json').exists():
        raise RuntimeError('Do not restart the qualification run')
    expected = load(RUN/'qualification-freeze.json')
    if binding() != expected:
        raise ValueError('N36 qualification freeze drift')
    space(); cpu,wall=process_time(),perf_counter()
    save(RUN/'qualification-started.json', {'freeze_SHA256': sha256((RUN/'qualification-freeze.json').read_bytes())})
    rows = []; failure = None
    try:
        for name, qc in fixtures():
            state = Statevector(qc).data
            for mode in MODES:
                out, evidence = compile_tket(qc, mode)
                err = state_error(state,Statevector(out).data)
                prob = float(np.sum(np.abs(np.abs(state)**2-np.abs(Statevector(out).data)**2)))
                unitary_err = state_error(Operator(qc).data.reshape(-1), Operator(out).data.reshape(-1)) if mode=='UNITARY' else None
                if err > expected['tolerance'] or prob > expected['tolerance'] or (unitary_err is not None and unitary_err > expected['tolerance']):
                    raise ValueError((name,mode,err,prob,unitary_err))
                rows.append({'fixture':name,'mode':mode,'state_error_up_to_global_phase':err,
                             'probability_L1':prob,'all_input_unitary_error_up_to_global_phase':unitary_err,
                             'input_CX':int(qc.count_ops().get('cx',0)),
                             'output_CX':int(out.count_ops().get('cx',0)),'evidence':evidence})
        rejects = []
        for op in ('reset','measure','initialize'):
            q = QuantumCircuit(2,1) if op=='measure' else QuantumCircuit(2)
            if op=='measure':q.measure(0,0)
            elif op=='reset':q.reset(0)
            else:q.initialize([1,0],0)
            try:
                compile_tket(q,'ZERO_INPUT')
            except ValueError:
                rejects.append(op)
            else:
                raise AssertionError('Nonunitary input was silently accepted')
        status = 'PASSED_16_INDEPENDENT_SMALL_COMPARISONS_3_NONUNITARY_REJECTIONS'
    except Exception as exc:
        import traceback
        failure = {'error':repr(exc),'traceback':traceback.format_exc()}
        rejects = []; status = 'FAILED_ENGINEERING_PRESERVED'
    terminal = {'status':status,'rows':rows,'expected_rejections':rejects,'failure':failure,
                'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-wall,
                'new_model_shots':0,'new_QPU_shots':0,'new_CI':0,'new_energy_evaluations':0}
    save(RUN/'qualification-terminal.json',terminal)
    print(json.dumps(terminal))
    if failure:raise RuntimeError('Qualification failed; preserve evidence, use additive successor')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    if args.action=='freeze':
        if (RUN/'qualification-freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'qualification-freeze.json',binding());print('N36 small qualifier frozen')
    else:
        resource.setrlimit(resource.RLIMIT_CPU,(600,600));run()
