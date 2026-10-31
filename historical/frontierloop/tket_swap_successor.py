"""Additive N36-v2: narrow, exact SWAP grammar repair; old adapter immutable."""
import json
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from qiskit import QuantumCircuit,qasm2
from qiskit.circuit.library import SwapGate
from qiskit.quantum_info import Statevector,Operator
from pytket import Circuit
from pytket.circuit import OpType
from pytket.passes import FullPeepholeOptimise,SimplifyInitial
from pytket.qasm import circuit_from_qasm_str,circuit_to_qasm_str
from .tket_qualification import fixtures,state_error,binding as old_binding,MODES
from .tket_runtime_preflight import RUN as OLDQUALIFY
from .paths import ROOT
from .io import load,save,sha256,space

RUN=ROOT/'runs/N36-TKET-SWAP-QUALIFICATION-v2'


def parse_export(text):
    # qiskit's strict qelib1 intentionally omits nonstandard extensions.
    # Register ONLY the precisely named, zero-angle, 2-qubit SWAP gate.
    return qasm2.loads(text,custom_instructions=[qasm2.CustomInstruction('swap',0,2,SwapGate,builtin=True)])


def compile_tket(qc,mode):
    if mode not in MODES:raise ValueError('Unknown compiler mode')
    if qc.num_clbits or any(x.operation.name in ('reset','measure','initialize') for x in qc.data):
        raise ValueError('Only explicit unitary input accepted')
    tk=circuit_from_qasm_str(qasm2.dumps(qc));passes=[]
    if mode=='ZERO_INPUT':
        p=SimplifyInitial(allow_classical=False,create_all_qubits=True);p.apply(tk);passes.append(p.to_dict())
    p=FullPeepholeOptimise(allow_swaps=False,target_2qb_gate=OpType.CX);p.apply(tk);passes.append(p.to_dict())
    if mode=='ZERO_INPUT':
        p=SimplifyInitial(allow_classical=False,create_all_qubits=True);p.apply(tk);passes.append(p.to_dict())
    implicit=tk.implicit_qubit_permutation();tk.replace_implicit_wire_swaps()
    declared=tk.to_dict()['created_qubits'];text=circuit_to_qasm_str(tk);out=parse_export(text)
    if out.num_qubits!=qc.num_qubits or out.num_clbits or any(x.operation.name in ('reset','measure','initialize') for x in out.data):
        raise ValueError('Unexpected nonunitary or width change')
    return out,{'passes':passes,'created_qubits':declared,
                'implicit_permutation_before_explicit_swaps':[[str(k),str(v)] for k,v in implicit.items()],
                'implicit_swaps_materialized':True,'output_qasm_SHA256':sha256(text.encode()),
                'qasm_repair':'EXACT_SWAP_ONLY_NO_OPERATION_REMOVAL',
                'global_unitary_preservation_claimed':mode=='UNITARY',
                'zero_state_preservation_claimed':True}


def binding():
    if old_binding()!=load(OLDQUALIFY/'qualification-freeze.json'):raise ValueError('Old qualifier drift')
    return {'implementation_SHA256':sha256(Path(__file__).read_bytes()),
            'old_qualifier_SHA256':sha256((OLDQUALIFY/'qualification-terminal.json').read_bytes()),
            'fixture_names':[n for n,_ in fixtures()]+['explicit_swap_export','implicit_permutation_export'],
            'tolerance':1e-8,'unknown_gate_policy':'REJECT','only_new_custom_instruction':'swap/0parameters/2qubits/QiskitSwapGate',
            'new_shots_CI_energy_optimizer':[0,0,0,0]}


def run():
    if (RUN/'terminal.json').exists():raise RuntimeError('No duplicate qualifier')
    if binding()!=load(RUN/'freeze.json'):raise ValueError('Qualifier v2 drift')
    space();cpu,wall=process_time(),perf_counter();rows=[]
    for name,qc in fixtures():
        for mode in MODES:
            out,_=compile_tket(qc,mode)
            err=state_error(Statevector(qc).data,Statevector(out).data)
            un=state_error(Operator(qc).data.reshape(-1),Operator(out).data.reshape(-1)) if mode=='UNITARY' else None
            if err>1e-8 or (un is not None and un>1e-8):raise AssertionError('Qualifier mismatch')
            rows.append({'fixture':name,'mode':mode,'state_error':err,'unitary_error':un})
    for implicit in (False,True):
        tk=Circuit(4).X(3).H(0).SWAP(0,2)
        expected=QuantumCircuit(4);expected.x(3);expected.h(0);expected.swap(0,2)
        if implicit:
            # Construct an explicit fixture with a known nonidentity output
            # permutation using the documented serializable Circuit interface.
            tk=Circuit(4).X(3).H(0)
            spec=tk.to_dict()
            for pair in spec['implicit_permutation']:
                if pair[1]==['q',[0]]:pair[1]=['q',[2]]
                elif pair[1]==['q',[2]]:pair[1]=['q',[0]]
            tk=Circuit.from_dict(spec)
            if all(k==v for k,v in tk.implicit_qubit_permutation().items()):raise AssertionError('No nontrivial permutation')
            tk.replace_implicit_wire_swaps()
        text=circuit_to_qasm_str(tk)
        if 'swap ' not in text:raise AssertionError('Missing actual swap grammar test')
        out=parse_export(text);err=state_error(Operator(expected).data.reshape(-1),Operator(out).data.reshape(-1))
        if err>1e-8:raise AssertionError('SWAP meaning changed')
        rows.append({'fixture':'implicit_permutation_export' if implicit else 'explicit_swap_export','unitary_error':err})
    try:parse_export('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; unknown_gate q[0],q[1];')
    except qasm2.QASM2ParseError:unknown=True
    else:raise AssertionError('Unknown grammar silently accepted')
    terminal={'status':'ALL18_SMALL_CHECKS_PASSED_UNKNOWN_GATE_REJECTED','rows':rows,'unknown_gate_rejected':unknown,
              'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-wall,
              'new_shots_CI_energy_optimizer':[0,0,0,0]}
    save(RUN/'terminal.json',terminal);print(json.dumps(terminal))


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['freeze']:
        if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'freeze.json',binding());print('N36 exact SWAP qualifier v2 frozen')
    else:run()
