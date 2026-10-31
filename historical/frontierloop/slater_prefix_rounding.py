"""Precision-guarded prefix rounding; known Clifford/virtual-SWAP techniques.

Only the Slater prefix changes logically. Odd-pi XX+YY is exactly SWAP times
diagonal local phases and CZ. The SWAP is propagated as a qubit wire map;
fermionic signs are NOT discarded: CZ remains. Decoder is mandatory.
"""
import numpy as np
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator
from .slater_prefix_compression import assemble

TARGETS=(0.,float(np.pi),float(-np.pi),float(2*np.pi))


def matrix_and_infidelity(prefix,actions):
    overlap=1.
    for records,occupied,target in zip(prefix.records,prefix.occupied,prefix.target):
        u=np.eye(prefix.norb,dtype=complex)
        for r in records:
            matrix=r['matrix']
            if r['index'] in actions:matrix=Operator(XXPlusYYGate(actions[r['index']],r['params'][1])).data[np.ix_([1,2],[1,2])]
            u[r['wires']]=matrix@u[r['wires']]
        overlap*=abs(np.linalg.det(target.conj().T@u[:,occupied]))**2
    if not np.isfinite(overlap) or overlap>1+1e-10:raise ValueError('Invalid overlap')
    return max(0.,float(1-overlap))


def nominal_saved_CX(target):return 2 if np.isclose(abs(target),np.pi,atol=1e-14) else 4


def trajectory(prefix,policy,max_infidelity=.01):
    if policy not in ['STATE_OVERLAP_ROUND','ANGLE_ONLY_ROUND']:raise ValueError('Unknown rounding policy')
    records={r['index']:r for r in prefix.records[0]};actions={};remaining=list(prefix.pairs)
    steps=[dict(actions=[],infidelity=matrix_and_infidelity(prefix,actions),nominal_saved_CNOT=0)];evaluations=1;saved=0
    while remaining:
        options=[];current=steps[-1]['infidelity']
        for pair in remaining:
            theta=records[pair[0]]['params'][0]
            for target in TARGETS:
                trial=dict(actions);trial.update({pair[0]:target,pair[1]:target})
                value=matrix_and_infidelity(prefix,trial);evaluations+=1
                if value>max_infidelity+1e-12:continue
                gain=nominal_saved_CX(target)
                score=(value-current)/gain if policy=='STATE_OVERLAP_ROUND' else 4*(1-np.cos((theta-target)/2))/gain
                options.append((score,value,-gain,pair,target))
        if not options:break
        _,value,negative_gain,pair,target=min(options)
        actions.update({pair[0]:target,pair[1]:target});remaining.remove(pair);saved-=negative_gain
        steps.append(dict(actions=[[i,t] for i,t in sorted(actions.items())],infidelity=value,nominal_saved_CNOT=saved))
    return dict(policy=policy,steps=steps,determinant_overlap_evaluations=evaluations,not_global_optimality=True)


def choose(trace,tier):
    good=[s for s in trace['steps'] if s['infidelity']<=tier+1e-12]
    return max(good,key=lambda s:(s['nominal_saved_CNOT'],-s['infidelity']))


def logical_candidate(source,prefix,actions):
    out=prefix.circuit.copy_empty_like()
    for i,item in enumerate(prefix.circuit):
        gate=item.operation
        if i in actions:
            if gate.name!='xx_plus_yy':raise ValueError('Only registered Givens can be rounded')
            gate=XXPlusYYGate(actions[i],float(gate.params[1]))
        out.append(gate,[prefix.circuit.find_bit(q).index for q in item.qubits])
    return assemble(source,out)


def lower_virtual_swaps(logical,prefix_length,actions):
    out=logical.copy_empty_like();mapping=list(range(logical.num_qubits))
    for index,item in enumerate(logical):
        wires=[logical.find_bit(q).index for q in item.qubits];physical=[mapping[w] for w in wires]
        if index<prefix_length and index in actions:
            theta=actions[index]
            if not any(abs(theta-t)<1e-14 for t in TARGETS):raise ValueError('Unknown rounded target')
            matrix=Operator(item.operation).data;a,b=physical
            if np.isclose(abs(theta),np.pi,atol=1e-14):
                out.p(float(np.angle(matrix[2,1])),a);out.p(float(np.angle(matrix[1,2])),b);out.cz(a,b)
                mapping[wires[0]],mapping[wires[1]]=mapping[wires[1]],mapping[wires[0]]
            else:
                if abs(matrix[1,1]-1)>1e-14:out.p(float(np.angle(matrix[1,1])),a)
                if abs(matrix[2,2]-1)>1e-14:out.p(float(np.angle(matrix[2,2])),b)
        else:out.append(item.operation,physical)
    if sorted(mapping)!=list(range(logical.num_qubits)):raise RuntimeError('Invalid decoder')
    return out,mapping


def decode_full_vector(vector,mapping):
    # Independent small-fixture helper only; large states use blocked decoders.
    if len(mapping)>12:raise ValueError('Tiny full-Fock decoder only')
    index=np.arange(1<<len(mapping),dtype=np.uint64);physical=np.zeros_like(index)
    for logical,p in enumerate(mapping):physical|=((index>>logical)&1)<<p
    return np.asarray(vector)[physical]
