"""N15 known-style Slater prefix pruning with an unchanged correlated suffix.

The determinant overlap is the exact many-electron overlap for the registered
HF input, not a one-particle HS proxy. Unitarity preserves this overlap through
the unchanged suffix. This does not certify a CI energy or arbitrary-input map.
"""
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


@dataclass
class Prefix:
    circuit: object
    norb: int
    occupied: tuple
    records: tuple
    pairs: tuple
    target: tuple


def parse_prefix(circuit,norb):
    occupied=[[],[]];records=[[],[]];started=[False,False]
    for index,item in enumerate(circuit):
        wires=[circuit.find_bit(q).index for q in item.qubits];gate=item.operation
        if not wires:
            if gate.name!='global_phase':raise ValueError('Unknown zero-wire operation')
            continue
        spin=0 if all(w<norb for w in wires) else 1 if all(norb<=w<2*norb for w in wires) else None
        if spin is None:raise ValueError('Prefix is not spin-separable')
        local=[w-spin*norb for w in wires]
        if gate.name=='x':
            if started[spin] or len(local)!=1 or local[0] in occupied[spin]:raise ValueError('X must prepare a fixed initial configuration')
            occupied[spin].append(local[0]);continue
        started[spin]=True
        if gate.name=='p' and len(local)==1:matrix=np.array([[np.exp(1j*float(gate.params[0]))]])
        elif gate.name=='xx_plus_yy' and len(local)==2:
            matrix=Operator(gate).data[np.ix_([1,2],[1,2])]
        else:raise ValueError('Unqualified Slater primitive '+gate.name)
        records[spin].append(dict(index=index,wires=local,matrix=matrix,name=gate.name,params=list(map(float,gate.params))))
    if len(occupied[0])!=len(occupied[1]) or not occupied[0]:raise ValueError('Registered balanced sector only')
    def signature(r):return (r['name'],r['wires'],r['params'])
    if [signature(r) for r in records[0]]!=[signature(r) for r in records[1]] or occupied[0]!=occupied[1]:raise ValueError('Registered spin-balanced pair order differs')
    pairs=tuple((a['index'],b['index']) for a,b in zip(*records) if a['name']=='xx_plus_yy')
    target=tuple(matrix_for(rs,set(),norb)[:,occ] for rs,occ in zip(records,occupied))
    return Prefix(circuit,norb,tuple(occupied),tuple(records),pairs,target)


def matrix_for(records,removed,norb):
    matrix=np.eye(norb,dtype=complex)
    for r in records:
        if r['index'] not in removed:matrix[r['wires']]=r['matrix']@matrix[r['wires']]
    return matrix


def infidelity(prefix,removed):
    overlap=1.
    for records,occupied,target in zip(prefix.records,prefix.occupied,prefix.target):
        candidate=matrix_for(records,removed,prefix.norb)[:,occupied]
        overlap*=abs(np.linalg.det(target.conj().T@candidate))**2
    if not np.isfinite(overlap) or overlap>1+1e-10:raise ValueError('Invalid determinant overlap')
    return float(max(0.,1-overlap))


def materialize(prefix,removed):
    out=prefix.circuit.copy_empty_like()
    for index,item in enumerate(prefix.circuit):
        if index not in removed:out.append(item.operation,[prefix.circuit.find_bit(q).index for q in item.qubits])
    return out


def trajectory(prefix,policy,max_infidelity=.01):
    if policy not in ['STATE_OVERLAP_GREEDY','ANGLE_ONLY']:raise ValueError('Unknown policy')
    removed=set();remaining=list(prefix.pairs);records=[dict(removed=[],infidelity=infidelity(prefix,removed))];evaluations=1
    order={r['index']:abs(np.sin(r['params'][0]/4)) for r in prefix.records[0] if r['name']=='xx_plus_yy'}
    while remaining:
        if policy=='STATE_OVERLAP_GREEDY':
            scores=[(infidelity(prefix,removed|set(pair)),pair) for pair in remaining];evaluations+=len(scores)
            value,pair=min(scores,key=lambda x:(x[0],x[1]))
        else:
            pair=min(remaining,key=lambda x:(order[x[0]],x));value=infidelity(prefix,removed|set(pair));evaluations+=1
        if value>max_infidelity+1e-12:break
        removed.update(pair);remaining.remove(pair);records.append(dict(removed=sorted(removed),infidelity=value))
    return dict(policy=policy,steps=records,determinant_overlap_evaluations=evaluations,
        stopped_at_first_no_admissible_next_step=True,not_global_optimality=True)


def choose(trace,tier):
    eligible=[r for r in trace['steps'] if r['infidelity']<=tier+1e-12]
    if not eligible:raise ValueError('Original point missing')
    return max(eligible,key=lambda r:(len(r['removed']),-r['infidelity']))


def assemble(source,prefix):
    # source is the original PRE_INIT circuit, with exactly one Slater prefix.
    if source.data[0].operation.name!='slater_jw' or prefix.num_qubits!=source.num_qubits:raise ValueError('Unexpected source prefix')
    out=source.copy_empty_like();out.compose(prefix,inplace=True)
    for item in source.data[1:]:out.append(item.operation,[source.find_bit(q).index for q in item.qubits])
    return out
