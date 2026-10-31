"""Exact, audit-friendly serialization of materialized number-preserving gates.

QPY is the circuit evidence. This plan is only an execution representation of
that SAME circuit, not an approximation or a different UCJ parameterization.
"""
import numpy as np
import ffsim
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator


def expand_for_execution(circuit):
    """QPY restores custom gates as generic Instruction plus a definition.

    Execute the stored definition rather than relying on the Python subclass.
    Expanding exact diagonal gates changes no state or sampling probability.
    """
    result=circuit.copy()
    for _ in range(3):
        if not any(i.operation.name=='diag_coulomb_jw' for i in result):return result
        result=result.decompose(gates_to_decompose=['diag_coulomb_jw'])
    raise RuntimeError('Unresolved exact diagonal circuit definition')


def plan(circuit, norb, nelec):
    if circuit.num_qubits != 2*norb or norb > 31:
        raise ValueError('Registered blocked spin layout requires n <= 31')
    circuit=expand_for_execution(circuit)
    dag=circuit_to_dag(circuit);occupied=[[],[]]
    for node in list(dag.front_layer()):
        if node.op.name=='x':
            wire=dag.find_bit(node.qargs[0]).index
            occupied[wire//norb].append(wire%norb);dag.remove_op_node(node)
    if tuple(map(len,occupied)) != tuple(nelec):
        raise RuntimeError('Initial particle sector differs from problem')
    remaining=dag_to_circuit(dag);ops=[];diagonal=[]
    pending=np.zeros((3,norb,norb),float)
    def flush():
        nonlocal pending
        if np.any(pending):
            diagonal.append(pending.copy());ops.append({'kind':'jastrow','index':len(diagonal)-1,'time':-1.})
            pending=np.zeros_like(pending)
    for item in remaining:
        gate=item.operation;w=[remaining.find_bit(q).index for q in item.qubits]
        if gate.name in ('global_phase','id','barrier'):
            continue
        if gate.name=='xx_plus_yy':
            flush()
            if len(w)!=2 or w[0]//norb!=w[1]//norb or abs(w[0]-w[1])!=1:
                raise ValueError('Only adjacent same-spin XX+YY qualified')
            block=Operator(gate).data[np.ix_([1,2],[1,2])]
            c=float(block[0,0].real);s=block[0,1].conjugate()
            if not np.allclose(block,[[c,s.conjugate()],[-s,c]],atol=1e-12):
                raise RuntimeError('XX+YY convention mismatch')
            ops.append({'kind':'givens','spin':w[0]//norb,'i':w[0]%norb,'j':w[1]%norb,
                'c':c,'sr':float(s.real),'si':float(s.imag)})
        elif gate.name=='p':
            spin=w[0]//norb;i=w[0]%norb
            pending[0 if spin==0 else 2,i,i]+=2*float(gate.params[0])
        elif gate.name=='cp':
            theta=float(gate.params[0]);spins=[i//norb for i in w];i,j=[x%norb for x in w]
            if spins[0]==spins[1]:
                k=0 if spins[0]==0 else 2
                pending[k,i,j]+=theta;pending[k,j,i]+=theta
            elif spins==[0,1]:pending[1,i,j]+=theta
            else:pending[1,j,i]+=theta
        else:
            raise ValueError('Unsupported materialized primitive: '+gate.name)
    flush()
    return {'norb':norb,'nelec':list(nelec),'occupied':occupied,'ops':ops,
        'global_phase_ignored_only':True,'initial_X_only':True}, {'diagonal_matrices':np.array(diagonal)}


def strings(n,electrons):
    from itertools import combinations
    return np.array(sorted(sum(1<<i for i in t) for t in combinations(range(n),electrons)),dtype=np.uint32)


def pairs(values,i,j):
    first=np.flatnonzero(((values>>i)&1) & (1-((values>>j)&1)))
    second=np.searchsorted(values,values[first]^np.uint32((1<<i)|(1<<j)))
    if not np.array_equal(values[second],values[first]^np.uint32((1<<i)|(1<<j))):
        raise RuntimeError('Incomplete number-sector pairing')
    return first,second


def numpy_state(config, arrays):
    n=config['norb'];sa,sb=[strings(n,e) for e in config['nelec']]
    if len(sa)*len(sb)>1_000_000: raise ValueError('Bounded CPU primitive emulator only')
    state=np.zeros((len(sa),len(sb)),complex)
    ia,ib=[sum(1<<i for i in occupied) for occupied in config['occupied']]
    state[np.searchsorted(sa,ia),np.searchsorted(sb,ib)]=1.
    wa,wb=[((x[:,None]>>np.arange(n))&1).astype(float) for x in (sa,sb)]
    for op in config['ops']:
        if op['kind']=='givens':
            spin=op['spin'];a,b=pairs((sa,sb)[spin],op['i'],op['j'])
            c=op['c'];s=complex(op['sr'],op['si'])
            x=state[a,:].copy() if spin==0 else state[:,a].copy()
            y=state[b,:].copy() if spin==0 else state[:,b].copy()
            if spin==0:state[a,:],state[b,:]=c*x+s.conjugate()*y,-s*x+c*y
            else:state[:,a],state[:,b]=c*x+s.conjugate()*y,-s*x+c*y
        elif op['kind']=='phase':
            p=np.exp(1j*op['theta']*(wa if op['spin']==0 else wb)[:,op['i']])
            state*=p[:,None] if op['spin']==0 else p[None,:]
        elif op['kind']=='jastrow':
            aa,ab,bb=arrays['diagonal_matrices'][op['index']]
            e=.5*np.einsum('bi,ij,bj->b',wa,aa,wa)[:,None]
            e=e+.5*np.einsum('bi,ij,bj->b',wb,bb,wb)[None,:]+wa@ab@wb.T
            state*=np.exp(-1j*op['time']*e)
        else:raise ValueError('Unknown primitive')
    return state.ravel()
