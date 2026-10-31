"""N16 prerequisite: explicit same-spin *qubit* XX+YY, including nonadjacent.

Old executors remain immutable. This is not a nonlocal fermionic hopping
operator: there is no implicit Jordan-Wigner string. Exact full-Fock fixtures
must qualify this distinction before a separate CUDA qualification/production.
"""
import types
from .paths import ROOT


def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Pinned primitive context drift')
    return text.replace(before,after)


def generated_plan():
    text=(ROOT/'frontierloop/primitive_plan.py').read_text()
    text=replace(text,"i.operation.name=='diag_coulomb_jw'","i.operation.name in ('diag_coulomb_jw','orb_rot_jw','slater_jw')")
    text=replace(text,"gates_to_decompose=['diag_coulomb_jw']","gates_to_decompose=['diag_coulomb_jw','orb_rot_jw','slater_jw']")
    text=replace(text," or abs(w[0]-w[1])!=1",'')
    text=replace(text,'Only adjacent same-spin XX+YY qualified','Only same-spin qubit XX+YY qualified by successor')
    text=replace(text,"elif gate.name=='cp':\n            theta=float(gate.params[0]);", "elif gate.name in ('cp','cz'):\n            theta=float(gate.params[0]) if gate.name=='cp' else float(np.pi);")
    return text


def generated_worker():
    text=(ROOT/'frontierloop/primitive_gpu_worker.py').read_text()
    text=replace(text,"if abs(i-j)!=1 or spin not in (0,1):","if i==j or min(i,j)<0 or max(i,j)>=n or spin not in (0,1):")
    text=replace(text,"'independent_ffsim_parity':parity", "'independent_state_reference_parity':parity,'primitive_semantics':'HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP'")
    return text


def plan_module():
    module=types.ModuleType('qualified_qubit_primitive_candidate')
    exec(compile(generated_plan(),'<qubit-primitive-successor>','exec'),module.__dict__)
    return module


def fixtures():
    import numpy as np
    import ffsim
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from .slater_prefix_compression import parse_prefix
    from .slater_prefix_rounding import logical_candidate,lower_virtual_swaps
    out=[]
    for n,e in [(4,2),(6,5)]:
        q=QuantumCircuit(2*n)
        for spin in range(2):
            for i in range(e):q.x(spin*n+i)
        q.append(XXPlusYYGate(.49,.37),[0,n-1]);q.append(XXPlusYYGate(-.61,.23),[n,2*n-1])
        q.cz(n-1,2*n-1);q.cp(.31,1,n+1);q.p(.19,0)
        out.append((f'nonadjacent-{n}-{e}',q,n,(e,e)))
    n=4;u=ffsim.random.random_unitary(n,seed=918)
    prep=ffsim.qiskit.PrepareSlaterDeterminantJW(n,(range(2),range(2)),(u,u));p=parse_prefix(prep.definition,n)
    source=QuantumCircuit(2*n);source.append(prep,range(2*n));source.cp(.73,0,n);source.append(ffsim.qiskit.OrbitalRotationJW(n,(u,u)),range(2*n))
    actions={p.pairs[0][0]:float(np.pi),p.pairs[0][1]:float(np.pi)}
    logical=logical_candidate(source,p,actions);actual,mapping=lower_virtual_swaps(logical,len(p.circuit),actions)
    out.append(('rounded-virtual-swap',actual,n,(2,2)))
    return out
