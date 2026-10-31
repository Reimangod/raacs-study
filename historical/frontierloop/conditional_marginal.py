"""Candidate direction F2: a positive one-spin mixture for ONE UCJ layer.

This is a partial-trace identity, NOT an exact half-qubit preparation of the
full entangled state, nor an assertion of novelty or a finite-shot advantage.
Only the alpha marginal is reproduced. Matched shots/CI and ensemble costs
must still be tested. No Hamiltonian, energy reference or learned data enters.
"""
from itertools import product
import numpy as np
import ffsim
from qiskit import QuantumCircuit
from .primitive_plan import strings


def components(op, nelec, max_partner_columns=6):
    if len(op.diag_coulomb_mats)!=1:
        raise ValueError('Positive Slater-partner identity here is one-layer only')
    n=op.norb;na,nb=nelec
    u=op.orbital_rotations[0];jaa,jab=op.diag_coulomb_mats[0]
    columns=np.flatnonzero(np.any(jab!=0,axis=0))
    if len(columns)>max_partner_columns:
        raise ValueError('Partner-pattern enumeration cap; no tiny entries silently dropped')
    partner_orbitals=u.conj().T[:,:nb]
    kernel=partner_orbitals@partner_orbitals.conj().T
    restricted=kernel[np.ix_(columns,columns)]
    identity=np.eye(len(columns));out=[]
    for bits in product((0,1),repeat=len(columns)):
        # DPP occupation pattern: occupied rows use K, empty rows use I-K.
        matrix=np.array([restricted[i] if bit else identity[i]-restricted[i]
            for i,bit in enumerate(bits)],complex).reshape(len(columns),len(columns))
        value=np.linalg.det(matrix) if len(columns) else 1.+0j
        if abs(value.imag)>1e-10 or value.real < -1e-10:
            raise RuntimeError('Invalid exact Slater partner-pattern probability')
        out.append({'partner_columns':columns.tolist(),'bits':list(bits),
            'probability':float(max(0,value.real)),
            'conditional_phases':jab[:,columns]@np.asarray(bits,float)})
    total=sum(x['probability'] for x in out)
    if abs(total-1)>1e-10:raise RuntimeError('Partner-pattern probabilities not normalized')
    for x in out:x['probability']/=total
    return out


def conditional_state(op,nelec,component):
    n=op.norb;na,_=nelec;u=op.orbital_rotations[0];jaa=op.diag_coulomb_mats[0,0]
    state=ffsim.slater_determinant(n,list(range(na)),orbital_rotation=u.conj().T)
    occupations=((strings(n,na)[:,None]>>np.arange(n))&1).astype(float)
    phase=.5*np.einsum('bi,ij,bj->b',occupations,jaa,occupations)
    phase+=occupations@component['conditional_phases']
    state*=np.exp(1j*phase)
    final=np.eye(n) if op.final_orbital_rotation is None else op.final_orbital_rotation
    return ffsim.apply_orbital_rotation(state,final@u,n,na,copy=False)


def circuit(op,nelec,component):
    n=op.norb;na,_=nelec;u=op.orbital_rotations[0];jaa=op.diag_coulomb_mats[0,0]
    q=QuantumCircuit(n)
    q.append(ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(n,list(range(na)),u.conj().T),q.qubits)
    q.append(ffsim.qiskit.DiagCoulombEvolutionSpinlessJW(n,jaa,-1.),q.qubits)
    for i,phase in enumerate(component['conditional_phases']):q.p(float(phase),i)
    final=np.eye(n) if op.final_orbital_rotation is None else op.final_orbital_rotation
    q.append(ffsim.qiskit.OrbitalRotationSpinlessJW(n,final@u),q.qubits)
    return q
