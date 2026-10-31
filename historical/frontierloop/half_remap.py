"""HSQD-inspired spin-collapse comparator, not the paper's full HCI workflow.

For a balanced source diag generator .5*a*Jaa*a + .5*b*Jaa*b + a*Jab*b,
retain the alpha within-spin term and map every cross term a_i*b_j -> a_i*a_j.
The resulting spinless generator is .5*a*(Jaa+2*Jab)*a. Preserve diagonal
terms n_i**2=n_i as one-body phases; do not double the retained beta block.
This remapping is not an identity for the original reduced density matrix.
"""
import numpy as np
import ffsim
from qiskit import QuantumCircuit


def operator(source):
    if len(source.diag_coulomb_mats)!=1:raise ValueError('Registered comparator is one layer')
    aa,ab=source.diag_coulomb_mats[0]
    if not np.allclose(ab,ab.T,rtol=0,atol=1e-12):raise ValueError('Symmetric cross-spin matrix required')
    return ffsim.UCJOpSpinless(np.array([aa+2*ab]),source.orbital_rotations.copy(),
        None if source.final_orbital_rotation is None else source.final_orbital_rotation.copy())


def state(source,nelec):
    if nelec[0]!=nelec[1]:raise ValueError('Balanced comparator only')
    return ffsim.apply_unitary(ffsim.hartree_fock_state(source.norb,nelec[0]),operator(source),source.norb,nelec[0])


def circuit(source,nelec):
    if nelec[0]!=nelec[1]:raise ValueError('Balanced comparator only')
    q=QuantumCircuit(source.norb)
    q.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(source.norb,nelec[0]),q.qubits)
    q.append(ffsim.qiskit.UCJOpSpinlessJW(operator(source)),q.qubits)
    return q


def qualify():
    from qiskit.quantum_info import Statevector
    from .primitive_plan import strings
    checks=[]
    for seed,na in [(9811,1),(9812,2),(9813,3)]:
        n=4;rng=np.random.default_rng(seed)
        aa=rng.normal(0,.2,(n,n));aa=(aa+aa.T)/2
        ab=rng.normal(0,.2,(n,n));ab=(ab+ab.T)/2
        u=ffsim.random.random_unitary(n,seed=seed)
        f=ffsim.random.random_unitary(n,seed=seed+1)
        source=ffsim.UCJOpSpinBalanced(np.array([[aa,ab]]),np.array([u]),f)
        # Independent occupation-space action of the remapped cross generator.
        addresses=strings(n,na);occupations=((addresses[:,None]>>np.arange(n))&1).astype(float)
        initial=ffsim.slater_determinant(n,list(range(na)),orbital_rotation=u.conj().T)
        phase=.5*np.einsum('bi,ij,bj->b',occupations,aa,occupations)+np.einsum('bi,ij,bj->b',occupations,ab,occupations)
        expected=ffsim.apply_orbital_rotation(initial*np.exp(1j*phase),f@u,n,na)
        actual=state(source,(na,na))
        full=Statevector.from_instruction(circuit(source,(na,na)).decompose(reps=5)).data[addresses]
        error=float(max(abs(expected-actual)));infidelity=float(abs(1-abs(np.vdot(full,actual))))
        if error>1e-11 or infidelity>1e-10:raise RuntimeError('Half-remapping state/circuit mismatch')
        checks.append({'seed':seed,'occupation_formula_error':error,'independent_Qiskit_infidelity':infidelity})
    return checks
