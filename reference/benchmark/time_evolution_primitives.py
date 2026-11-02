"""Direct JW Pauli time evolution primitives for small-system qualification.

Pauli-qDRIFT is explicitly not the grouped-fermionic SqDRIFT implementation.
No FCI information is accepted by Hamiltonian construction or circuit selection.
"""
from __future__ import annotations
import numpy as np
from numba import njit
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from .integral_hamiltonian import IntegralHamiltonian


def pauli_hamiltonian(h1, eri, hf_id, hf_energy):
    backend = IntegralHamiltonian.from_saved_integrals(h1, eri, hf_id, hf_energy)
    one, two = spinorb_from_spatial(h1, eri.transpose(0, 2, 3, 1))
    jw = of.jordan_wigner(of.get_fermion_operator(of.InteractionOperator(backend.core, one, .5*two)))
    n = 2*len(h1); rows = []; constant = 0.
    for term, value in sorted(jw.terms.items()):
        if abs(value.imag) > 1e-10:
            raise ValueError("Non-real Pauli Hamiltonian coefficient")
        if not term:
            constant += value.real; continue
        if abs(value.real) < 1e-12:
            continue
        label = ["I"]*n
        for q, p in term:
            label[q] = p
        rows.append(("".join(label), float(value.real)))
    return rows, float(constant)


@njit(cache=True)
def _rotation(state, flip, phase_mask, ny, angle):
    output = np.cos(angle)*state
    factor = -1j*np.sin(angle)*(1j**ny)
    for x in range(len(state)):
        z = x & phase_mask; parity = 0
        while z:
            parity ^= 1; z &= z-1
        output[x ^ flip] += factor*(-1 if parity else 1)*state[x]
    return output


def apply_sequence(state, sequence):
    result = np.array(state, dtype=complex, copy=True)
    for label, angle in sequence:
        n = len(label)
        flip = sum(1 << (n-1-q) for q, p in enumerate(label) if p in "XY")
        phase = sum(1 << (n-1-q) for q, p in enumerate(label) if p in "YZ")
        result = _rotation(result, flip, phase, label.count("Y"), float(angle))
    if abs(np.linalg.norm(result)-1) > 1e-9:
        raise RuntimeError("Pauli unitary norm failed")
    return result


def evolution_circuit(n, hf_id, sequence):
    qc = QuantumCircuit(n)
    for wire in range(n):
        if hf_id >> wire & 1:
            qc.x(wire)
    for label, angle in sequence:
        qc.append(PauliEvolutionGate(SparsePauliOp(label), time=float(angle)), range(n))
    return qc


def trotter_sequence(terms, time, steps):
    if steps < 1:
        raise ValueError("Positive Trotter steps required")
    return [(label, coefficient*time/steps) for _ in range(steps) for label, coefficient in terms]


def qdrift_sequence(terms, time, steps, rng):
    coefficients = np.array([x[1] for x in terms])
    weights = abs(coefficients); norm = float(weights.sum())
    if norm <= 0 or steps < 1:
        raise ValueError("Positive qDRIFT norm / steps required")
    indices = rng.choice(len(terms), size=steps, p=weights/norm)
    return [(terms[i][0], float(np.sign(coefficients[i])*norm*time/steps)) for i in indices], indices.tolist(), norm
