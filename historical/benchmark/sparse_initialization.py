"""Exact two-configuration preparation, including actual gates, not an oracle.

This ordinary sparse/GHZ construction is an engineering alternative to a
restricted greedy Pauli initializer, not a claimed new scientific algorithm.
It is not installed into any frozen B2 trajectory in place.
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit


def prepare_two_configurations(n_qubits, ids, amplitudes):
    ids = list(map(int, ids)); amplitudes = np.asarray(amplitudes, complex)
    if len(ids) not in (1, 2) or len(ids) != len(amplitudes) or len(set(ids)) != len(ids):
        raise ValueError("Exactly one or two distinct named configurations")
    if min(ids) < 0 or max(ids) >= 1 << n_qubits:
        raise ValueError("Configuration outside register")
    if not np.isclose(np.linalg.norm(amplitudes), 1., atol=1e-12):
        raise ValueError("Unnormalized target")
    qc = QuantumCircuit(n_qubits)
    first = ids[0]
    if len(ids) == 2:
        difference = first ^ ids[1]
        changed = [wire for wire in range(n_qubits) if difference >> wire & 1]
        pivot = changed[0]
        a, b = amplitudes
        qc.ry(2*np.arctan2(abs(b), abs(a)), pivot)
        phase_a, phase_b = np.angle(a), np.angle(b)
        qc.p(float(phase_b-phase_a), pivot)
        qc.global_phase = float(phase_a)
        # Balanced broadcast: d−1 CX, ceil(log2(d)) two-qubit layers all-to-all.
        entangled = [pivot]; remaining = changed[1:]
        while remaining:
            old_controls = entangled.copy()
            for control in old_controls:
                if not remaining:
                    break
                target = remaining.pop(0)
                qc.cx(control, target); entangled.append(target)
    else:
        qc.global_phase = float(np.angle(amplitudes[0]))
    for wire in range(n_qubits):
        if first >> wire & 1:
            qc.x(wire)
    return qc
