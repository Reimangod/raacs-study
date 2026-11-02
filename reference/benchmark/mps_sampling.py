"""Bounded actual-circuit MPS sampling, without full-statevector allocation.

This is an explicitly approximate classical backend. Finite bond/cutoff results
are not QPU evidence or certified exact distributions; use bond convergence.
"""
from __future__ import annotations
from collections import Counter
from time import perf_counter
import numpy as np
import quimb.tensor as qtn
from qiskit.quantum_info import Operator
from .study.core import rss_peak


def construct_mps(qc, max_bond=64, cutoff=1e-12, seconds_cap=120, rss_cap=2*2**30):
    if max_bond < 1 or cutoff < 0 or qc.num_clbits or not 1 <= qc.num_qubits <= 62:
        raise ValueError("Positive bond, nonnegative cutoff, unmeasured static circuit required")
    started = perf_counter()
    state = qtn.CircuitMPS(qc.num_qubits, max_bond=max_bond, cutoff=cutoff, dtype="complex128",
                          gate_opts={"method": "svd", "renorm": False})
    applied = 0
    for item in qc.data:
        if item.operation.name == "barrier":
            continue
        if item.operation.num_qubits not in (1, 2) or item.operation.name in ("reset", "measure"):
            raise ValueError("Compile to static 1/2-qubit unitary gates first")
        if perf_counter()-started > seconds_cap or rss_peak() > rss_cap:
            raise RuntimeError("MPS construction resource cap at gate boundary")
        matrix = Operator(item.operation).data
        # Qiskit matrix arguments are little-endian; quimb tensor argument order
        # is most-significant first. Reverse arguments AND raw site numbering.
        where = tuple(qc.num_qubits-1-qc.find_bit(q).index for q in reversed(item.qubits))
        state.apply_gate_raw(matrix, where)
        applied += 1
    norm = float(np.real(state.psi.H @ state.psi))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid retained MPS norm")
    metadata = {"backend": "quimb CircuitMPS", "max_bond_cap": max_bond, "cutoff": cutoff,
                "retained_norm_squared": norm, "norm_loss_not_a_global_error_certificate": 1-norm,
                "max_observed_final_bond": int(state.psi.max_bond()), "applied_gates": applied,
                "construction_seconds": perf_counter()-started, "peak_process_rss": rss_peak(),
                "full_statevector_allocated": False, "hardware_shots": 0,
                "scope": "approximate tensor-network simulation; convergence and small exact parity required"}
    return state, metadata


def sample_mps(state, shots, seed, seconds_cap=120, rss_cap=2*2**30):
    if shots < 1:
        raise ValueError("Positive shots required")
    counts = Counter(); started = perf_counter(); status = "COMPLETED"
    # Preserve the pre-normalization norm in construction metadata, then sample
    # an explicitly normalized copy. Never claim truncation was physically free.
    psi = state.psi.copy(); psi.normalize()
    samples = psi.sample(shots, seed=seed)
    for index in range(shots):
        if index % 32 == 0 and (perf_counter()-started > seconds_cap or rss_peak() > rss_cap):
            status = "RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED"; break
        configuration, _ = next(samples)
        value = 0
        for bit in configuration:
            value = (value << 1) | int(bit)
        counts[value] += 1
    ids = np.array(sorted(counts), dtype=np.int64)
    ns = np.array([counts[int(x)] for x in ids], dtype=np.int64)
    return ids, ns, {"status": status, "planned_shots": shots, "raw_simulated_shots": int(ns.sum()), "hardware_shots": 0,
                     "sampling_seconds": perf_counter()-started, "seed": str(seed),
                     "normalization": "samples from normalized retained approximate MPS; retained norm reported separately"}
