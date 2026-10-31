"""Unmodified author pools/decompositions; common independent ADAPT optimizer."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from functools import lru_cache
import sys
import numpy as np
from scipy.sparse.linalg import expm_multiply
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector
from .common import ROOT, digest

sys.path.insert(0, str(ROOT / "vendor/ceo"))
from adaptvqe import pools  # noqa:E402

METHODS = {"ceo_ovp": pools.OVP_CEO, "qeb": pools.QE,
           "qubit": pools.NoZPauliPool, "gsd": pools.GSD, "ceo_mvp":pools.MVP_CEO}
LOGICAL_BASIS = ["rz", "ry", "rx", "h", "s", "sdg", "x", "cx"]


@lru_cache(maxsize=10)
def make_pool(method, n):
    if method not in METHODS:
        raise ValueError("Unsupported pool: " + method)
    pool = METHODS[method](n=n)
    pool.imp_type = pools.ImplementationType.SPARSE
    return pool


def pool_catalog(pool):
    operators = []
    for i in range(pool.size):
        terms = [{"pauli": [[int(q), p] for q, p in term], "coefficient": [float(c.real), float(c.imag)]}
                 for term, c in pool.get_q_op(i).terms.items()]
        operators.append({"index": i, "terms": terms,
                          "support": [int(q) for q in pool.get_qubits(i)]})
    data = {"class": type(pool).__name__, "name": pool.name, "n_qubits": pool.n,
            "operators": operators, "normalization": "unmodified author PoolOperator.arrange"}
    return {**data, "catalog_digest": digest(data)}


def hf_state(n, hf_id):
    state = np.zeros(1 << n, dtype=complex)
    state[hf_id] = 1
    return state


def apply_op(pool, index, theta, state):
    if theta == 0:
        return state.copy()
    # Author analytic trigonometric kernels; independently checked against generic
    # expm_multiply and actual Qiskit circuits. Avoid repeated generic Krylov work.
    return np.asarray(pool.expm_mult(theta,index,state)).reshape(-1)


def state_from_parameters(pool, indices, theta, initial, keep=False):
    history = [initial]
    state = initial
    for index, angle in zip(indices, theta):
        state = apply_op(pool, index, angle, state)
        if keep:
            history.append(state)
    return history if keep else state


def energy_and_gradient(pool, indices, theta, initial, h):
    history = state_from_parameters(pool, indices, theta, initial, keep=True)
    state = history[-1]
    adjoint = h @ state
    energy = float(np.vdot(state, adjoint).real)
    grad = np.empty(len(indices))
    for j in range(len(indices)-1, -1, -1):
        grad[j] = 2*np.vdot(adjoint, pool.get_imp_op(indices[j]) @ history[j+1]).real
        adjoint = apply_op(pool, indices[j], -theta[j], adjoint)
    return energy, grad


def selection_gradients(pool, state, h):
    hstate = h @ state
    # Do not materialize/cache a 2**n sparse matrix for EVERY pool operator.
    # Only selected operators need native sparse matrices during optimization.
    source=np.flatnonzero(state != 0)
    amplitudes=state[source]
    expectations={}
    gradients=[]
    for i in range(pool.size):
        value=0j
        for term,coefficient in pool.get_q_op(i).terms.items():
            if term not in expectations:
                flip=0;phase_mask=0;ny=0
                for q,pauli in term:
                    mask=1 << (pool.n-1-q)
                    if pauli in ("X","Y"):
                        flip |= mask
                    if pauli in ("Y","Z"):
                        phase_mask |= mask
                    ny+=pauli=="Y"
                parity=np.bitwise_count(source & phase_mask)&1
                phase=np.where(parity==0,1.,-1.)*(1j**ny)
                expectations[term]=np.vdot(hstate[source ^ flip],phase*amplitudes)
            value+=coefficient*expectations[term]
        gradients.append(2*value.real)
    return np.array(gradients)


def build_circuit(pool, indices, theta, hf_id):
    qc = QuantumCircuit(pool.n)
    for wire in range(pool.n):
        if hf_id >> wire & 1:
            qc.x(wire)
    qc.compose(pool.get_circuit(indices, list(theta)), inplace=True)
    qc.remove_final_measurements(inplace=True)
    # All methods use the same all-to-all logical basis and compiler contract.
    # Barriers are not physical gates; remove them before common optimization.
    return compile_logical(qc)


def compile_logical(qc):
    from qiskit.transpiler.passes import RemoveBarriers
    from qiskit.transpiler import PassManager
    qc = PassManager(RemoveBarriers()).run(qc)
    return transpile(qc,basis_gates=LOGICAL_BASIS,optimization_level=1,seed_transpiler=8137)


def measurement_equivalent_circuit(qc):
    """Common exact Z-sampling optimization. Keep preparation and sampler distinct."""
    # Operate before logical decomposition when possible (e.g., a terminal CZ).
    diagonal = {"rz","z","s","sdg","t","tdg","p","cp","cz","rzz","id"}
    blocked=set(); keep=[]; removed=0
    for item in reversed(qc.data):
        wires={qc.find_bit(q).index for q in item.qubits}
        if item.operation.name in diagonal and not (wires & blocked):
            removed+=1
            continue
        keep.append(item)
        if item.operation.name not in diagonal and item.operation.name != "barrier":
            blocked.update(wires)
    sampler=qc.copy_empty_like()
    for item in reversed(keep):
        sampler.append(item.operation,[sampler.qubits[qc.find_bit(q).index] for q in item.qubits],[])
    return compile_logical(sampler),removed


def resources(qc, n_parameters, n_operators):
    count = qc.count_ops().get("cx", 0)
    depth2 = qc.depth(filter_function=lambda item: item.operation.num_qubits == 2)
    return {"cnot_count": int(count), "cnot_depth": int(qc.depth(filter_function=lambda x: x.operation.name == "cx")),
            "total_depth": int(qc.depth()), "two_qubit_count": sum(v for k,v in qc.count_ops().items() if k == "cx"),
            "two_qubit_depth": int(depth2), "parameter_count": int(n_parameters),
            "selected_operator_count": int(n_operators),
            "native_2q_count": None, "native_2q_depth": None,
            "native_status": "NOT_CONFIGURED_NO_HARDWARE_TARGET",
            "compiler": {"basis": LOGICAL_BASIS, "optimization_level": 1,
                         "coupling": "all-to-all", "seed": 8137}}


def verify_circuit_state(qc, state, atol=1e-8):
    independent = Statevector.from_instruction(qc).data
    overlap = np.vdot(independent, state)
    phase = overlap / abs(overlap) if abs(overlap) else 1.
    error = float(np.linalg.norm(state-phase*independent))
    if error > atol or abs(np.vdot(state,state).real - 1) > atol:
        raise RuntimeError(f"Sparse/Qiskit state mismatch: {error}")
    return error


def circuit_bytes(qc):
    buffer = BytesIO()
    qpy.dump(qc, buffer)
    return buffer.getvalue()

