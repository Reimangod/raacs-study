"""Author-connectivity LUCJ adapter and independent small-circuit verification."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import ffsim
import numpy as np
from pyscf.ci.cisd import cisdvec_to_amplitudes
from pyscf.fci import cistring
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector

from .input_adapter import array_bytes
from .io import ROOT, atomic_bytes, digest, save_json, sha256

BASIS = ["rz", "ry", "rx", "h", "s", "sdg", "x", "cx"]


def interaction_pairs(norb):
    # Same definition as the licensed author utility and current ffsim.
    pairs = ([(p, p+1) for p in range(norb-1)], [(p, p) for p in range(norb) if p % 4 == 0])
    if pairs != ffsim.variational.util.interaction_pairs_spin_balanced("heavy-hex", norb):
        raise RuntimeError("Author/library heavy-hex definitions disagree")
    return pairs


def amplitudes_from_saved(path, norb, na):
    with np.load(path, allow_pickle=False) as archive:
        if "ccsd_t2" in archive:
            return archive["ccsd_t1"].copy(), archive["ccsd_t2"].copy(), "SAVED_CCSD_PROVENANCE_ONLY"
        if "cisd_vec" not in archive:
            raise ValueError("No CCSD or registered CISD fallback")
        c0, c1, c2 = cisdvec_to_amplitudes(archive["cisd_vec"], norb, na)
    if abs(c0) <= 1e-8:
        raise ValueError("Author CISD c0 threshold failed")
    t1 = c1/c0
    t2 = c2/c0 - np.einsum("ia,jb->ijab", t1, t1)
    return t1, t2, "SAVED_CISD_REGISTERED_FALLBACK_PROVENANCE_ONLY"


def naive(t1, t2, norb, n_reps=1):
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=n_reps,
                                                   interaction_pairs=interaction_pairs(norb), optimize=False)


def make_circuit(op, norb, nelec):
    # SDK native blocked qubit ordering: alpha0..alphaN, beta0..betaN.
    qc = QuantumCircuit(2*norb)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(2*norb))
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), range(2*norb))
    return qc


def common_resources(qc, native_pre_init=False):
    qc = qc.remove_final_measurements(inplace=False)
    if native_pre_init:
        qc = ffsim.qiskit.PRE_INIT.run(qc)
    qc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=8137)
    if qc.count_ops().get("reset", 0):
        raise ValueError("Reset in a unitary resource circuit")
    return qc, {"cnot_count": int(qc.count_ops().get("cx", 0)),
                "cnot_depth": int(qc.depth(filter_function=lambda x: x.operation.name == "cx")),
                "total_unitary_depth": int(qc.depth()), "basis": BASIS,
                "coupling": "all-to-all", "optimization_level": 1, "seed_transpiler": 8137,
                "ffsim_state_preserving_PRE_INIT": native_pre_init}


def circuit_views(qc):
    preparation, prep_resources = common_resources(qc, native_pre_init=True)
    diagonal = {"rz", "z", "s", "sdg", "t", "tdg", "p", "cp", "cz", "rzz", "id"}
    blocked = set()
    keep = []
    for item in reversed(preparation.data):
        wires = {preparation.find_bit(q).index for q in item.qubits}
        if item.operation.name in diagonal and not wires & blocked:
            continue
        keep.append(item)
        blocked |= wires
    sampling = preparation.copy_empty_like()
    for item in reversed(keep):
        sampling.append(item.operation, [sampling.qubits[preparation.find_bit(q).index] for q in item.qubits])
    sampling, sampling_resources = common_resources(sampling)
    sampling_resources["ffsim_state_preserving_PRE_INIT"] = True
    return preparation, sampling, {"preparation": prep_resources, "Z_sampling_equivalent": sampling_resources,
                                  "Z_sampling_equivalence_not_full_state_equivalence": True}


def sector_to_blocked_full(vec, norb, nelec):
    if 2*norb > 12:
        raise ValueError("Independent full-Fock check is restricted to small fixtures")
    sa = cistring.make_strings(range(norb), nelec[0])
    sb = cistring.make_strings(range(norb), nelec[1])
    result = np.zeros(1 << (2*norb), complex)
    result[(sa[:, None] | (sb[None, :] << norb)).reshape(-1)] = vec.reshape(-1)
    return result


def small_parity(norb=4, nelec=(2, 2), seed=101):
    rng = np.random.default_rng(seed)
    nocc = nelec[0]
    t1 = rng.normal(0, .02, (nocc, norb-nocc))
    t2 = rng.normal(0, .02, (nocc, nocc, norb-nocc, norb-nocc))
    t2 = (t2+t2.transpose(1, 0, 3, 2))/2
    op = naive(t1, t2, norb)
    sector = ffsim.apply_unitary(ffsim.hartree_fock_state(norb, nelec), op, norb, nelec)
    expected = sector_to_blocked_full(sector, norb, nelec)
    qc, resources = common_resources(make_circuit(op, norb, nelec))
    actual = Statevector.from_instruction(qc).data
    overlap = np.vdot(expected, actual)
    actual = actual*np.exp(-1j*np.angle(overlap))
    state_error = float(np.max(abs(actual-expected)))
    prob_error = float(np.sum(abs(abs(actual)**2-abs(expected)**2)))
    if state_error > 1e-8 or prob_error > 1e-8:
        raise RuntimeError("Independent ffsim/Qiskit state or distribution mismatch")
    return {"schema": "paper-LUCJ-small-parity-1", "norb": norb, "nelec": list(nelec), "seed": seed,
            "status": "INDEPENDENT_QISKIT_FFSIM_PARITY_PASSED", "state_max_error": state_error,
            "probability_L1_error": prob_error, "resources": resources,
            "heavy_hex_pairs": interaction_pairs(norb), "coefficient_vector_length": len(op.to_parameters(interaction_pairs=interaction_pairs(norb))),
            "nonredundant_parameter_dimension_proved": False, "actual_QPU_shots": 0}


def profile_naive(case_id="N2-32-R1p2-P2", shots=1000):
    root = ROOT / "qualification" / (case_id+"-naive")
    path = root / "result.json"
    if path.exists():
        return json.loads(path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "data/physical" / case_id / "problem.json").read_text())
    n, na, nb = meta["norb"], meta["nalpha"], meta["nbeta"]
    if n > 16:
        raise ValueError("Local qualification profile limited to at most16 spatial orbitals")
    save_json(root / "started.json", {"case_id": case_id, "shots": shots, "phase": "S1_TIMING_FIXTURE",
                                      "not_primary_scientific_run": True})
    started = perf_counter()
    t1, t2, init_kind = amplitudes_from_saved(ROOT / "data/initialization" / case_id / "amplitudes.npz", n, na)
    op = naive(t1, t2, n)
    build = perf_counter()-started
    atomic_bytes(root / "operator.npz", array_bytes(dict(diag_coulomb_mats=op.diag_coulomb_mats,
                  orbital_rotations=op.orbital_rotations, final_orbital_rotation=op.final_orbital_rotation)))
    qc, resources = common_resources(make_circuit(op, n, (na, nb)))
    stream = BytesIO(); qpy.dump(qc, stream)
    atomic_bytes(root / "preparation.qpy", stream.getvalue())
    save_json(root / "construction.json", {"resources": resources, "initialization": init_kind,
               "initialization_creation_CPU_seconds": None, "naive_factorization_seconds": build,
               "coefficient_vector_length": len(op.to_parameters(interaction_pairs=interaction_pairs(n)))})
    tick = perf_counter()
    vec = ffsim.hartree_fock_state(n, (na, nb))
    vec = ffsim.apply_unitary(vec, op, n, (na, nb), copy=False)
    simulation = perf_counter()-tick
    tick = perf_counter()
    samples = ffsim.sample_state_vector(vec, norb=n, nelec=(na, nb), shots=shots,
                                      bitstring_type=ffsim.BitstringType.INT, seed=8191)
    ids, counts = np.unique(samples, return_counts=True)
    atomic_bytes(root / "timing-counts.npz", array_bytes({"blocked_ids": ids.astype(np.uint64), "counts": counts.astype(np.int64)}))
    result = {"status": "EXACT_NUMBER_SECTOR_LUCJ_TIMING_COMPLETE", "case_id": case_id,
              "problem_id": meta["problem_id"], "phase": "S1_TIMING_FIXTURE_NOT_PRIMARY",
              "vector_norm_squared": float(np.vdot(vec, vec).real), "vector_size": int(vec.size),
              "simulation_seconds": simulation, "sampling_seconds": perf_counter()-tick,
              "factorization_seconds": build, "total_seconds": perf_counter()-started,
              "requested_model_shots": shots, "actual_model_shots": int(counts.sum()),
              "actual_QPU_shots": 0, "resources": resources, "candidate_energy_evaluated": False,
              "whole_study_complete": False, "statevector_saved": False}
    save_json(path, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    if a.profile:
        output = profile_naive()
    else:
        output = {"fixtures": [small_parity(seed=s) for s in (101, 211, 307)]}
        save_json(ROOT / "qualification/LUCJ-small-parity.json", output)
    print(json.dumps(output, indent=2))
