"""Independent native-circuit/physical-sparse fixture before molecular draws."""
import json
import numpy as np

from .input_adapter import array_bytes, hf_energy
from .io import ROOT, atomic_bytes, digest, save_json, sha256
from .sqdrift_execution import Session, prepare_member, source_request


def run():
    case = "SQDRIFT-SMALL-8Q-QUALIFICATION"
    n, na = 4, 2
    rng = np.random.default_rng(3927)
    h1 = rng.normal(0, .1, (n, n))
    h1 = (h1+h1.T)/2+np.diag(np.linspace(-1.2, -.1, n))
    L = rng.normal(0, .15, (3, n, n))
    L = (L+L.transpose(0, 2, 1))/2
    eri = np.einsum("kpq,krs->pqrs", L, L)
    hf = sum(1 << (2*n-1-2*i) for i in range(na))+sum(1 << (2*n-2-2*i) for i in range(na))
    physical = {"norb": n, "nalpha": na, "nbeta": na, "core_energy": 0., "hf_id": hf,
                "physical_arrays_sha256": {k: sha256(v.astype("<f8").tobytes()) for k, v in (("h1", h1), ("eri", eri))},
                "mapping": "Jordan-Wigner", "orbital_order": "interleaved alpha/beta, orbital0 at most-significant logical bit"}
    meta = {"schema": "paper-physical-input-1", **physical, "problem_id": digest(physical), "case_id": case,
            "n_qubits": 2*n, "hf_energy_Ha": hf_energy(h1, eri, 0., na, na), "basis": "synthetic-real-integral-fixture",
            "geometry": None, "source": {"synthetic_seed": 3927}, "qualification": {"not_a_primary_problem": True}}
    folder = ROOT/"data/physical"/case
    save_json(folder/"problem.json", meta)
    atomic_bytes(folder/"integrals.npz", array_bytes({"h1": h1, "eri": eri}))
    directory = ROOT/"native-SqDRIFT"/case
    save_json(directory/"base.json", source_request(case))
    session = Session(directory/"base.json", directory)
    results = []
    try:
        for seed in (101, 211, 307):
            spec = {"time": 1., "num_terms": 20, "seed": seed, "k": 1, "member_index": 0}
            _, result = prepare_member(case, spec, session, qualify=True)
            results.append(result)
    finally:
        session.close()
    report = {"status": "NATIVE_SQDRIFT_SPARSE_CIRCUIT_PARITY_PASSED", "fixtures": results,
              "full_Fock_verification_max_qubits": 8, "large_molecule_full_Fock_used": False,
              "model_quantum_shots": 0, "whole_study_complete": False}
    save_json(ROOT/"qualification/SqDRIFT-native-sparse.json", report)
    print(json.dumps({"status": report["status"], "parities": [x["independent_small_parity"] for x in results]}, indent=2))
    return report


if __name__ == "__main__":
    run()
