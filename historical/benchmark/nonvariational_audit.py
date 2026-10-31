"""B3 qualification: stored Hamiltonian entries, states, actual circuit parity."""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, rng_for, save_json
from .chemistry import load_reference
from .time_evolution_primitives import apply_sequence, qdrift_sequence, trotter_sequence


def named_pauli_entries(terms, constant, ids):
    ids = list(map(int, ids)); index = {x: i for i, x in enumerate(ids)}
    matrix = np.eye(len(ids), dtype=complex)*constant
    for label, coefficient in terms:
        n = len(label)
        flip = sum(1 << (n-1-q) for q, p in enumerate(label) if p in "XY")
        mask = sum(1 << (n-1-q) for q, p in enumerate(label) if p in "YZ")
        for col, x in enumerate(ids):
            y = x ^ flip
            if y in index:
                matrix[index[y], col] += coefficient*(1j**label.count("Y"))*(-1 if (x & mask).bit_count() % 2 else 1)
    return matrix


def audit(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text())
    for name, sha in mf["implementation"].items():
        assert file_digest(ROOT / name) == sha, name
    n_cells = physical = matrix_entries = 0; failures = []
    with threadpool_limits(limits=1):
        for case in mf["case_ids"]:
            ref = load_reference(Path(mf["source"]) / "references" / case); meta = ref["meta"]
            setup = json.loads((root / "case-setup" / (case+".json")).read_text())
            terms = setup["terms"]
            rng = np.random.default_rng(7143)
            ids = np.unique(np.r_[rng.integers(0, 1 << meta["n_qubits"], size=48), ref["ids"][:16]])
            matrix = named_pauli_entries(terms, setup["omitted_identity_global_phase"], ids)
            assert np.allclose(matrix, ref["h"][ids][:, ids].toarray(), atol=1e-9)
            matrix_entries += len(ids)**2
            for spec in mf["methods"]:
                d = root / "cells" / case / spec["name"]
                row = json.loads((d / "result.json").read_text()); n_cells += 1
                if row["status"] == "FAILED_PRESERVED":
                    failures.append({"case_id": case, "method": spec["name"], "error": row["error"]}); continue
                assert file_digest(d / "final.npz") == row["final_sha256"]
                with np.load(d / "final.npz") as z:
                    state, p = z["state"], z["probability"]
                assert abs(np.linalg.norm(state)-1) < 1e-9 and np.allclose(abs(state)**2, p, atol=1e-12)
                assert abs(np.vdot(state, ref["h"] @ state).real-row["input_state_energy"]) < 1e-9
                assert not row["FCI_used_for_selection"] and row["n_optimized_variables"] == 0
                if spec["family"] != "orbital":
                    rng, _ = rng_for("B3-nonvariational-circuit-v1", case, spec)
                    if spec["family"] == "TE":
                        sequence = trotter_sequence(terms, spec["time"], spec["steps"])
                    else:
                        sequence, indices, norm = qdrift_sequence(terms, spec["time"], spec["steps"], rng)
                        assert indices == row["details"]["sampled_term_indices"]
                    assert sequence == [tuple(x) for x in row["details"]["sequence"]]
                # Production checks every circuit; fresh audit uses all quick or
                # one per family per case, independently decoded from saved QPY.
                check = mf["mode"] == "quick" or spec["name"] in ("TE_t1_steps1", "Pauli_qDRIFT_t1_steps64_s101", "orbital_brickwork_layers2_s101")
                if check:
                    prep = qpy.load(BytesIO((d / "preparation.qpy").read_bytes()))[0]
                    sampler = qpy.load(BytesIO((d / "sampling.qpy").read_bytes()))[0]
                    assert 1-abs(np.vdot(Statevector.from_instruction(prep).data, state))**2 < 1e-8
                    assert np.allclose(abs(Statevector.from_instruction(sampler).data)**2, p, atol=1e-9)
                    assert sampler.count_ops().get("cx", 0) == row["resources"]["cnot_count"]
                    physical += 1
    result = {"status": "UNEXPECTED_FAILURES_PRESERVED" if failures else ("QUICK_NONVARIATIONAL_CIRCUITS_VALIDATED" if mf["mode"] == "quick" else "STANDARD_NONVARIATIONAL_CIRCUITS_VALIDATED"),
              "cells": n_cells, "failures": failures, "independent_named_Hamiltonian_entry_checks": matrix_entries,
              "independent_saved_QPY_circuit_checks": physical, "whole_study_complete": False,
              "scope": "all stored states/energies and saved distributions; all quick or3family/case fresh circuit parity; production parity everycell",
              "SqDRIFT_implemented_by_this_cohort": False, "hardware_shots": 0}
    save_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    result = audit(p.parse_args().output); print(json.dumps(result))
    if result["failures"]:
        raise SystemExit(1)
