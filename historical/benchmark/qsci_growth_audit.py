"""Independent checkpoint/shot/sector/Qiskit parity audit for B2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from .common import ROOT, file_digest, rng_for, save_json
from .chemistry import load_reference
from .qsci_growth import restore_state, selected_subspace
from .adapt_qsci_native import circuit


def audit(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text()); failures = []; records = []
    for name, sha in mf["code"].items():
        if file_digest(ROOT / name) != sha:
            failures.append("implementation drift " + name)
    for case in mf["cases"]:
        ref = load_reference(Path(mf["source"]) / "references" / case)
        for d in sorted((root / "trajectories" / case).glob("*/*")):
            if (d / "failure.json").exists():
                row = json.loads((d / "failure.json").read_text())
                failures.append(str(d.relative_to(root))+":"+row["error"])
                continue
            terminal = json.loads((d / "terminal.json").read_text())
            cps = sorted(d.glob("step-*.json")); total_shots = 0
            for step, path in enumerate(cps):
                cp = json.loads(path.read_text())
                assert cp["step"] == step and cp["task_id"] == terminal["task_id"]
                with np.load(d / f"counts-{step:04d}.npz") as z:
                    ids, counts = z["ids"], z["counts"]
                assert counts.sum() == cp["Ns"] and len(ids) == len(set(ids))
                total_shots += int(counts.sum())
                n, hf = ref["meta"]["n_qubits"], ref["meta"]["hf_id"]
                state = restore_state(n, hf, cp["rotations"])
                independent = Statevector.from_instruction(circuit(n, hf, cp["rotations"])).data
                assert abs(abs(np.vdot(state, independent))-1.) < 1e-8
                p = abs(independent)**2
                rng, seed = rng_for("B2-native-growth-draw-v1", cp["task_id"], step)
                simulated = rng.multinomial(cp["Ns"], p/p.sum())
                # Tiny compiler roundoff can change RNG categorical boundaries;
                # exact-count parity uses reconstructed native-state probabilities.
                rng, _ = rng_for("B2-native-growth-draw-v1", cp["task_id"], step)
                p_native = abs(state)**2
                replay = rng.multinomial(cp["Ns"], p_native/p_native.sum())
                assert np.array_equal(replay[ids], counts) and np.count_nonzero(replay) == len(ids)
                S = selected_subspace(ids, counts, ref["ids"], hf, cp["R_cap"], cp["native_HF_insertion"])
                assert S.tolist() == cp["selected_ids"]
                c = np.array(cp["ci_coefficients_real"])
                hs = ref["h"][S][:, S].toarray()
                assert abs(np.linalg.norm(c)-1.) < 1e-9
                assert np.linalg.norm(hs @ c-cp["native_qsci_energy"]*c) < 1e-8
                assert abs(np.linalg.eigvalsh(hs)[0]-cp["native_qsci_energy"]) < 1e-9
                assert total_shots == cp["cumulative"]["raw_simulated_shots"]
            assert len(cps) == terminal["checkpoints"]
            assert total_shots == terminal["cumulative"]["raw_simulated_shots"]
            records.append({"case": case, "method": terminal["method"], "Ns": terminal["Ns"], "R": terminal["R"],
                            "seed": terminal["seed"], "checkpoints": len(cps), "shots": total_shots,
                            "status": terminal["status"]})
    expected = len(mf["cases"])*(2 if mf["mode"] == "quick" else 24)
    if len(records) != expected:
        failures.append(f"successful audit coverage {len(records)}/{expected}; preserve missingness")
    result = {"status": "FAILED" if failures else "QUICK_B2_VALIDATED" if mf["mode"] == "quick" else "B2_VALIDATED_WITH_LIMITS",
              "records": records, "failures": failures, "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / "validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    if audit(p.parse_args().output)["failures"]:
        raise SystemExit(1)
