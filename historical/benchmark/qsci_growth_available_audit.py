"""Validate B2 available trajectories while retaining proven initializer missingness.

This successor does not overwrite the stricter all-success v1 audit. A failed
greedy initializer is not a valid sampler and cannot win a performance comparison.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from .common import ROOT, digest, file_digest, rng_for, save_json
from .chemistry import load_reference
from .qsci_growth import restore_state, selected_subspace, orbital_energies
from .adapt_qsci_native import circuit, NamedHamiltonian, paper_pool, perturbative_target


def audit(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text())
    failures, missing, records = [], [], []
    checked_checkpoints = independent_parity = 0
    for name, sha in mf["code"].items():
        if file_digest(ROOT / name) != sha:
            raise RuntimeError("B2 frozen code drift: " + name)
    for case in mf["cases"]:
        ref = load_reference(Path(mf["source"]) / "references" / case); meta = ref["meta"]
        paths = sorted((root / "trajectories" / case).glob("*/*"))
        all_checkpoints = sorted((p for d in paths for p in d.glob("step-*.json")), key=lambda p: digest(str(p.relative_to(root))))
        parity_sample = set(all_checkpoints[:5])
        for d in paths:
            if (d / "failure.json").exists():
                f = json.loads((d / "failure.json").read_text())
                is_initializer = (f["method"] == "hf_separated_adapt_qsci_MP_m0_2" and
                                  "INITIAL_PREPARATION_FIDELITY_NOT_REACHED" in f["error"] and
                                  f.get("partial_checkpoint_count") == 0 and f.get("durable_raw_simulated_shots") == 0)
                if is_initializer:
                    ids, c = perturbative_target(NamedHamiltonian(ref["h"]), ref["ids"], meta["hf_id"], orbital_energies(ref), meta["n_qubits"])
                    diff = int(ids[0]) ^ int(ids[1]) if len(ids) == 2 else 0
                    direct = any(p.flip_mask == diff for p in paper_pool(meta["n_qubits"]))
                    if len(ids) != 2 or direct or not np.all(abs(c) > 1e-12):
                        failures.append(str(d.relative_to(root))+": initializer failure not explained by registered greedy limitation")
                        continue
                    missing.append({"case_id": case, "method": f["method"], "Ns": f["Ns"], "R": f["R"], "seed": f["seed"],
                                    "status": "INITIALIZER_UNAVAILABLE_NO_SAMPLER_OR_ENERGY", "target_ids": ids.tolist(),
                                    "target_difference_bits": diff.bit_count(), "direct_pair_rotation_in_pool": False,
                                    "raw_simulated_shots": 0, "classical_failure_wall_seconds": None,
                                    "cost_limit": "original failure handler did not record per-attempt wall; not assumed zero",
                                    "explanation": "first greedy rotation optimizes fidelity to one nonHF target; no pool element connects the two targets, so strict greedy improvement stalls. Not global unpreparability or failure of HF-separated objective."})
                    continue
                failures.append(str(d.relative_to(root))+":"+f["error"])
                continue
            terminal = json.loads((d / "terminal.json").read_text())
            cps = sorted(d.glob("step-*.json")); shots = 0
            for step, path in enumerate(cps):
                cp = json.loads(path.read_text()); checked_checkpoints += 1
                assert cp["task_id"] == terminal["task_id"] and cp["step"] == step
                with np.load(d / f"counts-{step:04d}.npz") as z:
                    ids, counts = z["ids"], z["counts"]
                assert counts.sum() == cp["Ns"] and len(ids) == len(set(ids))
                state = restore_state(meta["n_qubits"], meta["hf_id"], cp["rotations"])
                rng, _ = rng_for("B2-native-growth-draw-v1", cp["task_id"], step)
                p = abs(state)**2; actual = rng.multinomial(cp["Ns"], p/p.sum())
                assert np.array_equal(actual[ids], counts) and np.count_nonzero(actual) == len(ids)
                S = selected_subspace(ids, counts, ref["ids"], meta["hf_id"], cp["R_cap"], cp["native_HF_insertion"])
                assert S.tolist() == cp["selected_ids"]
                c = np.array(cp["ci_coefficients_real"]); hs = ref["h"][S][:, S]
                assert abs(np.linalg.norm(c)-1.) < 1e-9
                assert np.linalg.norm(hs @ c-cp["native_qsci_energy"]*c) < 1e-8
                assert abs(np.linalg.eigvalsh(hs.toarray())[0]-cp["native_qsci_energy"]) < 1e-9
                shots += int(counts.sum())
                assert shots == cp["cumulative"]["raw_simulated_shots"]
                if path in parity_sample:
                    independent = Statevector.from_instruction(circuit(meta["n_qubits"], meta["hf_id"], cp["rotations"])).data
                    assert abs(abs(np.vdot(state, independent))-1.) < 1e-8
                    independent_parity += 1
            # A terminal no-valid-sample draw may have counts but no CI checkpoint.
            raw_shots = 0
            for f in d.glob("counts-*.npz"):
                with np.load(f) as z:
                    raw_shots += int(z["counts"].sum())
            assert raw_shots == terminal["cumulative"]["raw_simulated_shots"]
            assert len(cps) == terminal["checkpoints"]
            records.append({"case_id": case, "method": terminal["method"], "Ns": terminal["Ns"], "R": terminal["R"],
                            "seed": terminal["seed"], "status": terminal["status"], "checkpoints": len(cps), "raw_shots": raw_shots})
    expected = len(mf["cases"])*(2 if mf["mode"] == "quick" else 24)
    if len(records)+len(missing) != expected:
        failures.append(f"cohort coverage {len(records)} available+{len(missing)} unavailable/{expected}")
    result = {"status": "FAIL_UNEXPLAINED_DATA_OR_EXECUTION" if failures else "AVAILABLE_B2_VALIDATED_WITH_EXPLICIT_INITIALIZATION_MISSINGNESS",
              "planned_trajectories": expected, "validated_available": len(records), "initializer_missing": len(missing),
              "checkpoint_count": checked_checkpoints, "additional_independent_Qiskit_parity_checks": independent_parity,
              "audit_scope": "all counts/RNG/selection/CI residuals, deterministic up-to5-per-case additional circuit parity; production already checked circuit parity at every savedcheckpoint",
              "records": records, "missingness": missing, "unexpected_failures": failures,
              "not_claimed": ["failed_initializers_have_valid_energy", "method_is_globally_unpreparable", "HF_separation_criterion_failed", "complete_failure_wall_accounting"],
              "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / "validation-available-v2.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "missingness")}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    if audit(p.parse_args().output)["unexpected_failures"]:
        raise SystemExit(1)
