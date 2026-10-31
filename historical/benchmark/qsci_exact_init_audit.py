"""Independent saved count, CI, and physical-circuit qualification of B2 successor."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, rng_for, save_json
from .chemistry import load_reference
from .qsci_exact_initialization import restore, preparation
from .qsci_growth import selected_subspace


def audit(root):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text())
    for path, sha in mf["implementation"].items():
        assert file_digest(ROOT / path) == sha, path
    checkpoints = physical = available = 0; failed = []; status_counts = {}
    with threadpool_limits(limits=1):
        for case in mf["cases"]:
            ref = load_reference(Path(mf["source"]) / "references" / case); meta = ref["meta"]
            paths = sorted((root / "trajectories" / case).glob("*"))
            assert len(paths) == (1 if mf["mode"] == "quick" else 12)
            for d in paths:
                if (d / "failure.json").exists():
                    failed.append({"path": str(d), **json.loads((d / "failure.json").read_text())}); continue
                terminal = json.loads((d / "terminal.json").read_text()); available += 1
                status_counts[terminal["status"]] = status_counts.get(terminal["status"], 0)+1
                cps = sorted(d.glob("step-*.json")); shots = 0
                assert cps and len(cps) == terminal["checkpoints"]
                for step, path in enumerate(cps):
                    cp = json.loads(path.read_text()); init = cp["initialization"]
                    ids = np.array(init["target_ids"]); coefficients = np.array(init["coefficients_real"])+1j*np.array(init["coefficients_imag"])
                    assert cp["task_id"] == terminal["task_id"] and cp["step"] == step
                    assert not cp["FCI_used_for_selection"] and set(ids).issubset(set(ref["ids"]))
                    state = restore(meta["n_qubits"], ids, coefficients, cp["rotations"])
                    rng, seed = rng_for("B2-exactprep-growth-v1", cp["task_id"], step)
                    p = abs(state)**2; ns = rng.multinomial(cp["Ns"], p/p.sum())
                    with np.load(d / f"counts-{step:04d}.npz") as z:
                        assert np.array_equal(z["ids"], np.flatnonzero(ns)) and np.array_equal(z["counts"], ns[z["ids"]])
                    S = selected_subspace(np.flatnonzero(ns), ns[ns > 0], ref["ids"], meta["hf_id"], cp["R_cap"], True)
                    assert S.tolist() == cp["selected_ids"]
                    c = np.array(cp["ci_coefficients_real"]); hs = ref["h"][S][:, S].toarray()
                    assert abs(np.linalg.norm(c)-1) < 1e-9 and np.linalg.norm(hs @ c-cp["native_qsci_energy"]*c) < 1e-8
                    assert abs(np.linalg.eigvalsh(hs)[0]-cp["native_qsci_energy"]) < 1e-9
                    shots += int(ns.sum()); assert shots == cp["cumulative"]["raw_simulated_shots"]
                    checkpoints += 1
                    if step in (0, len(cps)-1):
                        qc = preparation(meta["n_qubits"], ids, coefficients, cp["rotations"])
                        q = Statevector.from_instruction(qc).data
                        assert 1-abs(np.vdot(q, state))**2 < 1e-8
                        physical += 1
                assert shots == terminal["cumulative"]["raw_simulated_shots"]
    status = "UNEXPECTED_FAILURES_PRESERVED" if failed else ("QUICK_PHYSICAL_INIT_GROWTH_VALIDATED" if mf["mode"] == "quick" else "STANDARD_PHYSICAL_INIT_GROWTH_VALIDATED")
    result = {"status": status, "available_trajectories": available, "failed": failed, "checkpoints": checkpoints,
              "independent_physical_circuit_checks": physical, "status_counts": status_counts,
              "scope": "all saved count/RNG/selection/CI checks; extra first/last circuit parity; production parity everycheckpoint",
              "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    result = audit(p.parse_args().output); print(json.dumps(result))
    if result["failed"]:
        raise SystemExit(1)
