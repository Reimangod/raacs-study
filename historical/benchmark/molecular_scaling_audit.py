"""Independent all-record molecular scaling audit; no full Fock objects."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .raacs_counts_audit import audit_cell
from .local_sampler_primitives import unpack_counts64, observed_ci
from .resource_adaptive_sampler import integral_problem
from .molecular_mps_scaling import no_full_state_guards


def audit(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    for name, sha in cfg["implementation"].items(): assert file_digest(ROOT/name) == sha
    state = json.loads((root/"execution-status.json").read_text())
    assert state["terminal"] == state["planned"] == len(cfg["queue"])*4
    rows = []; failures = []; bond_rows = []; raw = diagnostic = 0
    with threadpool_limits(limits=1), no_full_state_guards():
        for item in cfg["queue"]:
            case = root/"cases"/item["id"]; problem = case/"physical-problem"
            if (case/"physical-failure.json").exists(): failures.append({"case": item["id"], "method": "ALL", "kind": "PHYSICAL_INPUT_FAILURE"}); continue
            pm = json.loads((problem/"problem.json").read_text())
            assert not pm["FCI_computed"] and not pm["full_Fock_H_constructed"] and not pm["full_sector_enumerated"]
            assert pm["ground_state_reference_energy_Ha"] is None
            for name, sha in pm["artifact_sha256"].items(): assert file_digest(problem/name) == sha
            backend, info = integral_problem(problem)
            hf = json.loads((case/"HF/result.json").read_text()); blob = case/"HF"/hf["counts"]["path"]
            assert file_digest(blob) == hf["counts"]["sha256"]
            ids, ns = unpack_counts64(blob.read_bytes()); assert set(map(int, ids)) == {info["hf_id"]}
            assert int(ns.sum()) == hf["counts"]["shots"] == cfg["shots_per_cell"]
            assert abs(hf["energy_Ha"]-info["HF_energy_Ha"]) < 1e-8; raw += int(ns.sum())
            rows.append({"case": item["id"], "method": "HF", "n_qubits": info["n_qubits"], "energy_Ha": hf["energy_Ha"],
                         "dimension": 1, "status": hf["status"], "cnot": 0, "depth": 0,
                         "wall_seconds": hf["metadata"]["call_wall_seconds"], "pilot_shots": 0, "final_shots": int(ns.sum())})
            for method in cfg["scores"]:
                cell = case/method
                if (cell/"failure.json").exists(): failures.append({"case": item["id"], "method": method, "kind": "ENGINEERING_FAILURE"}); continue
                a = audit_cell(problem, cell); raw += a["pilot_shots"]+a["final_shots"]
                r = json.loads((cell/"result.json").read_text()); final = a["final_resources"][0]
                rows.append({"case": item["id"], "method": method, "n_qubits": info["n_qubits"], "energy_Ha": a["energy_Ha"],
                             "HF_minus_QSCI_mHa": 1000*(info["HF_energy_Ha"]-a["energy_Ha"]), "dimension": a["actual_dimension"],
                             "status": a["status"], "cnot": final["cnot_count"], "depth": final["cnot_depth"],
                             "wall_seconds": r["actual_wall_seconds"], "integral_build_seconds": pm["construction_wall_seconds"],
                             "named_H_requests": r["classical_named_H_requests"], **{k: a[k] for k in ("pilot_shots", "final_shots", "peak_process_RSS_bytes", "CNOT_times_all_raw_shots", "depth_times_all_raw_shots")}})
                bond_dir = case/"bond-diagnostics"/method; b = json.loads((bond_dir/"result.json").read_text())
                assert b["binding"]["source_database_sha256"] == r["event_database_sha256"]
                assert b["binding"]["preparation_id"] == r["final_circuit_allocations"][0]["circuit_id"]
                assert b["identical_physical_circuit"] and not b["used_for_policy_selection"]
                assert [x["bond"] for x in b["rows"]] == cfg["bond_checks"]
                for row in b["rows"]:
                    record = {"case": item["id"], "method": method, "n_qubits": info["n_qubits"], **row}
                    if "counts" in row:
                        path = bond_dir/row["counts"]["path"]; assert file_digest(path) == row["counts"]["sha256"]
                        ids, ns = unpack_counts64(path.read_bytes()); assert int(ns.sum()) == row["counts"]["shots"] <= cfg["diagnostic_shots_per_bond"]
                        fresh = observed_ci(backend, dict(zip(map(int, ids), map(int, ns))), info["hf_id"], info["nalpha"], info["nbeta"], 64)
                        assert abs(fresh["energy_Ha"]-row["energy_Ha"]) < 1e-9
                        diagnostic += int(ns.sum())
                        assert not row["build"]["full_statevector_allocated"]
                    bond_rows.append(record)
    import pandas as pd
    pd.DataFrame(rows).to_csv(root/"audited-scaling-cells.csv", index=False)
    pd.json_normalize(bond_rows).to_csv(root/"audited-bond-diagnostics.csv", index=False)
    out = {"status": "SCALING_COUNTS_AND_BINDINGS_VALIDATED", "planned": state["planned"], "available_rows": len(rows),
           "failures": failures, "raw_construction_and_final_model_shots": raw, "extra_bond_diagnostic_model_shots": diagnostic,
           "maximum_actual_molecular_qubits": max(r["n_qubits"] for r in rows),
           "ground_state_error_and_chemical_accuracy": "UNKNOWN_NO_REFERENCE", "hardware_shots": 0,
           "all_records_checked_not_all_draws_reexecuted": True, "whole_study_complete": False,
           "auditor_sha256": file_digest(__file__)}
    save_json(root/"validation.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
