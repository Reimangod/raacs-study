"""Reconstruct actual pilot/final records; proposals never become free samples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .ansatz import resources
from .common import ROOT, digest, file_digest, rng_for, save_json
from .circuit_sampling_runtime import CircuitCountSampler, SamplingSpec
from .local_sampler_primitives import LocalQE, observed_ci, unpack_counts64
from .resource_adaptive_sampler import integral_problem


def audit(root):
    from io import BytesIO
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); source = Path(cfg["source"])
    for p, sha in cfg["implementation"].items(): assert file_digest(ROOT/p) == sha
    cells = draws = total = pilots = finals = circuits_checked = 0; rows = []; min_final = None
    max_probability_discrepancy = 0.
    with threadpool_limits(limits=1):
        for result_path in sorted((root/"cells").glob("*/*/result.json")):
            folder = result_path.parent; case = folder.parent.name; result = json.loads(result_path.read_text())
            config = json.loads((folder/"config.json").read_text()); algorithm = config["algorithm"]
            backend, info = integral_problem(source/"references"/case); assert info == config["problem"] == result["problem"]
            assert file_digest(folder/"events.sqlite") == result["event_database_sha256"]
            db = sqlite3.connect(f"file:{folder/'events.sqlite'}?mode=ro", uri=True)
            circuit_map = {}; samplers = CircuitCountSampler(SamplingSpec(**config["sampling"]))
            for cid, raw, sample_raw, actions_raw, resources_raw in db.execute("SELECT * FROM circuits"):
                raw = zlib.decompress(raw); sample_raw = zlib.decompress(sample_raw)
                assert hashlib.sha256(raw).hexdigest() == cid
                prep = qpy.load(BytesIO(raw))[0]; sampling = qpy.load(BytesIO(sample_raw))[0]; actions = json.loads(actions_raw)
                assert prep.num_qubits == sampling.num_qubits == info["n_qubits"] <= 16
                rebuilt = QuantumCircuit(info["n_qubits"])
                for wire in range(info["n_qubits"]):
                    if info["hf_id"] >> wire & 1: rebuilt.x(wire)
                for action in actions:
                    op = LocalQE(tuple(action["source"]), tuple(action["target"]), action["n_qubits"])
                    rebuilt.compose(op.circuit(action["theta"]), inplace=True)
                p = abs(Statevector.from_instruction(prep).data)**2
                independent = abs(Statevector.from_instruction(rebuilt).data)**2
                sampling_p = abs(Statevector.from_instruction(sampling).data)**2
                difference = float(max(np.max(abs(p-independent)), np.max(abs(p-sampling_p))))
                assert difference < 1e-9; max_probability_discrepancy = max(max_probability_discrepancy, difference)
                assert resources(sampling, len(actions), len(actions)) == json.loads(resources_raw)
                circuit_map[cid] = (prep, actions); circuits_checked += 1
            observed_draws = {}; cell_pilot = cell_final = 0
            for did, cid, role, step, M, n, seed, blob, metadata_raw in db.execute("SELECT * FROM draws"):
                prep, actions = circuit_map[cid]; metadata = json.loads(metadata_raw)
                if role == "initial_pilot": extra = "HF"
                elif role == "candidate_pilot":
                    a = actions[-1]; extra = [LocalQE(tuple(a["source"]), tuple(a["target"]), a["n_qubits"]).identity, a["theta"]]
                elif role == "independent_final":
                    matches = [j for j, a in enumerate(result["final_circuit_allocations"]) if a["circuit_id"] == cid and a["shots"] == M]
                    assert len(matches) == 1; extra = matches[0]
                else: raise AssertionError("Unknown physical draw role")
                fresh_seed = rng_for("RAACS-physical-counts-v1", digest(config), role, step, extra, M)[1]
                assert str(fresh_seed) == seed == metadata["seed"]
                ids, counts = unpack_counts64(blob)
                assert int(counts.sum()) == n == metadata["actual_raw_model_shots"] <= M
                assert did == digest([digest(config), role, step, extra, M, cid])
                a, b, fresh = samplers.draw(prep, M, fresh_seed, len(actions), len(actions))
                assert fresh["sample_metadata"]["status"] == "COMPLETED" and n == M
                assert np.array_equal(ids, a.astype(np.uint64)) and np.array_equal(counts, b.astype(np.uint64))
                obs = {int(x): int(n) for x, n in zip(ids, counts)}
                ci = observed_ci(backend, obs, info["hf_id"], info["nalpha"], info["nbeta"], algorithm["CI_cap"])
                assert all(x == info["hf_id"] or obs.get(x, 0) > 0 for x in ci["selected_ids"])
                observed_draws[did] = {"ci": ci, "obs": obs, "role": role, "metadata": metadata}
                if role == "independent_final": cell_final += n
                else: cell_pilot += n
                draws += 1; total += n
            for step, raw in db.execute("SELECT * FROM decisions"):
                d = json.loads(raw); assert d["step"] == step and not d["FCI_used"]
                assert all(observed_draws[did]["role"] == "candidate_pilot" for did in d["all_measured_draw_ids"])
                assert d["measured_candidates_and_angles"] == len(d["all_measured_draw_ids"])
                if d["accepted"]:
                    assert d["winner_draw_id"] in d["all_measured_draw_ids"]
                    assert d["winner_gain_Ha_selection_biased"] > algorithm["pilot_acceptance_improvement_Ha"]
            merged = {}
            for did in result["final_draw_ids"]:
                assert observed_draws[did]["role"] == "independent_final"
                for x, n in observed_draws[did]["obs"].items(): merged[x] = merged.get(x, 0)+n
            final_ci = observed_ci(backend, merged, info["hf_id"], info["nalpha"], info["nbeta"], algorithm["CI_cap"])
            assert final_ci["selected_ids"] == result["final_selected_ids"]
            assert abs(final_ci["energy_Ha"]-result["final_QSCI_energy_Ha"]) < 1e-9
            assert cell_pilot == result["pilot_raw_model_shots"] and cell_final == result["independent_final_raw_model_shots"]
            assert cell_pilot+cell_final == result["requested_total_model_shots"] == result["actual_total_model_shots"]
            assert cell_final >= int(np.ceil(algorithm["total_model_shots"]*algorithm["final_reserve_fraction"]))
            assert not result["FCI_used"] and not result["CI_state_freely_prepared"] and result["hardware_shots"] == 0
            if config["sampling"]["backend"] == "mps": assert not result["full_Fock_statevector_used_in_backend"]
            pilots += cell_pilot; finals += cell_final; cells += 1; min_final = cell_final if min_final is None else min(min_final, cell_final)
            rows.append({"case_id": case, "cell": folder.name, "energy_Ha": final_ci["energy_Ha"], "pilot_shots": cell_pilot,
                         "final_shots": cell_final, "dimension": final_ci["actual_dimension"], "accepted_steps": result["accepted_steps"]})
            db.close()
    expected = cfg.get("exact_small_cells", 0)+cfg.get("MPS_cells", 0)
    assert cells == expected
    import pandas as pd
    pd.DataFrame(rows).to_csv(root/"audited-cell-energies.csv", index=False)
    result = {"status": "RAACS_QUICK_COUNTS_PHYSICS_AND_FINAL_SUPPORT_VALIDATED", "cells": cells,
              "all_raw_draws_rebuilt": draws, "raw_model_shots": total, "pilot_raw_model_shots": pilots,
              "independent_final_raw_model_shots": finals, "minimum_independent_final_shots": min_final,
              "all_physical_circuits_checked": circuits_checked, "max_sampling_probability_difference": max_probability_discrepancy,
              "FCI_used_in_sampler": False, "classical_proposals_not_inserted_into_CI": True, "hardware_shots": 0,
              "actual_40qubit_molecular_scaling_demonstrated": False, "whole_study_complete": False,
              "audit_sha256": file_digest(__file__)}
    save_json(root/"validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
