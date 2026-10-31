"""Actual-growth endpoint physics, all paid probes and shared-prefix audit."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .ansatz import make_pool, hf_state, state_from_parameters, selection_gradients, resources
from .common import ROOT, file_digest, rng_for, save_json
from .molecular_problem_no_fci import load_problem
from .online_stopping_rules import ObservedFrame, OnlineStopRule, registered_rules
from .stopping_pilot_bank import ci_features


def audit(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    for p, sha in cfg["implementation"].items(): assert file_digest(ROOT/p) == sha
    cells = endpoints = probes = rawshots = prefixes = missing_native = 0
    statuses = Counter(); paired_rows = []; problems_checked = 0
    max_state_difference = max_gradient_difference = 0.
    with threadpool_limits(limits=1):
        for item in cfg["physical_problems"]:
            case = item["case_id"]; problem = load_problem(root/"problems"/case); meta = problem["meta"]
            assert not meta["FCI_computed"] and meta["ground_state_reference_energy_Ha"] is None
            for name, sha in meta["artifact_sha256"].items(): assert file_digest(problem["directory"]/name) == sha
            problems_checked += 1; valid = set(map(int, problem["ids"])); hf = meta["hf_id"]
            hsector = problem["h"][problem["ids"]][:, problem["ids"]].tocsr()
            for method in cfg["methods"]:
                pool = make_pool(method, meta["n_qubits"]); initial = hf_state(meta["n_qubits"], hf)
                native_dir = root/"cells"/case/method/"native_terminal"
                if not (native_dir/"result.json").exists(): raise RuntimeError("Native trajectory unavailable; preserve failure")
                native = json.loads((native_dir/"result.json").read_text())
                native_cps = {p.name: json.loads(p.read_text()) for p in (native_dir/"steps").glob("step-*.json")}
                for policy in cfg["policies"]:
                    folder = root/"cells"/case/method/policy
                    if not (folder/"result.json").exists(): raise RuntimeError("Actual cell failed/unfinished: "+str(folder))
                    r = json.loads((folder/"result.json").read_text()); statuses[r["status"]] += 1
                    assert not r["FCI_used"] and r["hardware_shots"] == 0 and r["problem_id"] == meta["problem_id"]
                    assert not r["unknown_interrupted_inflight_work"], "Interrupted cost requires separate nonzero/unknown accounting"
                    for name, sha in r["artifact_sha256"].items(): assert file_digest(folder/name) == sha
                    cps = [json.loads(p.read_text()) for p in sorted((folder/"steps").glob("step-*.json"))]
                    assert [c["step"] for c in cps] == list(range(r["last_step"]+1))
                    assert cps[-1]["terminal"] == r["status"] and all(c["terminal"] is None for c in cps[:-1])
                    cumulative = {}; seen = set(); spent = 0; cell_prefix_missing = 0
                    rule = None if policy in ("native_terminal", "gradient") else OnlineStopRule(next(s for s in registered_rules() if s.name == policy))
                    for cp in cps:
                        name = f"step-{cp['step']:04d}.json"
                        if name in native_cps:
                            n = native_cps[name]
                            assert cp["selected_operators"] == n["selected_operators"]
                            assert np.allclose(cp["theta"], n["theta"], atol=1e-8, rtol=0)
                            assert abs(cp["energy_Ha"]-n["energy_Ha"]) < 1e-8
                            assert cp["resources"] == n["resources"]
                            prefixes += 1
                        else:
                            assert native["status"] == "TIME_CENSORED"
                            cell_prefix_missing += 1; missing_native += 1
                        ev = cp["pilot"]
                        if ev is None: continue
                        psi = state_from_parameters(pool, cp["selected_operators"], cp["theta"], initial)
                        p = abs(psi)**2
                        rng, seed = rng_for("PhaseC-actual-policy-pilots-v1", meta["problem_id"], method, policy, cfg["construction_seed"], cp["step"])
                        counts = rng.multinomial(64, p/p.sum()); ids = np.flatnonzero(counts)
                        assert ev["seed"] == str(seed) and ev["raw_counts"] == [[int(x), int(counts[x])] for x in ids]
                        pairs = [[int(x), int(counts[x])] for x in ids if int(x) in valid and int(x) != hf]
                        assert pairs == ev["valid_nonHF_counts"]
                        now = {x for x, _ in pairs}; new = len(now-seen); seen.update(now)
                        for x, n in pairs: cumulative[x] = cumulative.get(x, 0)+n
                        selected = [hf]+[x for x, _ in sorted(cumulative.items(), key=lambda a: (-a[1], a[0]))][:63]
                        assert selected == ev["selected_ids"]
                        fresh = ci_features(hsector, problem["ids"], selected, hf, cumulative)
                        f = ObservedFrame(**ev["frame"]); spent += 64
                        assert f.cumulative_pilot_raw_shots == spent and f.new_observed_valid_nonHF == new
                        for field in ("qsci_energy_Ha", "external_residual_norm_Ha", "pt2_proxy_Ha"):
                            assert abs(getattr(f, field)-fresh[field]) < 1e-8
                        assert f.cnot_count == cp["resources"]["cnot_count"] and f.cnot_depth == cp["resources"]["cnot_depth"]
                        assert f.current_pool_gradient_norm == cp["current_pool_gradient_norm"]
                        assert f.hf_raw_fraction == counts[hf]/64
                        decision = rule.observe(f); assert decision == ev["decision"]
                        if decision: assert cp is cps[-1] and r["status"] == "POLICY_STOPPED"
                        probes += 1; rawshots += 64
                    assert spent == r["pilot_raw_model_shots"]
                    if policy in ("native_terminal", "gradient"): assert spent == 0
                    last = cps[-1]
                    psi = state_from_parameters(pool, last["selected_operators"], last["theta"], initial)
                    g = selection_gradients(pool, psi, problem["h"]); gn = float(np.linalg.norm(g))
                    diff = abs(gn-r["current_pool_gradient_norm"]); assert diff < 1e-8
                    max_gradient_difference = max(max_gradient_difference, diff)
                    if r["status"] == "NATIVE_GRADIENT_CONVERGED": assert gn <= 1e-3
                    elif r["status"] == "MAX_STEPS": assert r["last_step"] == cfg["max_steps"] and gn > 1e-3
                    with (folder/"final.preparation.qpy").open("rb") as stream: prep = qpy.load(stream)[0]
                    with (folder/"final.sampling.qpy").open("rb") as stream: sample = qpy.load(stream)[0]
                    prep_state = Statevector.from_instruction(prep).data
                    overlap = np.vdot(prep_state, psi); phase = overlap/abs(overlap) if abs(overlap) else 1
                    state_diff = float(np.linalg.norm(psi-phase*prep_state)); assert state_diff < 1e-8
                    max_state_difference = max(max_state_difference, state_diff)
                    sampling_p = abs(Statevector.from_instruction(sample).data)**2
                    with np.load(folder/"final.npz") as z: stored_p = z["probability"]
                    assert np.allclose(stored_p, abs(psi)**2, atol=1e-9) and np.allclose(stored_p, sampling_p, atol=1e-9)
                    assert abs(float(stored_p.sum())-1) < 1e-9
                    assert abs(float(stored_p[problem["ids"]].sum())-1) < 1e-8
                    assert resources(sample, len(last["theta"]), len(last["selected_operators"])) == r["resources"]
                    for field in ("objective_evaluations", "gradient_evaluations", "optimizer_iterations", "pool_gradient_components_computed"):
                        assert sum(c[field] for c in cps) == r[field]
                    paired_rows.append({"case_id": case, "method": method, "policy": policy, "status": r["status"],
                        "last_step": r["last_step"], "native_last_step": native["last_step"], "native_status": native["status"],
                        "all_selected_prefix_verified": cell_prefix_missing == 0,
                        "CNOT_reduction_percent": 100*(1-r["resources"]["cnot_count"]/native["resources"]["cnot_count"]),
                        "depth_reduction_percent": 100*(1-r["resources"]["cnot_depth"]/native["resources"]["cnot_depth"]),
                        "native_minus_policy_objective_calls": native["objective_evaluations"]-r["objective_evaluations"],
                        "native_minus_policy_optimizer_iterations": native["optimizer_iterations"]-r["optimizer_iterations"],
                        "native_minus_policy_gradient_components": native["pool_gradient_components_computed"]-r["pool_gradient_components_computed"],
                        "pilot_raw_model_shots": spent, "hardware_shots": 0})
                    cells += 1; endpoints += 1
                make_pool.cache_clear()
    assert cells == len(cfg["physical_problems"])*len(cfg["methods"])*len(cfg["policies"])
    import pandas as pd
    pd.DataFrame(paired_rows).to_csv(root/"construction-pairs.csv", index=False)
    result = {"status": "ACTUAL_GROWTH_PREFIX_AND_PHYSICS_VALIDATED", "problems_checked": problems_checked,
              "actual_cells": cells, "statuses": dict(statuses), "fresh_final_preparation_and_sampling_circuits": endpoints*2,
              "all_available_native_prefix_points_checked": prefixes, "prefix_points_beyond_native_timecap": missing_native,
              "all_pilot_draws_rebuilt": probes, "actual_pilot_raw_model_shots": rawshots,
              "maximum_prepared_state_difference": max_state_difference, "maximum_current_gradient_difference": max_gradient_difference,
              "N_and_Sz_sector_checked_not_total_S2": True, "FCI_used": False, "hardware_shots": 0,
              "scalable_backend_demonstrated": False, "whole_study_complete": False, "auditor_sha256": file_digest(__file__)}
    save_json(root/"validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
