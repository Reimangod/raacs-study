"""Independent fixed-sequence control audit, including actual saved QPY states.

Full vectors are restricted to <=16q *auditor* checks, never optimization.
Rebuild every recorded draw from its seed and physical circuit distribution.
The CI selector/eigensolver and coordinate replay below are not the producer's.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .local_sampler_primitives import LocalQE, unpack_counts64, valid_sector
from .resource_adaptive_sampler import integral_problem
from .sparse_qe_circuit import SparseQECircuit
from .ansatz import resources
from .raacs_counts_audit import circuit_signature

NAMESPACE = "RAACS-fixed-operator-parameter-control-v1"
SUCCESS = "PARAMETER_CONTROL_COUNTS_AND_CIRCUITS_VALIDATED"


def close(a, b, tolerance=1e-9):
    assert np.allclose(a, b, atol=tolerance, rtol=0), (a, b)


def check_coordinate(control, initial, draws):
    """Replay the registered one-pass coordinate sweep without running it anew."""
    theta = initial.copy()
    first = draws[control["initial_draw"]]
    assert first["role"] == "initial" and first["slot"] == 0 and first["shots"] == 64
    close(first["theta"], theta, 0)
    current_E = first["energy"]
    expected_count = min(4*len(theta), 32)
    assert len(control["trials"]) == expected_count
    seen = [control["initial_draw"]]
    for j in range(min(len(theta), 8)):
        base = theta.copy()
        candidates = [(current_E, theta.copy())]
        for offset, shift in enumerate((-np.pi/4, -np.pi/8, np.pi/8, np.pi/4)):
            index = j*4+offset
            if index >= expected_count: break
            trial = control["trials"][index]
            d = draws[trial["draw"]]
            target = base.copy(); target[j] += shift
            assert trial["trial"] == d["slot"] == index+1
            assert trial["coordinate"] == j and d["role"] == "trial" and d["shots"] == 64
            close(trial["theta"], target, 1e-14); close(d["theta"], target, 1e-14)
            close(d["energy"], trial["energy_Ha_pilot"])
            candidates.append((d["energy"], target)); seen.append(trial["draw"])
        best = min(candidates, key=lambda x: (x[0], tuple(x[1])))
        if best[0] < current_E-1e-6: current_E, theta = best[0], best[1]
    close(control["theta"], theta, 1e-14)
    close(control["pilot_energy_Ha_selection_biased"], current_E)
    assert control["pilot_evaluations"] == len(seen) == len(set(seen))
    assert set(seen) == {did for did, d in draws.items() if d["role"] != "independent_final"}


def audit_cell(root, cfg, item, mode):
    root = Path(root); folder = root/"cells"/item["case"]/f"s{item['seed']}"/mode
    result = json.loads((folder/"result.json").read_text())
    control = json.loads((folder/"optimization.json").read_text())
    source = Path(item["source_cell"])
    parent = json.loads((source/"result.json").read_text())
    assert file_digest(source/"events.sqlite") == parent["event_database_sha256"]
    backend, info = integral_problem(Path(cfg["source"])/"references"/item["case"])
    ops = [LocalQE(tuple(a["source"]), tuple(a["target"]), a["n_qubits"]) for a in parent["accepted_operators"]]
    initial = np.array([a["theta"] for a in parent["accepted_operators"]], float)
    task = {"case": item["case"], "source_seed": item["seed"], "mode": mode,
            "source_database_sha256": parent["event_database_sha256"], "problem": info,
            "operators": [o.identity for o in ops], "initial_theta": initial.tolist(),
            "inherited_pilot_shots": parent["pilot_raw_model_shots"],
            "experiment_manifest_sha256": file_digest(root/"manifest.json")}
    assert result["task"] == task and result["mode"] == mode and result["status"] == "COMPLETED"
    assert file_digest(folder/"events.sqlite") == result["events_sha256"]
    for name, sha in result["artifact_sha256"].items(): assert file_digest(folder/name) == sha
    assert not result["FCI_used"] and not result["full_Fock_vector_used_in_control"]
    assert result["hardware_shots"] == 0 and not result["whole_study_complete"]
    assert not info["FCI_labels_passed_to_algorithm"]
    sim = SparseQECircuit(info["n_qubits"], info["hf_id"], ops, cfg["support_cap"], backend)
    assert len(sim.ids) == result["physical_sparse_support_size"] <= cfg["support_cap"]
    assert all(valid_sector(int(x), info["n_qubits"], info["nalpha"], info["nbeta"]) for x in sim.ids)
    theta = np.asarray(result["final_theta"], float)
    close(initial, result["initial_theta"], 0); close(control["theta"], theta, 0)
    E0, _ = sim.energy_gradient(initial); E, gradient = sim.energy_gradient(theta)
    close(E0, result["physical_VQE_energy_initial_Ha"]); close(E, result["physical_VQE_energy_final_Ha"])
    close(max(abs(gradient), default=0.), result["final_full_gradient_inf"])
    assert result["n_independent_ansatz_parameters"] == len(ops)

    # Different implementation: rank measured configurations, then a fresh named
    # integral eigensolve per unique selected space; no producer observed_ci().
    ci_cache = {}; probabilities = {}; draws = {}; final = []; pilot_shots = 0; final_shots = 0
    db = sqlite3.connect(f"file:{folder/'events.sqlite'}?mode=ro", uri=True)
    try:
        for did, role, slot, rawtheta, shots, seed, blob, energy, rawselected in db.execute("SELECT * FROM draws ORDER BY rowid"):
            angles = np.array(json.loads(rawtheta), float)
            assert role in ("initial", "trial", "independent_final")
            assert did == digest([task, role, slot, angles.tolist(), shots])
            assert seed == str(rng_for(NAMESPACE, did)[1])
            ids, counts = unpack_counts64(blob)
            assert len(ids) == len(set(map(int, ids))) and all(int(n) > 0 for n in counts)
            assert int(counts.sum()) == shots
            assert all(valid_sector(int(x), info["n_qubits"], info["nalpha"], info["nbeta"]) for x in ids)
            pkey = tuple(angles)
            if pkey not in probabilities:
                p = abs(sim.state(angles))**2; probabilities[pkey] = p/p.sum()
            predicted = np.random.default_rng(int(seed)).multinomial(shots, probabilities[pkey])
            assert np.array_equal(ids, sim.ids[predicted > 0])
            assert np.array_equal(counts, predicted[predicted > 0])
            observed = dict(zip(map(int, ids), map(int, counts)))
            ordered = sorted((x for x in observed if x != info["hf_id"]), key=lambda x: (-observed[x], x))
            selected = tuple(sorted([info["hf_id"]]+ordered[:cfg["R"]-1]))
            assert list(selected) == json.loads(rawselected)
            if selected not in ci_cache:
                h = backend.matrix(selected, selected)
                close(h, h.T.conj(), 1e-10)
                ci_cache[selected] = float(np.linalg.eigvalsh(h)[0])
            close(ci_cache[selected], energy)
            draws[did] = {"theta": angles, "energy": energy, "role": role, "slot": slot, "shots": shots}
            if role == "independent_final":
                assert slot == len(final); close(angles, theta, 0)
                final.append((did, energy)); final_shots += shots
            else: pilot_shots += shots
    finally: db.close()
    assert [d[0] for d in final] == result["final_draw_ids"] and len(set(result["final_draw_ids"])) == len(final)
    close([d[1] for d in final], result["independent_QSCI_energies_Ha"])
    assert len(final) == result["final_repeats"] == cfg["final_repeats"]
    assert pilot_shots == result["new_QSCI_pilot_shots"]
    assert all(d["shots"] == result["final_shots_per_rep"] for d in draws.values() if d["role"] == "independent_final")
    inherited = parent["pilot_raw_model_shots"]
    assert inherited == result["inherited_pilot_shots"]
    assert inherited+pilot_shots+result["final_shots_per_rep"] == cfg["M"] == result["known_per_branch_model_budget"]
    assert final_shots+pilot_shots == result["new_actual_model_shots"]
    trace_count = 0
    if mode.endswith("_vqe"):
        indices = list(range(len(ops))) if mode == "full_vqe" else list(range(max(0, len(ops)-(2 if mode == "blockwise_vqe" else 1)), len(ops)))
        assert indices == control["optimized_variable_indices"]
        assert result["n_optimized_variables"] == control["n_optimized_variables"] == len(indices)
        assert control["optimizer_initial_state"] == "RESET_IDENTITY_BFGS"
        assert result["VQE_objective_QPU_shots"] is None and control["QPU_objective_measurement_shots"] is None
        close(control["initial_theta"], initial, 0)
        trace = control["trace"]; trace_count = len(trace)
        assert 0 < trace_count <= cfg["BFGS"]["max_calls"]
        assert trace_count == control["actual_objective_evaluations"] == control["actual_complete_gradient_evaluations"]
        assert control["n_independent_ansatz_parameters"] == len(ops)
        best = min(trace, key=lambda t: t["energy_Ha"])
        close(best["theta"], theta, 0); close(best["energy_Ha"], E)
        close(trace[0]["theta"], initial, 0)
        inactive = [j for j in range(len(ops)) if j not in indices]
        for i, row in enumerate(trace):
            assert row["call"] == i+1
            close(np.array(row["theta"])[inactive], initial[inactive], 0)
            fresh_E, fresh_g = sim.energy_gradient(row["theta"])
            close(fresh_E, row["energy_Ha"])
            close(max(abs(fresh_g[indices]), default=0.), row["active_gradient_inf"])
            close(max(abs(fresh_g), default=0.), row["full_gradient_inf"])
        active = max(abs(gradient[indices]), default=0.)
        assert control["active_stationary"] == (active <= cfg["BFGS"]["gtol"])
        assert control["full_stationary"] == (max(abs(gradient), default=0.) <= cfg["BFGS"]["gtol"])
        assert E <= E0+1e-10 and pilot_shots == 0
    elif mode == "qsci_coordinate":
        check_coordinate(control, initial, draws)
        assert pilot_shots == (1+min(4*len(ops), 32))*64
        assert result["VQE_objective_QPU_shots"] == 0
    else:
        close(theta, initial if mode == "source_qsci_grid" else np.full(len(ops), np.pi/4), 0)
        assert pilot_shots == 0 and result["n_optimized_variables"] == 0 and result["VQE_objective_QPU_shots"] == 0

    # Entire numerical optimization uses named support. Only this independent
    # small-case auditor allocates full QPY vectors (never a full Hamiltonian).
    assert info["n_qubits"] <= 16
    prep = qpy.load(BytesIO((folder/"preparation.qpy").read_bytes()))[0]
    sampling = qpy.load(BytesIO((folder/"sampling.qpy").read_bytes()))[0]
    reconstructed = QuantumCircuit(info["n_qubits"])
    for q in range(info["n_qubits"]):
        if info["hf_id"] >> q & 1: reconstructed.x(q)
    for op, angle in zip(ops, theta): reconstructed = op.append(reconstructed, angle)
    a, b = circuit_signature(prep), circuit_signature(reconstructed)
    assert a["n_qubits"] == b["n_qubits"] and len(a["gates"]) == len(b["gates"])
    close(np.exp(1j*a["global_phase"]), np.exp(1j*b["global_phase"]))
    for ga, gb in zip(a["gates"], b["gates"]):
        assert ga[0] == gb[0] and ga[2] == gb[2]; close(ga[1], gb[1], 1e-10)
    actual = Statevector.from_instruction(prep).data
    vector = np.zeros_like(actual); vector[sim.ids.astype(np.int64)] = sim.state(theta)
    close(actual, vector, 1e-9)
    close(abs(Statevector.from_instruction(sampling).data)**2, abs(actual)**2, 1e-9)
    close(np.vdot(actual[sim.ids.astype(np.int64)], backend.matrix(sim.ids, sim.ids)@actual[sim.ids.astype(np.int64)]).real, E)
    for j in range(len(theta)):
        plus = theta.copy(); minus = theta.copy(); plus[j] += 1e-6; minus[j] -= 1e-6
        close((sim.energy_gradient(plus)[0]-sim.energy_gradient(minus)[0])/2e-6, gradient[j], 3e-7)
    assert resources(prep, len(ops), len(ops)) == result["preparation_resources"]
    assert resources(sampling, len(ops), len(ops)) == result["sampling_resources"]
    assert result["physical_sparse_support_not_padded_into_QSCI"]
    return {"case": item["case"], "source_seed": item["seed"], "mode": mode,
            "draws_rebuilt": len(draws), "new_pilot_shots": pilot_shots, "new_final_shots": final_shots,
            "inherited_source_pilot_shots_not_new": inherited,
            "unique_named_CI_spaces_freshly_solved": len(ci_cache), "physical_QPY_vectors_checked": 2,
            "objective_trace_calls_checked": trace_count, "physical_support_size": len(sim.ids),
            "full_vectors_only_in_small_auditor_not_optimization": True,
            "result_sha256": file_digest(folder/"result.json")}


def audit(root):
    root = Path(root).resolve(); started = perf_counter()
    cfg = json.loads((root/"manifest.json").read_text())
    for name, sha in cfg["implementation"].items(): assert file_digest(ROOT/name) == sha, name
    assert file_digest(ROOT/"study-v1/PARAMETER_CONTROLS.md") == cfg["spec_sha256"]
    assert cfg["post_construction_conditional_not_new_online_discovery"] and cfg["hardware_shots"] == 0
    rows = []; failures = []
    with threadpool_limits(limits=1):
        for item in cfg["queue"]:
            for mode in cfg["modes"]:
                try: rows.append(audit_cell(root, cfg, item, mode))
                except Exception as exc:
                    failures.append({"case": item["case"], "source_seed": item["seed"], "mode": mode, "error": repr(exc)})
            print(json.dumps({"case": item["case"], "seed": item["seed"], "checked": len(rows), "failed": len(failures)}), flush=True)
    status = json.loads((root/"execution-status.json").read_text())
    expected = len(cfg["queue"])*len(cfg["modes"])
    assert status["terminal"] == status["planned"] == expected
    totals = {key: sum(r[key] for r in rows) for key in ("draws_rebuilt", "new_pilot_shots", "new_final_shots", "unique_named_CI_spaces_freshly_solved", "physical_QPY_vectors_checked", "objective_trace_calls_checked")}
    report = {"status": SUCCESS if not failures and len(rows) == expected else "PARAMETER_CONTROL_AUDIT_FAILED",
              "manifest_sha256": file_digest(root/"manifest.json"), "planned": expected, "validated": len(rows),
              "failures": failures, "totals": totals, "rows": rows, "wall_seconds": perf_counter()-started,
              "inherited_pilots_are_not_new_acquisitions": True, "hardware_shots": 0,
              "VQE_QPU_objective_cost_unknown": True, "D5_online_requirement_completed": False,
              "auditor_code_sha256": file_digest(Path(__file__)), "whole_study_complete": False}
    path = root/"validation.json"
    if path.exists():
        old = json.loads(path.read_text())
        # Keep prior audit evidence on revalidation, including failed audits.
        path = root/f"validation-recheck-{digest(report)[:16]}.json"
    save_json(path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("root")
    result = audit(parser.parse_args().root)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if result["failures"]: raise SystemExit(1)
