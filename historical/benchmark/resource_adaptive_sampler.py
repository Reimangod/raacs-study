"""RAACS hypothesis: bounded integral proposals, actual circuits and paid pilots.

No claim of novelty or superiority. Practical MPS mode never requests a full
Fock vector/Hamiltonian, a reference energy, or a freely prepared CI state.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import hashlib
import importlib.metadata
import zlib
import numpy as np
from qiskit import QuantumCircuit
from .ansatz import circuit_bytes, resources, measurement_equivalent_circuit
from .circuit_sampling_runtime import CircuitCountSampler, SamplingSpec
from .common import digest, file_digest, rng_for, save_json
from .integral_hamiltonian import IntegralHamiltonian
from .integral_connections import IntegralConnections
from .local_sampler_primitives import local_proposals, observed_ci, pack_counts64, unpack_counts64

SCORES = ("ci_gradient_local", "residual", "residual_cost", "residual_cost_diversity", "full_cost")


@dataclass(frozen=True)
class RAACSSpec:
    score: str = "full_cost"
    angle_mode: str = "qsci_local_grid"
    final_allocation: str = "last_only"
    total_model_shots: int = 10000
    final_reserve_fraction: float = .5
    probe_shots: int = 32
    max_steps: int = 8
    CI_cap: int = 64
    parent_connection_cap: int = 128
    target_cap: int = 128
    candidate_bank_cap: int = 32
    candidates_measured_per_step: int = 4
    maximum_named_H_requests: int = 20000000
    pilot_acceptance_improvement_Ha: float = 1e-6
    max_shallow_circuits: int = 4
    seed: int = 101


def integral_problem(directory):
    """Whitelist only physical integrals and HF, not FCI/reference arrays."""
    directory = Path(directory)
    fresh = (directory/"problem.json").exists()
    m = json.loads((directory/("problem.json" if fresh else "reference.json")).read_text())
    archive = directory/("integrals.npz" if fresh else "reference.npz")
    with np.load(archive) as z: h1 = z["h1"].copy(); eri = z["eri"].copy()
    hf_energy = m["hf_energy_Ha"] if fresh else m["rhf_energy"]
    backend = IntegralHamiltonian.from_saved_integrals(h1, eri, m["hf_id"], hf_energy)
    physical_binding = {"h1": hashlib.sha256(backend.h1.astype("<f8").tobytes()).hexdigest(),
                        "eri": hashlib.sha256(backend.eri.astype("<f8").tobytes()).hexdigest(), "core": backend.core,
                        "n_qubits": m["n_qubits"], "nalpha": m["nalpha"], "nbeta": m["nbeta"], "hf_id": m["hf_id"]}
    info = {"problem_id": digest(physical_binding), "n_qubits": m["n_qubits"],
            "nalpha": m["nalpha"], "nbeta": m["nbeta"], "hf_id": m["hf_id"], "HF_energy_Ha": hf_energy,
            "physical_integral_binding": physical_binding,
            "FCI_labels_passed_to_algorithm": False, "physical_integrals_only": True}
    return backend, info


def allocation_counts(total, weights):
    weights = np.maximum(np.asarray(weights, float), 0)
    if total < 1 or not len(weights) or not np.isfinite(weights).all(): raise ValueError("Invalid shot allocation")
    if weights.sum() == 0: weights = np.ones(len(weights))
    exact = total*weights/weights.sum(); n = np.floor(exact).astype(int)
    for i in sorted(range(len(n)), key=lambda i: (-(exact[i]-n[i]), i))[:total-int(n.sum())]: n[i] += 1
    assert int(n.sum()) == total
    return n.tolist()


def run_sampler(directory, output, spec, sampling_spec):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if directory == output or directory in output.parents or output in directory.parents:
        raise ValueError("Protect physical input problem; write to a separate result directory")
    for ancestor in directory.parents:
        if (ancestor/"REPRODUCIBILITY.json").exists() and (output == ancestor or ancestor in output.parents):
            raise ValueError("Do not write inside a sealed source dataset")
    output.mkdir(parents=True, exist_ok=True)
    if spec.score not in SCORES or spec.angle_mode not in ("fixed_angle", "qsci_local_grid"):
        raise NotImplementedError("Only qualified angle modes; blockwise/full-VQE controls require separate implementation")
    if spec.final_allocation not in ("last_only", "uniform", "discovery", "energy_guided"):
        raise ValueError("Unregistered multishallow allocation")
    if spec.total_model_shots < 2*spec.probe_shots or not 0 < spec.final_reserve_fraction < 1 or spec.max_steps < 1:
        raise ValueError("Reserve a positive independent final budget")
    backend, info = integral_problem(directory)
    config = {"schema": "RAACS-prototype-v1", "problem": info, "algorithm": asdict(spec), "sampling": asdict(sampling_spec),
              "runtime_versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "qiskit", "quimb", "numba")},
              "implementation": {str(Path(p).name): file_digest(p) for p in (__file__,
                  Path(__file__).with_name("local_sampler_primitives.py"), Path(__file__).with_name("circuit_sampling_runtime.py"),
                  Path(__file__).with_name("integral_hamiltonian.py"), Path(__file__).with_name("integral_connections.py"),
                  Path(__file__).with_name("ansatz.py"), Path(__file__).with_name("mps_sampling.py"),
                  Path(__file__).resolve().parents[1]/"vendor/ceo/adaptvqe/circuits.py")},
              "gradient_control_scope": "CI Ritz commutator on common local bank, not native full-pool ADAPT gradient",
              "FCI_used": False, "hardware_shots": 0}
    path = output/"config.json"
    if path.exists() and json.loads(path.read_text()) != config: raise RuntimeError("RAACS code/config drift; use successor")
    save_json(path, config)
    if (output/"result.json").exists():
        result = json.loads((output/"result.json").read_text())
        if file_digest(output/"events.sqlite") != result["event_database_sha256"]:
            raise RuntimeError("Completed adaptive event database digest mismatch")
        return result
    archive = directory/("integrals.npz" if (directory/"problem.json").exists() else "reference.npz")
    raw_metadata = directory/("problem.json" if (directory/"problem.json").exists() else "reference.json")
    save_json(output/"input-provenance.json", {"archive_sha256": file_digest(archive), "metadata_sha256": file_digest(raw_metadata),
               "physical_integral_binding": info["physical_integral_binding"],
               "raw_archive_and_metadata_hashes_excluded_from_algorithm_RNG": True})
    # One task owns a cell. Partial trial evidence survives; resume of a failed
    # adaptive decision requires a separately audited successor, not free reruns.
    if (output/"events.sqlite").exists(): raise RuntimeError("Partial adaptive task evidence exists; do not silently rerun")
    db = sqlite3.connect(output/"events.sqlite"); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
      CREATE TABLE circuits(id TEXT PRIMARY KEY,preparation_qpy BLOB,sampling_qpy BLOB,actions TEXT,resources TEXT);
      CREATE TABLE draws(id TEXT PRIMARY KEY,circuit TEXT,role TEXT,step INTEGER,requested INTEGER,raw_shots INTEGER,seed TEXT,counts64 BLOB,metadata TEXT);
      CREATE TABLE decisions(step INTEGER PRIMARY KEY,metadata TEXT);
    """)
    start = perf_counter(); sampler = CircuitCountSampler(sampling_spec); any_sampling_censored = False
    connections = IntegralConnections(backend, parent_cap=spec.parent_connection_cap)
    parent = QuantumCircuit(info["n_qubits"], name="raacs-hf")
    for wire in range(info["n_qubits"]):
        if info["hf_id"] >> wire & 1: parent.x(wire)
    actions = []; pilot_used = 0; minimum_final = int(np.ceil(spec.total_model_shots*spec.final_reserve_fraction))
    construction_cap = spec.total_model_shots-minimum_final; terminated = "STEP_CAP"; history = []

    def measure(circuit, acts, M, role, step, extra):
        nonlocal any_sampling_censored
        circuit = circuit.copy(); circuit.name = "raacs-"+digest(acts)[:16]
        ids, counts, meta = sampler.draw(circuit, M, rng_for("RAACS-physical-counts-v1", digest(config), role, step, extra, M)[1], len(acts), len(acts))
        any_sampling_censored |= meta["sample_metadata"]["status"] != "COMPLETED"
        sampling, _ = measurement_equivalent_circuit(circuit); raw = circuit_bytes(circuit); sha = hashlib.sha256(raw).hexdigest()
        db.execute("INSERT OR IGNORE INTO circuits VALUES(?,?,?,?,?)", (sha, zlib.compress(raw, 3), zlib.compress(circuit_bytes(sampling), 3),
                   json.dumps(acts), json.dumps(meta["sampling_resources"])))
        drawid = digest([digest(config), role, step, extra, M, sha])
        db.execute("INSERT INTO draws VALUES(?,?,?,?,?,?,?,?,?)", (drawid, sha, role, step, M, int(counts.sum()), meta["seed"],
                    pack_counts64(ids, counts), json.dumps(meta)))
        db.commit()
        obs = {int(x): int(n) for x, n in zip(ids, counts)}
        ci = observed_ci(backend, obs, info["hf_id"], info["nalpha"], info["nbeta"], spec.CI_cap)
        return ci, obs, meta, drawid, sha

    current, current_counts, firstmeta, drawid, parentid = measure(parent, actions, spec.probe_shots, "initial_pilot", 0, "HF")
    pilot_used += firstmeta["actual_raw_model_shots"]
    history.append({"circuit": parent.copy(), "actions": list(actions), "id": parentid, "gain": 0., "discovery": 1, "pilot_energy_Ha": current["energy_Ha"]})
    for step in range(1, spec.max_steps+1):
        if firstmeta["sample_metadata"]["status"] != "COMPLETED": terminated = "SAMPLING_CENSORED"; break
        if backend.matrix_entry_requests >= spec.maximum_named_H_requests: terminated = "CLASSICAL_WORK_CAP"; break
        proposals, proposal_scope = local_proposals(backend, connections, current, spec.target_cap, spec.candidate_bank_cap,
                                                    include_ci_gradients=spec.score == "ci_gradient_local")
        if not proposals: terminated = "NO_LOCAL_PROPOSALS"; break
        parent_sampling, _ = measurement_equivalent_circuit(parent)
        parent_resources = resources(parent_sampling, len(actions), len(actions))
        for p in proposals:
            trial = p["operator"].append(parent, np.pi/8); ts, _ = measurement_equivalent_circuit(trial)
            p["nominal_resources"] = resources(ts, len(actions)+1, len(actions)+1)
            gate_cost = 1+max(0, p["nominal_resources"]["cnot_count"]-parent_resources["cnot_count"])+.5*max(0, p["nominal_resources"]["cnot_depth"]-parent_resources["cnot_depth"])
            action_ids = set(p["operator"].action(dict(zip(current["selected_ids"], current["coefficients"]))))
            novelty = len(action_ids-set(current_counts)-{info["hf_id"]})/max(1, len(action_ids))
            priority = abs(p["ci_gradient_proxy"]) if spec.score == "ci_gradient_local" else p["residual_priority"]
            if spec.score in ("residual_cost", "residual_cost_diversity", "full_cost"): priority /= gate_cost
            if spec.score in ("residual_cost_diversity", "full_cost"): priority *= 1+novelty
            p["ranking_priority"] = priority
        ranked = sorted(proposals, key=lambda p: (-p["ranking_priority"], p["operator"]))[:spec.candidates_measured_per_step]
        angles = (np.pi/4,) if spec.angle_mode == "fixed_angle" else (-np.pi/4, -np.pi/8, np.pi/8, np.pi/4)
        measured = []; budget_exhausted = False
        for p in ranked:
            for angle in angles:
                if pilot_used+spec.probe_shots > construction_cap:
                    budget_exhausted = True; break
                op = p["operator"]; trial = op.append(parent, angle); acts = actions+[{**asdict(op), "theta": float(angle)}]
                before_requests = backend.matrix_entry_requests
                ci, obs, meta, did, cid = measure(trial, acts, spec.probe_shots, "candidate_pilot", step, [op.identity, float(angle)])
                pilot_used += meta["actual_raw_model_shots"]
                gain = current["energy_Ha"]-ci["energy_Ha"]
                dc = max(0, meta["sampling_resources"]["cnot_count"]-parent_resources["cnot_count"])
                work_increment = backend.matrix_entry_requests-before_requests
                utility = gain/(1+dc*spec.probe_shots/32+work_increment/64) if spec.score == "full_cost" else gain
                measured.append({"operator": op, "angle": float(angle), "ci": ci, "obs": obs, "meta": meta,
                                 "draw_id": did, "circuit_id": cid, "gain": gain, "utility": utility,
                                 "circuit": trial, "actions": acts, "work_entries": work_increment})
                if meta["sample_metadata"]["status"] != "COMPLETED": budget_exhausted = True; break
            if budget_exhausted: break
        eligible = [m for m in measured if m["gain"] > spec.pilot_acceptance_improvement_Ha and m["meta"]["sample_metadata"]["status"] == "COMPLETED"]
        winner = max(eligible, key=lambda m: (m["utility"], m["gain"], -m["meta"]["sampling_resources"]["cnot_count"],
                                             m["operator"].source, m["operator"].target, m["angle"])) if eligible else None
        decision = {"step": step, "proposal_scope": proposal_scope, "measured_candidates_and_angles": len(measured),
                    "all_measured_draw_ids": [m["draw_id"] for m in measured], "pilot_raw_model_shots": pilot_used,
                    "accepted": winner is not None, "winner_draw_id": winner["draw_id"] if winner else None,
                    "winner_gain_Ha_selection_biased": winner["gain"] if winner else None,
                    "classical_named_H_requests": backend.matrix_entry_requests, "FCI_used": False,
                    "acceptance_is_not_accuracy_certificate": True}
        db.execute("INSERT INTO decisions VALUES(?,?)", (step, json.dumps(decision))); db.commit()
        if winner is None:
            terminated = "PILOT_BUDGET_CAP" if budget_exhausted else "NO_PILOT_ENERGY_IMPROVEMENT"; break
        discovery = len(set(winner["obs"])-set(current_counts))
        parent = winner["circuit"]; actions = winner["actions"]; current = winner["ci"]; current_counts = winner["obs"]
        history.append({"circuit": parent.copy(), "actions": list(actions), "id": winner["circuit_id"],
                        "gain": winner["gain"], "discovery": discovery, "pilot_energy_Ha": current["energy_Ha"]})
        if budget_exhausted: terminated = "PILOT_BUDGET_CAP"; break
    remaining = spec.total_model_shots-pilot_used
    chosen = history[-spec.max_shallow_circuits:] if spec.final_allocation != "last_only" else history[-1:]
    weights = [1. if spec.final_allocation in ("uniform", "last_only") else h["discovery"] if spec.final_allocation == "discovery" else h["gain"] for h in chosen]
    allocations = allocation_counts(remaining, weights); merged = {}; final_ids = []; final_shots = 0
    for j, (h, M) in enumerate(zip(chosen, allocations)):
        if not M: continue
        ci, obs, meta, did, cid = measure(h["circuit"], h["actions"], M, "independent_final", len(actions), j)
        for x, n in obs.items(): merged[x] = merged.get(x, 0)+n
        final_shots += meta["actual_raw_model_shots"]; final_ids.append(did)
    final_ci = observed_ci(backend, merged, info["hf_id"], info["nalpha"], info["nbeta"], spec.CI_cap)
    result = {"status": "COMPLETED" if not any_sampling_censored and pilot_used+final_shots == spec.total_model_shots else "SAMPLING_CENSORED_PARTIAL_COUNTS",
              "construction_stop_reason": terminated, "problem": info, "algorithm": asdict(spec), "sampling": asdict(sampling_spec),
              "accepted_operators": actions, "accepted_steps": len(actions), "final_QSCI_energy_Ha": final_ci["energy_Ha"],
              "final_selected_ids": final_ci["selected_ids"], "actual_final_CI_dimension": final_ci["actual_dimension"],
              "final_energy_not_a_ground_state_error_certificate": True,
              "pilot_raw_model_shots": pilot_used, "independent_final_raw_model_shots": final_shots,
              "requested_total_model_shots": spec.total_model_shots, "actual_total_model_shots": pilot_used+final_shots,
              "final_draw_ids": final_ids, "final_circuit_allocations": [{"circuit_id": h["id"], "shots": M} for h, M in zip(chosen, allocations)],
              "circuit_switches_in_final_allocation": max(0, sum(n>0 for n in allocations)-1), "hardware_switch_latency_seconds": None,
              "classical_named_H_requests": backend.matrix_entry_requests, "connection_statistics": asdict(connections.statistics),
              "named_H_work_cap_checked_at_iteration_boundaries": True,
              "actual_wall_seconds": perf_counter()-start, "model_backend_builds": sampler.MPS_builds+sampler.exact_small_builds,
              "CI_state_freely_prepared": False, "FCI_used": False, "hardware_shots": 0,
              "primary_selection_support": "raw observed IDs+knownHF; proposed external IDs never padded",
              "full_Fock_statevector_used_by_algorithm": False,
              "full_Fock_statevector_used_in_backend": sampling_spec.backend == "exact_small",
              "whole_study_complete": False}
    db.close(); result["event_database_sha256"] = file_digest(output/"events.sqlite")
    save_json(output/"result.json", result)
    return result
