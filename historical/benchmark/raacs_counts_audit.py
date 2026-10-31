"""All-record integrity/seed/budget/final-support audit, not a new physics run.

Works on censored counts and >16qubit molecular integral-only tasks. It never
redraws all pilots or invokes full-statevector verification. Quick quantum
circuit parity is a separate, explicitly stronger small-system qualification.
"""
from __future__ import annotations
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np
from qiskit import QuantumCircuit, qpy
from .common import ROOT, digest, file_digest, rng_for
from .local_sampler_primitives import LocalQE, observed_ci, unpack_counts64
from .resource_adaptive_sampler import integral_problem
from .ansatz import resources


def circuit_signature(qc):
    return {"n_qubits": qc.num_qubits, "global_phase": float(qc.global_phase),
            "gates": [(g.operation.name, [float(p) for p in g.operation.params],
                       [qc.find_bit(q).index for q in g.qubits]) for g in qc.data]}


def audit_cell(problem, folder, rebuild_final=True):
    folder = Path(folder); result = json.loads((folder/"result.json").read_text())
    config = json.loads((folder/"config.json").read_text()); algorithm = config["algorithm"]
    for name, sha in config["implementation"].items():
        path = ROOT/"benchmark"/name if name != "circuits.py" else ROOT/"vendor/ceo/adaptvqe/circuits.py"
        assert file_digest(path) == sha, f"Numerical implementation drift: {path}"
    backend, info = integral_problem(problem)
    assert info == config["problem"] == result["problem"]
    assert file_digest(folder/"events.sqlite") == result["event_database_sha256"]
    db = sqlite3.connect(f"file:{folder/'events.sqlite'}?mode=ro", uri=True)
    try:
        circuits = {}; final_cids = {a["circuit_id"] for a in result["final_circuit_allocations"] if a["shots"] > 0}
        circuit_count = 0; symbolic_rebuilds = 0
        for cid, prep_blob, sample_blob, actions_raw, resource_raw in db.execute("SELECT * FROM circuits"):
            raw = zlib.decompress(prep_blob); assert hashlib.sha256(raw).hexdigest() == cid
            prep = qpy.load(BytesIO(raw))[0]; sample = qpy.load(BytesIO(zlib.decompress(sample_blob)))[0]
            actions = json.loads(actions_raw)
            assert prep.num_qubits == sample.num_qubits == info["n_qubits"]
            assert not prep.num_clbits and not sample.num_clbits
            assert resources(sample, len(actions), len(actions)) == json.loads(resource_raw)
            if cid in final_cids and rebuild_final:
                reconstructed = QuantumCircuit(info["n_qubits"])
                for wire in range(info["n_qubits"]):
                    if info["hf_id"] >> wire & 1: reconstructed.x(wire)
                for a in actions:
                    op = LocalQE(tuple(a["source"]), tuple(a["target"]), a["n_qubits"])
                    reconstructed = op.append(reconstructed, a["theta"])
                sa, sb = circuit_signature(prep), circuit_signature(reconstructed)
                assert sa["n_qubits"] == sb["n_qubits"] and len(sa["gates"]) == len(sb["gates"])
                assert abs(np.exp(1j*sa["global_phase"])-np.exp(1j*sb["global_phase"])) < 1e-9
                for ga, gb in zip(sa["gates"], sb["gates"]):
                    assert ga[0] == gb[0] and ga[2] == gb[2] and np.allclose(ga[1], gb[1], atol=1e-10, rtol=0)
                symbolic_rebuilds += 1
            circuits[cid] = {"actions": actions, "sampling_sha256": hashlib.sha256(zlib.decompress(sample_blob)).hexdigest(),
                             "resources": json.loads(resource_raw)}
            circuit_count += 1
        draws = {}; pilot = final = requested = partial = 0; gate_shots = depth_shots = 0; wall = 0.; peak = 0
        for did, cid, role, step, M, n, seed, blob, metadata_raw in db.execute("SELECT * FROM draws"):
            c = circuits[cid]; meta = json.loads(metadata_raw)
            assert meta["sampling_circuit_sha256"] == c["sampling_sha256"]
            assert meta["sampling_resources"] == c["resources"]
            if role == "initial_pilot": extra = "HF"
            elif role == "candidate_pilot":
                a = c["actions"][-1]
                extra = [LocalQE(tuple(a["source"]), tuple(a["target"]), a["n_qubits"]).identity, a["theta"]]
            elif role == "independent_final":
                matches = [j for j, a in enumerate(result["final_circuit_allocations"]) if a["circuit_id"] == cid and a["shots"] == M]
                assert len(matches) == 1; extra = matches[0]
            else: raise AssertionError("Unknown draw role")
            assert str(rng_for("RAACS-physical-counts-v1", digest(config), role, step, extra, M)[1]) == seed == meta["seed"]
            assert did == digest([digest(config), role, step, extra, M, cid])
            ids, ns = unpack_counts64(blob)
            assert len(ids) == len(set(map(int, ids))) and all(0 <= int(x) < 1 << info["n_qubits"] for x in ids)
            assert all(int(x) > 0 for x in ns)
            assert int(ns.sum()) == n == meta["actual_raw_model_shots"] <= M
            assert M == meta["requested_model_shots"] and meta["hardware_shots"] == 0 and not meta["FCI_used"]
            completed = meta["sample_metadata"]["status"] == "COMPLETED"
            if completed: assert n == M
            else: partial += 1
            assert not meta["full_statevector_exposed_to_algorithm"]
            if config["sampling"]["backend"] == "mps": assert not meta["original_build_metadata"]["full_statevector_allocated"]
            draws[did] = {"role": role, "obs": dict(zip(map(int, ids), map(int, ns))), "circuit": cid}
            if role == "independent_final": final += n
            else: pilot += n
            requested += M; gate_shots += n*c["resources"]["cnot_count"]; depth_shots += n*c["resources"]["cnot_depth"]
            wall += meta["call_wall_seconds"]; peak = max(peak, meta["process_RSS_hwm_bytes"])
        decision_pilots = []; accepted = 0
        for step, raw in db.execute("SELECT * FROM decisions"):
            d = json.loads(raw); assert d["step"] == step and not d["FCI_used"]
            assert d["measured_candidates_and_angles"] == len(d["all_measured_draw_ids"])
            for did in d["all_measured_draw_ids"]: assert draws[did]["role"] == "candidate_pilot"
            decision_pilots.extend(d["all_measured_draw_ids"])
            if d["accepted"]:
                assert d["winner_draw_id"] in d["all_measured_draw_ids"]
                assert d["winner_gain_Ha_selection_biased"] > algorithm["pilot_acceptance_improvement_Ha"]; accepted += 1
        assert len(decision_pilots) == len(set(decision_pilots))
        assert set(decision_pilots) == {did for did, d in draws.items() if d["role"] == "candidate_pilot"}
        assert accepted == result["accepted_steps"] == len(result["accepted_operators"])
        merged = {}; assert len(result["final_draw_ids"]) == len(set(result["final_draw_ids"]))
        assert set(result["final_draw_ids"]) == {did for did, d in draws.items() if d["role"] == "independent_final"}
        for did in result["final_draw_ids"]:
            for x, n in draws[did]["obs"].items(): merged[x] = merged.get(x, 0)+n
        ci = observed_ci(backend, merged, info["hf_id"], info["nalpha"], info["nbeta"], algorithm["CI_cap"])
        assert ci["selected_ids"] == result["final_selected_ids"] and ci["actual_dimension"] == result["actual_final_CI_dimension"]
        assert abs(ci["energy_Ha"]-result["final_QSCI_energy_Ha"]) < 1e-9
        assert pilot == result["pilot_raw_model_shots"] and final == result["independent_final_raw_model_shots"]
        assert pilot+final == result["actual_total_model_shots"] <= result["requested_total_model_shots"]
        if result["status"] == "COMPLETED":
            assert partial == 0 and pilot+final == result["requested_total_model_shots"]
            assert final >= int(np.ceil(algorithm["total_model_shots"]*algorithm["final_reserve_fraction"]))
        assert not result["FCI_used"] and not result["CI_state_freely_prepared"] and result["hardware_shots"] == 0
        return {"n_qubits": info["n_qubits"], "status": result["status"], "raw_draws_checked": len(draws),
                "circuits_decoded_and_resources_checked": circuit_count, "final_circuits_symbolically_rebuilt": symbolic_rebuilds,
                "pilot_shots": pilot, "final_shots": final, "all_requested_draw_shots": requested, "partial_draws": partial,
                "energy_Ha": ci["energy_Ha"], "actual_dimension": ci["actual_dimension"], "CNOT_times_all_raw_shots": gate_shots,
                "depth_times_all_raw_shots": depth_shots, "all_sampling_call_wall_seconds": wall, "peak_process_RSS_bytes": peak,
                "accepted_steps": accepted, "final_resources": [circuits[cid]["resources"] for cid in sorted(final_cids)],
                "no_all_draw_reexecution_claim": True}
    finally: db.close()
