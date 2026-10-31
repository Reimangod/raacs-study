"""Paid-prefix stopping replay with independent final draws; never live growth.

The selector accepts observed frames only. This execution layer may resolve the
selected physical checkpoint after that decision. No FCI target is loaded here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
from time import perf_counter, time_ns

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

from .common import ROOT, digest, file_digest, resolve_artifact, rng_for, save_json
from .online_stopping_rules import ObservedFrame, OnlineStopRule, registered_rules, remaining_final_budget
from .study.core import CAPS, RitzCache, open_db, pack_pairs, ranks, rss_peak, unpack_pairs
from .study.reanalysis import frozen_inventory

NAMESPACE = "PhaseC-final-independent-evaluation-v1"
POLICIES = ("native_terminal",) + tuple(s.name for s in registered_rules())


def select_prefix(episode, name):
    """Return consumed prefix only; native endpoint is resolved by caller.

    Native/gradient cost no QSCI pilots. Saved current-point gradients exist
    only at independently verified native converged endpoints in this bank.
    Consequently gradient is an endpoint diagnostic, not an inferred early
    gradient trajectory. No unobserved gradient is replaced by zero.
    """
    if name == "native_terminal":
        return {"stop": False, "reason": "NATIVE_TERMINAL_NO_QSCI_PROBES", "pilot_shots": 0,
                "observed_probes": 0, "selected_checkpoint_id": None, "FCI_used": False}
    if name == "gradient":
        threshold = next(s.current_gradient_threshold for s in registered_rules() if s.name == name)
        for frame, evidence in zip(episode["frames"], episode["evidence"]):
            norm = frame["current_pool_gradient_norm"]
            if norm is not None and norm <= threshold:
                return {"stop": True, "reason": "VERIFIED_NATIVE_CURRENT_GRADIENT", "pilot_shots": 0,
                        "observed_probes": 0, "selected_checkpoint_id": evidence["checkpoint_id"],
                        "gradient_scope": "only verified native endpoint available; no early-gradient claim", "FCI_used": False}
        return {"stop": False, "reason": "NO_VERIFIED_GRADIENT_STOP_NATIVE_ENDPOINT", "pilot_shots": 0,
                "observed_probes": 0, "selected_checkpoint_id": None, "FCI_used": False}
    spec = next((s for s in registered_rules() if s.name == name), None)
    if spec is None:
        raise ValueError("Unregistered policy")
    rule = OnlineStopRule(spec)
    for raw, evidence in zip(episode["frames"], episode["evidence"]):
        frame = ObservedFrame(**raw)
        decision = rule.observe(frame)
        if decision is not None:
            return {**decision, "pilot_shots": frame.cumulative_pilot_raw_shots,
                    "selected_checkpoint_id": evidence["checkpoint_id"],
                    "pilot_classical_seconds": frame.cumulative_classical_seconds}
    last = rule.frames[-1] if rule.frames else None
    return {"stop": False, "reason": "OBSERVED_STREAM_ENDED_NOT_RULE_CONVERGENCE", "FCI_used": False,
            "pilot_shots": 0 if last is None else last.cumulative_pilot_raw_shots,
            "observed_probes": len(rule.frames), "selected_checkpoint_id": None,
            "pilot_classical_seconds": 0 if last is None else last.cumulative_classical_seconds}


def bind_manifest(root, source, bank, mode):
    cfg = json.loads((bank / "manifest.json").read_text())
    validation = json.loads((bank / "validation.json").read_text())
    if validation["status"] != "PILOT_BANK_VALIDATED" or Path(cfg["source"]).resolve() != source:
        raise RuntimeError("Audited pilot/source binding required")
    for path, sha in cfg["implementation"].items():
        if file_digest(ROOT / path) != sha:
            raise RuntimeError("Frozen pilot implementation drift")
    manifest = {"schema": "PhaseC-independent-paid-prefix-replay-v1", "mode": mode,
                "source": str(source), "pilot_bank": str(bank),
                "pilot_manifest_sha256": file_digest(bank / "manifest.json"),
                "pilot_validation_sha256": file_digest(bank / "validation.json"),
                "case_ids": cfg["case_ids"], "methods": cfg["methods"], "repeats": cfg["repeats"],
                "total_M": cfg["final_budget_M_grid"], "R_caps": list(CAPS), "primary_R": 64,
                "rules": [asdict(s) for s in registered_rules()], "policies": list(POLICIES),
                "namespace": NAMESPACE, "seed_key": "physical checkpoint / remaining M / replicate; no policy key",
                "final_support": "independent final counts only, knownHF insertion; no pilot support union",
                "pilot_cost_included": True, "FCI_used": False, "hardware_shots": 0,
                "actual_online_growth": False, "historical_full_growth_incurred": True,
                "free_floor_bytes": 3 * 2**30, "output_cap_bytes": 1024 * 2**20,
                "rss_boundary_bytes": 2 * 2**30,
                "spec_sha256": file_digest(ROOT / "study-v1/PHASE_C_REPLAY_EVALUATION_SPEC.md"),
                "implementation": {p: file_digest(ROOT / p) for p in (
                    "benchmark/stopping_replay.py", "benchmark/online_stopping_rules.py",
                    "benchmark/study/core.py", "benchmark/common.py")}}
    target = root / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise RuntimeError("Immutable replay binding mismatch; use a successor directory")
    if mode == "standard":
        quick = ROOT / "results/PhaseC-replay-quick-v1/validation.json"
        if not quick.exists() or json.loads(quick.read_text())["status"] != "INDEPENDENT_REPLAY_VALIDATED":
            raise RuntimeError("Qualify quick replay before standard")
    save_json(target, manifest)
    return manifest


def run_case(root, case, config):
    source, bank = Path(config["source"]), Path(config["pilot_bank"])
    refdir = source / "references" / case
    meta = json.loads((refdir / "reference.json").read_text())
    # Only physical IDs and H enter numerical final evaluation; no FCI arrays.
    with np.load(refdir / "reference.npz") as z:
        valid_ids = z["ids"].copy()
    h = load_npz(refdir / "hamiltonian.npz")
    db = open_db(root / "readouts" / case / "readouts.sqlite")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS decisions(
        id TEXT PRIMARY KEY, episode TEXT NOT NULL, policy TEXT NOT NULL,
        total_M INTEGER NOT NULL, rep INTEGER NOT NULL, pool TEXT NOT NULL,
        cp INTEGER NOT NULL, draw INTEGER NOT NULL, native_draw INTEGER NOT NULL,
        pilot_shots INTEGER NOT NULL, metadata TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS policy_pairs(
        decision TEXT NOT NULL, cap INTEGER NOT NULL, actual_d INTEGER NOT NULL,
        selected_subspace INTEGER NOT NULL, native_subspace INTEGER NOT NULL,
        PRIMARY KEY(decision,cap));
    """)
    cache = RitzCache(h, valid_ids, meta["request_digest"], db)
    probabilities = {}; checkpoint_by_sha = {}; terminal = {}; status = {}
    for method in config["methods"]:
        folder = source / "checkpoints" / case / method
        cps = [json.loads(p.read_text()) for p in sorted(folder.glob("step-*.json"))]
        terminal[method] = cps[-1]
        status[method] = json.loads((folder / "status.json").read_text())["status"]
        for c in cps:
            checkpoint_by_sha[c["checkpoint_id"]] = c
            db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                       (c["checkpoint_id"], method, c["step"], 1, json.dumps(c)))
    cp_ids = dict(db.execute("SELECT sha,id FROM checkpoints"))

    def final_draw(c, M, rep):
        sha = c["checkpoint_id"]; cp = cp_ids[sha]
        row = db.execute("SELECT id,counts,seed FROM draws WHERE cp=? AND shots=? AND rep=?", (cp, M, rep)).fetchone()
        rng, seed = rng_for(NAMESPACE, sha, M, rep)
        if row:
            draw, blob, recorded = row
            if str(seed) != recorded:
                raise RuntimeError("Final draw identity drift")
            ids, counts = unpack_pairs(blob)
        else:
            if sha not in probabilities:
                path = resolve_artifact(source, c["distribution"])
                if file_digest(path) != c["distribution_sha256"]:
                    raise RuntimeError("Source distribution mismatch")
                with np.load(path) as z:
                    probabilities[sha] = z["probability"].copy()
            start = perf_counter(); p = probabilities[sha]
            counts_full = rng.multinomial(M, p / p.sum())
            ids = np.flatnonzero(counts_full); counts = counts_full[ids]
            draw = db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)",
                              (cp, M, rep, str(seed), pack_pairs(ids, counts), perf_counter()-start)).lastrowid
        order = ranks(ids, counts, valid_ids, meta["hf_id"], [sha, M, rep])["frequency"]
        for R in config["R_caps"]:
            if db.execute("SELECT 1 FROM outcomes WHERE draw=? AND cap=? AND rule='frequency'", (draw, R)).fetchone():
                continue
            sid, _, hit, sec = cache.solve(order[:R])
            db.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?)", (draw, R, "frequency", sid, int(hit), sec))
        return draw, order

    for method in config["methods"]:
        native = terminal[method]
        for rep in range(config["repeats"]):
            ep_path = bank / "episodes" / case / method / f"rep-{rep:03d}.json"
            ep = json.loads(ep_path.read_text())
            if ep["source_terminal_status"] != status[method]:
                raise RuntimeError("Native terminal identity mismatch")
            choices = {name: select_prefix(ep, name) for name in config["policies"]}
            for total in config["total_M"]:
                native_draw, native_order = final_draw(native, total, rep)
                for name, choice in choices.items():
                    did = digest([ep["trajectory_id"], name, total, digest(config)])
                    if db.execute("SELECT 1 FROM decisions WHERE id=?", (did,)).fetchone():
                        continue
                    selected = checkpoint_by_sha[choice["selected_checkpoint_id"]] if choice["selected_checkpoint_id"] else native
                    if selected["method"] != method:
                        raise RuntimeError("Selected checkpoint belongs to another physical trajectory")
                    remaining = remaining_final_budget(total, choice["pilot_shots"])
                    draw, order = final_draw(selected, remaining, rep)
                    selected_resources, native_resources = selected["resources"], native["resources"]
                    record = {"decision": choice, "selected_checkpoint_id": selected["checkpoint_id"],
                              "source_checkpoint_sha256": digest(selected), "episode_file_sha256": file_digest(ep_path),
                              "source_terminal_status": status[method], "selected_step": selected["step"],
                              "native_last_saved_step": native["step"], "remaining_final_shots": remaining,
                              "selected_resources": selected_resources, "native_resources": native_resources,
                              "selected_source_prefix_seconds": selected.get("trajectory_seconds"),
                              "historical_full_growth_seconds_incurred": native.get("trajectory_seconds"),
                              "selected_gate_shots": remaining*selected_resources["cnot_count"],
                              "paid_pilot_gate_shots": sum(e["full_raw_shots"]*f["cnot_count"] for f, e in
                                  zip(ep["frames"][:choice["observed_probes"]], ep["evidence"][:choice["observed_probes"]])),
                              "hardware_shots": 0, "FCI_used": False, "actual_growth_avoided": False}
                    db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                               (did, ep["trajectory_id"], name, total, rep, method, cp_ids[selected["checkpoint_id"]],
                                draw, native_draw, choice["pilot_shots"], json.dumps(record, allow_nan=False)))
                    for R in config["R_caps"]:
                        d = min(R, len(order), len(native_order))
                        sa = cache.solve(order[:d])[0]; sb = cache.solve(native_order[:d])[0]
                        db.execute("INSERT INTO policy_pairs VALUES(?,?,?,?,?)", (did, R, d, sa, sb))
                db.commit()
            if shutil.disk_usage(root).free < config["free_floor_bytes"] or rss_peak() > config["rss_boundary_bytes"]:
                raise RuntimeError("Boundary resource guard; committed replay prefix preserved")
        print(f"independent paid-prefix replay {case} {method}", flush=True)
    count = db.execute("SELECT count(*) FROM decisions").fetchone()[0]
    expected = len(config["methods"])*config["repeats"]*len(config["total_M"])*len(config["policies"])
    if count != expected:
        raise RuntimeError("Incomplete policy decision coverage")
    draws, shots = db.execute("SELECT count(*),sum(shots) FROM draws").fetchone()
    status_case = {"status": "INDEPENDENT_READOUT_COMPLETE", "case_id": case, "decisions": count,
                   "unique_physical_final_draws": draws, "unique_final_raw_model_shots": shots,
                   "peak_process_RSS_bytes": rss_peak(), "hardware_shots": 0,
                   "FCI_used": False, "actual_online_growth": False}
    save_json(root / "case-status" / f"{case}.json", status_case)
    db.close()
    return status_case


def run(root, bank, mode):
    root, bank = Path(root).resolve(), Path(bank).resolve()
    source = Path(json.loads((bank / "manifest.json").read_text())["source"]).resolve()
    if root == source or source in root.parents or root == bank or bank in root.parents:
        raise ValueError("Protect frozen source and pilot bank")
    root.mkdir(exist_ok=True, parents=True)
    with (root / "run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = bind_manifest(root, source, bank, mode)
        attempt = root / "attempts" / str(time_ns()); start = perf_counter()
        save_json(attempt / "started.json", {"config_digest": digest(config), "status": "STARTED"})
        frozen_inventory(source, root)
        rows = []
        for case in config["case_ids"]:
            if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > config["output_cap_bytes"]:
                raise RuntimeError("Replay output cap at case boundary")
            t = perf_counter(); row = run_case(root, case, config)
            save_json(attempt / f"{case}.json", {"case_wall_seconds_this_attempt": perf_counter()-t, **row})
            rows.append(row)
            save_json(root / "execution-status.json", {"status": "RUNNING", "completed_cases": len(rows),
                      "planned_cases": len(config["case_ids"]), "whole_study_complete": False})
        result = {"status": "INDEPENDENT_REPLAY_EXECUTED_PENDING_AUDIT", "completed_cases": len(rows),
                  "policy_decisions": sum(r["decisions"] for r in rows),
                  "unique_physical_final_draws": sum(r["unique_physical_final_draws"] for r in rows),
                  "unique_final_raw_model_shots": sum(r["unique_final_raw_model_shots"] for r in rows),
                  "pilot_collection_raw_model_shots": json.loads((bank/"validation.json").read_text())["raw_model_pilot_shots"],
                  "hardware_shots": 0, "FCI_used_in_selection": False, "actual_online_growth": False,
                  "whole_study_complete": False}
        save_json(root / "execution-status.json", result)
        save_json(attempt / "finished.json", {**result, "attempt_wall_seconds": perf_counter()-start})
        return result


def rows_from_db(root):
    """Whitelisted absolute-energy calibration rows; no FCI labels/report fields."""
    import sqlite3
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text()); all_rows = []
    for case in cfg["case_ids"]:
        db = sqlite3.connect(f"file:{root/'readouts'/case/'readouts.sqlite'}?mode=ro", uri=True)
        frame = pd.read_sql_query("""SELECT d.id,d.episode,d.policy,d.total_M,d.rep,d.pool,d.pilot_shots,d.metadata,
            o.cap,s.energy AS energy_Ha,n.energy AS native_energy_Ha,s.dimension AS actual_d,
            ns.energy AS matched_energy_Ha,nn.energy AS matched_native_energy_Ha,p.actual_d AS matched_d
            FROM decisions d JOIN outcomes o ON o.draw=d.draw AND o.rule='frequency'
            JOIN subspaces s ON s.id=o.subspace
            JOIN outcomes no ON no.draw=d.native_draw AND no.cap=o.cap AND no.rule='frequency'
            JOIN subspaces n ON n.id=no.subspace
            JOIN policy_pairs p ON p.decision=d.id AND p.cap=o.cap
            JOIN subspaces ns ON ns.id=p.selected_subspace JOIN subspaces nn ON nn.id=p.native_subspace""", db)
        for r in frame.to_dict("records"):
            m = json.loads(r.pop("metadata"))
            r.update(case_id=case, molecule=case.split("_R")[0], selected_step=m["selected_step"],
                     native_last_saved_step=m["native_last_saved_step"],
                     cnot_count=m["selected_resources"]["cnot_count"], cnot_depth=m["selected_resources"]["cnot_depth"],
                     native_cnot_count=m["native_resources"]["cnot_count"], native_cnot_depth=m["native_resources"]["cnot_depth"],
                     remaining_final_shots=m["remaining_final_shots"],
                     historical_full_growth_seconds_incurred=m["historical_full_growth_seconds_incurred"],
                     selected_source_prefix_seconds=m["selected_source_prefix_seconds"],
                     total_deployment_gate_shots=m["selected_gate_shots"]+m["paid_pilot_gate_shots"],
                     rule_stop=m["decision"]["stop"], reason=m["decision"]["reason"],
                     pilot_classical_seconds=m["decision"].get("pilot_classical_seconds", 0.))
            all_rows.append(r)
        db.close()
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--output"); p.add_argument("--pilot-bank")
    a = p.parse_args()
    print(json.dumps(run(a.output or ROOT/f"results/PhaseC-replay-{a.mode}-v1",
                         a.pilot_bank or ROOT/f"results/PhaseC-pilot-bank-{a.mode}-v1", a.mode)))
