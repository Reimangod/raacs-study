"""Fresh FCI-free pilot observations on already-grown trajectories.

This bank supports controlled replay only. It cannot establish that historical
growth was avoided, or replace actual stopped growth on fresh geometries.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter, time_ns
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .chemistry import load_reference
from .online_stopping_rules import ObservedFrame
from .study.core import ADAPT, open_db, pack_pairs, unpack_pairs, rss_peak

PROBES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)


def ci_features(h, sector_ids, selected_ids, hf_id, observed_counts, regularizer_Ha=1e-4):
    """Small saved-H backend, with no FCI vector/energy argument or padding."""
    index = {int(x): i for i, x in enumerate(sector_ids)}
    selected = sorted(set(int(x) for x in selected_ids))
    if not selected or any(x not in index for x in selected):
        raise ValueError("CI basis must contain only observed valid IDs/knownHF")
    if int(hf_id) not in index or any(x != int(hf_id) and observed_counts.get(x, 0) <= 0 for x in selected):
        raise ValueError("Unobserved configuration padding is forbidden")
    positions = [index[x] for x in selected]
    hs = h[positions][:, positions].toarray()
    if np.max(abs(hs-hs.conj().T)) > 1e-10:
        raise ValueError("Non-Hermitian physical CI matrix")
    values, vectors = eigh(hs, subset_by_index=[0, 0], check_finite=False)
    energy, coefficients = float(values[0]), vectors[:, 0]
    residual = np.asarray(h[:, positions] @ coefficients).reshape(-1)
    residual[positions] -= energy*coefficients
    residual[positions] = 0  # The external norm, not Ritz's near-zero internal residual.
    diagonal = np.asarray(h.diagonal()).real
    denominators = energy-diagonal
    safe = np.where(denominators >= 0, 1., -1.)*np.maximum(abs(denominators), regularizer_Ha)
    pt2 = float(np.sum(abs(residual)**2/safe).real)
    return {"qsci_energy_Ha": energy, "actual_dimension": len(selected), "selected_ids": selected,
            "external_residual_norm_Ha": float(np.linalg.norm(residual)), "pt2_proxy_Ha": pt2,
            "residual_scope": "exact external residual of saved small-sector H; regularized EN-like PT2 proxy, not error bound",
            "regularizer_Ha": regularizer_Ha,
            "external_near_zero_denominators": int(np.count_nonzero((abs(denominators) < regularizer_Ha) & (abs(residual) > 0)))}


def build(root, source, mode):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents:
        raise ValueError("Protect immutable source")
    root.mkdir(exist_ok=True, parents=True)
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source / "references").iterdir() if p.is_dir())
    methods = ["ceo_ovp", "ceo_mvp"] if mode == "quick" else list(ADAPT)
    config = {"schema": "PhaseC-retrospective-independent-pilot-bank-v1", "source": str(source), "mode": mode,
              "case_ids": cases, "methods": methods, "repeats": 4 if mode == "quick" else 20,
              "pilot_raw_shots_per_probe": 64, "probe_steps": list(PROBES), "R_cap": 64,
              "include_natural_terminal_probe": True, "counts_selection": "frequency rank cumulative observed valid pilot IDs; knownHF occupies1slot",
              "final_budget_M_grid": [1000, 10000, 100000], "final_readout_is_not_in_this_bank": True,
              "namespace": "PhaseC-independent-pilot-observations-v1", "regularizer_Ha": 1e-4,
              "free_floor_bytes": 3*2**30, "output_cap_bytes": 512*2**20, "rss_boundary_bytes": 2*2**30,
              "implementation": {p: file_digest(ROOT / p) for p in ("benchmark/stopping_pilot_bank.py", "benchmark/online_stopping_rules.py")},
              "FCI_used": False, "scope": "fresh observations on HISTORICAL trajectory; NOT actual online construction saving"}
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError("Pilot config/code drift; create successor")
    if mode == "standard":
        quick = ROOT / "results/PhaseC-pilot-bank-quick-v1/validation.json"
        if not quick.exists() or json.loads(quick.read_text())["status"] != "PILOT_BANK_VALIDATED":
            raise RuntimeError("Quick observation/count validation required")
    save_json(manifest, config)
    checks = {(r["case_id"], r["method"]): r for r in json.loads((source / "validation/trajectory-checks.json").read_text())}
    episodes = []; attempt_id = str(time_ns()); total_started = perf_counter()
    save_json(root / "attempts" / attempt_id / "started.json", {"config_digest": digest(config), "status": "STARTED", "hardware_shots": 0})
    import fcntl
    with (root / "run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in cases:
            if shutil.disk_usage(root).free < config["free_floor_bytes"] or rss_peak() > config["rss_boundary_bytes"]:
                raise RuntimeError("Boundary resource guard; preserve pilot prefix")
            if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > config["output_cap_bytes"]:
                raise RuntimeError("Pilot output boundary cap")
            start = perf_counter(); ref = load_reference(source / "references" / case)
            ids = ref["ids"]; h = ref["h"][ids][:, ids].tocsr(); valid = set(map(int, ids)); hf = ref["meta"]["hf_id"]
            setup = {"reference_load_and_sector_projection_seconds": perf_counter()-start,
                       "source_setup_not_rebuilt": True, "full_FCI_not_recomputed": True, "backend_scope": "small saved sparse sector H"}
            save_json(root / "attempts" / attempt_id / f"{case}-setup.json", setup)
            if not (root / "shared-setup" / f"{case}.json").exists():
                save_json(root / "shared-setup" / f"{case}.json", setup)
            cache = {}
            db = open_db(root / "counts" / case / "pilots.sqlite")
            for method in methods:
                folder = source / "checkpoints" / case / method
                checkpoints = [json.loads(p.read_text()) for p in sorted(folder.glob("step-*.json"))]
                final_step = checkpoints[-1]["step"]
                probes = [c for c in checkpoints if c["step"] in PROBES or c["step"] == final_step]
                status = json.loads((folder / "status.json").read_text())["status"]
                for c in probes:
                    db.execute("INSERT OR IGNORE INTO checkpoints(sha,method,step,primary_record,metadata) VALUES(?,?,?,?,?)",
                               (c["checkpoint_id"], method, c["step"], 1, "{}"))
                cp_indices = dict(db.execute("SELECT sha,id FROM checkpoints"))
                for rep in range(config["repeats"]):
                    trajectory_id = digest([ref["meta"]["request_digest"], method, rep, digest(config)])
                    path = root / "episodes" / case / method / f"rep-{rep:03d}.json"
                    if path.exists():
                        episode = json.loads(path.read_text()); assert episode["trajectory_id"] == trajectory_id
                        episodes.append(episode); continue
                    cumulative = {}; seen = set(); spent = 0; elapsed = 0.; frames = []; evidence = []
                    for cp in probes:
                        M = config["pilot_raw_shots_per_probe"]; key = cp_indices[cp["checkpoint_id"]]
                        rng, seed = rng_for(config["namespace"], case, method, cp["checkpoint_id"], M, rep)
                        existing = db.execute("SELECT id,counts,seconds,seed FROM draws WHERE cp=? AND shots=? AND rep=?", (key, M, rep)).fetchone()
                        if existing:
                            draw, blob, seconds, recorded_seed = existing; assert recorded_seed == str(seed)
                            obs_ids, obs_counts = unpack_pairs(blob)
                        else:
                            if file_digest(cp["distribution"]) != cp["distribution_sha256"]:
                                raise RuntimeError("Source probability changed")
                            with np.load(cp["distribution"]) as z: p = z["probability"]
                            start = perf_counter(); counts = rng.multinomial(M, p/p.sum()); obs_ids = np.flatnonzero(counts); obs_counts = counts[obs_ids]
                            seconds = perf_counter()-start; blob = pack_pairs(obs_ids, obs_counts)
                            db.execute("INSERT INTO draws(cp,shots,rep,seed,counts,seconds) VALUES(?,?,?,?,?,?)", (key, M, rep, str(seed), blob, seconds))
                            draw = db.execute("SELECT last_insert_rowid()").fetchone()[0]; db.commit()
                        assert int(obs_counts.sum()) == M
                        this_valid = {int(x) for x in obs_ids if int(x) in valid and int(x) != hf}
                        new = len(this_valid-seen); seen.update(this_valid)
                        for x, count in zip(obs_ids, obs_counts):
                            if int(x) in valid and int(x) != hf: cumulative[int(x)] = cumulative.get(int(x), 0)+int(count)
                        selected = [hf]+[x for x, _ in sorted(cumulative.items(), key=lambda v: (-v[1], v[0]))][:config["R_cap"]-1]
                        identity = tuple(sorted(selected)); start = perf_counter(); hit = identity in cache
                        if not hit:
                            cache[identity] = ci_features(h, ids, selected, hf, cumulative, config["regularizer_Ha"])
                        features = cache[identity]; ci_seconds = perf_counter()-start
                        spent += M; elapsed += seconds+ci_seconds
                        gradient = checks[(case, method)]["terminal_pool_gradient_norm"] if cp["step"] == final_step and status == "CONVERGED" else None
                        frame = ObservedFrame(ref["meta"]["request_digest"], trajectory_id, cp["step"], features["qsci_energy_Ha"],
                                              features["actual_dimension"], new, spent, float(dict(zip(map(int, obs_ids), map(int, obs_counts))).get(hf, 0)/M),
                                              cp["resources"]["cnot_count"], cp["resources"]["cnot_depth"], elapsed,
                                              gradient, features["external_residual_norm_Ha"], features["pt2_proxy_Ha"], features["residual_scope"])
                        frames.append(frame.__dict__)
                        evidence.append({"checkpoint_id": cp["checkpoint_id"], "draw_id": draw, "seed": str(seed), "selected_ids": features["selected_ids"],
                                         "source_distribution_sha256": cp["distribution_sha256"], "CI_cache_hit": hit, "CI_lookup_or_solve_seconds": ci_seconds,
                                         "source_historical_trajectory_seconds": cp.get("trajectory_seconds"), "full_raw_shots": M,
                                         "valid_raw_shots": sum(int(n) for x, n in zip(obs_ids, obs_counts) if int(x) in valid),
                                         "external_near_zero_denominators": features["external_near_zero_denominators"]})
                    episode = {"case_id": case, "method": method, "rep": rep, "trajectory_id": trajectory_id, "problem_id": ref["meta"]["request_digest"],
                               "source_terminal_status": status, "source_last_saved_step": final_step, "frames": frames, "evidence": evidence,
                               "pilot_raw_shots_for_full_bank_episode": spent, "FCI_used": False, "hardware_shots": 0,
                               "scope": "historical checkpoint replay bank; policy may consume only the observed prefix; native endpoint is not future-feature input"}
                    save_json(path, episode); episodes.append(episode)
            db.close(); print("FCI-free historical pilot bank completed "+case, flush=True)
    status = {"status": "PILOT_OBSERVATIONS_GENERATED_PENDING_AUDIT", "episodes": len(episodes),
              "frames": sum(len(e["frames"]) for e in episodes), "new_model_raw_shots": sum(e["pilot_raw_shots_for_full_bank_episode"] for e in episodes),
              "FCI_used": False, "hardware_shots": 0, "actual_online_construction_executed": False, "whole_study_complete": False}
    save_json(root / "generation-status.json", status)
    save_json(root / "attempts" / attempt_id / "finished.json", {**status, "actual_attempt_wall_seconds": perf_counter()-total_started})
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--output")
    p.add_argument("--source", default=str(ROOT / "results/standard-v1")); a = p.parse_args()
    print(json.dumps(build(a.output or ROOT / f"results/PhaseC-pilot-bank-{a.mode}-v1", a.source, a.mode)))
