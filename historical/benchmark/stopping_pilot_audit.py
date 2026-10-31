"""Rebuild pilot frames from raw counts; no FCI accuracy label consumed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .chemistry import load_reference
from .online_stopping_rules import ObservedFrame, remaining_final_budget
from .stopping_pilot_bank import ci_features
from .study.core import unpack_pairs


def audit(root):
    root = Path(root); cfg = json.loads((root / "manifest.json").read_text()); source = Path(cfg["source"])
    for p, sha in cfg["implementation"].items():
        assert file_digest(ROOT / p) == sha
    frames_checked = episodes = total_shots = reused_ci = 0; minimum_final_remaining = min(cfg["final_budget_M_grid"])
    with threadpool_limits(limits=1):
        for case in cfg["case_ids"]:
            ref = load_reference(source / "references" / case); sector = ref["ids"]; valid = set(map(int, sector)); hf = ref["meta"]["hf_id"]
            h = ref["h"][sector][:, sector].tocsr(); cache = {}
            db = sqlite3.connect(f"file:{root/'counts'/case/'pilots.sqlite'}?mode=ro", uri=True)
            keys = {sha: cp for cp, sha in db.execute("SELECT id,sha FROM checkpoints")}
            for method in cfg["methods"]:
                source_cps = [json.loads(p.read_text()) for p in sorted((source / "checkpoints" / case / method).glob("step-*.json"))]
                last = source_cps[-1]["step"]
                expected_cps = [c for c in source_cps if c["step"] in cfg["probe_steps"] or c["step"] == last]
                for rep in range(cfg["repeats"]):
                    ep = json.loads((root / "episodes" / case / method / f"rep-{rep:03d}.json").read_text())
                    assert ep["trajectory_id"] == digest([ref["meta"]["request_digest"], method, rep, digest(cfg)])
                    assert not ep["FCI_used"] and ep["hardware_shots"] == 0
                    assert len(ep["frames"]) == len(ep["evidence"]) == len(expected_cps)
                    seen = set(); cumulative = {}; spent = 0
                    for c, raw_frame, ev in zip(expected_cps, ep["frames"], ep["evidence"]):
                        f = ObservedFrame(**raw_frame); assert f.step == c["step"] and ev["checkpoint_id"] == c["checkpoint_id"]
                        M = cfg["pilot_raw_shots_per_probe"]
                        row = db.execute("SELECT id,counts,seed FROM draws WHERE cp=? AND shots=? AND rep=?", (keys[c["checkpoint_id"]], M, rep)).fetchone()
                        assert row and row[0] == ev["draw_id"]
                        ids, counts = unpack_pairs(row[1]); assert int(counts.sum()) == M
                        rng, seed = rng_for(cfg["namespace"], case, method, c["checkpoint_id"], M, rep)
                        assert str(seed) == row[2] == ev["seed"]
                        assert file_digest(c["distribution"]) == ev["source_distribution_sha256"]
                        with np.load(c["distribution"]) as z: p = z["probability"]
                        rebuilt = rng.multinomial(M, p/p.sum())
                        assert np.array_equal(np.flatnonzero(rebuilt), ids) and np.array_equal(rebuilt[ids], counts)
                        current = {int(x) for x in ids if int(x) in valid and int(x) != hf}
                        assert len(current-seen) == f.new_observed_valid_nonHF; seen.update(current)
                        for x, n in zip(ids, counts):
                            if int(x) in valid and int(x) != hf: cumulative[int(x)] = cumulative.get(int(x), 0)+int(n)
                        selected = [hf]+[x for x, _ in sorted(cumulative.items(), key=lambda v: (-v[1], v[0]))][:cfg["R_cap"]-1]
                        key = tuple(sorted(selected))
                        if key not in cache:
                            cache[key] = ci_features(h, sector, selected, hf, cumulative, cfg["regularizer_Ha"])
                        else:
                            reused_ci += 1
                        fresh = cache[key]
                        assert sorted(selected) == ev["selected_ids"]
                        for field in ("qsci_energy_Ha", "external_residual_norm_Ha", "pt2_proxy_Ha"):
                            assert abs(getattr(f, field)-fresh[field]) < 1e-9
                        assert f.actual_dimension == len(selected)
                        spent += M; assert f.cumulative_pilot_raw_shots == spent
                        assert f.hf_raw_fraction == dict(zip(map(int, ids), map(int, counts))).get(hf, 0)/M
                        assert f.cnot_count == c["resources"]["cnot_count"] and f.cnot_depth == c["resources"]["cnot_depth"]
                        if c["step"] != last or ep["source_terminal_status"] != "CONVERGED":
                            assert f.current_pool_gradient_norm is None
                        frames_checked += 1; total_shots += M
                    assert spent == ep["pilot_raw_shots_for_full_bank_episode"]
                    minimum_final_remaining = min(minimum_final_remaining, remaining_final_budget(min(cfg["final_budget_M_grid"]), spent))
                    episodes += 1
            db.close()
    assert episodes == len(cfg["case_ids"])*len(cfg["methods"])*cfg["repeats"]
    result = {"status": "PILOT_BANK_VALIDATED", "episodes": episodes, "all_frames_rebuilt_from_counts": frames_checked,
              "raw_model_pilot_shots": total_shots, "minimum_final_remaining_at_smallest_M": minimum_final_remaining,
              "cached_identical_CI_spaces_in_audit": reused_ci, "FCI_used": False, "hardware_shots": 0,
              "actual_online_construction_savings_validated": False, "whole_study_complete": False,
              "audit_code_sha256": file_digest(__file__)}
    save_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
