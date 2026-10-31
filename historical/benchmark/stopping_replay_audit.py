"""Raw final-draw rebuilding and paid-prefix reconciliation for Phase C replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, resolve_artifact, rng_for, save_json
from .stopping_replay import NAMESPACE, POLICIES, select_prefix
from .study.core import unpack_pairs


def audit(root):
    root = Path(root); cfg = json.loads((root/"manifest.json").read_text())
    bank, source = Path(cfg["pilot_bank"]), Path(cfg["source"])
    for path, sha in cfg["implementation"].items():
        assert file_digest(ROOT/path) == sha
    assert file_digest(bank/"manifest.json") == cfg["pilot_manifest_sha256"]
    assert file_digest(bank/"validation.json") == cfg["pilot_validation_sha256"]
    ndecisions = ndraws = shots = noutcomes = independent_solves = episode_checks = pairs_checked = 0
    maximum_solver_difference = 0.; minimum_remaining = min(cfg["total_M"])
    with threadpool_limits(limits=1):
        for case in cfg["case_ids"]:
            meta = json.loads((source/"references"/case/"reference.json").read_text())
            with np.load(source/"references"/case/"reference.npz") as z: sector = set(map(int, z["ids"]))
            hf = meta["hf_id"]
            db = sqlite3.connect(f"file:{root/'readouts'/case/'readouts.sqlite'}?mode=ro", uri=True)
            cp_map = {}; cp_json = {}
            for cp, sha, method, step, raw in db.execute("SELECT id,sha,method,step,metadata FROM checkpoints"):
                actual = json.loads((source/"checkpoints"/case/method/f"step-{step:04d}.json").read_text())
                assert actual["checkpoint_id"] == sha and actual == json.loads(raw)
                cp_map[cp] = (sha, actual); cp_json[sha] = actual
            orders = {}; draw_meta = {}; p_cache = {}
            for draw, cp, M, rep, seed, blob in db.execute("SELECT id,cp,shots,rep,seed,counts FROM draws"):
                sha, actual = cp_map[cp]
                if cp not in p_cache:
                    path = resolve_artifact(source, actual["distribution"])
                    assert file_digest(path) == actual["distribution_sha256"]
                    with np.load(path) as z: p_cache[cp] = z["probability"].copy()
                p = p_cache[cp]; rng, fresh_seed = rng_for(NAMESPACE, sha, M, rep)
                expected = rng.multinomial(M, p/p.sum()); ids, counts = unpack_pairs(blob)
                assert seed == str(fresh_seed) and int(counts.sum()) == M
                assert np.array_equal(ids, np.flatnonzero(expected)) and np.array_equal(counts, expected[ids])
                order = [hf]+[int(x) for x, n in sorted(zip(ids, counts), key=lambda a: (-a[1], a[0])) if int(x) in sector and int(x) != hf]
                orders[draw] = order; draw_meta[draw] = (cp, M, rep)
                ndraws += 1; shots += M
            subspaces = {sid: (np.frombuffer(zlib.decompress(raw), dtype="<i4").tolist(), d, E)
                         for sid, raw, d, E in db.execute("SELECT id,ids,dimension,energy FROM subspaces")}
            for draw, R, sid in db.execute("SELECT draw,cap,subspace FROM outcomes WHERE rule='frequency'"):
                ids, d, _ = subspaces[sid]
                assert ids == sorted(orders[draw][:R]) and d == len(ids) and set(ids) <= sector
                noutcomes += 1
            # Eigenvalues are independently recomputed on a deterministic spread
            # of saved spaces. Identity/count/selection checks above cover all.
            all_sid = sorted(subspaces)
            selected_sid = [all_sid[i] for i in sorted(set(np.linspace(0, len(all_sid)-1, min(5, len(all_sid)), dtype=int)))]
            h = load_npz(source/"references"/case/"hamiltonian.npz")
            for sid in selected_sid:
                ids, _, E = subspaces[sid]; hs = h[ids][:, ids].toarray()
                fresh = float(eigh(hs, subset_by_index=(0, 0), eigvals_only=True)[0])
                difference = abs(fresh-E); assert difference < 1e-8
                maximum_solver_difference = max(maximum_solver_difference, difference); independent_solves += 1
            episodes = {}
            for method in cfg["methods"]:
                for rep in range(cfg["repeats"]):
                    path = bank/"episodes"/case/method/f"rep-{rep:03d}.json"
                    ep = json.loads(path.read_text()); episodes[ep["trajectory_id"]] = (ep, path)
                    episode_checks += 1
            decisions_per_episode = {}
            for did, epid, policy, total, rep, method, cp, draw, native_draw, pilot, raw in db.execute("SELECT * FROM decisions"):
                record = json.loads(raw); ep, path = episodes[epid]
                expected = select_prefix(ep, policy)
                assert expected == record["decision"] and file_digest(path) == record["episode_file_sha256"]
                assert pilot == expected["pilot_shots"] and total-pilot == draw_meta[draw][1] > 0
                assert draw_meta[draw][0] == cp and draw_meta[draw][2] == rep
                assert draw_meta[native_draw][1:] == (total, rep)
                assert cp_map[cp][1]["method"] == method == ep["method"] and rep == ep["rep"]
                assert not record["FCI_used"] and record["hardware_shots"] == 0 and not record["actual_growth_avoided"]
                assert record["selected_checkpoint_id"] == cp_map[cp][0]
                if expected["selected_checkpoint_id"] is not None:
                    assert cp_map[cp][0] == expected["selected_checkpoint_id"]
                else:
                    assert cp_map[cp][1]["step"] == ep["source_last_saved_step"]
                if policy in ("native_terminal", "gradient"):
                    assert pilot == expected["observed_probes"] == 0
                else:
                    n = expected["observed_probes"]
                    assert pilot == sum(e["full_raw_shots"] for e in ep["evidence"][:n])
                    assert record["paid_pilot_gate_shots"] == sum(e["full_raw_shots"]*f["cnot_count"] for e, f in zip(ep["evidence"][:n], ep["frames"][:n]))
                pair_rows = list(db.execute("SELECT cap,actual_d,selected_subspace,native_subspace FROM policy_pairs WHERE decision=?", (did,)))
                assert len(pair_rows) == len(cfg["R_caps"])
                for R, d, sa, sb in pair_rows:
                    assert d == min(R, len(orders[draw]), len(orders[native_draw]))
                    assert subspaces[sa][0] == sorted(orders[draw][:d]) and subspaces[sb][0] == sorted(orders[native_draw][:d])
                    pairs_checked += 1
                minimum_remaining = min(minimum_remaining, total-pilot)
                decisions_per_episode[epid] = decisions_per_episode.get(epid, 0)+1
                ndecisions += 1
            assert len(decisions_per_episode) == len(episodes)
            assert all(n == len(POLICIES)*len(cfg["total_M"]) for n in decisions_per_episode.values())
            db.close()
    result = {"status": "INDEPENDENT_REPLAY_VALIDATED", "raw_draws_rebuilt": ndraws,
              "unique_final_raw_model_shots": shots, "all_policy_decisions_reconciled": ndecisions,
              "pilot_episodes_verified": episode_checks, "all_common_R_bases_verified": noutcomes,
              "all_actual_d_pairs_verified": pairs_checked, "fresh_independent_Ritz_solves": independent_solves,
              "maximum_fresh_Ritz_difference_Ha": maximum_solver_difference, "minimum_remaining_final_shots": minimum_remaining,
              "pilot_collection_shots_are_separate": True, "FCI_used_in_selection": False, "hardware_shots": 0,
              "actual_online_growth_validated": False, "whole_study_complete": False,
              "audit_code_sha256": file_digest(__file__)}
    save_json(root/"validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
