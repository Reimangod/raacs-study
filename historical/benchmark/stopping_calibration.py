"""Training-only, molecule-cluster policy choice from absolute QSCI energies.

This module does not load a reference wavefunction or reference energy. Final
ground-state comparisons belong to the separate outer report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .common import digest, file_digest, save_json
from .stopping_replay import POLICIES, rows_from_db

WHITELIST = ("id", "case_id", "molecule", "pool", "policy", "total_M", "rep", "cap", "energy_Ha", "native_energy_Ha",
             "cnot_count", "cnot_depth", "native_cnot_count", "native_cnot_depth", "pilot_shots")


def choose_training_rule(records, heldout_molecule):
    if not records or any(set(r) != set(WHITELIST) for r in records):
        raise ValueError("Calibration accepts the explicit energy/resource/paid-shot whitelist only")
    if any(r["molecule"] == heldout_molecule for r in records):
        raise ValueError("Heldout molecular cluster leaked into policy selection")
    f = pd.DataFrame(records)
    if f.pool.nunique() != 1 or f.total_M.nunique() != 1 or set(f.cap) != {64}:
        raise ValueError("Calibrate one pool/M at the registered primary R64")
    if set(f.policy) != set(POLICIES):
        raise ValueError("Preserve every registered rule and native fallback")
    f["regret_mHa"] = 1000*(f.energy_Ha-f.native_energy_Ha)
    for metric, native, target in (("cnot_count", "native_cnot_count", "cnot_ratio"), ("cnot_depth", "native_cnot_depth", "depth_ratio")):
        if (f[native] <= 0).any():
            raise ValueError("Undefined native-resource ratio; do not silently replace")
        f[target] = f[metric]/f[native]
    numeric = ["regret_mHa", "cnot_ratio", "depth_ratio", "pilot_shots"]
    if not np.isfinite(f[numeric].to_numpy()).all():
        raise ValueError("Nonfinite training evidence")
    # First average repeats, then geometries within each molecular cluster.
    cases = f.groupby(["policy", "molecule", "case_id"], as_index=False)[numeric].mean()
    clusters = cases.groupby(["policy", "molecule"], as_index=False)[numeric].mean()
    score = []
    for policy, g in clusters.groupby("policy"):
        row = {"policy": policy, "mean_molecular_regret_mHa": float(g.regret_mHa.mean()),
               "worst_molecular_mean_regret_mHa": float(g.regret_mHa.max()),
               "mean_CNOT_ratio": float(g.cnot_ratio.mean()), "mean_depth_ratio": float(g.depth_ratio.mean()),
               "mean_pilot_shots": float(g.pilot_shots.mean())}
        row["eligible"] = row["mean_molecular_regret_mHa"] <= .5+1e-10 and row["worst_molecular_mean_regret_mHa"] <= 1.6+1e-10
        score.append(row)
    eligible = [s for s in score if s["eligible"]]
    winner = min(eligible, key=lambda s: (s["mean_CNOT_ratio"], s["mean_depth_ratio"], s["mean_pilot_shots"], s["policy"]))
    return {"selected_policy": winner["policy"], "training_molecules": sorted(set(f.molecule)),
            "training_case_ids": sorted(set(f.case_id)), "training_input_digest": digest(sorted(records, key=lambda r: r["id"])),
            "training_input_records": len(records), "training_scores": score, "FCI_used": False,
            "threshold_scope": "registered training-regret management thresholds; not absolute accuracy certificates"}


def calibrate(root):
    root = Path(root); validation = json.loads((root/"validation.json").read_text())
    assert validation["status"] == "INDEPENDENT_REPLAY_VALIDATED"
    f = rows_from_db(root)
    directory = root/"LOMO-v1"; directory.mkdir(exist_ok=True)
    binding = {"selector_sha256": file_digest(__file__), "manifest_sha256": file_digest(root/"manifest.json"),
               "validation_sha256": file_digest(root/"validation.json"), "whitelist": list(WHITELIST)}
    path = directory/"binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise RuntimeError("Calibration implementation/binding drift")
    save_json(path, binding)
    # This bank contains old molecular identities: LOMO is not newly unseen chemistry.
    selections = []; heldout_rows = []
    for molecule in sorted(f.molecule.unique()):
        for pool in sorted(f.pool.unique()):
            for M in sorted(f.total_M.unique()):
                train = f[(f.molecule != molecule)&(f.pool == pool)&(f.total_M == M)&(f.cap == 64)]
                records = train[list(WHITELIST)].to_dict("records")
                choice = choose_training_rule(records, molecule)
                result = {"heldout_molecule": molecule, "pool": pool, "total_M": int(M), **choice}
                save_json(directory/"folds"/f"{molecule}-{pool}-M{M}.json", result)
                pd.DataFrame(records).to_parquet(directory/"folds"/f"{molecule}-{pool}-M{M}-training.parquet", index=False)
                selections.append(result)
                test = f[(f.molecule == molecule)&(f.pool == pool)&(f.total_M == M)&(f.policy == choice["selected_policy"])].copy()
                test["fold_selected_policy"] = choice["selected_policy"]
                test["training_input_digest"] = choice["training_input_digest"]
                heldout_rows.append(test)
    heldout = pd.concat(heldout_rows, ignore_index=True)
    f.to_parquet(root/"all-policy-independent-readouts.parquet", index=False)
    heldout.to_parquet(directory/"heldout-readouts.parquet", index=False)
    heldout["regret_mHa"] = 1000*(heldout.energy_Ha-heldout.native_energy_Ha)
    heldout["CNOT_reduction_percent"] = 100*(1-heldout.cnot_count/heldout.native_cnot_count)
    heldout["depth_reduction_percent"] = 100*(1-heldout.cnot_depth/heldout.native_cnot_depth)
    metrics = ["regret_mHa", "CNOT_reduction_percent", "depth_reduction_percent", "pilot_shots"]
    case_means = heldout.groupby(["pool", "total_M", "cap", "molecule", "case_id"], as_index=False)[metrics].mean()
    mol_means = case_means.groupby(["pool", "total_M", "cap", "molecule"], as_index=False)[metrics].mean()
    mol_means.to_csv(directory/"heldout-molecular-means.csv", index=False)
    cluster_rows = []
    for keys, group in mol_means.groupby(["pool", "total_M", "cap"]):
        row = dict(zip(("pool", "total_M", "cap"), keys)); row["molecular_clusters"] = len(group)
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[metric+"_t90_halfwidth"] = float(t.ppf(.95, len(group)-1)*group[metric].std(ddof=1)/np.sqrt(len(group))) if len(group)>1 else None
        cluster_rows.append(row)
    pd.DataFrame(cluster_rows).to_csv(directory/"cluster-summary.csv", index=False)
    status = {"status": "TRAINING_ONLY_LOMO_EXECUTED", "folds": len(selections), "heldout_rows": len(heldout),
              "molecular_clusters": int(f.molecule.nunique()), "FCI_used_in_calibration": False,
              "new_quantum_draws_for_calibration": 0, "bank_reuse_not_multiplied_by_folds": True,
              "uncertainty_scope": "descriptive t90 across molecular cluster means; folds share training data; only6standardclusters",
              "historically_studied_molecular_identities": True, "actual_online_growth": False, "whole_study_complete": False}
    save_json(directory/"status.json", status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(calibrate(p.parse_args().output)))
