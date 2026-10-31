"""Recompute LOMO scores from the exported training whitelist independently."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import digest, file_digest, save_json
from .stopping_calibration import WHITELIST


def audit(root):
    root = Path(root); folder = root/"LOMO-v1"
    allrows = pd.read_parquet(root/"all-policy-independent-readouts.parquet")
    heldout = pd.read_parquet(folder/"heldout-readouts.parquet")
    expected_heldout = set(); selections = Counter(); scorechecks = 0; folds = 0
    for path in sorted((folder/"folds").glob("*.json")):
        record = json.loads(path.read_text()); molecule = record["heldout_molecule"]
        f = pd.read_parquet(path.with_name(path.stem+"-training.parquet"))
        assert set(f.columns) == set(WHITELIST) and molecule not in set(f.molecule)
        assert f.id.is_unique and set(f.cap) == {64}
        assert set(f.pool) == {record["pool"]} and set(f.total_M) == {record["total_M"]}
        expected = allrows[(allrows.molecule != molecule)&(allrows.pool == record["pool"])&
                           (allrows.total_M == record["total_M"])&(allrows.cap == 64)][list(WHITELIST)]
        a = sorted(f.to_dict("records"), key=lambda r: r["id"])
        b = sorted(expected.to_dict("records"), key=lambda r: r["id"])
        assert a == b and digest(a) == record["training_input_digest"]
        assert sorted(set(f.case_id)) == record["training_case_ids"]
        assert sorted(set(f.molecule)) == record["training_molecules"]
        scores = []
        for policy, group in f.groupby("policy"):
            molecule_means = []
            for _, cluster in group.groupby("molecule"):
                case_means = []
                for _, g in cluster.groupby("case_id"):
                    case_means.append([np.mean(1000*(g.energy_Ha-g.native_energy_Ha)),
                                       np.mean(g.cnot_count/g.native_cnot_count),
                                       np.mean(g.cnot_depth/g.native_cnot_depth), np.mean(g.pilot_shots)])
                molecule_means.append(np.mean(case_means, axis=0))
            mm = np.array(molecule_means); mean = mm.mean(axis=0)
            eligible = mean[0] <= .5+1e-10 and mm[:, 0].max() <= 1.6+1e-10
            saved = next(s for s in record["training_scores"] if s["policy"] == policy)
            assert np.allclose(mean, [saved[k] for k in ("mean_molecular_regret_mHa", "mean_CNOT_ratio", "mean_depth_ratio", "mean_pilot_shots")], atol=1e-10, rtol=0)
            assert abs(mm[:, 0].max()-saved["worst_molecular_mean_regret_mHa"]) < 1e-10
            assert bool(eligible) == saved["eligible"]
            if eligible: scores.append((mean[1], mean[2], mean[3], policy))
            scorechecks += 1
        winner = min(scores)[-1]
        assert winner == record["selected_policy"]
        selections[winner] += 1; folds += 1
        test = allrows[(allrows.molecule == molecule)&(allrows.pool == record["pool"])&
                       (allrows.total_M == record["total_M"])&(allrows.policy == winner)]
        expected_heldout.update(zip(test.id, test.cap))
    assert expected_heldout == set(zip(heldout.id, heldout.cap)) and len(expected_heldout) == len(heldout)
    result = {"status": "LOMO_SELECTION_AND_HELDOUT_BINDING_VALIDATED", "folds_rebuilt": folds,
              "rule_scores_independently_rebuilt": scorechecks, "heldout_readouts_verified": len(heldout),
              "selected_policies": dict(selections), "FCI_in_training_whitelist": False,
              "new_quantum_evaluations": 0, "actual_online_growth": False, "whole_study_complete": False,
              "auditor_sha256": file_digest(__file__)}
    save_json(folder/"independent-validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(audit(p.parse_args().output)))
