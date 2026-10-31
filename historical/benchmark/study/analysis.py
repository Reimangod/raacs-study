from __future__ import annotations

import json
import math
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr, t

from ..common import save_json
from ..qsci import CHEMICAL_ACCURACY_HA


class Welford:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0., 0.

    def step(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def finalize(self):
        return math.sqrt(max(0., self.m2 / (self.n - 1))) if self.n > 1 else None


def analyze(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["source"])
    summaries, metrics, joint, individual, statuses = [], [], [], [], []
    for path in sorted((root / "finite").glob("*/readouts.sqlite")):
        case = path.parent.name
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.create_aggregate("std_welford", 1, Welford)
        ref = json.loads((source / "references" / case / "reference.json").read_text())
        expression = "(s.energy - ?) * 1000"
        sql = f"""SELECT c.sha,c.method,c.step,c.primary_record,d.shots,o.cap,o.rule,
        count(*),avg({expression}),std_welford({expression}),avg(s.dimension),
        sum(CASE WHEN s.energy-? <= ? THEN 1 ELSE 0 END),avg(o.cache_hit),
        avg(o.lookup_seconds),max(s.rss_hwm),avg(s.projected_bytes)
        FROM outcomes o JOIN draws d ON o.draw=d.id JOIN checkpoints c ON d.cp=c.id
        JOIN subspaces s ON o.subspace=s.id
        GROUP BY c.id,d.shots,o.cap,o.rule"""
        for row in db.execute(sql, (ref["fci_energy"], ref["fci_energy"], ref["fci_energy"], CHEMICAL_ACCURACY_HA)):
            sha, method, step, primary, M, R, rule, n, mean, std, dimension, success, hit, seconds, rss, nbytes = row
            half = float(t.ppf(.95, n - 1) * std / math.sqrt(n)) if std is not None else None
            lo = float(beta.ppf(.05, success, n - success + 1)) if success else 0.
            hi = float(beta.ppf(.95, success + 1, n - success)) if success < n else 1.
            summaries.append({"case_id": case, "molecule": case.split("_R")[0], "method": method, "step": step,
                              "checkpoint_id": sha, "primary_record": bool(primary), "shots": M, "R_cap": R, "selection_rule": rule,
                              "repeats": n, "planned_repeats": manifest["repeats"], "complete_group": n == manifest["repeats"],
                              "mean_error_mHa": mean, "std_error_mHa": std, "ci90_low_mHa": mean-half if half is not None else None,
                              "ci90_high_mHa": mean+half if half is not None else None, "mean_dimension": dimension,
                              "successes": success, "success_probability_estimate": success/n,
                              "success_probability_ci90_low": lo, "success_probability_ci90_high": hi,
                              "cache_hit_fraction": hit, "actual_lookup_or_solve_seconds_mean": seconds,
                              "process_peak_rss_bytes": rss, "mean_projected_matrix_storage_bytes": nbytes,
                              "uncertainty_scope": "pointwise_draw_uncertainty; NOT molecule-level or simultaneous"})
        for method, step, sha, primary, payload in db.execute("SELECT c.method,c.step,c.sha,c.primary_record,a.payload FROM analytics a JOIN checkpoints c ON a.cp=c.id"):
            d = json.loads(zlib.decompress(payload))
            prefix = {"case_id": case, "method": method, "step": step, "checkpoint_id": sha, "primary_record": bool(primary)}
            metrics.extend({**prefix, **r} for r in d["rows"])
            joint.extend({**prefix, **r} for r in d["joint"])
            individual.extend({**prefix, **r} for r in d["individual"])
        db.close()
        status_file = path.parent / "status.json"
        statuses.append(json.loads(status_file.read_text()) if status_file.exists() else {"case": case, "status": "INCOMPLETE_NO_TERMINAL"})
    directory = root / "analysis"
    directory.mkdir(exist_ok=True)
    for name, rows in (("readout-summary", summaries), ("discovery", metrics), ("joint-m90", joint), ("individual-m90", individual)):
        if rows:
            pd.DataFrame(rows).to_parquet(directory / f"{name}.parquet", index=False)
    if summaries:
        df = pd.DataFrame(summaries)
        primary = df[df.primary_record & df.complete_group & (df.selection_rule == "frequency")]
        shots_to_target = []
        for (case, method, step, R), g in primary.groupby(["case_id", "method", "step", "R_cap"]):
            qualified = g[g.success_probability_estimate >= .9].sort_values("shots")
            certified = g[g.success_probability_ci90_low >= .9].sort_values("shots")
            shots_to_target.append({"case_id": case, "method": method, "step": step, "R_cap": R,
                                    "smallest_tested_M_with_observed_p90": int(qualified.iloc[0].shots) if len(qualified) else None,
                                    "smallest_tested_M_with_ci_lower90": int(certified.iloc[0].shots) if len(certified) else None,
                                    "status": "OBSERVED_THRESHOLD" if len(qualified) else "RIGHT_CENSORED_ON_TESTED_GRID",
                                    "maximum_M_tested": int(g.shots.max()),
                                    "note": "p-hat is not certified success; rank-truncation need not be monotone in shots"})
        pd.DataFrame(shots_to_target).to_parquet(directory / "shots-to-observed-p90.parquet", index=False)
        met = pd.DataFrame(metrics)
        cor = []
        if len(met):
            joined = primary[primary.R_cap == 32].merge(met, on=["case_id", "method", "step", "checkpoint_id", "shots", "primary_record"])
            for (molecule, method, M), g in joined.groupby(["case_id", "method", "shots"]):
                for name in ("nonhf_weighted_coverage", "expected_unique_valid", "hf_probability", "recall_top3", "recall_top10"):
                    usable = g[[name, "mean_error_mHa"]].dropna()
                    rho = float(spearmanr(usable[name], usable.mean_error_mHa).statistic) if len(usable) > 2 and usable[name].nunique() > 1 and usable.mean_error_mHa.nunique() > 1 else None
                    cor.append({"case_id": molecule, "method": method, "shots": M, "metric": name, "rho": rho,
                                "checkpoints": len(usable), "scope": "descriptive_within_trajectory; correlated_checkpoints_not_independent_samples"})
            pd.DataFrame(cor).to_parquet(directory / "discovery-energy-correlations.parquet", index=False)
    status = {"finite_case_statuses": statuses, "readout_summary_groups": len(summaries), "analytic_metric_rows": len(metrics),
              "joint_m90_groups": len(joint), "hardware_quantum_shots": 0, "performance_claim": "NOT_ESTABLISHED"}
    save_json(directory / "status.json", status)
    return status
