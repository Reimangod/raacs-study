"""Post-audit error/success/cluster diagnostics, not confirmatory superiority."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import file_digest, rng_for, save_json


def family(method):
    return re.sub(r"-s\d+-(exact_small|mps)$", r"-\1", method)


def wilson(k, n, z=1.6448536269514722):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0., c-h), min(1., c+h)


def cluster_interval(values, identity):
    """Equal molecule weighting; six clusters are not a large-population proof."""
    values = np.asarray(values, float)
    if len(values) < 2: return None, None
    rng, _ = rng_for("RAACS-molecule-bootstrap-report-v1", identity)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return float(np.quantile(means, .05)), float(np.quantile(means, .95))


def report(output):
    output = Path(output); cfg = json.loads((output/"manifest.json").read_text())
    validation = json.loads((output/"validation.json").read_text())
    assert validation["status"] == "RAACS_INDEPENDENT_COMPARISON_COUNTS_AND_PAIRS_VALIDATED"
    destination = output/"derived-report-v1"; destination.mkdir(exist_ok=True)
    summary = []; pair_summary = []; missing = []; coverage = []
    for case in cfg["cases"]:
        meta = json.loads((Path(cfg["source"])/"references"/case/"reference.json").read_text())
        molecule = meta["case"]["molecule"]
        frame = pd.read_parquet(output/"readouts"/case/"individual-readouts.parquet")
        frame["error_mHa"] = (frame.energy-meta["fci_energy"])*1000
        frame["chemical_success"] = frame.error_mHa <= 1.5936014376
        for (method, scope, M, R), g in frame.groupby(["method", "scope", "total_M", "cap"]):
            k = int(g.chemical_success.sum()); n = len(g); lo, hi = wilson(k, n)
            se = float(g.error_mHa.std(ddof=1)/np.sqrt(n)) if n > 1 else None
            summary.append({"case_id": case, "molecule": molecule, "distance_A": meta["case"]["distance"],
                "method": method, "family": family(method), "scope": scope, "M": M, "R": R,
                "mean_error_mHa": float(g.error_mHa.mean()), "readout_t90_halfwidth_mHa": float(t.ppf(.95, n-1)*se) if n>1 else None,
                "repeats": n, "success_fraction": k/n, "success_Wilson90_low": lo, "success_Wilson90_high": hi,
                "mean_actual_d": float(g.dimension.mean()), "mean_final_shots": float(g.final_shots.mean()),
                "build_model_shots": None if g.build_shots.isna().all() else float(g.build_shots.iloc[0]),
                "correlation_energy_recovery_percent": 100*(1-float(g.error_mHa.mean())/(1000*(meta["rhf_energy"]-meta["fci_energy"]))),
                "scope_of_uncertainty": "conditional independent final draws, not construction/molecular uncertainty"})
        for row in json.loads((output/"readouts"/case/"source-records.json").read_text()):
            coverage.append({"case_id": case, **row})
        miss = pd.read_csv(output/"readouts"/case/"budget-missingness.csv"); miss["case_id"] = case; missing.append(miss)
        db = sqlite3.connect(f"file:{output/'readouts'/case/'readouts.sqlite'}?mode=ro", uri=True)
        pairs = pd.read_sql_query("""SELECT p.method,p.total_M,p.scope,p.rep,p.cap,p.baseline,p.actual_d,
              (a.energy-b.energy)*1000 AS RAACS_minus_baseline_mHa FROM pairs p
              JOIN subspaces a ON a.id=p.RAACS_subspace JOIN subspaces b ON b.id=p.baseline_subspace""", db); db.close()
        for (method, baseline, scope, M, R), g in pairs.groupby(["method", "baseline", "scope", "total_M", "cap"]):
            n = len(g); se = float(g.RAACS_minus_baseline_mHa.std(ddof=1)/np.sqrt(n)) if n > 1 else None
            pair_summary.append({"case_id": case, "molecule": molecule, "method": method, "family": family(method),
                                 "baseline": baseline, "scope": scope, "M": M, "R": R,
                                 "mean_RAACS_minus_baseline_mHa": float(g.RAACS_minus_baseline_mHa.mean()),
                                 "conditional_readout_t90_halfwidth_mHa": float(t.ppf(.95, n-1)*se) if n>1 else None,
                                 "mean_paired_actual_d": float(g.actual_d.mean()), "readout_repeats": n})
    summary = pd.DataFrame(summary); paired = pd.DataFrame(pair_summary)
    summary.to_csv(destination/"all-cell-common-R-summary.csv", index=False)
    paired.to_csv(destination/"all-cell-actual-d-pairs.csv", index=False)
    pd.concat(missing, ignore_index=True).to_csv(destination/"all-budget-missingness.csv", index=False)
    pd.DataFrame(coverage).to_csv(destination/"baseline-coverage-including-failures.csv", index=False)
    ninety = []
    for (case, method, scope, R), g in summary.groupby(["case_id", "method", "scope", "R"]):
        hits = g[g.success_fraction >= .9].sort_values("M")
        ninety.append({"case_id": case, "method": method, "scope": scope, "R": R,
                       "empirical90_first_tested_M": int(hits.M.iloc[0]) if len(hits) else None,
                       "status": "EMPIRICAL_GRID_HIT_NOT_CONFIDENCE_CERTIFICATE" if len(hits) else "RIGHT_CENSORED_AT_TESTED_GRID",
                       "not_continuous_M_threshold": True})
    pd.DataFrame(ninety).to_csv(destination/"empirical-shots-to90.csv", index=False)
    primary = "RAACS:full_cost-qsci_local_grid-last_only-exact_small" if cfg["mode"] == "standard" else "RAACS:full_cost-exact_small"
    # Average construction seeds WITHIN case first, then distances WITHIN molecule.
    by_case = summary[summary.family == primary].groupby(["case_id", "molecule", "scope", "M", "R"], as_index=False).mean(numeric_only=True)
    clusters = []; diagnostic_rows = []
    for baseline in cfg["paired_baselines"]:
        bs = summary[summary.method == baseline]
        for (scope, M, R), g in by_case.groupby(["scope", "M", "R"]):
            b = bs[(bs.scope == scope) & (bs.M == M) & (bs.R == R)]
            merged = g.merge(b[["case_id", "mean_error_mHa"]], on="case_id", suffixes=("_RAACS", "_baseline"))
            if not len(merged): continue
            merged["delta"] = merged.mean_error_mHa_RAACS-merged.mean_error_mHa_baseline
            values = merged.groupby("molecule").delta.mean(); low, high = cluster_interval(values.to_numpy(), [baseline, scope, int(M), int(R)])
            clusters.append({"baseline": baseline, "scope": scope, "M": M, "R": R, "molecular_clusters": len(values),
                             "available_conditions": len(merged), "equal_molecule_mean_delta_mHa": float(values.mean()),
                             "exploratory_cluster_bootstrap90_low": low, "exploratory_cluster_bootstrap90_high": high})
            diagnostic_rows.extend({"molecule": mol, "baseline": baseline, "scope": scope, "M": M, "R": R, "delta_mHa": float(v)} for mol,v in values.items())
    pd.DataFrame(clusters).to_csv(destination/"primary-molecule-cluster-diagnostics.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(destination/"primary-molecule-means.csv", index=False)
    figures = []
    for molecule in sorted(summary.molecule.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for method in [primary]+cfg["paired_baselines"]:
            subset = summary[(summary.molecule == molecule) & (summary.family == method) & (summary.R == 64)]
            if not len(subset): continue
            for ax, scope in zip(axes, cfg["readout_scopes"]):
                g = subset[subset.scope == scope].groupby("M").mean(numeric_only=True)
                if len(g): ax.plot(g.index, g.mean_error_mHa, "o-", ms=3, label=method)
                ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=.01); ax.set_xlabel("Model shots M"); ax.set_ylabel("Mean QSCI error (mHa)")
                ax.set_title(scope); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6)
        fig.suptitle(f"{molecule}: R64; geometry/seed means descriptive, not independent molecules")
        fig.tight_layout(rect=(0, .24, 1, .94)); name = f"independent-comparison-{molecule}"
        for suffix in ("png", "svg", "pdf"): fig.savefig(destination/f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    text = f"""# RAACS independent conditional comparison\n\n
Mode: {cfg['mode']}. Cases: {len(cfg['cases'])}. Final-readout repetitions:
{cfg['repeats']} per saved sampler/budget. This protocol is post-construction
exploratory; it is not a preregistered confirmatory superiority test.

All {validation['all_physical_draw_records_checked']:,} raw draw records checked;
{validation['new_draws_rebuilt']:,} new independent draws rebuilt and
{validation['reused_draws_checked_against_original_acquisition']:,} old baseline
draws matched byte-for-byte to original acquisition. Reuse is not new evidence.
All {validation['all_actual_d_pairs_checked']:,} actual-d pairs and budget links
are checked. Hardware shots0. No FCI selection/stopping; FCI used only here.

## Read the two cost scopes separately

`readout_only` holds final readout M equal, with historical construction cost
separate. `known_build_plus_readout` subtracts known construction pilots from
M. Unknown VQE/pool QPU construction cost is not0 and produces no certified
total-budget comparison. Budget-infeasible native growth is reported, not
silently removed or rerun using a cheaper proxy. CSVs preserve missingness.

Common-R and pairwise actual-d equality answer different questions. No global
minimum is forced by HF-only. Lower final energy can be due to a larger useful
CI subspace; the actual-d diagnostic is provided rather than claiming identical
classical work. R, method and M ablations do not multiply the physical shots.

## Uncertainty and interpretation

Readout intervals condition on the trained/fixed circuit. They do not include
construction stochasticity or system-to-system uncertainty. Construction seeds
are averaged within case; five geometries are averaged within each molecule.
Equal-molecule bootstrap diagnostics use at most six clusters, not30 independent
molecules, and are exploratory, small-sample and not multiplicity-adjusted.
No best-method/seed is picked. The empirical90% grid table is not a confidence
certificate or an interpolated minimum-shot theorem; misses stay right-censored.

All840 RAACS construction results and baseline failures remain in their source
cohorts. OriginalB2 initialization failures/SqDRIFT/SCI reports remain separate
and must enter final synthesis; this cohort does not redefine their scope.
This small<=16q evaluator can expand probabilities for qualification only.
The practical20–44q algorithm never does so. Full-study completion, universal
quantum advantage and optimized-ansatz superiority remain unproven.
"""
    (destination/"REPORT.md").write_text(text)
    result = {"status": "RAACS_INDEPENDENT_COMPARISON_REPORTED", "common_R_summary_rows": len(summary),
              "actual_d_summary_rows": len(paired), "figures": figures, "hardware_shots": 0,
              "reporter_sha256": file_digest(__file__), "whole_study_complete": False}
    save_json(destination/"manifest.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
