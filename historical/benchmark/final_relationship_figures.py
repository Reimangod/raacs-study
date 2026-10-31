"""Read-only missing relationship figures, using completed A/B1/B4 evidence.

No optimizer, eigensolver, new state preparation, random draw or reference solve.
In particular the oracle is an ANALYTIC diagnostic of saved FCI Born weights,
not a practical sampler or an upper bound on discovery quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("ceo_ovp", "ceo_mvp", "qeb", "qubit", "gsd")
COLORS = dict(zip(POOLS, ("#2563a6", "#b05c23", "#278452", "#b04266", "#7857a6")))
SHOTS = (10, 25, 50, 100, 250, 500, 1000, 10000, 100000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_nonhf_discovery(target_ids, weights, hf, probabilities, shots):
    """Exact marginal expectation, no product-of-events/joint-discovery claim."""
    ids = np.asarray(target_ids, dtype=np.int64)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(ids) != len(w) or len(ids) != len(p) or len(set(map(int, ids))) != len(ids):
        raise ValueError("Unique aligned configuration IDs required")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or abs(w.sum() - 1) > 1e-8:
        raise ValueError("Saved target weights not normalized")
    if not np.all(np.isfinite(p)) or np.any(p < -1e-13) or np.any(p > 1+1e-13) or p.sum() > 1+1e-8:
        raise ValueError("Invalid aligned Born probabilities")
    if int(shots) != shots or shots < 0:
        raise ValueError("Nonnegative integer shots required")
    keep = ids != int(hf)
    mass = float(w[keep].sum())
    if mass <= 1e-12:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = -np.expm1(int(shots) * np.log1p(-np.clip(p[keep], 0, 1))) if shots else np.zeros(keep.sum())
    return float(np.dot(w[keep], observed) / mass)


def adjacent_improvements(group):
    """Energy gain after addition AND full reoptimization, not a causal effect."""
    g = group.sort_values("step")
    if g.step.duplicated().any():
        raise ValueError("Ambiguous checkpoint per step")
    rows = []
    for a, b in zip(list(g.itertuples())[:-1], list(g.itertuples())[1:]):
        if b.step != a.step + 1:
            raise ValueError("Missing intermediate checkpoint; do not silently bridge")
        rows.append({"case_id": b.case_id, "method": b.method, "step": int(b.step),
                     "checkpoint_id": b.checkpoint_id, "prior_checkpoint_id": a.checkpoint_id,
                     "delta_cnot": b.cnot-a.cnot, "delta_depth": b.depth-a.depth,
                     "qsci_gain_mHa": a.mean_error_mHa-b.mean_error_mHa,
                     "interpretation": "ADDITION_AND_FULL_REOPTIMIZATION_ASSOCIATION"})
    return rows


def primary_readouts(df, M=1000, R=32):
    return df[df.primary_record & df.complete_group & (df.selection_rule == "frequency") &
              (df.shots == M) & (df.R_cap == R)].copy()


def build(output, mode="standard"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        existing = json.loads((output/"manifest.json").read_text())
        if existing["script_sha256"] != sha(__file__):
            raise RuntimeError("Completed derivative is version bound; use a successor directory")
        for name, value in existing["source_sha256"].items():
            if sha(ROOT/name) != value:
                raise RuntimeError("Bound source changed")
        for name, value in existing["output_sha256"].items():
            if sha(output/name) != value:
                raise RuntimeError("Completed derivative output changed")
        return existing
    start = time.perf_counter()
    source = ROOT/"results/standard-v1"
    A = ROOT/"results/sampler-study-standard-v1"
    B1 = ROOT/"results/sampler-baselines-B1-standard-v1"
    B4 = ROOT/"results/random-B4-standard-v1"
    inputs = {}
    def bind(path):
        path = Path(path)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return path
    def read(path):
        return json.loads(bind(path).read_text())
    inventory = pd.DataFrame(read(A/"checkpoint-inventory.json"))
    rows = pd.read_parquet(bind(A/"analysis/readout-summary.parquet"))
    discovery = pd.read_parquet(bind(A/"analysis/discovery.parquet"))
    B1rows = pd.read_csv(bind(B1/"readout-summary.csv"))
    refs = {p.parent.name: read(p) for p in sorted((source/"references").glob("*/reference.json"))}
    cases = [c for c in refs if mode == "standard" or c in ("H2_R0.7", "H4_R1.1")]
    cp = inventory[inventory.primary & inventory.case_id.isin(cases)].copy()
    curves = primary_readouts(rows).merge(cp, on=["case_id", "method", "step", "checkpoint_id"], validate="one_to_one")
    curves["vqe_error_mHa"] = [(r.vqe_energy-refs[r.case_id]["fci_energy"])*1000 for r in curves.itertuples()]
    curves["QSCI_evaluation_namespace"] = "sampler-study-A-independent-counts-v1"
    # All numerical plotted values are exported without rounding.
    cols = ["case_id", "method", "step", "checkpoint_id", "vqe_energy", "vqe_error_mHa", "mean_error_mHa", "ci90_low_mHa", "ci90_high_mHa", "cnot", "depth", "QSCI_evaluation_namespace"]
    curves[cols].to_csv(output/"growth-curves.csv", index=False)
    transitions = []
    for (_, _), group in curves[curves.method.isin(POOLS)].groupby(["case_id", "method"]):
        transitions.extend(adjacent_improvements(group))
    transition = pd.DataFrame(transitions)
    # Keep actual added-operator identity, not only the increment amount.
    added = cp[["checkpoint_id", "added_group"]].copy()
    added["added_group_json"] = added.added_group.map(json.dumps)
    transition = transition.merge(added[["checkpoint_id", "added_group_json"]], on="checkpoint_id", validate="one_to_one")
    transition.to_csv(output/"addition-reoptimization-transitions.csv", index=False)
    primary_discovery = discovery[discovery.primary_record & discovery.case_id.isin(cases) & (discovery.shots == 1000)].copy()
    primary_discovery.to_csv(output/"HF-vs-nonHF-coverage.csv", index=False)
    outputs = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    def save(fig, stem, note):
        # This is small derivative reporting, not a lower production floor.
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 32*2**20:
            raise RuntimeError("Derived figure output size boundary (preserve partial evidence)")
        fig.text(.015, .015, note, fontsize=7, va="bottom")
        fig.tight_layout(rect=(0, .15, 1, .97))
        for ext in ("png", "svg", "pdf"):
            p = output/f"{stem}.{ext}"
            fig.savefig(p, dpi=160, facecolor="white")
            outputs.append(p.name)
        plt.close(fig)
    for case in cases:
        g = curves[curves.case_id == case]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        for ax, methods in zip(axes, (POOLS[:2], POOLS[2:])):
            for method in methods:
                a = g[g.method == method].sort_values("step")
                ax.plot(a.step, a.vqe_error_mHa, "-", color=COLORS[method], label=method+" VQE")
                ax.plot(a.step, a.mean_error_mHa, "--", color=COLORS[method], label=method+" QSCI")
            ax.set(xlabel="Saved growth step", ylabel="Error vs saved FCI (mHa)", yscale="symlog", title=case)
            ax.set_yscale("symlog", linthresh=.01)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=.15)
        save(fig, case+"_Fig05_growth", "VQE: original preparation state; QSCI: independent A M1000/R32 mean. Curves are saved observations, not an online stopping rule.\nFCI subtraction is evaluation-only. MAX_STEPS endpoints are not convergence. Pointwise draw intervals retained in CSV, not drawn here.")
        t = transition[transition.case_id == case]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        for ax, feature, label in zip(axes, ("delta_cnot", "delta_depth"), ("Added logical sampling CNOT", "Change in logical sampling CNOT depth")):
            for method in POOLS:
                a = t[t.method == method]
                ax.scatter(a[feature], a.qsci_gain_mHa, s=12, alpha=.65, color=COLORS[method], label=method)
            ax.axhline(0, color="gray", lw=.7)
            ax.set(xlabel=label, ylabel="QSCI gain after growth step (mHa)", title=case)
            ax.set_yscale("symlog", linthresh=.01)
            ax.legend(fontsize=7)
        save(fig, case+"_Fig12_operator_gain", "M1000/R32; new A independent-draw means. Positive gain = lower error; negative points retained.\nOperator addition AND full-ansatz reoptimization change together: associations, NOT isolated operator causal effects. IDs/operators saved in CSV.")
        met = primary_discovery[primary_discovery.case_id == case]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for method, arm in met.groupby("method"):
            arm = arm.sort_values("step")
            ax.plot(arm.hf_probability*100, arm.nonhf_weighted_coverage*100,
                    "o-" if method in POOLS else "s", ms=3, lw=.8, label=method, color=COLORS.get(method))
        ax.set(xlabel="HF measurement probability (%)", ylabel="Expected non-HF FCI-weighted coverage (%)", title=case+" / M1000")
        ax.legend(fontsize=6, ncol=3)
        ax.grid(alpha=.15)
        save(fig, case+"_Fig13_HF_coverage", "Analytic marginal expectations from saved probabilities; FCI target weights are oracle evaluation, not selection.\nLines follow saved checkpoints, not monotone/causal laws. Fixed controls are points. HF probability != HF weight in FCI.")
    # Unified current-method heatmap. No per-cell best method/seed selection.
    heatrows = []
    final = cp.groupby(["case_id", "method"]).step.max().to_dict()
    a64 = primary_readouts(rows, R=64)
    for case in cases:
        for method in (*POOLS, "lucj_ccsd_2_vqe"):
            if method in POOLS:
                r = a64[(a64.case_id == case) & (a64.method == method) & (a64.step == final[(case, method)])]
                assert len(r) == 1
                q = r.iloc[0]
                value, d, status = q.mean_error_mHa, q.mean_dimension, "AVAILABLE_SAVED_FINAL"
                namespace = "A independent draws"
            else:
                folder = B1/"cells"/case/method
                result = read(folder/"result.json")
                r = B1rows[(B1rows.case_id == case) & (B1rows.method == method) & (B1rows.shots == 1000) & (B1rows.R_cap == 64)]
                if len(r):
                    assert len(r) == 1 and result["problem_id"] == refs[case]["request_digest"]
                    q = r.iloc[0]
                    value, d, status = q.mean_error_mHa, q.mean_actual_d, result["status"]
                else:
                    value, d, status = None, None, result["status"]
                namespace = "B1 independent draws (not paired with A)"
            heatrows.append({"case_id": case, "method": method, "M": 1000, "R_cap": 64,
                "mean_error_mHa": value, "mean_actual_d": d, "status": status,
                "readout_scope": namespace, "matched_realized_dimension": False})
    heat = pd.DataFrame(heatrows)
    heat.to_csv(output/"unified-method-heatmap.csv", index=False)
    molecules = sorted({refs[c]["case"]["molecule"] for c in cases})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=.1, vmin=0, vmax=max(1., float(heat.mean_error_mHa.max())))
    image = None
    for ax, molecule in zip(axes.flat, molecules):
        cs = sorted([c for c in cases if refs[c]["case"]["molecule"] == molecule], key=lambda c: refs[c]["case"]["distance"])
        names = [*POOLS, "lucj_ccsd_2_vqe"]
        values = heat[heat.case_id.isin(cs)].pivot(index="method", columns="case_id", values="mean_error_mHa").reindex(index=names, columns=cs)
        image = ax.imshow(values.to_numpy(float), norm=norm, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(cs)), [f'{refs[c]["case"]["distance"]:g}' for c in cs])
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel="Distance / equal spacing (A)", title=molecule)
        for i in range(len(names)):
            for j in range(len(cs)):
                value = values.iloc[i, j]
                ax.text(j, i, "N/A" if pd.isna(value) else f"{value:.2g}", ha="center", va="center", fontsize=7, color="black", bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=.2))
    for ax in list(axes.flat)[len(molecules):]: ax.set_visible(False)
    if image is not None: fig.colorbar(image, ax=list(axes.flat), shrink=.75, label="Mean QSCI error (mHa)", location="right", fraction=.025, pad=.015)
    # Explicit axes positions retain colorbar; no tight_layout incompatibility.
    fig.subplots_adjust(left=.13, right=.87, top=.93, bottom=.19, hspace=.42, wspace=.72)
    fig.text(.015, .025, "M1000/R64, saved final circuits; common cap NOT matched realized d. FCI only for evaluation. Missing CCSD stays N/A.\nLUCJ is bounded CCSD-VQE, not best possible LUCJ. A/B1 readout namespaces differ; heatmap is descriptive, NOT paired inference.", fontsize=8)
    for ext in ("png", "svg", "pdf"):
        p=output/f"Fig09_unified_method_heatmap.{ext}";fig.savefig(p,dpi=160,facecolor="white");outputs.append(p.name)
    plt.close(fig)
    # Explicit oracle-vs-trained-vs-random analytic comparison; no new shots.
    oracle_rows = []
    for case in cases:
        ref = refs[case]
        with np.load(bind(source/"references"/case/"reference.npz"), allow_pickle=False) as z:
            ids, weights = z["ids"].copy(), z["weights"].copy()
        arms = []
        for method in POOLS[:2]:
            r = cp[(cp.case_id == case) & (cp.method == method)].sort_values("step").iloc[-1]
            path = bind(source/r.distribution_path)
            assert sha(path) == r.distribution_sha
            with np.load(path, allow_pickle=False) as z: probabilities = z["probability"][ids]
            arms.append((method, None, probabilities, "SAVED_TRAINED_CIRCUIT"))
        for family in ("random_ceo", "trained_ceo_random_angles"):
            for seed in (101, 211, 307):
                method = f"{family}_s{seed}"
                folder = B4/"cells"/case/method
                result = read(folder/"result.json")
                assert result["problem_id"] == ref["request_digest"]
                with np.load(bind(folder/"final.npz"), allow_pickle=False) as z: probabilities = z["probability"][ids]
                arms.append((family, seed, probabilities, "RETROSPECTIVE_FINAL_LENGTH_MATCHED"))
        arms.append(("FCI_Born_oracle", None, weights, "INFEASIBLE_ORACLE_DIAGNOSTIC_NOT_UPPER_BOUND"))
        for method, seed, probability, scope in arms:
            for M in SHOTS:
                oracle_rows.append({"case_id": case, "method": method, "construction_seed": seed, "M": M,
                    "nonhf_expected_weighted_coverage": weighted_nonhf_discovery(ids, weights, ref["hf_id"], probability, M),
                    "scope": scope, "new_draws": 0, "quantum_construction_shots": None})
        data = pd.DataFrame([r for r in oracle_rows if r["case_id"] == case])
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for method, arm in data.groupby("method"):
            values = arm.groupby("M").nonhf_expected_weighted_coverage.agg(["mean", "min", "max"])
            ax.plot(values.index, 100*values["mean"], "o-", ms=3, label=method)
            if arm.construction_seed.notna().any():ax.fill_between(values.index, 100*values["min"], 100*values["max"], alpha=.12)
        ax.set(xlabel="Hypothetical number of computational-basis draws", ylabel="Expected non-HF weighted coverage (%)", title=case, xscale="log")
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
        save(fig, case+"_Fig15_random_trained_oracle", "FCI Born curve uses pre-existing ground-state weights; not a practical sampler, not a discovery upper bound, and NOT zero-cost state preparation.\nAnalytic expectations, no new draws. Random: all3 seeds, mean/range NOT CI. Final CEO length/order is retrospective construction information.")
    pd.DataFrame(oracle_rows).to_csv(output/"oracle-random-trained-discovery.csv", index=False)
    result = {"schema": "final-relationships-derived-v1", "mode": mode, "cases": cases,
        "implemented_requested_forms": [5, 9, 12, 13, 15], "figures": len(outputs)//3,
        "source_sha256": dict(sorted(inputs.items())), "script_sha256": sha(__file__),
        "no_new_optimizer_or_reference_solve": True, "new_quantum_draws": 0, "hardware_shots": 0,
        "oracle_is_analytic_infeasible_diagnostic_not_upper_bound": True,
        "new_classical_postprocessing_seconds": time.perf_counter()-start, "whole_goal_complete": False}
    pd.DataFrame([{"case_id": c, "Fig05": True, "Fig12": True, "Fig13": True, "Fig15": True} for c in cases]).to_csv(output/"figure-coverage.csv",index=False)
    index=["# Missing-relationship figure supplement", "", "Derived from completed A/B1/B4 only. No new draws, optimizer runs, molecular states or FCI solves.",
        "Common R is not equal realized d. Oracle is saved FCI Born-distribution analysis, NOT practical preparation or an optimal-discovery bound.",
        "Figures8/18 complete total-cost frontiers and whole-study completion remain unresolved. Large E/F jobs are not claimed complete.", ""]
    for stem in sorted({Path(p).stem for p in outputs}):
        index.append(f"- {stem}: [PNG]({stem}.png) / [SVG]({stem}.svg) / [PDF]({stem}.pdf)")
    (output/"INDEX.md").write_text("\n".join(index)+"\n")
    result["output_sha256"]={str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="manifest.json"}
    (output/"manifest.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--mode",choices=("quick","standard"),default="standard")
    args=parser.parse_args();r=build(args.output,args.mode);print(json.dumps({k:r[k] for k in ("mode","figures","implemented_requested_forms","new_quantum_draws","whole_goal_complete")}))
