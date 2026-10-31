"""Figure-only successor: persistent method colors across unequal-cost panels.

No numerical data changed. The original report/figures remain preserved.
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, file_digest, save_json
from .raacs_independent_evaluation import PAIRED


def render():
    root = ROOT/"results/RAACS-independent-comparison-standard-sharded-v1"
    frame_path = root/"cost-corrected-report-v2/all-cell-common-R-summary-cost-corrected.csv"
    frame = pd.read_csv(frame_path); output = root/"figures-v2"; output.mkdir(exist_ok=True)
    primary = "RAACS:full_cost-qsci_local_grid-last_only-exact_small"
    methods = [primary]+PAIRED
    labels = ["RAACS full cost", "CEO OVP", "CEO MVP", "QEB", "qubit ADAPT", "GSD", "LUCJ CCSD2-VQE", "TE t1/4 steps", "Pauli qDRIFT256 s101", "HF-separated ADAPT-QSCI s101"]
    colors = {method: plt.get_cmap("tab10")(i) for i, method in enumerate(methods)}
    figures = []
    for molecule in sorted(frame.molecule.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for method, label in zip(methods, labels):
            subset = frame[(frame.molecule == molecule) & (frame.family == method) & (frame.R == 64)]
            for ax, scope in zip(axes, ["readout_only", "known_build_plus_readout"]):
                g = subset[subset.scope == scope].groupby("M").mean(numeric_only=True)
                if len(g): ax.plot(g.index, g.mean_error_mHa, "o-", color=colors[method], label=label, ms=4)
                ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=.01)
                ax.set_xlabel("Model shots M"); ax.set_ylabel("Mean QSCI error (mHa)"); ax.grid(alpha=.2)
        axes[0].set_title("Equal final readout shots")
        axes[1].set_title("Known construction + readout budget")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"{molecule}: R64, geometry/seed means; construction-unknown methods absent on right")
        fig.tight_layout(rect=(0, .19, 1, .94)); name = "independent-comparison-"+molecule
        for suffix in ("png", "svg", "pdf"): fig.savefig(output/f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig); figures.append(name)
    save_json(output/"manifest.json", {"status": "FIGURE_ONLY_COLOR_MAPPING_CORRECTED", "figures": figures,
              "correction": "same persistent method color on both panels, even when unknown-budget methods are absent",
              "data_sha256": file_digest(frame_path), "renderer_sha256": file_digest(__file__),
              "numerical_results_changed": False, "old_figures_preserved": True, "whole_study_complete": False})
    (output/"README.md").write_text("""# Current comparison figures\n\n
Prefer these six PNG/SVG/PDF figures over the preserved v1 figures. In v1,
matplotlib restarted its color cycle separately on each panel, so a method's
color could change when unknown-build-cost methods were absent on the right.
This successor fixes method-to-color identity; all numerical data are unchanged.
Read the cost-corrected-report-v2 supplement and derived-report-v1/REPORT.md.
Missing curves on the right are unavailable cost certifications, not zero error.
""")
    return {"figures": figures, "numerical_results_changed": False}


if __name__ == "__main__": print(json.dumps(render()))
