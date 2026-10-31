"""Computed scaling figures only; no invented ground-state references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .common import save_json, file_digest


def report(root):
    root = Path(root); validation = json.loads((root/"validation.json").read_text())
    assert validation["status"] == "SCALING_COUNTS_AND_BINDINGS_VALIDATED"
    out = root/"derived-report-v1"; out.mkdir(exist_ok=True)
    frame = pd.read_csv(root/"audited-scaling-cells.csv"); bonds = pd.read_csv(root/"audited-bond-diagnostics.csv")
    figlist = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"): fig.savefig(out/f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig); figlist.append(name)
    for name, y, label, log in (("01_construction_scaling", "wall_seconds", "Observed task wall time (s)", True),
                                ("02_CNOT_scaling", "cnot", "Logical sampling CNOT count", False),
                                ("03_depth_scaling", "depth", "Logical sampling CNOT depth", False),
                                ("04_named_H_work", "named_H_requests", "Named integral-H entry requests", True),
                                ("05_peak_memory", "peak_process_RSS_bytes", "Process RSS high water (bytes)", True),
                                ("06_energy_lowering", "HF_minus_QSCI_mHa", "HF minus QSCI energy (mHa; NOT ground-state error)", False)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if y in frame:
            for method, group in frame[frame.method != "HF"].groupby("method"):
                ax.scatter(group.n_qubits, group[y], label=method, s=23, alpha=.7)
        ax.set_xlabel("Molecular qubits (different physical systems)"); ax.set_ylabel(label)
        if log: ax.set_yscale("log")
        ax.legend(fontsize=8); ax.grid(alpha=.2); save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if "normalized_MPS_overlap_squared_to_highest_completed_bond" in bonds:
        for (case, method), group in bonds.groupby(["case", "method"]):
            axes[0].plot(group.bond, group.normalized_MPS_overlap_squared_to_highest_completed_bond, "o-", alpha=.35)
            if "energy_Ha" in group:
                e = group.energy_Ha; axes[1].plot(group.bond, 1000*(e-e.iloc[-1]), "o-", alpha=.35)
    for ax in axes: ax.set_xlabel("Maximum MPS bond, identical circuit"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Normalized overlap² vs highest completed bond")
    axes[1].set_ylabel("Finite-shot CI energy minus highest-bond draw (mHa)")
    save(fig, "07_same_circuit_bond_diagnostics")
    text = f"""# Molecular integral-only/MPS scaling\n\n
{validation['planned']} planned cells; {validation['available_rows']} available rows.
Failures are retained in validation.json, not changed to zero cost/energy.
Maximum actually constructed molecular register: {validation['maximum_actual_molecular_qubits']} qubits.
Construction/final model shots: {validation['raw_construction_and_final_model_shots']:,};
additional diagnostic shots: {validation['extra_bond_diagnostic_model_shots']:,}. Hardware shots0.

## Interpretation boundary

No FCI solver, full Fock Hamiltonian/vector or full-sector enumeration is used
on this practical path. A bounded observed-only CI is still diagonalized.
HF energy lowering is not ground-state error and does not establish chemical
accuracy. These are approximate classical MPS simulations, not QPU evidence.
The bond overlap compares the same circuit to the highest completed approximate
bond; it is not an exact error certificate. Finite-shot energy differences also
contain independent sampling noise. A shallow circuit can have converged MPS
representation and still be a poor electronic-structure sampler.

## Comparison limits

Local-bank gradient is not global ADAPT-QSCI. No strong large-molecule LUCJ/TE
baseline is supplied by this cohort alone. One construction seed,16 standard
problems and geometry-dependent molecular clusters are not independent broad
generalization. End-to-end native-QPU cost and chemical accuracy remain unknown.
The total2000 shots include pilots;384-per-bond diagnostics are separately paid.
Do not compare this total2000 scaling budget to the standardD10000 cohort as
if they were matched. CPU time/RSS are measured on this machine, not A100 gains.
Full six-milestone completion remains unproven.
"""
    (out/"REPORT.md").write_text(text)
    result = {"figures": figlist, "figure_formats": ["png", "svg", "pdf"], "whole_study_complete": False,
              "report_sha256": file_digest(__file__), "validation_sha256": file_digest(root/"validation.json")}
    save_json(out/"manifest.json", result); return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(report(p.parse_args().output)))
