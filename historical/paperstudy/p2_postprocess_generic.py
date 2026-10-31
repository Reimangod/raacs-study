"""P2 source-specific continuation with actual matrix-free workspace checks.

Parameterization of p2_resume.py only; existing bytes and outcomes unchanged.
"""
import argparse
from importlib.metadata import version
import json
from time import perf_counter

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

from .io import ROOT, save_json, sha256


def run(case="N2-32-R2p4-P2", kind="naive", workspace_mib=512):
    source = ROOT/"paper-anchors/P2"/(case+"-"+kind)
    output = source/"postprocessing-v3"
    output.mkdir(parents=True, exist_ok=True)
    if (output/"result.json").exists():
        return json.loads((output/"result.json").read_text())
    physical = ROOT/"data/physical"/case
    meta = json.loads((physical/"problem.json").read_text())
    with np.load(physical/"integrals.npz") as z:
        h1, eri = z["h1"].copy(), z["eri"].copy()
    with np.load(source/"raw-counts.npz") as z:
        ids, counts = z["blocked_ids"].copy(), z["counts"].copy()
    bit_array = BitArray.from_counts({int(x): int(c) for x, c in zip(ids, counts)}, num_bits=2*meta["norb"])
    trace = []
    estimates = []
    start = perf_counter()
    save_json(output/"started.json", {"source_counts_sha256": sha256((source/"raw-counts.npz").read_bytes()),
               "source_model_shots": int(counts.sum()), "new_model_shots": 0,
               "workspace_budget_mib": workspace_mib,
               "postprocess_order": "canonical count expansion; new postprocess RNG namespace, no new measurement or selection outcome"})

    def bounded(strings, one, two, norb, nelec):
        dimensions = [len(a)*len(b) for a, b in strings]
        largest = max(dimensions)
        largest_r = max(max(len(a), len(b)) for a, b in strings)
        # Conservative working-array estimate, not a full dense matrix. The
        # last field records actual runtime memory separately when available.
        estimated = largest*8*32 + largest_r*norb*norb*max(nelec)*8*20 + norb**4*8*16
        record = {"batch_dimensions": dimensions, "largest_dimension": largest,
                  "single_vector_bytes": largest*8, "estimated_working_bytes": estimated,
                  "full_dense_matrix_not_allocated": True}
        estimates.append(record)
        save_json(output/"subspace-precheck.json", {"checks": estimates})
        if estimated > workspace_mib*2**20:
            raise MemoryError("Estimated fixed-space working memory exceeds registered local allocation")
        return solve_sci_batch(strings, one, two, norb, nelec, spin_sq=None,
                               max_cycle=200, tol=1e-10, max_memory=workspace_mib, verbose=0)

    def callback(results):
        trace.append({"batch_energies_Ha": [float(x.energy)+meta["core_energy"] for x in results],
                      "batch_dimensions": [int(x.sci_state.amplitudes.size) for x in results]})
        save_json(output/"history.json", {"iterations": trace})

    result = diagonalize_fermionic_hamiltonian(h1, eri, bit_array, samples_per_batch=4000,
              norb=16, nelec=(5, 5), num_batches=10, energy_tol=1e-5, occupancies_tol=1e-3,
              max_iterations=1, sci_solver=bounded, symmetrize_spin=True, carryover_threshold=1e-3,
              max_dim=4000, callback=callback, seed=0)
    values = np.array(trace[-1]["batch_energies_Ha"])
    report = {"status": "P2_COUNTS_POSTPROCESSING_RESUMED_CURRENT_RUNTIME", "lane": "P", "case_id": case,
              "source_counts_sha256": sha256((source/"raw-counts.npz").read_bytes()),
              "source_model_shots": int(counts.sum()), "new_model_shots": 0, "actual_QPU_shots": 0,
              "historical_environment_reproduced": False, "published_curve_match_verified": False,
              "source_solver_deviation": "Current PySCF solver instead of author Dice",
              "mean_energy_Ha": float(values.mean()), "minimum_energy_Ha": float(values.min()),
              "maximum_energy_Ha": float(values.max()), "returned_energy_Ha": float(result.energy)+meta["core_energy"],
              "batch_range_is_not_confidence_interval": True, "postprocess_wall_seconds": perf_counter()-start,
              "qiskit_addon_sqd": version("qiskit-addon-sqd"), "subspace_estimates": estimates,
              "settings_unchanged": {"samples_per_batch": 4000, "num_batches": 10, "max_dim_per_spin": 4000, "max_iterations": 1},
              "postprocess_order_changed_after_engineering_failure": True,
              "no_energy_observed_in_failed_postprocess_attempt": True,
              "iterations": trace, "main_C_lane_cell_completed": False, "whole_experiment_complete": False}
    save_json(output/"result.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace-mib", type=int, default=512)
    p.add_argument("--case", default="N2-32-R2p4-P2")
    p.add_argument("--kind", choices=["naive","compressed"], default="naive")
    a = p.parse_args()
    r = run(a.case, a.kind, a.workspace_mib)
    print(json.dumps({k: v for k, v in r.items() if k != "iterations"}, indent=2))
