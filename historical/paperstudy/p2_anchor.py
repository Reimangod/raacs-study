"""P2 N2 anchor: author inputs/settings, current SDK, explicit solver deviation.

This is NOT a C-lane budget-matched run and NOT a claim of reproducing a
historical software environment or of matching a published numerical table.
"""
import argparse
from functools import partial
from importlib.metadata import version
import json
from time import perf_counter

import ffsim
import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

from .author_compressed import load_author
from .input_adapter import array_bytes
from .io import ROOT, atomic_bytes, save_json
from .lucj import amplitudes_from_saved, interaction_pairs, make_circuit, circuit_views


def run(case="N2-32-R1p2-P2", kind="naive"):
    if not case.startswith("N2-"):
        raise ValueError("First native anchor restricted to N2; Fe gauge reproduction unresolved")
    output = ROOT/"paper-anchors/P2"/(case+"-"+kind)
    if (output/"result.json").exists():
        return json.loads((output/"result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    physical = ROOT/"data/physical"/case
    meta = json.loads((physical/"problem.json").read_text())
    with np.load(physical/"integrals.npz") as z:
        h1, eri = z["h1"].copy(), z["eri"].copy()
    n, ne = meta["norb"], (meta["nalpha"], meta["nbeta"])
    tick = perf_counter()
    save_json(output/"started.json", {"lane": "P", "case": case, "kind": kind,
                                     "source_amplitudes_not_fresh_C_lane": True})
    if kind == "naive":
        t1, t2, initialization = amplitudes_from_saved(ROOT/"data/initialization"/case/"amplitudes.npz", n, ne[0])
        op, _, _ = load_author().from_t_amplitudes_compressed(t2, t1=t1, n_reps=1,
                                interaction_pairs=interaction_pairs(n), optimize=False)
    elif kind == "compressed":
        init_path = ROOT/"qualification"/(case+"-author-compressed")/"operator.npz"
        with np.load(init_path) as z:
            op = ffsim.UCJOpSpinBalanced(z["diag_coulomb_mats"], z["orbital_rotations"], z["final_orbital_rotation"])
        initialization = "UNMODIFIED_AUTHOR_INITIALIZER_SAVED_P2_AMPLITUDES_STAGE_CAPS_RECORDED"
    else:
        raise ValueError(kind)
    _, _, resource_views = circuit_views(make_circuit(op, n, ne))
    construct_seconds = perf_counter()-tick
    tick = perf_counter()
    state = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne, copy=False)
    simulation_seconds = perf_counter()-tick
    samples = ffsim.sample_state_vector(state, norb=n, nelec=ne, shots=100000,
                                       bitstring_type=ffsim.BitstringType.INT, seed=0)
    ids, counts = np.unique(samples, return_counts=True)
    atomic_bytes(output/"raw-counts.npz", array_bytes({"blocked_ids": ids.astype(np.uint64), "counts": counts.astype(np.int64)}))
    pool = BitArray.from_samples(samples, num_bits=2*n)
    history = []

    def callback(results):
        record = {"batch_energies_Ha": [float(x.energy)+meta["core_energy"] for x in results],
                  "batch_dimensions": [int(x.sci_state.amplitudes.size) for x in results]}
        history.append(record)
        save_json(output/"history.json", {"iterations": history})

    def bounded_solver(strings, one, two, norb, nelec):
        if any(len(a)*len(b) > 65536 for a, b in strings):
            raise MemoryError("Anchor selected-space cost limit, not a replacement with smaller R")
        return solve_sci_batch(strings, one, two, norb, nelec, spin_sq=None,
                               max_cycle=200, tol=1e-10, max_memory=1024, verbose=0)

    tick = perf_counter()
    result = diagonalize_fermionic_hamiltonian(h1, eri, pool, samples_per_batch=4000,
               norb=n, nelec=ne, num_batches=10, energy_tol=1e-5, occupancies_tol=1e-3,
               max_iterations=1, sci_solver=bounded_solver, symmetrize_spin=True,
               carryover_threshold=1e-3, max_dim=4000, callback=callback, seed=0)
    all_energies = np.array(history[-1]["batch_energies_Ha"])
    report = {"status": "P2_NATIVE_SETTINGS_EXECUTED_CURRENT_RUNTIME", "case_id": case,
              "problem_id": meta["problem_id"], "kind": kind, "lane": "P",
              "model_quantum_shots": 100000, "actual_QPU_shots": 0,
              "samples_per_batch": 4000, "num_batches": 10, "max_dim_per_spin": 4000,
              "max_iterations": 1, "source_initialization": initialization,
              "source_amplitude_creation_cost_seconds": None,
              "current_runtime": {p: version(p) for p in ("ffsim", "pyscf", "qiskit", "qiskit-addon-sqd")},
              "solver_deviation": "Current PySCF fixed-space solver instead of author script Dice solver",
              "historical_environment_reproduced": False, "published_numeric_curve_match_verified": False,
              "mean_energy_Ha": float(all_energies.mean()), "minimum_energy_Ha": float(all_energies.min()),
              "maximum_energy_Ha": float(all_energies.max()), "returned_best_energy_Ha": float(result.energy)+meta["core_energy"],
              "batch_range_is_not_confidence_interval": True, "resource_views": resource_views,
              "construction_seconds_this_call": construct_seconds, "exact_sector_simulation_seconds": simulation_seconds,
              "SQD_postprocessing_seconds": perf_counter()-tick, "raw_observed_unique": len(ids),
              "iterations": history, "main_C_lane_cell_completed": False, "whole_experiment_complete": False}
    save_json(output/"result.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="N2-32-R1p2-P2")
    p.add_argument("--kind", choices=["naive", "compressed"], default="naive")
    a = p.parse_args()
    r = run(a.case, a.kind)
    print(json.dumps({k: v for k, v in r.items() if k not in ("iterations", "resource_views")}, indent=2))
