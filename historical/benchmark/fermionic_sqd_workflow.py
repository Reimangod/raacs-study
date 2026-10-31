"""Official full fermionic SQD with Cartesian-basis and recovery provenance.

Unlike legacy explicit-determinant recovery, calls the complete official
self-consistency loop and bounded alpha×beta SCI solver. No reference FCI
labels enter. Selected-CI fixed-space diagonalization is not a full FCI solve.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
from unittest.mock import patch
import importlib.metadata
import numpy as np
from qiskit.primitives import BitArray
import qiskit_addon_sqd.fermion as fermion
from .integral_hamiltonian import IntegralHamiltonian
from .sqd import ids_to_blocked, blocked_to_ids
from .local_sampler_primitives import valid_sector
from .study.core import rss_peak


@dataclass(frozen=True)
class FermionicSQDSpec:
    max_spin_strings: int = 8
    samples_per_batch: int = 32
    batches: int = 3
    max_iterations: int = 5
    energy_tol: float = 1e-8
    occupancies_tol: float = 1e-5
    max_davidson_iterations: int = 100
    wall_boundary_seconds: float = 120.
    RSS_boundary_bytes: int = 2*2**30
    seed: int = 101


def cartesian_ids(alpha, beta, norb):
    """PySCF little-orbital-index spin strings -> original interleaved IDs."""
    return sorted(sum(((int(a) >> j) & 1) << (2*norb-1-2*j) for j in range(norb))+
                  sum(((int(b) >> j) & 1) << (2*norb-2-2*j) for j in range(norb))
                  for a in alpha for b in beta)


def full_fermionic_sqd(h1, eri, core, hf_id, nalpha, nbeta, ids, counts, spec=FermionicSQDSpec()):
    ids, counts = list(ids), list(counts)
    if len(ids) != len(counts) or len(set(map(int, ids))) != len(ids): raise ValueError("Unique actual raw configurations")
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) for x in ids+counts):
        raise ValueError("Integer raw IDs/counts required")
    if any(int(c) <= 0 for c in counts) or not 1 <= sum(map(int, counts)) <= 200000:
        raise ValueError("Positive bounded raw sample budget")
    n = 2*len(h1); norb = n//2
    if not 1 <= n <= 62 or any(not 0 <= int(x) < 1 << n for x in ids): raise ValueError("Invalid register")
    if not valid_sector(int(hf_id), n, nalpha, nbeta): raise ValueError("Invalid known HF sector")
    if not 1 <= spec.max_spin_strings <= 22 or spec.max_spin_strings**2 > 512:
        raise ValueError("Bound the Cartesian dimension by512")
    if min(spec.batches, spec.max_iterations, spec.samples_per_batch) < 1: raise ValueError("Positive workflow bounds")
    raw_ids = set(map(int, ids)); ids = np.asarray(ids, dtype=np.int64); counts = np.asarray(counts, dtype=np.int64)
    # Multiplicities are retained exactly. Passing only unique rows would silently
    # change the empirical sample distribution used by official SQD.
    expanded = np.repeat(ids_to_blocked(ids, norb), counts, axis=0)
    bit_array = BitArray.from_bool_array(expanded)
    backend = IntegralHamiltonian(h1=np.asarray(h1), eri=np.asarray(eri), core_energy=float(core))
    started = perf_counter(); recovery_records = []; solver_records = []; iteration_records = []
    original_recover = fermion.recover_configurations
    def guard():
        if perf_counter()-started > spec.wall_boundary_seconds or rss_peak() > spec.RSS_boundary_bytes:
            raise RuntimeError("Fermionic SQD solver/recovery boundary cap; partial trace preserved by caller")
    def recover(*args, **kwargs):
        guard(); result = original_recover(*args, **kwargs)
        recovered_ids = blocked_to_ids(result[0]); weights = np.asarray(result[1], float)
        recovery_records.append({"call": len(recovery_records)+1, "returned_unique_ids": list(map(int, recovered_ids)),
            "returned_classical_weights": weights.tolist(), "weights_are_not_new_quantum_counts": True,
            "new_recovered_ids_not_raw_observed": sorted(set(map(int, recovered_ids))-raw_ids),
            "new_quantum_shots": 0})
        return result
    def solver(strings, one, two, orbitals, nelec):
        guard()
        for a, b in strings:
            if len(a)*len(b) > spec.max_spin_strings**2: raise RuntimeError("Unbounded Cartesian subspace")
        results = fermion.solve_sci_batch(strings, one, two, orbitals, nelec, spin_sq=None,
                                         max_cycle=spec.max_davidson_iterations, tol=1e-10, verbose=0)
        for result in results:
            state = result.sci_state
            if state is None: raise RuntimeError("Bounded PySCF solver must expose selected-space state")
            selected = cartesian_ids(state.ci_strs_a, state.ci_strs_b, orbitals)
            matrix = backend.matrix(selected, selected)
            exact_space_energy = float(np.linalg.eigvalsh(matrix)[0])
            total_energy = float(result.energy)+float(core)
            if abs(total_energy-exact_space_energy) > 1e-7:
                raise RuntimeError("Official fermionic/independent Slater-Condon projection parity failure")
            if any(not valid_sector(x, n, nalpha, nbeta) for x in selected): raise RuntimeError("Wrong solver sector")
            solver_records.append({"call": len(solver_records)+1, "energy_Ha": total_energy,
                "independent_named_CI_energy_Ha": exact_space_energy, "dimension": len(selected),
                "alpha_strings": list(map(int, state.ci_strs_a)), "beta_strings": list(map(int, state.ci_strs_b)),
                "selected_interleaved_ids": selected,
                "cartesian_ids_not_raw_observed": sorted(set(selected)-raw_ids),
                "cartesian_completion_is_classical_not_quantum_observation": True,
                "spin_square_measured": float(state.spin_square()),
                "S_squared_enforced": False})
        return results
    def callback(results):
        guard()
        iteration_records.append({"iteration": len(iteration_records)+1,
            "batch_energies_Ha": [float(r.energy)+float(core) for r in results],
            "batch_dimensions": [int(r.sci_state.amplitudes.size) for r in results],
            "occupancies": [[list(map(float, r.orbital_occupancies[0])), list(map(float, r.orbital_occupancies[1]))] for r in results]})
    hf_a = sum(((int(hf_id) >> (n-1-2*j)) & 1) << j for j in range(norb))
    hf_b = sum(((int(hf_id) >> (n-2-2*j)) & 1) << j for j in range(norb))
    with patch.object(fermion, "recover_configurations", recover):
        result = fermion.diagonalize_fermionic_hamiltonian(np.asarray(h1), np.asarray(eri), bit_array,
            samples_per_batch=spec.samples_per_batch, norb=norb, nelec=(nalpha, nbeta),
            num_batches=spec.batches, max_iterations=spec.max_iterations,
            energy_tol=spec.energy_tol, occupancies_tol=spec.occupancies_tol,
            sci_solver=solver, max_dim=spec.max_spin_strings, symmetrize_spin=False,
            include_configurations=([hf_a], [hf_b]), initial_occupancies=None,
            callback=callback, seed=np.random.default_rng(spec.seed))
    selected = cartesian_ids(result.sci_state.ci_strs_a, result.sci_state.ci_strs_b, norb)
    return {"schema": "official-full-fermionic-SQD-bounded-v1", "settings": asdict(spec),
        "qiskit_addon_sqd_version": importlib.metadata.version("qiskit-addon-sqd"),
        "energy_Ha": float(result.energy)+float(core), "dimension": len(selected), "selected_interleaved_ids": selected,
        "raw_model_shots": int(counts.sum()), "raw_observed_ids": list(map(int, ids)),
        "raw_valid_sector_fraction": sum(int(c) for x, c in zip(ids, counts) if valid_sector(int(x), n, nalpha, nbeta))/int(counts.sum()),
        "self_consistency_iterations": iteration_records, "recovery_trace": recovery_records, "solver_trace": solver_records,
        "iterations_used": len(iteration_records), "max_iterations_reached": len(iteration_records) >= spec.max_iterations,
        "max_iteration_label_not_convergence_certificate": True,
        "returned_selection": "official best observed batch across iterations, not independently evaluated next sample",
        "observed_counts_reused_across_iterations": True, "new_quantum_shots_for_recovery": 0,
        "observed_recovered_and_cartesian_generated_configurations_distinct": True,
        "actual_wall_seconds": perf_counter()-started, "process_RSS_hwm_bytes": rss_peak(),
        "FCI_labels_used": False, "full_FCI_solve": False, "hardware_shots": 0,
        "workflow_scope": "official complete bounded fermionic SQD loop, not exact paper settings/hardware reproduction"}
