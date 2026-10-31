"""Bounded matrix-free common Cartesian CI. No full-Fock or FCI solve."""
from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np
from pyscf.fci import selected_ci


def blocked_to_interleaved(value, norb):
    return sum(((int(value) >> i) & 1) << (2*norb-1-2*i) for i in range(norb)) + sum(
        ((int(value) >> (norb+i)) & 1) << (2*norb-2-2*i) for i in range(norb))


def interleaved_to_blocked(value, norb):
    return sum(((int(value) >> (2*norb-1-2*i)) & 1) << i for i in range(norb)) + sum(
        ((int(value) >> (2*norb-2-2*i)) & 1) << (norb+i) for i in range(norb))


def common_strings(ids, counts, norb, nelec, r_cap):
    if nelec[0] != nelec[1]:
        raise ValueError("Primary common spin-symmetrization requires equal Nalpha,Nbeta")
    if not 1 <= r_cap <= 64:
        raise ValueError("Registered maximum64 spin strings")
    if len(ids) != len(counts) or len(set(map(int, ids))) != len(ids):
        raise ValueError("Unique count IDs required")
    mask = (1 << norb)-1
    frequency = Counter()
    valid_shots = 0
    total_shots = 0
    raw_ids = set()
    for x, count in zip(ids, counts):
        if isinstance(x, (bool, np.bool_)) or isinstance(count, (bool, np.bool_)):
            raise ValueError("Boolean count/ID")
        if not isinstance(x, (int, np.integer)) or not isinstance(count, (int, np.integer)):
            raise ValueError("Integer count/ID")
        x, count = int(x), int(count)
        if count <= 0 or not 0 <= x < (1 << (2*norb)):
            raise ValueError("Positive counts within register required")
        total_shots += count
        a, b = x & mask, x >> norb
        raw_ids.add(x)
        if (a.bit_count(), b.bit_count()) == nelec:
            valid_shots += count
            frequency[a] += count
            frequency[b] += count
    if total_shots <= 0:
        raise ValueError("Nonempty counts required")
    hf = (1 << nelec[0])-1
    ranked = sorted((x for x in frequency if x != hf), key=lambda x: (-frequency[x], x))
    strings = np.array(sorted([hf, *ranked[:r_cap-1]]), np.int64)
    return strings, {"total_raw_model_shots": total_shots, "valid_sector_shots": valid_shots,
                     "valid_sector_fraction": valid_shots/total_shots, "r_cap": r_cap,
                     "actual_d_alpha": len(strings), "actual_d_beta": len(strings), "actual_d": len(strings)**2,
                     "spin_string_cap_reached": len(strings) == r_cap,
                     "raw_observed_unique": len(raw_ids),
                     "cartesian_generated_not_observed": sum((int(a) | int(b) << norb) not in raw_ids for a in strings for b in strings),
                     "configuration_recovery_enabled": False, "new_quantum_shots_for_Cartesian_completion": 0}


class CountedSelectedCI(selected_ci.SelectedCI):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline
        self.matvec_calls = 0

    def contract_2e(self, eri, civec_strs, norb, nelec, link_index=None, **kwargs):
        if perf_counter() > self.deadline:
            raise TimeoutError("Common CI matvec boundary time limit")
        self.matvec_calls += 1
        return super().contract_2e(eri, civec_strs, norb, nelec, link_index=link_index, **kwargs)


def solve(h1, eri, core, nelec, ids, counts, r_cap, seconds=300.):
    started = perf_counter()
    n = len(h1)
    strings, info = common_strings(ids, counts, n, nelec, r_cap)
    solver = CountedSelectedCI(started+seconds)
    solver.verbose = 0
    solver.max_memory = 1024
    energy, vector = solver.kernel_fixed_space(h1, eri, n, nelec, (strings, strings),
                        tol=1e-10, max_cycle=200, max_space=20, davidson_only=True, ecore=float(core), verbose=0)
    absorbed = solver.absorb_h1e(h1, eri, n, nelec, .5)
    residual = solver.contract_2e(absorbed, vector, n, nelec) - (float(energy)-core)*vector
    residual_norm = float(np.linalg.norm(residual))
    spin_square, multiplicity = solver.spin_square(vector, n, nelec)
    if not np.isfinite(energy) or not np.isfinite(residual_norm):
        raise RuntimeError("Nonfinite CI result")
    return {"schema": "common-Cartesian-CI-1", **info, "energy_Ha": float(energy),
            "selected_alpha_strings": strings.tolist(), "selected_beta_strings": strings.tolist(),
            "spin_square": float(spin_square), "spin_square_enforced": False,
            "solver": "PySCF SelectedCI.kernel_fixed_space matrix-free Davidson",
            "converged": bool(solver.converged), "residual_norm": residual_norm,
            "matvec_calls": solver.matvec_calls, "solver_calls": 1,
            "classical_wall_seconds": perf_counter()-started, "counts_consumed_once_no_new_shots": True,
            "full_FCI_solver_used": False, "large_dense_eigensolve_used": False}
