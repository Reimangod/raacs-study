"""Exact sparse physical QE simulation; unchanged native RAACS decisions.

Only circuit-action metadata enters the simulator. No classical CI vector is
prepared as a quantum state. This module is a separate bounded runtime provider.
"""
from collections import OrderedDict
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from qiskit.quantum_info import Operator

from benchmark import resource_adaptive_sampler as native
from benchmark.ansatz import circuit_bytes, measurement_equivalent_circuit, resources
from benchmark.circuit_sampling_runtime import SamplingSpec
from benchmark.common import digest as native_digest
from benchmark.local_sampler_primitives import LocalQE, _zero_phase
from benchmark.study.core import rss_peak
from adaptvqe.circuits import qe_circuit

from .io import ROOT, save_json


@lru_cache(maxsize=256)
def local_unitary(source, target, angle):
    support = tuple(sorted(source+target))
    src = tuple(support.index(x) for x in source)
    dst = tuple(support.index(x) for x in target)
    matrix = Operator(qe_circuit(list(src), list(dst), angle, len(support), big_endian=True)).data
    matrix = matrix/_zero_phase(src, dst, len(support))
    matrix[abs(matrix) < 1e-12] = 0
    if np.max(abs(matrix.conj().T@matrix-np.eye(len(matrix)))) > 1e-10:
        raise RuntimeError("Source QE floating structural cleanup broke unitarity")
    return support, matrix


def evolve(state, action, n, max_states=100000):
    source, target = tuple(action["source"]), tuple(action["target"])
    support, matrix = local_unitary(source, target, float(action["theta"]))
    k = len(support)
    mask = sum(1 << (n-1-q) for q in support)
    output = {}
    for raw, amplitude in state.items():
        column = sum(((raw >> (n-1-q)) & 1) << (k-1-i) for i, q in enumerate(support))
        for row in np.flatnonzero(matrix[:, column]):
            value = raw & ~mask
            for i, q in enumerate(support):
                value |= ((int(row) >> (k-1-i)) & 1) << (n-1-q)
            output[value] = output.get(value, 0j)+matrix[row, column]*amplitude
        if len(output) > max_states:
            raise MemoryError("Exact sparse physical support cap; no amplitude truncation substituted")
    return {x: a for x, a in output.items() if a != 0}


class ExactSparseSampler:
    def __init__(self, spec, hf_id):
        if spec.backend != "exact_sparse_QE":
            raise ValueError("Explicit exact-sparse runtime identity required")
        self.spec = spec
        self.hf_id = hf_id
        self.cache = OrderedDict()
        self.calls = self.raw_model_shots = self.MPS_builds = self.exact_small_builds = 0
        self.exact_sparse_builds = 0

    def draw(self, preparation, shots, seed, n_parameters=0, n_operators=0):
        tick = perf_counter()
        actions = (preparation.metadata or {}).get("paper_exact_sparse_actions", [])
        if len(actions) != n_operators:
            raise RuntimeError("Physical QE action trace missing/mismatched")
        state = {self.hf_id: 1.+0j}
        n = preparation.num_qubits
        built = 0
        for length, action in enumerate(actions, 1):
            key = native_digest(actions[:length])
            if key in self.cache:
                state = self.cache.pop(key)
                self.cache[key] = state
            else:
                state = evolve(state, action, n)
                self.exact_sparse_builds += 1
                built += 1
                self.cache[key] = state
                while len(self.cache) > 2:
                    self.cache.popitem(last=False)
            if perf_counter()-tick > self.spec.construction_seconds_cap:
                raise TimeoutError("Sparse physical circuit build boundary cap")
        ids = np.array(sorted(state), np.uint64)
        probability = np.array([abs(state[int(x)])**2 for x in ids])
        norm = float(probability.sum())
        if abs(norm-1) > 1e-8:
            raise RuntimeError("Exact sparse QE state lost norm")
        construction_seconds = perf_counter()-tick
        sampling_tick = perf_counter()
        counts = np.random.default_rng(int(seed)).multinomial(shots, probability/norm)
        nonzero = counts > 0
        ids, counts = ids[nonzero], counts[nonzero].astype(np.uint64)
        sampling, removed = measurement_equivalent_circuit(preparation)
        self.calls += 1
        self.raw_model_shots += int(counts.sum())
        metadata = {"sampling_circuit_sha256": __import__("hashlib").sha256(circuit_bytes(sampling)).hexdigest(),
                    "preparation_sampling_distinct": True, "removed_terminal_diagonal_gates": removed,
                    "prepared_simulator_cache_hit": built == 0, "construction_seconds_this_call": construction_seconds,
                    "original_build_metadata": {"backend": "exact sparse physical author-QE", "support_size": len(state),
                          "norm_squared": norm, "full_Fock_vector_allocated": False,
                          "CI_vector_prepared_as_quantum_state": False, "amplitude_truncation": False,
                          "local_structural_zero_tolerance": 1e-12},
                    "sample_metadata": {"status": "COMPLETED", "raw_simulated_shots": int(counts.sum()), "planned_shots": shots,
                                        "sampling_and_export_seconds": perf_counter()-sampling_tick},
                    "sampling_resources": resources(sampling, n_parameters, n_operators),
                    "preparation_resources": resources(preparation, n_parameters, n_operators),
                    "call_wall_seconds": perf_counter()-tick, "process_RSS_hwm_bytes": rss_peak(),
                    "seed": str(seed), "requested_model_shots": shots, "actual_raw_model_shots": int(counts.sum()),
                    "hardware_shots": 0, "FCI_used": False, "full_statevector_exposed_to_algorithm": False}
        return ids, counts, metadata


def traced_append(original, self, parent, theta):
    child = original(self, parent, theta)
    trace = list((parent.metadata or {}).get("paper_exact_sparse_actions", []))
    trace.append({"source": list(self.source), "target": list(self.target), "theta": float(theta)})
    child.metadata = {**(child.metadata or {}), "paper_exact_sparse_actions": trace}
    return child


def run_fixture(case_id="N2-32-R1p2-P2"):
    physical = ROOT/"data/physical"/case_id
    meta = json.loads((physical/"problem.json").read_text())
    result_path = ROOT/"qualification"/(case_id+"-RAACS-exact-sparse-v2")
    sampling = SamplingSpec(backend="exact_sparse_QE", cache_entries=2)
    spec = native.RAACSSpec(total_model_shots=1000, max_steps=2, candidates_measured_per_step=2, seed=8191)
    provider = ExactSparseSampler(sampling, meta["hf_id"])
    original = LocalQE.append
    with patch.object(native, "CircuitCountSampler", lambda config: provider), patch.object(
            LocalQE, "append", lambda self, parent, theta: traced_append(original, self, parent, theta)):
        result = native.run_sampler(physical, result_path, spec, sampling)
    report = {"phase": "S1_NATIVE_RAACS_FIXED_RULE_ENGINEERING_FIXTURE", "source_problem_id": meta["problem_id"],
              "native_result": result, "physical_simulator": "exact sparse source-QE, no MPS approximation",
              "exact_sparse_state_builds": provider.exact_sparse_builds,
              "legacy_model_backend_builds_field_counts_only_MPS_and_exact_small": True,
              "primary_scientific_run": False, "whole_experiment_complete": False}
    save_json(result_path/"wrapper-result.json", report)
    return report


if __name__ == "__main__":
    result = run_fixture()
    print(json.dumps({k: v for k, v in result.items() if k != "native_result"}, indent=2))
    print(json.dumps({k: result["native_result"][k] for k in ("status", "accepted_steps", "final_QSCI_energy_Ha", "actual_total_model_shots", "construction_stop_reason")}, indent=2))
