"""Physical-circuit count interface for RAACS and molecular MPS scaling.

The algorithm receives counts, never a full physical-statevector or FCI labels.
Exact small simulation is qualification only. MPS is an approximate classical
simulation, with retained norm and explicit caps, never hardware evidence.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import perf_counter
import hashlib
import numpy as np
from qiskit.quantum_info import Statevector
from .ansatz import circuit_bytes, measurement_equivalent_circuit, resources
from .common import digest
from .mps_sampling import construct_mps, sample_mps
from .study.core import rss_peak


@dataclass(frozen=True)
class SamplingSpec:
    backend: str = "mps"
    max_bond: int = 64
    cutoff: float = 1e-12
    construction_seconds_cap: float = 120.
    sampling_seconds_cap: float = 120.
    RSS_cap_bytes: int = 2*2**30
    cache_entries: int = 2


class CircuitCountSampler:
    def __init__(self, spec):
        if spec.backend not in ("mps", "exact_small") or not 0 <= spec.cache_entries <= 4:
            raise ValueError("Registered sampling backend and bounded cache required")
        if spec.max_bond < 1 or spec.cutoff < 0 or spec.construction_seconds_cap <= 0 or spec.sampling_seconds_cap <= 0:
            raise ValueError("Invalid simulation caps")
        self.spec = spec; self.cache = OrderedDict()
        self.calls = self.raw_model_shots = self.MPS_builds = self.exact_small_builds = 0

    def draw(self, preparation, shots, seed, n_parameters=0, n_operators=0):
        if preparation.num_clbits or not 1 <= preparation.num_qubits <= 62 or shots < 1:
            raise ValueError("Positive shots on a bounded unmeasured register required")
        if self.spec.backend == "exact_small" and preparation.num_qubits > 16:
            raise ValueError("Exact-small full-vector qualification is limited to16qubits")
        t = perf_counter(); sampling, removed = measurement_equivalent_circuit(preparation)
        raw = circuit_bytes(sampling); circuit_id = hashlib.sha256(raw).hexdigest()
        key = digest([circuit_id, asdict(self.spec)]); hit = key in self.cache
        if hit:
            prepared, build_metadata = self.cache.pop(key); self.cache[key] = (prepared, build_metadata)
            build_seconds_this_call = 0.
        else:
            start = perf_counter()
            if self.spec.backend == "mps":
                prepared, build_metadata = construct_mps(sampling, self.spec.max_bond, self.spec.cutoff,
                                                         self.spec.construction_seconds_cap, self.spec.RSS_cap_bytes)
                self.MPS_builds += 1
            else:
                prepared = abs(Statevector.from_instruction(sampling).data)**2
                if not np.isclose(prepared.sum(), 1., atol=1e-9): raise RuntimeError("Unnormalized exact-small simulator distribution")
                build_metadata = {"backend": "Qiskit exact-small qualification", "full_statevector_allocated_inside_backend": True,
                                  "full_vector_exposed_to_algorithm": False, "n_qubits": sampling.num_qubits,
                                  "retained_norm_squared": 1., "hardware_shots": 0}
                self.exact_small_builds += 1
            build_seconds_this_call = perf_counter()-start
            if self.spec.cache_entries:
                self.cache[key] = (prepared, build_metadata)
                while len(self.cache) > self.spec.cache_entries: self.cache.popitem(last=False)
        start = perf_counter()
        if self.spec.backend == "mps":
            ids, counts, sample_metadata = sample_mps(prepared, shots, seed,
                self.spec.sampling_seconds_cap, self.spec.RSS_cap_bytes)
        else:
            allcounts = np.random.default_rng(seed).multinomial(shots, prepared/prepared.sum())
            ids = np.flatnonzero(allcounts).astype(np.uint64); counts = allcounts[ids.astype(np.int64)].astype(np.uint64)
            sample_metadata = {"status": "COMPLETED", "raw_simulated_shots": int(counts.sum()),
                               "planned_shots": shots, "sampling_seconds": perf_counter()-start, "hardware_shots": 0}
        # This interface does not repair invalid sectors or replace partial counts
        # by requested shots. The caller pays every returned raw measurement.
        self.calls += 1; self.raw_model_shots += int(counts.sum())
        metadata = {"sampling_circuit_sha256": circuit_id, "preparation_sampling_distinct": True,
                    "removed_terminal_diagonal_gates": removed, "prepared_simulator_cache_hit": hit,
                    "construction_seconds_this_call": build_seconds_this_call,
                    "original_build_metadata": build_metadata, "sample_metadata": sample_metadata,
                    "sampling_resources": resources(sampling, n_parameters, n_operators),
                    "preparation_resources": resources(preparation, n_parameters, n_operators),
                    "call_wall_seconds": perf_counter()-t, "process_RSS_hwm_bytes": rss_peak(),
                    "seed": str(seed), "requested_model_shots": shots, "actual_raw_model_shots": int(counts.sum()),
                    "hardware_shots": 0, "FCI_used": False,
                    "full_statevector_exposed_to_algorithm": False}
        return ids, counts, metadata
