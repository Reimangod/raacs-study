"""Standalone CuPy number-sector LUCJ qualification. No full-Fock allocation.

Exterior-power orbital rotations are a qualification/reference kernel, not a
claim of a scalable fast implementation. Full compound matrices are bounded.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np


def run(input_path, output_path):
    import cupy as cp
    started = time.perf_counter()
    with np.load(input_path, allow_pickle=False) as a:
        n, ne = int(a["norb"]), tuple(map(int, a["nelec"]))
        rotations = a["orbital_rotations"].copy()
        diag = a["diag_coulomb_mats"].copy()
        final = a["final_orbital_rotation"].copy()
        expected = a["expected_sector_vector"].copy()
    if n > 6 or ne[0] != ne[1]:
        raise ValueError("This first GPU qualification is restricted to small balanced fixtures")
    occupation_tuples = sorted(itertools.combinations(range(n), ne[0]), key=lambda t: sum(1 << i for i in t))
    occ = cp.array(occupation_tuples, dtype=cp.int64)
    weights = cp.zeros((len(occupation_tuples), n), dtype=cp.float64)
    for j, indices in enumerate(occupation_tuples):
        weights[j, list(indices)] = 1.

    def compound(U):
        matrix = cp.asarray(U, dtype=cp.complex128)
        minors = matrix[occ[:, None, :, None], occ[None, :, None, :]]
        return cp.linalg.det(minors)

    def rotate(state, U):
        g = compound(U)
        return g @ state @ g.T

    state = cp.zeros((len(occ), len(occ)), dtype=cp.complex128)
    state[0, 0] = 1.
    current = np.eye(n)
    for (jaa, jab), U in zip(diag, rotations):
        state = rotate(state, U.conj().T @ current)
        same = cp.sum((weights @ cp.asarray(jaa))*weights, axis=1)/2
        energy = same[:, None] + same[None, :] + weights @ cp.asarray(jab) @ weights.T
        state *= cp.exp(1j*energy)
        current = U
    state = rotate(state, final @ current)
    cp.cuda.Stream.null.synchronize()
    actual = cp.asnumpy(state).reshape(-1)
    overlap = np.vdot(expected, actual)
    aligned = actual*np.exp(-1j*np.angle(overlap))
    error = float(np.max(abs(aligned-expected)))
    probability_error = float(np.sum(abs(abs(actual)**2-abs(expected)**2)))
    result = {"status": "GPU_SMALL_PARITY_PASSED" if max(error, probability_error) < 1e-8 else "GPU_SMALL_PARITY_FAILED",
              "state_max_error": error, "probability_L1_error": probability_error,
              "norm_squared": float(np.vdot(actual, actual).real), "norb": n, "nelec": list(ne),
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cupy": cp.__version__,
              "gpu_total_memory_bytes": int(cp.cuda.runtime.memGetInfo()[1]),
              "GPU_memory_pool_peak_retained_bytes": int(cp.get_default_memory_pool().total_bytes()),
              "seconds_including_CUDA_cold_start": time.perf_counter()-started,
              "actual_QPU_shots": 0, "production_qualified": False,
              "kernel_scope": "bounded exterior-power reference kernel, not a full campaign executor"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pending")
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(output_path)
    print(json.dumps(result), flush=True)
    if result["status"] != "GPU_SMALL_PARITY_PASSED":
        raise RuntimeError("GPU/independent CPU state mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    a = p.parse_args()
    run(a.input, a.output)
