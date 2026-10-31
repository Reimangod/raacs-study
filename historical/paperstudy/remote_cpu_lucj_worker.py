"""Exact ffsim on a high-RAM Slurm CPU allocation; not a GPU-speed claim."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

import ffsim
import numpy as np


def save(path, value):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(path)


def run(configuration, operator_file, output):
    config = json.loads(Path(configuration).read_text())
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"result.json").exists():
        print((directory/"result.json").read_text())
        return
    payload = Path(operator_file).read_bytes()
    if hashlib.sha256(payload).hexdigest() != config["operator_sha256"]:
        raise RuntimeError("Operator digest mismatch")
    tick, cpu = perf_counter(), process_time()
    qualification = None
    if config.get("CPU_parity_reference"):
        reference = config["CPU_parity_reference"]
        started = perf_counter()
        with np.load(reference["operator_file"], allow_pickle=False) as a:
            check_op = ffsim.UCJOpSpinBalanced(a["diag_coulomb_mats"], a["orbital_rotations"], a["final_orbital_rotation"])
        with np.load(reference["expected_file"], allow_pickle=False) as a:
            expected = a["expected_sector_vector"].copy()
        actual = ffsim.apply_unitary(ffsim.hartree_fock_state(16, (5, 5)), check_op, 16, (5, 5), copy=False)
        actual *= np.exp(-1j*np.angle(np.vdot(expected, actual)))
        error = float(np.max(abs(actual-expected)))
        if error > 1e-8:
            raise RuntimeError("Linux/Mac source ffsim whole32q sector parity failed")
        qualification = {"state_max_error": error, "wall_seconds": perf_counter()-started,
                         "compared_amplitudes": int(expected.size), "reference_is_fixed_LUCJ_not_FCI": True}
        del actual, expected, check_op
        __import__("gc").collect()
    n, ne = config["norb"], tuple(config["nelec"])
    with np.load(operator_file, allow_pickle=False) as a:
        op = ffsim.UCJOpSpinBalanced(a["diag_coulomb_mats"], a["orbital_rotations"], a["final_orbital_rotation"])
    vector = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne, copy=False)
    norm = float(np.vdot(vector, vector).real)
    if abs(norm-1) > 1e-8:
        raise RuntimeError("Exact high-RAM sector state norm failed")
    preparation = {"wall_seconds": perf_counter()-tick, "CPU_seconds": process_time()-cpu,
                   "sector_dimension": int(vector.size), "norm_squared": norm}
    records = []
    for request in config["requests"]:
        path = directory/(request["run_id"]+".json")
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        started, CPU = perf_counter(), process_time()
        values = ffsim.sample_state_vector(vector, norb=n, nelec=ne, shots=request["shots"],
                     bitstring_type=ffsim.BitstringType.INT, seed=request["seed"])
        ids, counts = np.unique(values, return_counts=True)
        target = directory/(request["run_id"]+".npz")
        temporary = target.with_suffix(".pending")
        with temporary.open("wb") as file:
            np.savez_compressed(file, blocked_ids=ids.astype(np.uint64), counts=counts.astype(np.uint64))
        temporary.replace(target)
        record = {**request, "counts_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "actual_model_shots": int(counts.sum()), "sampling_wall_seconds": perf_counter()-started,
                  "sampling_CPU_seconds": process_time()-CPU}
        save(path, record)
        records.append(record)
    result = {"status": "EXACT_REMOTE_CPU_LUCJ_COMPLETED", "config_id": config["config_id"],
              "problem_id": config["problem_id"], "operator_sha256": config["operator_sha256"],
              "preparation": preparation, "sampling_results": records,
              "wall_seconds": perf_counter()-tick, "CPU_seconds": process_time()-cpu,
              "actual_QPU_shots": 0, "GPU_used": False, "statevector_saved": False,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
              "runtime": {p: version(p) for p in ("numpy", "scipy", "ffsim", "pyscf", "qiskit")}}
    result["independent_platform_qualification"] = qualification
    save(directory/"result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:])
