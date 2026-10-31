import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from benchmark import gate_noise_study as study
from benchmark.gate_noise_runtime import compile_sampler, logical_probabilities


def test_routed_measurement_preserves_original_logical_bits():
    q = QuantumCircuit(3); q.x(0); q.ry(.37, 2); q.cx(0, 2)
    native, record = compile_sampler(q, "line_cx")
    reconstructed = logical_probabilities(Statevector(native.remove_final_measurements(inplace=False)).probabilities(), record["measurement_physical_to_logical_clbit"])
    assert np.allclose(reconstructed, Statevector(q).probabilities(), atol=1e-12)
    assert record["inserted_explicit_swaps_before_decomposition"] >= 1
    assert record["native_2q_count"] >= 3*record["inserted_explicit_swaps_before_decomposition"]


def test_committed_chunks_merge_once_without_losing_multiplicities(tmp_path):
    db = study.database(tmp_path)
    for i, counts in enumerate(({3: 20, 12: 5}, {3: 7, 12: 18})):
        db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?)", ("a", i, 25, str(i), study.pack(counts), study.pack({})))
    db.commit()
    assert study.merge_chunks(db, "a") == ({3: 27, 12: 23}, 50)
    assert study.merge_chunks(db, "a") == ({3: 27, 12: 23}, 50)
    db.close()


def test_capacity_floor_and_output_boundary_are_not_scientific_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "shutil", SimpleNamespace(disk_usage=lambda p: SimpleNamespace(free=10)))
    assert study.capacity(tmp_path, {"free_floor": 11, "output_cap": 100}) == "FREE_FLOOR_BOUNDARY"
    (tmp_path/"bounded-fixture").write_bytes(b"x"*20)
    assert study.capacity(tmp_path, {"free_floor": 5, "output_cap": 10}) == "OUTPUT_BOUNDARY"


def test_arbitrary_existing_process_is_not_an_owned_noise_worker(tmp_path):
    assert not study.pid_is_owned_worker(os.getpid(), tmp_path, "foreign-task-id")


def test_watchdog_terminates_only_its_own_new_process_group():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    method = study.stop_owned(proc)
    assert method == "SIGTERM" and proc.returncode == -15
