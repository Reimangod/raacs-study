from pathlib import Path
import json
import numpy as np
from qiskit.quantum_info import Statevector
from benchmark.common import ROOT
from benchmark.chemistry import load_reference
from benchmark.qsci_exact_initialization import preparation, restore, grow


def test_successor_actual_circuit_for_eight_bit_separated_target():
    n = 12; ids = np.array([int("110011000000", 2), int("000000110011", 2)])
    c = np.array([.6, -.8])
    rotations = [["IIIIIIIIIIXY", .17]]
    qc = preparation(n, ids, c, rotations)
    assert abs(np.vdot(Statevector.from_instruction(qc).data, restore(n, ids, c, rotations))) > 1-1e-10


def test_successor_checkpoint_reconstruction_resume_no_new_draw(tmp_path):
    ref = load_reference(ROOT / "results/standard-v1/references/H2_R0.7")
    config = {"max_steps": 2, "trajectory_seconds": 60, "worker_ram_bytes": 3*2**30, "free_floor_bytes": 0}
    result = grow(ref, 1000, 32, 101, tmp_path, config)
    assert result["checkpoints"] > 0
    saved = {p.name: p.read_bytes() for p in tmp_path.glob("counts-*.npz")}
    (tmp_path / "terminal.json").unlink()
    resumed = grow(ref, 1000, 32, 101, tmp_path, config)
    assert resumed["cumulative"]["raw_simulated_shots"] == result["cumulative"]["raw_simulated_shots"]
    assert saved == {p.name: p.read_bytes() for p in tmp_path.glob("counts-*.npz")}
    cp = json.loads((tmp_path / "step-0000.json").read_text())
    # H2 first-order target has one determinant, so it really needs zero CX.
    assert cp["initialization"]["resources"]["cnot_count"] == 0
    assert cp["initial_fixed_coefficient_variables"] == 0
    assert cp["resources"]["parameter_count"] == 0
