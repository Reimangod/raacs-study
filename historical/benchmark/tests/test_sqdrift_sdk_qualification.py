import numpy as np
import pytest
from scipy.linalg import expm
from benchmark.common import ROOT


def test_diagonal_hamiltonian_filtering_is_not_general_sampling_equivalence():
    X = np.array([[0., 1.], [1., 0.]])
    Z = np.diag([1., -1.]); initial = np.array([1., 0.])
    p_full = abs(expm(-1j*(X+Z)) @ initial)**2
    p_filtered = abs(expm(-1j*X) @ initial)**2
    assert abs(p_full[1]-p_filtered[1]) > .1
    # Removing a scalar identity is different: it only changes global phase.
    p_shift = abs(expm(-1j*(X+Z+3*np.eye(2))) @ initial)**2
    assert np.allclose(p_full, p_shift)


def test_real_official_sdk_refined_partition_and_physical_parity(tmp_path):
    overlay = ROOT / ".runtime-overlays/sqdrift-qiskit25-v1"
    if not overlay.exists():
        pytest.skip("Optional isolated Qiskit2.5/fermions0.1 runtime not installed")
    from benchmark.sqdrift_qualify import qualify
    result = qualify("H4_R1.1", tmp_path, ROOT / "results/standard-v1", num_terms=16)
    assert result["status"] == "SDK_GROUPED_FERMION_CIRCUIT_PARITY_VALIDATED"
    assert result["raw_SDK_first_moment_mismatch_before_refinement"] > 1e-3
    assert result["group_first_moment_error"] < 1e-10
    assert result["max_unit_group_Pauli_L1_norm_bound"] <= 1+1e-9
    assert result["state_parity_error"] < 1e-9
    assert abs(result["valid_N_Sz_sector_probability"]-1) < 1e-9
