from dataclasses import asdict
import numpy as np
import pytest
from benchmark.molecular_mps_scaling import problem_queue, no_full_state_guards
from benchmark.molecular_problem_no_fci import MolecularSpec


def test_actual_molecular_scaling_queue_not_synthetic_GHZ():
    quick = problem_queue("quick"); full = problem_queue("standard")
    assert len(quick) == 3 and len(full) == 16
    assert len({x["id"] for x in full}) == 16
    chains = [MolecularSpec(**x["spec"]) for x in full if x["spec"]["molecule"].startswith("H") and x["spec"]["molecule"][1:].isdigit()]
    assert {2*int(s.molecule[1:]) for s in chains} == {16, 20, 24, 32, 40, 44}
    assert all(len(s.geometry()) == int(s.molecule[1:]) for s in chains)
    assert problem_queue("standard") == full
    n2 = [x["spec"] for x in full if x["spec"]["molecule"] == "N2"]
    assert {s["active_orbitals"] for s in n2} == {12, 16}
    assert all(s["basis"] == "cc-pvdz" for s in n2)


def test_scaling_guards_reject_actual_full_state_and_FCI_calls():
    import pyscf.fci.direct_spin1
    import quimb.tensor as qtn
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    with no_full_state_guards():
        for call in (lambda: Statevector.from_instruction(QuantumCircuit(2)),
                     lambda: pyscf.fci.direct_spin1.kernel(None, None, 2, 2),
                     lambda: qtn.MPS_computational_state("00").to_dense()):
            with pytest.raises(RuntimeError, match="SCALING_NO_FCI"):
                call()
    assert len(Statevector.from_instruction(QuantumCircuit(2))) == 4


def test_small_MPS_and_named_CI_still_work_inside_scaling_guards():
    from qiskit import QuantumCircuit
    from benchmark.circuit_sampling_runtime import SamplingSpec, CircuitCountSampler
    from benchmark.local_sampler_primitives import LocalQE
    qc = QuantumCircuit(20); qc.x(19); qc.x(18)
    qc = LocalQE((0, 1), (18, 19), 20).append(qc, np.pi/4)
    with no_full_state_guards():
        ids, counts, meta = CircuitCountSampler(SamplingSpec(max_bond=8)).draw(qc, 32, 101)
    assert int(counts.sum()) == 32 and not meta["original_build_metadata"]["full_statevector_allocated"]
    assert meta["sampling_resources"]["cnot_count"] > 0
