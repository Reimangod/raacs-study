from itertools import combinations
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from benchmark.common import ROOT
from benchmark.resource_adaptive_sampler import integral_problem
from benchmark.nonorthogonal_ci import occupied_columns
from benchmark.multibasis_primitives import measurement_rotation,construct_bases,source_preparation,METHODS
from benchmark.tests.test_nonorthogonal_ci import exterior


def test_measurement_rotation_on_generic_complex_input_needs_both_fermionic_phases():
    rng=np.random.default_rng(971);n=3
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));u=expm(.3*(a-a.conj().T))
    ids=[sum(1<<(2*n-1-q) for q in occ) for occ in combinations(range(2*n),2) if sum(q%2==0 for q in occ)==1]
    amps=rng.normal(size=len(ids))+1j*rng.normal(size=len(ids));amps/=np.linalg.norm(amps)
    vec=np.zeros(1<<(2*n),complex);vec[ids]=amps
    qc=QuantumCircuit(2*n);qc.initialize(vec,range(2*n))
    actual=Statevector.from_instruction(measurement_rotation(qc,u)).data
    expected=sum(c*exterior(occupied_columns(u.conj().T,idx)) for c,idx in zip(amps,ids))
    assert np.max(abs(actual-expected))<1e-10
    assert np.max(abs(abs(actual)**2-abs(vec)**2))>1e-3


def test_source_loader_reads_full_phase_preparation_not_sampling():
    for method in METHODS:
        qc,origin=source_preparation('H4_R1.1',method)
        assert qc is not None and qc.num_qubits==8
        assert origin['state_scope'].startswith('full preparation')
        assert origin['historical_quantum_objective_shots'] is None


def test_classical_basis_construction_no_labels_or_sampler_inputs():
    from benchmark.nonorthogonal_ci import NonorthogonalHamiltonian
    old,info=integral_problem(ROOT/'results/standard-v1/references/H2_R0.7')
    h=NonorthogonalHamiltonian(old.h1,old.eri,old.core)
    u,record=construct_bases(h,info,evaluation_cap=8)
    assert len(u)==3 and record['basis_objective_evaluations']==16
    assert not record['FCI_used'] and not record['sampler_state_used']
    assert all(np.allclose(x.T@x,np.eye(len(x)),atol=1e-10) for x in u)
    assert record['classical_anchor_ground']['energy_Ha']<=info['HF_energy_Ha']+1e-9
