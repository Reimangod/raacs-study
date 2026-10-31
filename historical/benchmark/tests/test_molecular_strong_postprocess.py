from copy import deepcopy
import pytest
from qiskit import QuantumCircuit
from benchmark.gate_noise_runtime import compile_sampler
from benchmark.molecular_strong_postprocess import structural_native_check,diagnostic


def test_list_pair_readout_mapping_and_line_neighbors_checked():
    qc=QuantumCircuit(4);qc.x(0);qc.ry(.3,1);qc.cx(0,3)
    for target in ('all_to_all_cx','line_cx'):
        compiled,resource=compile_sampler(qc,target)
        assert structural_native_check(compiled,resource,4)
        bad=deepcopy(resource);bad['measurement_physical_to_logical_clbit'][0]=[0,0]
        bad['measurement_physical_to_logical_clbit'][1]=[1,0]
        with pytest.raises(AssertionError):structural_native_check(compiled,bad,4)


def test_same_circuit_bonds_do_not_acquire_new_counts():
    qc=QuantumCircuit(4);qc.ry(.3,1);qc.cx(1,3)
    result=diagnostic(qc,{'bonds':[4,8],'cutoff':1e-12,'construction_seconds_cap':20,'RSS_cap_bytes':2*2**30})
    assert result['new_model_shots']==0 and result['same_physical_circuit']
    assert result['normalized_overlap_squared']==pytest.approx(1,abs=1e-10)
