from dataclasses import asdict
import numpy as np
from qiskit import QuantumCircuit
from benchmark.circuit_sampling_runtime import SamplingSpec
from benchmark.spin_blocked_sampling import SpinBlockedCountSampler,BlockedApproximationUnqualified
from benchmark.mps_layout_qualification import logical_id
import pytest


def test_native_register_mapping_counts_cache_and_resource_scope():
    qc=QuantumCircuit(8);qc.x(1);qc.x(3);qc.ry(.4,2);qc.cx(2,7)
    sampler=SpinBlockedCountSampler(SamplingSpec(max_bond=128))
    ii,nn,meta=sampler.draw(qc,64,334)
    layout=meta['layout_evidence'];mapped={logical_id(x,layout['logical_to_simulator_qubit']):n for x,n in zip(layout['backend_ids'],layout['backend_counts'])}
    assert dict(zip(map(int,ii),map(int,nn)))==mapped
    assert sum(mapped.values())==64
    assert meta['sampling_resources']['parameter_count']==0
    ii2,nn2,meta2=sampler.draw(qc,64,334)
    assert np.array_equal(ii,ii2) and np.array_equal(nn,nn2)
    assert meta2['prepared_simulator_cache_hit'] and meta2['construction_seconds_this_call']==0


def test_approximation_rejection_happens_before_sampling(monkeypatch):
    from benchmark import spin_blocked_sampling as m
    original=m.construct_mps
    def inaccurate(*args,**kwargs):
        state,meta=original(*args,**kwargs);meta['retained_norm_squared']=.5;return state,meta
    monkeypatch.setattr(m,'construct_mps',inaccurate)
    def forbidden(*args,**kwargs):raise AssertionError('sampling after failed accuracy check')
    monkeypatch.setattr(m,'sample_mps',forbidden)
    with pytest.raises(BlockedApproximationUnqualified,match='BEFORE_SAMPLING'):
        SpinBlockedCountSampler(SamplingSpec()).draw(QuantumCircuit(4),32,22)
