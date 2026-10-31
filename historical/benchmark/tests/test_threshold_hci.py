import numpy as np
from benchmark.common import ROOT
from benchmark.resource_adaptive_sampler import integral_problem
from benchmark.integral_connections import _enumerate
from benchmark.threshold_hci import HeatBathConnections,HCISpec,variational
from benchmark.molecular_mps_scaling import no_full_state_guards


def test_presorted_heatbath_generator_equals_all_connections_threshold():
    h,info=integral_problem(ROOT/'results/standard-v1/references/H4_R1.1')
    hb=HeatBathConnections(h)
    allids,allvalues,_=_enumerate(np.uint64(info['hf_id']),h.n_qubits,h.h1,h.eri,h.core,0.)
    parents=[info['hf_id'],*map(int,allids[:6])]
    for p in parents:
        ii,vv,_=_enumerate(np.uint64(p),h.n_qubits,h.h1,h.eri,h.core,0.)
        for threshold in (1e-2,1e-3,1e-5):
            expected={int(x):float(v) for x,v in zip(ii,vv) if abs(v)>threshold}
            ii2,vv2=hb.connections(p,threshold)
            actual=dict(zip(map(int,ii2),map(float,vv2)))
            assert actual.keys()==expected.keys()
            assert all(abs(actual[x]-expected[x])<1e-12 for x in expected)


def test_variational_adds_entire_threshold_set_not_top_cardinality():
    h,info=integral_problem(ROOT/'results/standard-v1/references/H4_R1.1');hb=HeatBathConnections(h)
    ii,vv=hb.connections(info['hf_id'],1e-3)
    expected=sorted({info['hf_id'],*map(int,ii)})
    with no_full_state_guards():r=variational(h,info['hf_id'],HCISpec(epsilon1_Ha=1e-3),hb)
    assert r['history'][1]['selected_ids']==expected
    energies=[x['variational_energy_Ha'] for x in r['history']]
    assert all(b<=a+1e-10 for a,b in zip(energies,energies[1:]))
    assert r['FCI_used'] is False and r['PT2_energy_Ha'] is None and r['logical_CNOT'] is None


def test_dimension_boundary_keeps_last_solved_space_and_does_not_truncate_candidates():
    h,info=integral_problem(ROOT/'results/standard-v1/references/H4_R1.1')
    r=variational(h,info['hf_id'],HCISpec(epsilon1_Ha=1e-3,dimension_cap=2))
    assert r['status']=='DIMENSION_CAP_BEFORE_COMPLETE_THRESHOLD_EXPANSION'
    assert len(r['history'])==1 and r['history'][0]['dimension']==1
