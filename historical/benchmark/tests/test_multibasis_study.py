import numpy as np
from benchmark.multibasis_study import selected_nodes,uniform_counts,valid,configuration,capacity


def test_identical_bitstrings_in_different_bases_are_distinct_vectors():
    draws=[{'basis':0,'ids':[12,3],'counts':[5,1]},{'basis':1,'ids':[12,3],'counts':[6,2]}]
    nodes=selected_nodes(draws,'multibasis',12,8)
    assert nodes[:3]==[(0,12),(1,12),(2,12)]
    assert (0,3) in nodes and (1,3) in nodes
    assert selected_nodes(draws,'canonical',12,1)==[(0,12)]


def test_uniform_generation_has_fixed_spin_without_fullsector_list():
    info={'n_qubits':40,'nalpha':10,'nbeta':10}
    ids,counts=uniform_counts(20,info,100,np.random.default_rng(73))
    assert sum(counts)==100 and all(valid(x,info) for x in ids)


def test_anchors_consume_registered_cap_and_ties_are_deterministic():
    ds=[{'basis':2,'ids':[5,6],'counts':[3,3]},{'basis':0,'ids':[5],'counts':[3]}]
    assert selected_nodes(ds,'multibasis',12,4)==[(0,12),(1,12),(2,12),(0,5)]
    assert selected_nodes(ds,'multibasis',12,5)==selected_nodes(list(reversed(ds)),'multibasis',12,5)


def test_quick_standard_scope_and_no_free_production_validation(tmp_path):
    quick=configuration('quick');standard=configuration('standard')
    assert len(quick['cases'])==2 and len(standard['cases'])==30
    assert quick['seeds']==[101,211] and standard['seeds']==[101,211,307]
    assert standard['rank_relative_cutoff']==1e-9
    assert standard['hardware_shots']==0 and not standard['paper_CDF_NOCI_reproduction']
    cfg={**quick,'bounds':{**quick['bounds'],'free_floor_bytes':10**30}}
    assert capacity(tmp_path,cfg)=='PENDING_DISK_BOUNDARY'
