import numpy as np
from scipy.sparse import diags
from benchmark.sqd import ids_to_blocked,blocked_to_ids,recover_readout


def test_spin_blocked_roundtrip_asymmetric():
    ids=np.arange(64)
    bits=ids_to_blocked(ids,3)
    assert np.array_equal(blocked_to_ids(bits),ids)
    # logical alpha0,beta1 occupied => physical bits5 and2.
    bits=ids_to_blocked([36],3)
    assert bits.tolist()==[[False,True,False,False,False,True]]


def test_official_recovery_sector_and_counts_reuse():
    # Two orbitals, one alpha electron, no beta electron: deliberately asymmetric.
    ref={"meta":{"n_orbitals":2,"nalpha":1,"nbeta":0,"hf_id":8,"fci_energy":-1.},
         "ids":np.array([8,2]),"h":diags(np.r_[np.zeros(2),-.2,np.zeros(5),-1.,np.zeros(7)])}
    result=recover_readout(ref,[0,15],[4,6],2,513,3)
    assert all(r["valid_recovered_mass"]==1 for r in result)
    assert all(set(r["selected_ids"])<={2,8} for r in result)
    assert all(r["new_quantum_shots"]==0 for r in result)
