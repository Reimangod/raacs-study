import pandas as pd
from benchmark.analysis import nondominated_mask


def test_ritz_roundoff_does_not_hide_resource_dominance():
    d=pd.DataFrame({"cnot_count":[10,20],"shots":[1000,1000],"mean_energy_error_mHa":[1e-10,0.]})
    assert nondominated_mask(d,list(d.columns)).tolist()==[True,False]
    assert d.mean_energy_error_mHa.iloc[0]==1e-10  # raw values never rounded away
