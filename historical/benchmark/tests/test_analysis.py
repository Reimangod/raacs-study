import pandas as pd
from benchmark.analysis import nondominated_mask,finite_or_none


def test_frontier_does_not_drop_duplicates_or_tradeoffs():
    data=pd.DataFrame([[1,10,1],[2,10,2],[2,5,1],[1,10,1]],columns=["cnot","shots","error"])
    assert nondominated_mask(data,["cnot","shots","error"]).tolist()==[True,False,True,True]
    assert finite_or_none({"x":float("nan")})=={"x":None}
