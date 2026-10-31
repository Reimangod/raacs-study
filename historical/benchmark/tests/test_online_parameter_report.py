import json
import numpy as np
import pandas as pd
import pytest
from benchmark.common import ROOT,file_digest

QUICK=ROOT/"results/online-parameter-quick-v1"
REPORT=QUICK/"report-v1"
pytestmark=pytest.mark.skipif(not (REPORT/"report-manifest.json").exists(),reason="Actual quick online report not available")


def test_common_R_and_actual_d_are_not_shot_multipliers():
    r=json.loads((REPORT/"report-manifest.json").read_text())
    a=json.loads((QUICK/"validation.json").read_text())
    d=pd.read_csv(REPORT/"all-R-and-matched-d-readouts.csv.gz")
    assert len(d)==r["readout_rows"]==560 and r["cells"]==10
    assert r["totals"]["new_model_shots"]==a["new_pilot_model_shots"]+a["new_final_model_shots"]==379840
    assert r["totals"]["VQE_QPU_objective_cost"] is None and r["totals"]["hardware_shots"]==0
    matched=d[d.scope=="all_five_methods_actual_d_matched"]
    assert (matched.groupby(["case","seed","rep","R"]).actual_d.nunique()==1).all()
    common=d[d.scope=="common_R"]
    for _,g in common.groupby(["case","seed","mode","rep"]):
        assert np.all(np.diff(g.sort_values("R").error_mHa.to_numpy())<=1e-6)


def test_artifact_integrity_and_all_modes_kept():
    r=json.loads((REPORT/"report-manifest.json").read_text())
    f=pd.read_csv(REPORT/"all-constructed-cells.csv")
    assert f["mode"].nunique()==5 and len(f)==10 and f.case.nunique()==2
    assert not r["whole_study_complete"]
    for name,sha in r["artifact_sha256"].items():assert file_digest(REPORT/name)==sha
    assert len(list(REPORT.glob("*.png")))==len(list(REPORT.glob("*.svg")))==len(list(REPORT.glob("*.pdf")))==5
