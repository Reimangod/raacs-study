import json
from pathlib import Path
import shutil
import sqlite3
import pandas as pd
import pytest
from benchmark.common import ROOT,file_digest
from benchmark.fixed_bank_audit import audit

ROOT_STD=ROOT/"results/fixed-bank-allocation-standard-v1"
QUICK=ROOT/"results/fixed-bank-allocation-quick-v1"
pytestmark=pytest.mark.skipif(not (ROOT_STD/"validation.json").exists(),reason="Actual allocation cohort not available")


def test_all_sources_shared_costs_and_no_free_historical_growth():
    validation=json.loads((ROOT_STD/"validation.json").read_text())
    report=json.loads((ROOT_STD/"report-v1/report-manifest.json").read_text())
    cells=pd.read_csv(ROOT_STD/"report-v1/all-policy-cells.csv")
    old=pd.read_csv(ROOT_STD/"report-v1/unique-source-costs-NOT-new-acquisitions.csv")
    assert validation["bases"]==90 and len(cells)==validation["policy_cells"]==540
    assert len(old)==90 and old.historical_pilot_shots.sum()==validation["inherited_pilot_shots_unique_NOT_new"]==291008
    assert validation["new_pilot_shots_actual_once"]==20160
    assert cells.new_pilot_charge.sum()==5*validation["new_pilot_shots_actual_once"]  # Five hypothetical branches, NOT actual acquisitions.
    assert cells.no_historical_growth_avoidance_claim.all() and (cells.known_branch_model_shots==10000).all()
    assert (cells[cells.method=="single_deep_no_pilot"].new_pilot_charge==0).all()
    assert all(s["molecule_clusters"]==6 for s in report["summaries"])
    assert not report["whole_study_complete"]
    for name,sha in report["artifact_sha256"].items():assert file_digest(ROOT_STD/"report-v1"/name)==sha


def test_forged_policy_allocation_rejected_even_if_database_digest_updated(tmp_path):
    for name in ("events.sqlite","manifest.json","execution-status.json","source-audit.json"):shutil.copy2(QUICK/name,tmp_path/name)
    db=sqlite3.connect(tmp_path/"events.sqlite")
    bid,mode,raw=db.execute("SELECT base,method,record FROM policies ORDER BY rowid LIMIT 1").fetchone();record=json.loads(raw)
    record["allocation"][0]+=1
    db.execute("UPDATE policies SET record=? WHERE base=? AND method=?",(json.dumps(record),bid,mode));db.commit();db.close()
    path=tmp_path/"execution-status.json";s=json.loads(path.read_text());s["events_sha256"]=file_digest(tmp_path/"events.sqlite");path.write_text(json.dumps(s))
    with pytest.raises(AssertionError):audit(tmp_path)


def test_original_measurements_not_replaced_by_support_oracle():
    path=ROOT_STD/"support-saturation-v1/summary.json"
    if not path.exists():pytest.skip("Posthoc support diagnostic still running")
    d=json.loads(path.read_text())
    assert d["new_quantum_acquisitions"]==d["new_FCI_solves"]==0 and not d["practical_support_padding"]
    assert d["bases"]==90 and d["source_events_sha256"]==file_digest(ROOT_STD/"events.sqlite")
