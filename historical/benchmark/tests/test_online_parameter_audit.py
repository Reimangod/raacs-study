import json
from pathlib import Path
import shutil
import sqlite3
import pytest
from benchmark.common import ROOT,file_digest
from benchmark.local_sampler_primitives import unpack_counts64,pack_counts64
from benchmark.online_parameter_audit import audit
from benchmark.online_parameter_controls import compressed,decoded

QUICK=ROOT/"results/online-parameter-quick-v1"
pytestmark=pytest.mark.skipif(not (QUICK/"validation.json").exists(),reason="Actual online quick qualification not available")


@pytest.fixture
def tamper_copy(tmp_path):
    for name in ("manifest.json","execution-status.json","source-audit.json","events.sqlite"):
        shutil.copy2(QUICK/name,tmp_path/name)
    return tmp_path


def rehash(root):
    path=root/"execution-status.json";s=json.loads(path.read_text());s["events_sha256"]=file_digest(root/"events.sqlite");path.write_text(json.dumps(s))


def test_count_mutation_with_recomputed_digest_cannot_pass(tamper_copy):
    db=sqlite3.connect(tamper_copy/"events.sqlite")
    selected=None
    for did,raw in db.execute("SELECT id,counts FROM draws"):
        ids,ns=unpack_counts64(raw)
        if len(ns)>1 and ns.max()>1:selected=(did,ids,ns);break
    assert selected is not None
    did,ids,ns=selected;donor=int(ns.argmax());recipient=(donor+1)%len(ns);ns[donor]-=1;ns[recipient]+=1
    db.execute("UPDATE draws SET counts=? WHERE id=?",(pack_counts64(ids,ns),did));db.commit();db.close();rehash(tamper_copy)
    with pytest.raises(AssertionError):audit(tamper_copy)


def test_forged_candidate_score_with_recomputed_digest_cannot_pass(tamper_copy):
    db=sqlite3.connect(tamper_copy/"events.sqlite")
    cell,step,raw=db.execute("SELECT cell,step,record FROM decisions ORDER BY rowid LIMIT 1").fetchone()
    d=decoded(raw);d["ranked"][0]["score"]+=1
    db.execute("UPDATE decisions SET record=? WHERE cell=? AND step=?",(compressed(d),cell,step));db.commit();db.close();rehash(tamper_copy)
    with pytest.raises(AssertionError):audit(tamper_copy)


def test_quick_acquisition_and_hardware_accounting_are_explicit():
    v=json.loads((QUICK/"validation.json").read_text())
    assert v["validated"]==10 and not v["failures"]
    assert v["raw_draws_rebuilt"]==250 and v["new_pilot_model_shots"]+v["new_final_model_shots"]==379840
    assert v["hardware_shots"]==0 and v["exact_recorded_score_order_verified"]
    assert v["full_vectors_small_auditor_only_not_policy"]
