import sqlite3
import pytest
from benchmark.fixed_bank_allocation import allocation_weights,setting,SCHEMA,run_base,METHODS
from benchmark.resource_adaptive_sampler import allocation_counts


def test_weights_use_pilots_only_and_ties_are_declared():
    pilots=[{"energy_Ha":-1.,"observed_ids":[0,1]}, {"energy_Ha":-1.2,"observed_ids":[0,1,2]}, {"energy_Ha":-1.2,"observed_ids":[0,3]}]
    assert allocation_weights("greedy",pilots,0,-.9)==[0.,1.,0.]
    assert allocation_weights("discovery",pilots,0,-.9)==[0.,1.,1.]
    assert allocation_weights("single_last",pilots,0,-.9)==[0.,0.,1.]
    assert allocation_counts(9,[0,0,0])==[3,3,3]
    with pytest.raises(ValueError):allocation_weights("oracle",pilots,0,-.9)


def test_actual_H2_shared_bank_charges_once_and_resume_no_new_draws():
    cfg=setting("quick");cfg["repeats"]=2
    db=sqlite3.connect(":memory:");db.executescript(SCHEMA)
    item=next(q for q in cfg["queue"] if q["case"]=="H2_R0.7")
    r=run_base(db,cfg,item)
    assert r["status"]=="COMPLETED" and r["new_shared_pilot_shots_actual_once"]==64
    policies=db.execute("SELECT method,record FROM policies").fetchall()
    assert len(policies)==len(METHODS)==6
    import json
    for mode,raw in policies:
        p=json.loads(raw)
        assert p["source_pilot_charge"]+p["new_pilot_charge"]+p["final_shots"]==10000
        assert p["new_pilot_charge"]==(0 if mode=="single_deep_no_pilot" else 64)
    count=db.execute("SELECT count(*) FROM draws").fetchone()[0]
    assert run_base(db,cfg,item)==r and db.execute("SELECT count(*) FROM draws").fetchone()[0]==count
    assert r["new_total_model_shots"]==sum(row[0] for row in db.execute("SELECT shots FROM draws"))
    assert not r["FCI_used"] and not r["full_Fock_vector_used"] and r["hardware_shots"]==0
