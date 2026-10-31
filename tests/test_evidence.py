import hashlib
import json
from pathlib import Path
import pytest
from raacs_study.evidence import normalize,number,pick,safe_member_name,aggregate,verify_repository,reproduce,read_csv

ROOT=Path(__file__).resolve().parents[1]


def test_unknown_is_not_zero():
    row,original=normalize({"case":"x","method":"B4","energy_Ha":"-1","independent_parameters":""},"small-exact-singlet","x.csv",1)
    assert row["independent_parameters"]==""and number(row["independent_parameters"])is None
    assert row["display_name"]=="RAACS"and original["original_row"]["method"]=="B4"


def test_alias_does_not_rename_active():
    assert normalize({"method":"ACTIVE"},"large-85120","x",1)[0]["display_name"]=="ACTIVE"


def test_explicit_missing_is_preserved():
    assert pick({"x":"","y":"2"},"x","y")==""
    assert number("nan")is None and number("inf")is None


def test_identity_binds_cohort_ordinal_and_original():
    row={"method":"B4","energy_Ha":"-1.000000"}
    a=normalize(row,"a","x",1)[0]
    assert a["record_id"]!=normalize(row,"b","x",1)[0]["record_id"]
    assert a["record_id"]!=normalize(row,"a","x",2)[0]["record_id"]
    assert normalize(row,"a","x",1)[1]["original_row"]["energy_Ha"]=="-1.000000"


@pytest.mark.parametrize("path",["../x","/tmp/x","a/../x","a\\b","a\0b"])
def test_unsafe_archive_paths(path):assert not safe_member_name(path)


def test_safe_path():assert safe_member_name("outputs/a/counts.npz")


def test_failed_and_unreached_rows_remain_in_denominator():
    good=normalize({"case":"x","method":"B4","seed":"1","status":"COMPLETED","actual_D":"16","target_D":"16","energy_Ha":"-1","final_shots":"10"},"large-85120","x",1)[0]
    bad=dict(good,replicate="2",readout_status="D_UNREACHED",energy_Ha="",D_actual="9")
    r=aggregate([good,bad])[0]
    assert r["n_endpoint_rows"]==2 and r["n_D_matched"]==1 and not r["all_recorded_replicates_D_matched"]


def test_real_repository_hashes_and_faithful_rows():
    result=verify_repository(ROOT)
    assert result["passed"],result["errors"]


def test_real_population_and_cohort_boundary():
    rows=read_csv((ROOT/"data/catalog.csv").read_bytes())
    cohorts={r["cohort"]for r in rows}
    assert cohorts=={"small-exact-singlet","large-85120","naphthalene-additional"}
    assert {r["QPU_shots"]for r in rows}=={"0"}
    assert len({r["record_id"]for r in rows})==len(rows)
    assert len(json.loads((ROOT/"data/cases.json").read_text()))==10
    pop=read_csv((ROOT/"data/source-population.csv").read_bytes())
    assert len(pop)==144 and sum(r["readout_availability"]=="MISSING"for r in pop)==12


def test_reproduction_is_deterministic(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    reproduce(ROOT,a);reproduce(ROOT,b)
    assert sorted(p.relative_to(a)for p in a.rglob("*")if p.is_file())==sorted(p.relative_to(b)for p in b.rglob("*")if p.is_file())
    for p in a.rglob("*"):
        if p.is_file():assert p.read_bytes()==(b/p.relative_to(a)).read_bytes()


def test_refuses_evidence_overwrite():
    with pytest.raises(ValueError):reproduce(ROOT,ROOT/"data/invalid")


def test_policy_sources_bound_to_recorded_protocol():
    registry=json.loads((ROOT/"configs/policy-registry.json").read_text())
    assert registry["display_alias"]=={"B4":"RAACS"}
    freeze=json.loads((ROOT/"configs/original-B4/freeze-B4.json").read_text())
    for file in("beam.py","spin_beam.py","singlet_cache.py","singlet.py","journal.py","bank.py"):
        assert hashlib.sha256((ROOT/"historical/frontierloop"/file).read_bytes()).hexdigest()==freeze["files"][file]
