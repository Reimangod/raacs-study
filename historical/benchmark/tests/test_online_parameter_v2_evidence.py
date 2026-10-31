import json
import sqlite3
from benchmark.common import ROOT,file_digest
from benchmark.online_parameter_sequence import verify_prefix


def test_all450_online_records_independent_audit_and_exact_prefix_preservation():
    root=ROOT/"results/online-parameter-standard-v1"
    if not (root/"validation.json").exists():
        import pytest;pytest.skip("Actual online all-cohort v2 audit not finished")
    v=json.loads((root/"validation.json").read_text())
    assert v["validated"]==450 and not v["failures"] and v["raw_draws_rebuilt"]==23602
    assert v["independent_CI_energy_and_ground_subspace_solver"]=="NumPy eigh"
    assert v["objective_trace_calls_recomputed"]==22637 and v["hardware_shots"]==0
    assert v["events_sha256"]==file_digest(root/"events.sqlite")
    receipt=json.loads((root/"runtime-output-boundary-successor-v1.json").read_text())
    assert receipt["source_prefix"]["cells"]["count"]==407 and receipt["numeric_identity_unchanged"]
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True);verify_prefix(db,receipt["source_prefix"])
    counts=dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"));db.close()
    assert counts=={"COMPLETED":450}


def test_all450_online_report_has_all_seeds_and_cluster_not_geometry_inference():
    root=ROOT/"results/online-parameter-standard-v1";path=root/"report-v1/report-manifest.json"
    if not path.exists():
        import pytest;pytest.skip("Actual online outer report still running")
    import pandas as pd
    r=json.loads(path.read_text());frame=pd.read_csv(root/"report-v1/all-constructed-cells.csv")
    assert r["cells"]==len(frame)==450 and frame.case.nunique()==30 and set(frame.seed)=={101,211,307}
    assert r["readout_rows"]==126000
    assert r["totals"]["new_model_shots"]==81121984 and r["totals"]["VQE_QPU_objective_cost"] is None
    assert all(s["molecule_clusters"]==6 for s in r["summaries"])
    for name,sha in r["artifact_sha256"].items():assert file_digest(root/"report-v1"/name)==sha
