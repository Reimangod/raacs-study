"""Resume only audit/report, never the already completed450 constructions."""
import argparse
import json
from pathlib import Path
import sqlite3
from .common import file_digest,save_json,ROOT
from .online_parameter_audit_v2 import audit,SUCCESS
from .online_parameter_report import report
from .online_parameter_sequence import verify_prefix


def finish(root):
    root=Path(root).resolve()
    if not (root/"validation.json").exists():audit(root)
    v=json.loads((root/"validation.json").read_text());assert v["status"]==SUCCESS and v["validated"]==450 and not v["failures"]
    if not (root/"report-v1/report-manifest.json").exists():report(root)
    r=json.loads((root/"report-v1/report-manifest.json").read_text())
    assert r["cells"]==450 and r["readout_rows"]==126000
    assert v["events_sha256"]==r["events_sha256"]==file_digest(root/"events.sqlite")
    assert r["validation_sha256"]==file_digest(root/"validation.json")
    for name,sha in r["artifact_sha256"].items():assert file_digest(root/"report-v1"/name)==sha
    cfg=json.loads((root/"manifest.json").read_text())
    for name,sha in cfg["implementation"].items():assert file_digest(ROOT/name)==sha
    receipt=json.loads((root/"runtime-output-boundary-successor-v1.json").read_text())
    assert receipt["manifest_sha256_unchanged"]==file_digest(root/"manifest.json")
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True);verify_prefix(db,receipt["source_prefix"])
    assert dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))=={"COMPLETED":450};db.close()
    result={"status":"ONLINE_450_EXECUTED_AUDITED_V2_REPORTED","validated_cells":450,"common_R_and_actual_d_rows":126000,
            "prior_v1_coordinator_failure_preserved":True,"reexecuted_quantum_or_optimizer_outcomes":0,
            "events_sha256":v["events_sha256"],"manifest_sha256":file_digest(root/"manifest.json"),"validation_sha256":file_digest(root/"validation.json"),
            "report_manifest_sha256":file_digest(root/"report-v1/report-manifest.json"),"old_committed_cells_verified_unchanged":receipt["source_prefix"]["cells"]["count"],
            "postprocess_code_sha256":file_digest(Path(__file__)),"hardware_shots":0,"whole_study_complete":False}
    path=root/"completion-evidence-v2.json"
    if path.exists():assert json.loads(path.read_text())==result
    else:save_json(path,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();print(json.dumps(finish(a.root)))
