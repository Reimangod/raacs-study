"""One numerical worker: actual allocation execution -> audit -> outer report."""
import argparse
import json
from pathlib import Path
from .common import ROOT,save_json,file_digest


def complete(root,mode):
    from .fixed_bank_allocation import run
    from .fixed_bank_audit import audit,SUCCESS
    from .fixed_bank_report import report
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    status=root/"completion-sequence-v1.json"
    try:
        save_json(status,{"status":"RUNNING_EXECUTION","mode":mode,"whole_study_complete":False})
        run(root,mode)
        save_json(status,{"status":"RUNNING_INDEPENDENT_AUDIT","whole_study_complete":False})
        if not (root/"validation.json").exists():audit(root)
        else:assert json.loads((root/"validation.json").read_text())["status"]==SUCCESS
        save_json(status,{"status":"RUNNING_OUTER_REPORT","whole_study_complete":False})
        if not (root/"report-v1/report-manifest.json").exists():report(root)
        result={"status":"FIXED_BANK_EXECUTED_AUDITED_REPORTED","events_sha256":file_digest(root/"events.sqlite"),
                "report_manifest_sha256":file_digest(root/"report-v1/report-manifest.json"),"whole_study_complete":False}
        save_json(status,result);return result
    except Exception as exc:
        save_json(root/"sequence-failure-v1.json",{"status":"PRESERVED_STAGE_FAILURE","error":repr(exc),"no_automatic_outcome_retry":True,"whole_study_complete":False})
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["quick","standard"],default="standard");p.add_argument("--output");a=p.parse_args()
    print(json.dumps(complete(a.output or ROOT/f"results/fixed-bank-allocation-{a.mode}-v1",a.mode)))
