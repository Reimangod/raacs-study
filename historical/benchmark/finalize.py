"""Cohort status and observed hardware use; never overwrites raw scientific data."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
from .common import save_json,digest,ROOT,file_digest
from .config import cases_for
from .audit import METHODS


def finalize(root):
    root=Path(root);cases=[];intervals=[]
    for case in cases_for("full"):
        methods={}
        for method in METHODS:
            path=root/"checkpoints"/case.id/method/"status.json"
            failed=root/"failures"/case.id/(method+".json")
            if path.exists():methods[method]=json.loads(path.read_text())
            elif failed.exists():methods[method]=json.loads(failed.read_text())
            else:methods[method]={"status":"NOT_STARTED"}
            if failed.exists() and path.exists():methods[method]["historical_failure_preserved_not_current"]=str(failed)
        cases.append({"case":case.id,"methods":methods})
    for path in sorted((root/"gpu-production-v1").glob("*/sacct.txt")):
        job=json.loads((path.parent/"job.json").read_text())
        for line in path.read_text().splitlines():
            fields=line.split()
            if len(fields)>=6 and fields[0]==job["job_id"] and "T" in fields[2] and "T" in fields[3]:
                intervals.append({"case":path.parent.name,"job":fields[0],"state":fields[1],"start":fields[2],"end":fields[3]})
    events=[]
    for r in intervals:events += [(datetime.fromisoformat(r["start"]),1),(datetime.fromisoformat(r["end"]),-1)]
    current=maximum=0
    for _,delta in sorted(events,key=lambda x:(x[0],x[1])):current+=delta;maximum=max(maximum,current)
    hardware={"configured_max_concurrent_A100":2,"observed_max_concurrent_primary_A100":maximum,"primary_case_intervals":intervals,
              "cluster_clock":"Slurm cluster local timestamps; no timezone conversion inferred",
              "constraint":"normal QOS permits two GPUs per user total; pre-existing unrelated job5788 used one, left untouched",
              "qualification":"two independent N2 full64-step qualification jobs5786/5787 overlapped; primary cohort concurrency separately reported",
              "mac_platform":platform.platform(),"mac_python":platform.python_version(),"hardware_quantum_shots":0}
    try:hardware["mac_cpu"]=subprocess.check_output(["rtk","proxy","sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
    except Exception:hardware["mac_cpu"]="UNAVAILABLE"
    save_json(root/"gpu-production-v1/observed-utilization.json",hardware)
    fleet=root/"gpu-production-v1/fleet-terminal.json"
    if fleet.exists():
        save_json(root/"gpu-production-v1/fleet-terminal-v2.json",{"schema":"fleet-terminal-2","status":"COMPLETED",
                  "configured_max_concurrent_A100":2,"observed_max_concurrent_primary_A100":maximum,
                  "source_v1_sha256":file_digest(fleet),"v1_preserved":True,
                  "clarification":"v1 max_simultaneous_A100 field was the dispatcher cap, not observed utilization"})
    save_json(root/"run-status-final.json",cases)
    status=Counter(m["status"] for c in cases for m in c["methods"].values())
    save_json(root/"execution-manifest-final.json",{"schema":"completed-cohort-1","planned_cases":30,"planned_cells":330,
              "method_status_counts":dict(status),"all_cases_attempted":all(m["status"]!="NOT_STARTED" for c in cases for m in c["methods"].values()),
              "cpu_request_preserved":"request.json describes original CPU allocation; cohort_cases is all30",
              "gpu_allocation":"gpu-production-v1; actual CUDA state/energy work and per-case Slurm traces",
              "authoritative_current_status":"run-status-final.json","raw_prior_run_status_preserved":True,
              "hardware":hardware})
    print("cohort",len(cases),"cells",sum(status.values()),dict(status),"primary GPU concurrency",maximum)
    return cases


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);finalize(p.parse_args().output)
