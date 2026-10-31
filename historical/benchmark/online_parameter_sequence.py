"""Exact-engine continuation across a storage boundary, then audit/report.

Wait for the actual owner process to exit; never race its SQLite writer.
Only the operational output ceiling can extend; no numeric manifest edit.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from time import sleep
from .common import ROOT,digest,file_digest,save_json

TABLES=("cells","circuits","draws","optimizations","decisions","final_qpy")


def prefix_snapshot(db):
    snapshot={}
    for table in TABLES:
        maximum=db.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0
        h=hashlib.sha256();count=0
        for row in db.execute(f"SELECT rowid,* FROM {table} WHERE rowid<=? ORDER BY rowid",(maximum,)):
            h.update(json.dumps([hashlib.sha256(x).hexdigest() if isinstance(x,bytes) else x for x in row],separators=(",",":"),allow_nan=False).encode());h.update(b"\n");count+=1
        snapshot[table]={"max_rowid":maximum,"count":count,"sha256":h.hexdigest()}
    return snapshot


def verify_prefix(db,snapshot):
    for table,s in snapshot.items():
        h=hashlib.sha256();count=0
        for row in db.execute(f"SELECT rowid,* FROM {table} WHERE rowid<=? ORDER BY rowid",(s["max_rowid"],)):
            h.update(json.dumps([hashlib.sha256(x).hexdigest() if isinstance(x,bytes) else x for x in row],separators=(",",":"),allow_nan=False).encode());h.update(b"\n");count+=1
        assert count==s["count"] and h.hexdigest()==s["sha256"],table


def live_command(pid):
    result=subprocess.run(["ps","-p",str(pid),"-o","command="],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None


def complete(root,wait_pid,output_cap_mib=96):
    root=Path(root).resolve();statuspath=root/"completion-sequence-v1.json"
    previous=live_command(wait_pid)
    if previous and "benchmark.online_parameter_controls --mode standard" not in previous:raise RuntimeError("Wrong live process; refuse ownership transfer")
    while previous:
        save_json(statuspath,{"status":"WAITING_FOR_CONFIRMED_ORIGINAL_WRITER","pid":wait_pid,"command":previous,"no_competing_writer_started":True})
        sleep(10);previous=live_command(wait_pid)
        if previous and "benchmark.online_parameter_controls --mode standard" not in previous:raise RuntimeError("PID reuse; refuse continuation")
    from threadpoolctl import threadpool_limits
    from .online_parameter_controls import construct,setting,MODES,SCHEMA
    from .study.core import rss_peak
    cfg=json.loads((root/"manifest.json").read_text())
    expected=setting("standard");expected["free_floor"]=cfg["free_floor"]
    assert cfg==expected,"Numeric source or manifest changed"
    assert output_cap_mib==96 and cfg["output_cap"]==48*2**20
    with (root/"run.lock").open("a") as lock,threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=sqlite3.connect(root/"events.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL")
        assert not db.execute("SELECT count(*) FROM cells WHERE status='RUNNING'").fetchone()[0],"Partial cell needs explicit preservation, not retry"
        frozen=prefix_snapshot(db)
        receipt={"status":"STORAGE_ONLY_OUTPUT_BOUNDARY_SUCCESSOR","prior_owner_pid":wait_pid,"owner_confirmed_absent":True,
                 "original_cap_bytes":cfg["output_cap"],"successor_cap_bytes":output_cap_mib*2**20,"free_floor_unchanged_bytes":cfg["free_floor"],
                 "manifest_sha256_unchanged":file_digest(root/"manifest.json"),"numeric_identity_unchanged":True,"source_prefix":frozen,
                 "reason":"Complete all450planned cells without truncating cases or deleting evidence; observed compact database grows beyond48MiB,96MiB remains below available bounded headroom",
                 "code_sha256":file_digest(Path(__file__))}
        path=root/"runtime-output-boundary-successor-v1.json"
        if path.exists():
            old=json.loads(path.read_text());verify_prefix(db,old["source_prefix"])
            frozen=old["source_prefix"]
        else:save_json(path,receipt)
        cap=output_cap_mib*2**20
        planned=len(cfg["cases"])*len(cfg["seeds"])*len(MODES)
        for case in cfg["cases"]:
            for seed in cfg["seeds"]:
                for mode in MODES:
                    # Completed/failed cells are preserved; only previously
                    # unseen identities may execute. No rerun of pilot costs.
                    from .resource_adaptive_sampler import integral_problem
                    from .online_parameter_controls import task_identity
                    _,info=integral_problem(Path(cfg["source"])/"references"/case)
                    tid=digest(task_identity(cfg,info,seed,mode))
                    saved=db.execute("SELECT status FROM cells WHERE id=?",(tid,)).fetchone()
                    if saved:continue
                    size=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
                    if size>cap or shutil.disk_usage(root).free<cfg["free_floor"] or rss_peak()>cfg["RSS_cap"]:
                        verify_prefix(db,frozen);db.close();raise RuntimeError("Preserved successor resource boundary; no retry or threshold drift")
                    try:construct(db,cfg,case,seed,mode)
                    except Exception as exc:
                        db.execute("UPDATE cells SET status='FAILED_PRESERVED',result=? WHERE id=?",(json.dumps({"error":repr(exc),"cost_unknown_not_zero":True}),tid));db.commit()
                    states=dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
                    save_json(root/"execution-status.json",{"status":"RUNNING_STORAGE_SUCCESSOR","planned":planned,"terminal":sum(states.values()),"states":states})
                print(json.dumps({"case":case,"seed":seed,"states":dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))}),flush=True)
        verify_prefix(db,frozen);states=dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"));db.close()
        save_json(root/"execution-status.json",{"status":"ONLINE_PARAMETER_EXECUTION_PENDING_AUDIT","planned":planned,"terminal":sum(states.values()),"states":states,
                  "events_sha256":file_digest(root/"events.sqlite"),"old_committed_rows_preserved":True,"whole_study_complete":False})
    env=os.environ.copy();env.update({k:"1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS")})
    for module in ("benchmark.online_parameter_audit","benchmark.online_parameter_report"):
        logfile=root/(module.split(".")[-1]+"-sequence-v1.log")
        if module.endswith("audit") and (root/"validation.json").exists():continue
        if module.endswith("report") and (root/"report-v1/report-manifest.json").exists():continue
        save_json(statuspath,{"status":"RUNNING_CHILD","module":module,"log":str(logfile)})
        with logfile.open("x") as log:
            result=subprocess.run([sys.executable,"-m",module,str(root)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save_json(statuspath,{"status":"CHILD_FAILED_EVIDENCE_PRESERVED","module":module,"returncode":result.returncode,"log":str(logfile),"whole_study_complete":False})
            return
    save_json(statuspath,{"status":"ONLINE_CONTROLS_COMPLETE_AUDITED_REPORTED","events_sha256":file_digest(root/"events.sqlite"),
                          "whole_study_complete":False,"numeric_manifest_unchanged":True})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("--wait-pid",required=True,type=int);a=p.parse_args();complete(a.root,a.wait_pid)
