"""Mac CPU authority for at most two independent A100 case allocations.

Input and output hashes, durable claims, fcntl case locks, uncertain submission
reconciliation and additive imports prevent duplicate molecular computation.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
from time import perf_counter,sleep
from .common import ROOT,save_json,save_binary,file_digest,digest
from .gpu_remote import ssh,send,OPTIONS,HOST
from .engine_remote import REMOTE
from .engine_bundle import create_bundle
from .task_registry import Registry

METHODS=("ceo_ovp","ceo_mvp","qeb","qubit","gsd")


def execute(root,cases):
    root=Path(root).resolve();control=root/"gpu-production-v1";control.mkdir(parents=True,exist_ok=True)
    adoption=json.loads((ROOT/"results/engine-v2-qualification/adoption.json").read_text())
    if adoption["status"]!="QUALIFIED_N2_CUDA_NUMERICAL_ROUTE":raise RuntimeError("A100 backend not qualified")
    for name,sha in adoption["code_hashes"].items():
        if file_digest(ROOT/"benchmark"/name)!=sha:raise RuntimeError("Qualified kernel code changed: "+name)
    registry=Registry(ROOT/"results/task-registry.sqlite");pending=[];live={};locks={}
    send(ROOT/"gpu/case_driver.py",REMOTE+"/case_driver.py")
    for case in cases:
        directory=control/case;directory.mkdir(parents=True,exist_ok=True)
        lockpath=root/"locks"/(case+".lock");lockpath.parent.mkdir(parents=True,exist_ok=True)
        lock=lockpath.open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[case]=lock
        spec={"kind":"molecular-case-production-1","case_id":case,"output":str(root)}
        old=registry.get(spec)
        if old:
            if (directory/"postprocess-terminal.json").exists():continue
            if (directory/"job.json").exists():
                job=json.loads((directory/"job.json").read_text());live[case]=job;continue
            raise RuntimeError("Uncertain prior claim: reconcile before any new submit: "+case)
        owner="A100-case-"+case;task=registry.claim(spec,owner)
        save_json(directory/"claim.json",{"spec":spec,"task":task,"owner":owner,"adoption_sha256":file_digest(ROOT/"results/engine-v2-qualification/adoption.json")})
        started=perf_counter()
        for method in METHODS:create_bundle(root,case,method,directory/"bundles"/method,64,600)
        save_json(directory/"preparation.json",{"cpu_bundle_seconds":perf_counter()-started,
                  "hashes":{str(p.relative_to(directory)):file_digest(p) for p in (directory/"bundles").rglob("*") if p.is_file()}})
        pending.append(case)
    while pending or live:
        while pending and len(live)<2:
            case=pending.pop(0);directory=control/case;claim=json.loads((directory/"claim.json").read_text())
            rd=REMOTE+"/production/"+claim["task"];ssh("mkdir -p "+rd+"/bundles")
            started=perf_counter()
            subprocess.run(["rtk","proxy","scp","-r","-P","2222",*OPTIONS,str(directory/"bundles")+"/.",HOST+":"+rd+"/bundles"],check=True,timeout=120)
            transfer=perf_counter()-started
            script="\n".join(["#!/bin/bash","set -euo pipefail","export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                "cd "+REMOTE,"export CUDA_PATH="+REMOTE+"/venv/lib/python3.12/site-packages/nvidia/cuda_runtime",
                'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+REMOTE+'/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
                'export CUPY_ACCELERATORS=""',shlex.join([REMOTE+"/venv/bin/python",REMOTE+"/case_driver.py",rd+"/bundles",rd+"/results"])])+"\n"
            save_binary(directory/"job.sbatch",script.encode());send(directory/"job.sbatch",rd+"/job.sbatch")
            command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--gres=gpu:a100:1","--cpus-per-task=1",
                     "--mem=8G","--time=01:00:00","--job-name=ceoqsci-prod-"+case,"--output="+rd+"/slurm.out","--error="+rd+"/slurm.err",rd+"/job.sbatch"])
            save_json(directory/"submission-intent.json",{"command":command,"remote":rd,"input_upload_seconds":transfer})
            try:
                answer=ssh(command).strip();job_id=answer.split(";")[0]
                if not job_id.isdigit():raise RuntimeError("Uncertain Slurm reply: "+answer)
                job={"case":case,"job_id":job_id,"remote":rd,**claim};save_json(directory/"job.json",job)
                registry.update(claim["task"],claim["owner"],"SUBMITTED",{"job_id":job_id});live[case]=job
                print("SUBMITTED",case,job_id,flush=True)
            except BaseException as exc:
                registry.update(claim["task"],claim["owner"],"UNKNOWN_RECONCILE",{"error":repr(exc)});raise
        finished=[]
        for case,job in live.items():
            active=ssh("squeue -h -j "+job["job_id"]+" -o %T").strip()
            if active:print("LIVE",case,job["job_id"],active,flush=True);continue
            directory=control/case
            save_binary(directory/"sacct.txt",ssh("sacct -n -j "+job["job_id"]+" --format=JobID,State,Start,End,Elapsed,ExitCode,AllocTRES").encode())
            destination=directory/"results";destination.mkdir(exist_ok=True);started=perf_counter()
            subprocess.run(["rtk","proxy","scp","-r","-P","2222",*OPTIONS,HOST+":"+job["remote"]+"/results/.",str(destination)],check=True,timeout=120)
            save_json(directory/"collection.json",{"output_download_seconds":perf_counter()-started})
            terminal=json.loads((destination/"case-terminal.json").read_text())
            if terminal["status"]!="COMPLETED":
                registry.update(job["task"],job["owner"],"FAILED_PRESERVED",terminal)
                raise RuntimeError("Preserved remote failure: "+case)
            from .chemistry import load_reference
            from .engine_import import import_execution
            reference=load_reference(root/"references"/case)
            for method in METHODS:import_execution(reference,directory/"bundles"/method,destination/method,root)
            registry.update(job["task"],job["owner"],"COMPLETED",{"job_id":job["job_id"],"imported_methods":list(METHODS)})
            # GPU has finished. CPU now performs independent fixed LUCJ/HF controls and readout.
            from .run import _one_case_locked
            from .config import Case
            molecule,distance=case.split("_R")
            options={"output":str(root),"threads":1,"methods":[*METHODS,"lucj_1","lucj_2","lucj_truncated","lucj_random_101","hf","ceo_random_101"],
                     "max_steps":64,"task_seconds":600,"finite_shots":True,"repeats":20,"ci_size":32}
            result=_one_case_locked(Case(molecule,float(distance)),options)
            save_json(directory/"postprocess-terminal.json",result)
            print("IMPORTED_AND_READOUT",case,flush=True);finished.append(case)
        for case in finished:
            del live[case];locks[case].close()
        if live:sleep(10)
    save_json(control/"fleet-terminal.json",{"status":"COMPLETED","cases":cases,"one_A100_per_case":True,
              "configured_max_concurrent_A100":2,"observed_concurrency":"see observed-utilization.json; never infer it from the configured cap"})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--cases",nargs="+",required=True)
    a=p.parse_args();execute(a.output,a.cases)
