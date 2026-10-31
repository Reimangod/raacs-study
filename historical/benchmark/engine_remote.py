"""Versioned, CPU-owned A100 engine qualification/dispatch (never old env writes)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from .common import ROOT,save_json,save_binary,file_digest
from .gpu_remote import ssh,send,OPTIONS,HOST
from .task_registry import Registry

REMOTE="/share/reiwatanabe/ceo-qsci-engine-20261022-v2"
EVIDENCE=ROOT/"results/engine-v2-qualification"


def setup():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    if (EVIDENCE/"environment.json").exists():
        raise RuntimeError("Versioned runtime already exists; do not overwrite qualified files. Use a new runtime version for changes.")
    ssh("mkdir -p "+REMOTE+"/benchmark")
    sources={"gpu/setup.py":"setup.py","gpu/engine-v2-requirements.txt":"requirements.txt",
             "gpu/engine_driver.py":"engine_driver.py"}
    for name in ("__init__.py","common.py","cuda_backend.py","gpu_profile_worker.py","engine_worker.py"):
        sources["benchmark/"+name]="benchmark/"+name
    for source,target in sources.items():send(ROOT/source,REMOTE+"/"+target)
    answer=ssh("/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python "+REMOTE+"/setup.py",timeout=300)
    save_json(EVIDENCE/"environment.json",{"remote":REMOTE,"result":answer,
              "sources":{s:file_digest(ROOT/s) for s in sources}})
    print(answer)


def submit(bundle,case_id,purpose="QUALIFICATION_ONLY"):
    bundle=Path(bundle).resolve();request=json.loads((bundle/"request.json").read_text())
    owner="A100-engine-v2-"+case_id
    registry=Registry(ROOT/"results/task-registry.sqlite")
    task=registry.claim({"case_id":case_id,"request":request,"purpose":purpose},owner)
    directory=EVIDENCE/case_id;directory.mkdir(parents=True,exist_ok=True)
    rd=REMOTE+"/tasks/"+task;ssh("mkdir -p "+rd+"/bundle")
    transfer=0.
    for path in sorted(bundle.iterdir()):transfer+=send(path,rd+"/bundle/"+path.name)
    script="\n".join(["#!/bin/bash","set -euo pipefail",
         "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
         "cd "+REMOTE,
         "export CUDA_PATH="+REMOTE+"/venv/lib/python3.12/site-packages/nvidia/cuda_runtime",
         'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+REMOTE+'/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
         'export CUPY_ACCELERATORS=""',
         shlex.join([REMOTE+"/venv/bin/python",REMOTE+"/engine_driver.py","--bundle",rd+"/bundle","--output",rd+"/result","--backend","cupy"])])+"\n"
    path=directory/"job.sbatch";save_binary(path,script.encode());send(path,rd+"/job.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--gres=gpu:a100:1",
                        "--cpus-per-task=1","--mem=8G","--time=00:30:00",
                        "--job-name=ceoqsci-engine-"+case_id,"--output="+rd+"/slurm.out","--error="+rd+"/slurm.err",rd+"/job.sbatch"])
    save_json(directory/"intent.json",{"task":task,"owner":owner,"purpose":purpose,"command":command,
              "input_upload_seconds":transfer,"remote":rd})
    try:
        answer=ssh(command).strip();job=answer.split(";")[0]
        if not job.isdigit():raise RuntimeError("Uncertain submission: "+answer)
        registry.update(task,owner,"SUBMITTED",{"job_id":job})
        save_json(directory/"job.json",{"job_id":job,"task":task,"owner":owner,"remote":rd})
        print("SUBMITTED",case_id,job)
    except BaseException as exc:
        registry.update(task,owner,"UNKNOWN_RECONCILE",{"error":repr(exc)})
        raise


def collect():
    registry=Registry(ROOT/"results/task-registry.sqlite")
    for path in EVIDENCE.glob("*/job.json"):
        j=json.loads(path.read_text())
        active=ssh("squeue -h -j "+j["job_id"]+" -o %T").strip()
        if active:print("LIVE",j["job_id"],active);continue
        acct=ssh("sacct -n -j "+j["job_id"]+" --format=JobID,State,Elapsed,ExitCode,AllocTRES")
        save_binary(path.parent/"sacct.txt",acct.encode())
        dest=path.parent/"result";dest.mkdir(exist_ok=True)
        subprocess.run(["rtk","proxy","scp","-r","-P","2222",*OPTIONS,HOST+":"+j["remote"]+"/result/.",str(dest)],check=True,timeout=120)
        for name in ("slurm.out","slurm.err"):
            subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,HOST+":"+j["remote"]+"/"+name,str(path.parent/name)],check=True,timeout=120)
        launch=json.loads((dest/"cold-launch.json").read_text())
        state="COMPLETED" if launch["returncode"]==0 else "FAILED_PRESERVED"
        registry.update(j["task"],j["owner"],state,{"job_id":j["job_id"],"launch":launch})
        print("COLLECTED",path.parent.name,state)


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["setup","submit","collect"])
    p.add_argument("--bundle");p.add_argument("--case");a=p.parse_args()
    if a.action=="setup":setup()
    elif a.action=="submit":submit(a.bundle,a.case)
    else:collect()


if __name__=="__main__":main()
