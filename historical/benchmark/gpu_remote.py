"""Small, durable AIC profile dispatcher. Does not stop current jobs or auto-retry sbatch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from .common import ROOT,save_json,file_digest

REMOTE="/share/reiwatanabe/ceo-qsci-sampler-20261022-v1"
HOST="reiwatanabe@172.16.0.18"
KEY="/Users/rei/.ssh/id_ed25519"
PROXY=f"ssh -p 2222 -i {KEY} -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -W %h:%p reiwatanabe@casper.ai.hc.keio.ac.jp"
OPTIONS=["-i",KEY,"-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes",
         "-o","ConnectTimeout=20","-o","ProxyCommand="+PROXY]


def ssh(command,timeout=120):
    return subprocess.check_output(["rtk","proxy","ssh","-p","2222",*OPTIONS,HOST,command],text=True,timeout=timeout)


def send(path,target):
    started=time.perf_counter()
    subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,str(path),HOST+":"+target],check=True,timeout=120)
    return time.perf_counter()-started


def setup():
    evidence=ROOT/"results/gpu-remote";evidence.mkdir(parents=True,exist_ok=True)
    if (evidence/"environment-setup.json").exists():
        raise RuntimeError("Profile runtime already frozen; use a new version, not an overwrite")
    ssh("mkdir -p "+shlex.quote(REMOTE))
    for path in [ROOT/"gpu/setup.py",ROOT/"gpu/requirements.txt",ROOT/"gpu/profile_driver.py",ROOT/"benchmark/gpu_profile_worker.py"]:
        send(path,REMOTE+"/"+path.name)
    answer=ssh("/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python "+shlex.quote(REMOTE+"/setup.py"),timeout=300)
    save_json(evidence/"environment-setup.json",{"remote":REMOTE,"result":answer,
              "input_hashes":{p.name:file_digest(p) for p in [ROOT/"gpu/setup.py",ROOT/"gpu/requirements.txt"]}})
    print(answer)


def submit(fixture,case_id):
    fixture=Path(fixture).resolve();directory=ROOT/"results/gpu-remote"/case_id
    directory.mkdir(parents=True,exist_ok=True)
    attempt=directory/"submission-started.json"
    if attempt.exists():
        raise RuntimeError("Existing submission attempt: reconcile job ID/squeue/sacct, never duplicate")
    rd=REMOTE+"/profiles/"+case_id
    ssh("mkdir -p "+shlex.quote(rd))
    transfer=send(fixture,rd+"/fixture.npz")
    script="\n".join(["#!/bin/bash","set -euo pipefail",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export CUDA_PATH="+REMOTE+"/venv/lib/python3.12/site-packages/nvidia/cuda_runtime",
        'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+REMOTE+'/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
        'export CUPY_ACCELERATORS=""',
        shlex.join([REMOTE+"/venv/bin/python",REMOTE+"/profile_driver.py","--fixture",rd+"/fixture.npz","--output",rd+"/result"])])+"\n"
    path=directory/"job.sbatch";path.write_text(script)
    send(path,rd+"/job.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--gres=gpu:a100:1","--cpus-per-task=1",
                        "--mem=8G","--time=00:10:00","--job-name=ceoqsci-profile-"+case_id,
                        "--output="+rd+"/slurm.out","--error="+rd+"/slurm.err",rd+"/job.sbatch"])
    # Persist intent before external mutation. On uncertain network result inspect exact name.
    save_json(attempt,{"case_id":case_id,"fixture_sha256":file_digest(fixture),"command":command,
                       "fixture_upload_seconds":transfer,"purpose":"GPU_PROFILE_NOT_SCIENTIFIC_TRAJECTORY"})
    answer=ssh(command).strip();job=answer.split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Uncertain sbatch outcome: "+answer)
    save_json(directory/"job-id.json",{"job_id":job,"case_id":case_id,"remote":rd})
    print("SUBMITTED",case_id,job)


def status():
    for path in (ROOT/"results/gpu-remote").glob("*/job-id.json"):
        job=json.loads(path.read_text())
        result=ssh("sacct -n -j "+job["job_id"]+" --format=JobID,State,Elapsed,ExitCode")
        print(job["case_id"],result)


def collect():
    for path in (ROOT/"results/gpu-remote").glob("*/job-id.json"):
        job=json.loads(path.read_text());dest=path.parent/"result"
        active=ssh("squeue -h -j "+job["job_id"]+" -o %T").strip()
        if active:
            print("LIVE",job["job_id"],active);continue
        dest.mkdir(exist_ok=True)
        subprocess.run(["rtk","proxy","scp","-r","-P","2222",*OPTIONS,HOST+":"+job["remote"]+"/result/.",str(dest)],check=True,timeout=120)
        for name in ("slurm.out","slurm.err"):
            subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,HOST+":"+job["remote"]+"/"+name,str(path.parent/name)],check=True,timeout=120)
        print("COLLECTED",job["case_id"])


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["setup","submit","status","collect"])
    p.add_argument("--fixture");p.add_argument("--case");a=p.parse_args()
    if a.action=="setup":setup()
    elif a.action=="submit":submit(a.fixture,a.case)
    elif a.action=="status":status()
    else:collect()


if __name__=="__main__":main()
