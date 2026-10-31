"""P3 public-data reanalysis, isolated from every common-budget comparison.

The old 0.1.0 SDK cannot currently be retrieved. These jobs use the qualified
0.14.0 runtime and explicitly do NOT certify original-paper number matching.
"""
import json
from pathlib import Path
import shlex
import subprocess
from time import sleep
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from .gpu_qualification import REMOTE,PYTHON
from .io import ROOT,save_json,atomic_bytes,sha256,digest

CASES=("P3-N2-ratio1p00","P3-N2-ratio2p00","P3-FE2S2-40")
NAMESPACE="hardware-postprocessing-v2"


def submit(case):
    root=ROOT/NAMESPACE/case;root.mkdir(parents=True,exist_ok=True)
    if (root/"job.json").exists():return json.loads((root/"job.json").read_text())
    if (root/"submission-intent.json").exists():raise RuntimeError("Uncertain H submission: reconcile before retry")
    qualified=json.loads((ROOT/"hardware-runtime-v1/qualification.json").read_text())
    if qualified["version"]!="0.14.0":raise RuntimeError("H runtime must be import-qualified before submission")
    physical=ROOT/"hardware-physical-v1"/case
    meta=json.loads((physical/"physical.json").read_text())
    family="2Fe-2S" if case.startswith("P3-FE")else"N2_6-31G"
    distribution=ROOT/"hardware-reanalysis-v1"/family/meta["source_weight_label"]/"hardware-distribution.npz"
    remote=REMOTE+"/"+NAMESPACE+"/"+case;ssh("mkdir -p "+remote)
    worker=Path(__file__).with_name("hardware_postprocess_worker.py")
    for local,name in ((physical/"integrals.npz","integrals.npz"),(distribution,"distribution.npz"),(worker,"worker.py")):
        send(local,remote+"/"+name)
    fe=family=="2Fe-2S"
    config={"case_id":case,"problem_id":meta["problem_id"],"norb":meta["norb"],"nelec":meta["nelec"],
        "core_energy":meta["core_energy"],"wall_cap_seconds":3000,"no_CR_D_cap":4000000,
        "CR_samples_per_batch":250 if fe else 1000,"CR_D_cap":250000 if fe else 4000000,
        "configuration_recovery_iterations":10,"num_batches":10,"configuration_recovery_seed":41001,
        "subsampling_seed":42001,"runtime":"qiskit-addon-sqd0.14.0; PySCF2.14.0",
        "source_geometry_and_gauge":meta,"new_actual_QPU_shots":0,"original_hardware_shot_count_reconstructed":False,
        "paper_original_0p1_runtime_reproduced":False,
        "dimension_policy":"samples/batch chosen so spin-symmetrized Cartesian d cannot exceed registered D; actual d recorded, not asserted equal to D",
        "Fe_larger_9M_16M_paper_points_attempted":False,
        "input_sha256":{"integrals.npz":sha256((physical/"integrals.npz").read_bytes()),"distribution.npz":sha256(distribution.read_bytes())},
        "worker_sha256":sha256(worker.read_bytes()),"reference_energy_used_for_construction_or_recovery":False,
        "runtime_qualification":qualified}
    config["config_id"]=digest(config);save_json(root/"request.json",config);send(root/"request.json",remote+"/request.json")
    script="\n".join(["#!/bin/bash","set -euo pipefail","export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1",
        "export PYTHONPATH="+REMOTE+"/H-runtime-v1/packages:"+REMOTE+"/cpu-runtime-v1/packages",
        shlex.join([PYTHON,"-u",remote+"/worker.py",remote+"/request.json"])])+"\n"
    atomic_bytes(root/"job.sbatch",script.encode());send(root/"job.sbatch",remote+"/job.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--cpus-per-task=1","--mem=16G","--time=00:55:00",
        "--job-name=paper-H-"+case[3:16],"--output="+remote+"/slurm.out","--error="+remote+"/slurm.err",remote+"/job.sbatch"])
    intent={"case_id":case,"remote":remote,"config_id":config["config_id"],"command":command,
        "worker_sha256":config["worker_sha256"],"GPU_requested":False,"whole_experiment_complete":False}
    save_json(root/"submission-intent.json",intent);reply=ssh(command).strip().split(";")[0]
    if not reply.isdigit():raise RuntimeError("Unknown H sbatch response")
    job={**intent,"job_id":reply};save_json(root/"job.json",job);print("H_SUBMITTED",case,reply,flush=True);return job


def collect(case,job):
    root=ROOT/NAMESPACE/case
    if (root/"collected.json").exists():return True
    accounting=ssh("sacct -X -n -j "+job["job_id"]+" --format=JobID,State,Elapsed,ExitCode -P")
    save_json(root/"observed-accounting.json",{"accounting":accounting})
    if not any(x in accounting for x in ("COMPLETED","FAILED","TIMEOUT","OUT_OF_MEMORY","CANCELLED")):return False
    # Download only reports/checkpoints, never duplicate the large source distribution.
    raw=ssh("find "+job["remote"]+" -maxdepth 1 -type f -name '*.json' -printf '%f\\n'").splitlines()
    for name in [*raw,"slurm.out","slurm.err"]:
        if not name or "/"in name or name.startswith("."):raise ValueError("Unexpected remote report name")
        target=root/"output";target.mkdir(exist_ok=True)
        subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,HOST+":"+job["remote"]+"/"+name,str(target/name)],check=True,timeout=120)
    result=root/"output/result.json"
    if result.exists():
        report=json.loads(result.read_text())
        if report["configuration"]["config_id"]!=job["config_id"]:raise RuntimeError("H result/config binding failed")
        status=report["status"]
    else:status="PRESERVED_SLURM_TERMINAL_WITHOUT_FINAL_RESULT"
    save_json(root/"collected.json",{"status":status,"accounting":accounting,
        "reports_sha256":{p.name:sha256(p.read_bytes())for p in (root/"output").iterdir()},
        "paper_original_runtime_reproduction_claimed":False,"whole_experiment_complete":False})
    print("H_COLLECTED",case,status,flush=True);return True


def main():
    import fcntl
    with (ROOT/"hardware-controller.lock").open("a")as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            done=sum(collect(c,submit(c))for c in CASES)
            save_json(ROOT/NAMESPACE/"status.json",{"planned_cases":3,"terminal_cases":done,
                "current_runtime_reanalysis_not_original_SDK_reproduction":True,"new_QPU_shots":0})
            if done==3:return
            sleep(45)


if __name__=="__main__":main()
