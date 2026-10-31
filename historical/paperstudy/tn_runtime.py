"""Separate TN/NOMAD runtime and a tiny Slurm-only optimizer qualification."""
import json
from pathlib import Path
import shlex
from benchmark.gpu_remote import ssh,send
from .gpu_qualification import REMOTE,PYTHON
from .io import ROOT,atomic_bytes,save_json,sha256

PACKAGES=["PyNomadBBO==4.5.1","qiskit-quimb==0.0.10","quimb==1.15.0","autoray==0.11.0",
          "cotengra==0.8.3","numba==0.68.0","llvmlite==0.50.0","networkx==3.6.1","toolz==1.2.0",
          "cytoolz==1.2.0","opt_einsum==3.4.0","tqdm==4.70.1","psutil==7.2.2"]


def main():
    root=ROOT/"TN-runtime-v1";root.mkdir(exist_ok=True)
    remote=REMOTE+"/tn-runtime-v1";ssh("mkdir -p "+remote)
    existing=ssh("if test -f "+remote+"/runtime.json; then cat "+remote+"/runtime.json; fi").strip()
    if not existing:
        command=shlex.join([PYTHON,"-m","pip","install","--target",remote+"/packages","--no-deps","--no-cache-dir",*PACKAGES])
        result=ssh(command,timeout=180)
        save_json(root/"install-log.json",{"command":command,"output":result,"old_environments_modified":False})
        spec={"packages":PACKAGES,"new_target":remote+"/packages","CPU_runtime_shared_read_only":REMOTE+"/cpu-runtime-v1/packages"}
        save_json(root/"runtime.json",spec);send(root/"runtime.json",remote+"/runtime.json")
    else:save_json(root/"runtime.json",json.loads(existing))
    if (root/"job.json").exists():return json.loads((root/"job.json").read_text())
    if (root/"submission-intent.json").exists():raise RuntimeError("Uncertain TN qualification submission")
    code='''import json,PyNomad,os
from pathlib import Path
from time import perf_counter
def fun(x):
    v=sum((x.get_coord(i)-.1)**2 for i in range(x.size()))
    x.setBBO(str(v).encode());return 1
start=perf_counter()
options=["BB_OUTPUT_TYPE OBJ","MAX_BB_EVAL 20","DISPLAY_DEGREE 0","PSD_MADS_OPTIMIZATION true","PSD_MADS_NB_VAR_IN_SUBPROBLEM 5","PSD_MADS_SUBPROBLEM_MAX_BB_EVAL 4","PSD_MADS_NB_SUBPROBLEM 3","NB_THREADS_PARALLEL_EVAL 4"]
r=PyNomad.optimize(fun,[0.]*24,[],[],options)
result={"status":"PSD_MADS_OPENMP_RETURNED","optimizer_result":r,"options":options,"wall_seconds":perf_counter()-start,"CPU_allocation":4,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"molecular_outcomes_evaluated":False}
Path(__file__).with_name("qualification-result.json").write_text(json.dumps(result,indent=2)+"\\n")
print(json.dumps(result),flush=True)
'''
    atomic_bytes(root/"qualification.py",code.encode());send(root/"qualification.py",remote+"/qualification.py")
    script="\n".join(["#!/bin/bash","set -euo pipefail","export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export PYTHONPATH="+remote+"/packages:"+REMOTE+"/cpu-runtime-v1/packages",
        shlex.join([PYTHON,"-u",remote+"/qualification.py"])])+"\n"
    atomic_bytes(root/"qualification.sbatch",script.encode());send(root/"qualification.sbatch",remote+"/qualification.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--cpus-per-task=4","--mem=2G","--time=00:05:00",
        "--job-name=paper-TN-qual","--output="+remote+"/slurm.out","--error="+remote+"/slurm.err",remote+"/qualification.sbatch"])
    intent={"remote":remote,"command":command,"source_sha256":sha256(code.encode()),"model_shots":0}
    save_json(root/"submission-intent.json",intent);reply=ssh(command).strip().split(";")[0]
    if not reply.isdigit():raise RuntimeError("Unknown TN qualification submission")
    job={**intent,"job_id":reply};save_json(root/"job.json",job);print(json.dumps(job),flush=True);return job


if __name__=="__main__":main()
