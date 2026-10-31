"""Author-motivated TN common-readout controls; isolated bounded Slurm CPU jobs."""
import json
from pathlib import Path
import shlex
from benchmark.gpu_remote import ssh,send
from .common_execution import load_physical,prepare_lucj
from .gpu_qualification import REMOTE,PYTHON
from .io import ROOT,atomic_bytes,save_json,digest,sha256
from .lucj import interaction_pairs


def submit(case="FE2S2-40-P2"):
    root=ROOT/"TN-controls-v1"/case;root.mkdir(parents=True,exist_ok=True)
    if (root/"job.json").exists():return json.loads((root/"job.json").read_text())
    if (root/"submission-intent.json").exists():raise RuntimeError("Uncertain TN Slurm submission")
    qualification_job=json.loads((ROOT/"TN-runtime-v1/job.json").read_text())
    qualification=json.loads(ssh("cat "+qualification_job["remote"]+"/qualification-result.json"))
    if qualification["status"]!="PSD_MADS_OPENMP_RETURNED":raise RuntimeError("Linux PSD-MADS not qualified")
    save_json(ROOT/"TN-runtime-v1/qualification-result.json",qualification)
    meta,_,_=load_physical(case);_,construction=prepare_lucj(case,"C_LUCJ_COMPRESSED_L1")
    remote=REMOTE+"/TN-controls-v1/"+case;ssh("mkdir -p "+remote+"/paperstudy")
    worker=Path(__file__).with_name("tn_optimization_worker.py");send(worker,remote+"/worker.py")
    send(Path(__file__).with_name("ci.py"),remote+"/paperstudy/ci.py")
    send(Path(__file__).with_name("__init__.py"),remote+"/paperstudy/__init__.py")
    physical=ROOT/"data/physical"/case/"integrals.npz"
    operator=ROOT/"prepared"/case/"C_LUCJ_COMPRESSED_L1-L1/operator.npz"
    send(physical,remote+"/integrals.npz");send(operator,remote+"/initial-operator.npz")
    config={"case_id":case,"problem_id":meta["problem_id"],"norb":meta["norb"],"nelec":[meta["nalpha"],meta["nbeta"]],
        "core_energy":meta["core_energy"],"pairs":interaction_pairs(meta["norb"]),"surrogate_seed":31001,"optimizer_seed":31001,
        "initialization":construction,"MPS_max_bond":50,"MPS_cutoff":1e-10,"surrogate_samples_per_call":10000,
        "common_CI_r_cap":32,"maximum_valid_objective_calls":500,"objective_wall_cap_seconds":3000,
        "paper_native_Dice_objective_reproduced":False,"paper_subproblem_count20_reproduced":False,
        "subproblem_count":3,"parallel_evaluations":4,"variant_rationale":"Three subproblems and pollster fit four allocated CPU threads; common r32 postprocessing aligns the operational CI cap",
        "optimizer_state":"RESET","FCI_or_SCI_reference_used":False,"worker_sha256":sha256(worker.read_bytes()),
        "CI_code_sha256":sha256(Path(__file__).with_name("ci.py").read_bytes()),
        "input_sha256":{"integrals.npz":sha256(physical.read_bytes()),"initial-operator.npz":sha256(operator.read_bytes())},
        "evaluation_lane":"S_TN_COMMON_READOUT_VARIANT_NOT_PAPER_REPRODUCTION"}
    config["config_id"]=digest(config);save_json(root/"request.json",config);send(root/"request.json",remote+"/request.json")
    script="\n".join(["#!/bin/bash","set -euo pipefail","export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1",
        "export PYTHONPATH="+remote+":"+REMOTE+"/tn-runtime-v1/packages:"+REMOTE+"/cpu-runtime-v1/packages",
        shlex.join([PYTHON,"-u",remote+"/worker.py",remote+"/request.json"])])+"\n"
    atomic_bytes(root/"job.sbatch",script.encode());send(root/"job.sbatch",remote+"/job.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--cpus-per-task=4","--mem=16G","--time=00:55:00",
        "--job-name=paper-TN-"+case[:5],"--output="+remote+"/slurm.out","--error="+remote+"/slurm.err",remote+"/job.sbatch"])
    intent={"case_id":case,"remote":remote,"command":command,"config_id":config["config_id"],"worker_sha256":config["worker_sha256"],
            "GPU_requested":False,"model_QPU_shots":0,"classical_surrogate_draws_are_not_free_preprocessing":True}
    save_json(root/"submission-intent.json",intent);reply=ssh(command).strip().split(";")[0]
    if not reply.isdigit():raise RuntimeError("Unknown TN Slurm reply")
    job={**intent,"job_id":reply};save_json(root/"job.json",job);print(json.dumps(job),flush=True);return job


if __name__=="__main__":
    import argparse;p=argparse.ArgumentParser();p.add_argument("--case",default="FE2S2-40-P2");a=p.parse_args();submit(a.case)
