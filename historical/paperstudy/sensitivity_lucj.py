"""S5 L2/L4: author initializer, exact GPU sampling, independent CPU parity.

New namespace only. No primary input, method, count or frozen code is edited.
Five readouts share one exact state but have independent registered seeds.
"""
import argparse
import fcntl
from io import BytesIO
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
from time import perf_counter, process_time

import ffsim
import numpy as np
from qiskit import qpy

from benchmark.gpu_remote import ssh, send, OPTIONS, HOST
from .budget import BudgetLedger
from .common_execution import load_physical, final_readouts, run_identity
from .gpu_lucj_payload import payload
from .gpu_qualification import REMOTE, PYTHON, ENV
from .input_adapter import array_bytes
from .io import ROOT, PLAN_ROOT, atomic_bytes, digest, save_json, sha256
from .lucj import amplitudes_from_saved, naive, interaction_pairs, make_circuit, circuit_views

SEEDS = (21001,21011,21019,21023,21031)
CASES = ("NAPH20-P1","N2-32-R1p2-P2","N2-32-R2p4-P2","FE2S2-40-P2")


def method_id(compressed,n_reps):
    if n_reps not in (2,4):
        raise ValueError("Unregistered layer count")
    return ("C_LUCJ_COMPRESSED_L" if compressed else "B_LUCJ_NAIVE_L")+str(n_reps)


def freeze():
    filename=ROOT/"sensitivity-freeze-v1.json"
    files=[Path(__file__),Path(__file__).with_name("sensitivity_initialization.py"),
           Path(__file__).with_name("sensitivity_sqdrift.py"),Path(__file__).with_name("sensitivity_gpu_worker.py")]
    spec={"schema":"paper-S5-freeze-1","plan_sha256":sha256((PLAN_ROOT/"plan.json").read_bytes()),
          "code_sha256":{str(p):sha256(p.read_bytes()) for p in files},
          "cases":list(CASES),"seeds":list(SEEDS),"model_shots_per_run":100000,
          "lucj_layers":[2,4],"SqDRIFT_lengths":[5,50],"naph_additional_member_shots":512,
          "planned_cells":125,"primary_outcomes_used_for_settings":False,
          "selection": "All registered settings, no winner selection", "actual_QPU_shots":0}
    if filename.exists():
        if json.loads(filename.read_text()) != spec:
            raise RuntimeError("S5 code/settings changed: use explicit successor")
    else:
        save_json(filename,spec)
    return spec


def prepare(case,compressed,n_reps):
    mid=method_id(compressed,n_reps)
    directory=ROOT/"sensitivity-prepared"/case/mid
    filename=directory/"construction.json"
    if filename.exists():
        record=json.loads(filename.read_text())
        if sha256((directory/"operator.npz").read_bytes()) != record["operator_sha256"]:
            raise RuntimeError("S5 operator changed")
        with np.load(directory/"operator.npz",allow_pickle=False) as a:
            op=ffsim.UCJOpSpinBalanced(a["diag_coulomb_mats"],a["orbital_rotations"],a["final_orbital_rotation"])
        return directory,op,record
    meta,_,_=load_physical(case)
    tick,cpu=perf_counter(),process_time()
    if compressed:
        init=ROOT/"sensitivity-initialization"/case/("compressed-L"+str(n_reps))
        evidence=json.loads((init/"result.json").read_text())
        if evidence["status"] != "AUTHOR_INITIALIZER_RETURNED" or evidence["n_reps"]!=n_reps:
            raise RuntimeError("S5 compressed initialization unavailable; not replaced by naive")
        blob=(init/"operator.npz").read_bytes()
        if sha256(blob)!=evidence["operator_sha256"]:
            raise RuntimeError("S5 initializer operator binding failed")
        with np.load(BytesIO(blob),allow_pickle=False) as a:
            op=ffsim.UCJOpSpinBalanced(a["diag_coulomb_mats"],a["orbital_rotations"],a["final_orbital_rotation"])
        initial_cost={"source_result":str((init/"result.json").relative_to(ROOT)),
                      "CPU_seconds":evidence["CPU_process_seconds"],"wall_seconds":evidence["total_wall_seconds"],
                      "all_stages_converged":evidence["all_stages_optimizer_success"]}
    else:
        init=ROOT/"data/initialization-fresh"/case
        t1,t2,kind=amplitudes_from_saved(init/"amplitudes.npz",meta["norb"],meta["nalpha"])
        op=naive(t1,t2,meta["norb"],n_reps)
        blob=array_bytes({"diag_coulomb_mats":op.diag_coulomb_mats,"orbital_rotations":op.orbital_rotations,
                          "final_orbital_rotation":op.final_orbital_rotation})
        initial_cost={"amplitude_source":str((init/"result.json").relative_to(ROOT)),"kind":kind,
                      "factorization_CPU_seconds":process_time()-cpu,"factorization_wall_seconds":perf_counter()-tick}
    directory.mkdir(parents=True,exist_ok=True)
    atomic_bytes(directory/"operator.npz",blob)
    prep,sampling,resources=circuit_views(make_circuit(op,meta["norb"],(meta["nalpha"],meta["nbeta"])))
    for name,circuit in (("preparation",prep),("sampling",sampling)):
        stream=BytesIO();qpy.dump(circuit,stream);atomic_bytes(directory/(name+".qpy"),stream.getvalue())
    record={"case_id":case,"problem_id":meta["problem_id"],"method_id":mid,"n_reps":n_reps,
            "operator_sha256":sha256(blob),"resources":resources,"initialization_cost":initial_cost,
            "coefficient_slots":len(op.to_parameters(interaction_pairs=interaction_pairs(meta["norb"]))),
            "nonredundant_independent_parameters":None,"VQE_optimized_variables":0,
            "preparation_CPU_seconds":process_time()-cpu,"preparation_wall_seconds":perf_counter()-tick}
    save_json(filename,record)
    return directory,op,record


def submit(case,compressed,n_reps):
    freeze()
    directory,op,construction=prepare(case,compressed,n_reps)
    mid=method_id(compressed,n_reps)
    root=ROOT/"remote-S5-LUCJ-v1"/case/mid
    root.mkdir(parents=True,exist_ok=True)
    if (root/"job.json").exists():
        return json.loads((root/"job.json").read_text())
    if (root/"submission-intent.json").exists():
        raise RuntimeError("Unknown S5 Slurm submission: reconcile, do not duplicate")
    # Actual full32q qualification must precede production, for both initializers.
    for name in ("B_LUCJ_NAIVE_L1","C_LUCJ_COMPRESSED_L1"):
        obs=json.loads((ROOT/"qualification/GPU-Givens-v1"/name/"observed-status.json").read_text())
        qr=json.loads(obs["result_or_stderr"])
        if qr["status"]!="EXACT_SECTOR_GPU_LUCJ_COMPLETED" or max(qr["independent_ffsim_parity"].values())>1e-8:
            raise RuntimeError("Full32q independent implementation qualification incomplete")
    remote=REMOTE+"/S5-LUCJ-v1/"+case+"/"+mid
    ssh("mkdir -p "+remote)
    worker=Path(__file__).with_name("sensitivity_gpu_worker.py")
    send(worker,remote+"/worker.py")
    send(directory/"operator.npz",remote+"/operator.npz")
    meta,_,_=load_physical(case)
    requests=[]
    ledger=BudgetLedger(ROOT/"common-budget.sqlite")
    for seed in SEEDS:
        run_id=run_identity(case,mid,100000,seed,"S5_SENSITIVITY")
        ledger.register(run_id,100000)
        request={"operator_sha256":construction["operator_sha256"],"problem_id":meta["problem_id"],
                 "seed":seed,"shots":100000,"role":"independent-final"}
        reservation=ledger.reserve(run_id,"independent-final",request,100000)
        if reservation not in ("NEW_RESERVATION","RESERVED"):
            raise RuntimeError("S5 draw already executed; collect rather than submit")
        requests.append({"id":run_id,"seed":seed,"shots":100000})
    configuration,arrays=payload(op,meta["norb"],(meta["nalpha"],meta["nbeta"]),requests)
    configuration["independent_CPU_operator"]={"path":remote+"/operator.npz","sha256":construction["operator_sha256"]}
    configuration["problem_id"]=meta["problem_id"]
    configuration["worker_sha256"]=sha256(worker.read_bytes())
    array_blob=array_bytes(arrays)
    configuration["arrays_sha256"]=sha256(array_blob)
    configuration["input_id"]=digest({k:v for k,v in configuration.items() if k!="input_id"})
    save_json(root/"request.json",configuration)
    atomic_bytes(root/"arrays.npz",array_blob)
    send(root/"request.json",remote+"/request.json");send(root/"arrays.npz",remote+"/arrays.npz")
    script="\n".join(["#!/bin/bash","set -euo pipefail",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1",
        "export PYTHONPATH="+REMOTE+"/cpu-runtime-v1/packages",
        "export CUDA_PATH="+ENV+"/lib/python3.12/site-packages/nvidia/cuda_runtime",
        'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
        shlex.join([PYTHON,remote+"/worker.py",remote+"/request.json",remote+"/arrays.npz",remote+"/output"])])+"\n"
    atomic_bytes(root/"job.sbatch",script.encode());send(root/"job.sbatch",remote+"/job.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--gres=gpu:a100:1","--cpus-per-task=1",
        "--mem=32G","--time=00:20:00","--job-name=paper-S5-"+case[:4]+"-"+mid[0]+str(n_reps),
        "--output="+remote+"/slurm.out","--error="+remote+"/slurm.err",remote+"/job.sbatch"])
    intent={"case_id":case,"method_id":mid,"n_reps":n_reps,"remote":remote,"command":command,
            "input_id":configuration["input_id"],"worker_sha256":sha256(worker.read_bytes())}
    save_json(root/"submission-intent.json",intent)
    reply=ssh(command).strip().split(";")[0]
    if not reply.isdigit():
        raise RuntimeError("Unknown S5 Slurm reply "+reply)
    job={**intent,"job_id":reply};save_json(root/"job.json",job)
    print("S5_SUBMITTED",case,mid,reply,flush=True)
    return job


def collect(case,compressed,n_reps):
    mid=method_id(compressed,n_reps)
    directory=ROOT/"remote-S5-LUCJ-v1"/case/mid
    job=json.loads((directory/"job.json").read_text())
    queue=ssh("squeue -h -u reiwatanabe -o '%i %T %R'")
    live=[line for line in queue.splitlines() if line.split() and line.split()[0]==job["job_id"]]
    accounting=ssh("sacct -X -n -j "+job["job_id"]+" --format=JobID,State,Start,End,Elapsed,ExitCode,AllocTRES -P")
    save_json(directory/"slurm-observed.json",{"live":live,"accounting":accounting})
    if live:
        return "LIVE"
    if "COMPLETED" not in accounting:
        return "NEEDS_RECONCILIATION"
    record=json.loads(ssh("cat "+job["remote"]+"/output/result.json"))
    if record["input_id"]!=job["input_id"] or record["status"]!="EXACT_SECTOR_GPU_LUCJ_COMPLETED":
        raise RuntimeError("S5 GPU result binding failed")
    if record["independent_ffsim_parity"] is None or max(record["independent_ffsim_parity"].values())>1e-8:
        raise RuntimeError("Actual production-state CPU/GPU parity failed")
    save_json(directory/"remote-result.json",record)
    prepared,_,construction=prepare(case,compressed,n_reps)
    meta,h1,eri=load_physical(case)
    ledger=BudgetLedger(ROOT/"common-budget.sqlite")
    for request in record["sampling_results"]:
        seed=request["seed"];run_id=run_identity(case,mid,100000,seed,"S5_SENSITIVITY")
        if request["id"]!=run_id or request["actual_model_shots"]!=100000:
            raise RuntimeError("S5 readout binding failed")
        output=ROOT/"common-runs/S5_SENSITIVITY"/case/mid/("100000-"+str(seed))
        output.mkdir(parents=True,exist_ok=True)
        if (output/"result.json").exists():
            continue
        subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,HOST+":"+job["remote"]+"/output/"+request["count_file"],str(output/"final-counts.npz")],check=True,timeout=120)
        blob=(output/"final-counts.npz").read_bytes()
        with np.load(BytesIO(blob),allow_pickle=False) as a:
            ids,counts=a["blocked_ids"].copy(),a["counts"].copy()
        if int(counts.sum())!=100000:
            raise RuntimeError("Wrong S5 shot count")
        ledger.complete(run_id,"independent-final",100000,sha256(blob))
        tick,cpu=perf_counter(),process_time()
        readouts=final_readouts(output,meta,h1,eri,ids,counts)
        result={"status":"COMPLETED","lane":"S5_SENSITIVITY","run_id":run_id,"case_id":case,
            "method_id":mid,"problem_id":meta["problem_id"],"budget":100000,"seed":seed,
            "budget_ledger":ledger.snapshot(run_id),"final_CI":readouts,"pilot_model_shots":0,
            "independent_final_model_shots":100000,"actual_QPU_shots":0,
            "sampling_resources":construction["resources"]["Z_sampling_equivalent"],
            "preparation_resources":construction["resources"]["preparation"],
            "coefficient_slots":construction["coefficient_slots"],"nonredundant_independent_parameters":None,
            "VQE_optimized_variables":0,"construction":construction,"remote_shared_preparation_id":job["input_id"],
            "remote_result":str((directory/"remote-result.json").relative_to(ROOT)),"GPU_job_id":job["job_id"],
            "this_attempt_wall_seconds":perf_counter()-tick,"this_attempt_CPU_seconds":process_time()-cpu,
            "local_attempt_time_is_CI_only":True,"FCI_used_for_construction":False,
            "total_model_cnot_applications":100000*construction["resources"]["Z_sampling_equivalent"]["cnot_count"]}
        save_json(output/"result.json",result)
        print("S5_COLLECTED",case,mid,seed,flush=True)
    return "COMPLETED"


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["freeze","submit","collect"])
    p.add_argument("--case",default="NAPH20-P1");p.add_argument("--compressed",action="store_true")
    p.add_argument("--n-reps",type=int,choices=[2,4],default=2);a=p.parse_args()
    if a.action=="freeze": print(json.dumps(freeze()))
    elif a.action=="submit":print(json.dumps(submit(a.case,a.compressed,a.n_reps)))
    else:print(collect(a.case,a.compressed,a.n_reps))
