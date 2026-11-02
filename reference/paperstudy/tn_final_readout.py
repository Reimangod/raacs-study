"""Independent exact final readout after bounded TN surrogate optimization.

The CPU fallback uses the same pinned ffsim physics as the qualified GPU path.
Classical surrogate draws/optimization costs remain separate and are not free.
"""
from io import BytesIO
import json
from pathlib import Path
import shlex
import subprocess
from time import perf_counter,process_time,sleep
import ffsim
import numpy as np

from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from .budget import BudgetLedger
from .common_execution import load_physical,final_readouts,run_identity
from .gpu_qualification import REMOTE,PYTHON
from .input_adapter import array_bytes
from .io import ROOT,atomic_bytes,save_json,digest,sha256
from .lucj import circuit_views,make_circuit,interaction_pairs

CASES=("FE2S2-40-P2","NAPH20-P1","N2-32-R1p2-P2","N2-32-R2p4-P2")
SEEDS=(21001,21011,21019,21023,21031)
METHOD="E_LUCJ_TN_COMMON_L1"


def submit(case):
    directory=ROOT/"TN-controls-v1"/case
    effective_path=directory/"effective-source-v2.json"
    effective=json.loads(effective_path.read_text())if effective_path.exists()else None
    repair_root=ROOT/"TN-controls-v2"/case
    if not effective and (ROOT/"TN-successor-v2/contract.json").exists():
        # A bounded, already authorized engineering repair must be allowed to
        # publish its checkpoint. Do not mistake the original failed attempt
        # for the terminal state of the successor while it is still running.
        if (repair_root/"no-final-state.json").exists():
            raise RuntimeError("TN bounded successor terminal without usable final state")
        original=directory/"terminal-attention.json"
        if original.exists() and not (repair_root/"effective-source-published.json").exists():
            return None
    source_directory=ROOT/effective["local_source_directory"]if effective else directory
    job=effective["job"]if effective else json.loads((directory/"job.json").read_text());remote=job["remote"]
    final=directory/"final-readout";final.mkdir(exist_ok=True)
    if (final/"job.json").exists():return json.loads((final/"job.json").read_text())
    state=ssh("sacct -X -n -j "+job["job_id"]+" --format=JobID,State,Elapsed,ExitCode -P")
    save_json(directory/"optimizer-slurm-observed.json",{"accounting":state})
    if not effective and any(s in state for s in ("FAILED","TIMEOUT","OUT_OF_MEMORY","CANCELLED")):
        save_json(directory/"terminal-attention.json",{"actual_slurm_accounting":state,"missing_final_state_is_not_scientific_failure":True,
                  "no_automatic_reoptimization_or_resampling":True})
        raise RuntimeError("TN optimizer terminal requires checkpoint reconciliation: "+state)
    if "COMPLETED" not in state and not effective:return None
    metadata=effective["metadata_file"]if effective else"optimization-result.json"
    optimized=json.loads(ssh("cat "+remote+"/"+metadata))
    if optimized["status"]not in("TN_COMMON_READOUT_SURROGATE_RETURNED","TN_COMMON_READOUT_BOUNDED_CHECKPOINT_RECOVERED") or optimized["configuration"]["config_id"]!=job["config_id"]:
        raise RuntimeError("TN optimizer result binding failure")
    if effective and digest(optimized)!=effective["source_result_sha256"]:raise RuntimeError("TN successor metadata changed after source publication")
    metadata_local=source_directory/("selected-source-result-fetched.json"if effective else"optimization-result.json")
    save_json(metadata_local,optimized)
    operator=source_directory/("selected-operator.npz"if effective else"optimized-operator.npz")
    operator_remote=remote+"/"+(effective["remote_operator_file"]if effective else"optimized-operator.npz")
    subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,HOST+":"+operator_remote,str(operator)],check=True,timeout=120)
    if sha256(operator.read_bytes())!=optimized["operator_sha256"]:raise RuntimeError("TN optimized operator mismatch")
    with np.load(operator,allow_pickle=False) as a:op=ffsim.UCJOpSpinBalanced(a["diag_coulomb_mats"],a["orbital_rotations"],a["final_orbital_rotation"])
    m,_,_=load_physical(case)
    _,_,views=circuit_views(make_circuit(op,m["norb"],(m["nalpha"],m["nbeta"])))
    construction={"resources":views,"operator_sha256":optimized["operator_sha256"],
        "coefficient_slots":len(op.to_parameters(interaction_pairs=interaction_pairs(m["norb"]))),
        "optimization_source":str(metadata_local.relative_to(ROOT)),
        "engineering_successor_used":effective is not None,"original_failed_attempt_cost_retained":effective is not None,
        "nonredundant_independent_parameters":None,"VQE_optimized_variables":0,
        "surrogate_optimized_coordinate_slots":len(optimized["selected_coefficients"])}
    save_json(final/"construction.json",construction)
    if (final/"submission-intent.json").exists():raise RuntimeError("Unknown final-readout submission; reconcile first")
    ledger=BudgetLedger(ROOT/"common-budget.sqlite");requests=[]
    for seed in SEEDS:
        key=run_identity(case,METHOD,100000,seed,"S_TN_INDEPENDENT_FINAL")
        ledger.register(key,100000)
        request={"operator_sha256":construction["operator_sha256"],"problem_id":m["problem_id"],"seed":seed,"shots":100000,"role":"independent-final"}
        status=ledger.reserve(key,"independent-final",request,100000)
        if status not in ("NEW_RESERVATION","RESERVED"):raise RuntimeError("Already executed TN readout; do not duplicate")
        requests.append({"run_id":key,"seed":seed,"shots":100000})
    destination=remote+"/final-readout";ssh("mkdir -p "+destination)
    worker=Path(__file__).with_name("remote_cpu_lucj_worker.py");send(worker,destination+"/worker.py")
    config={"case_id":case,"problem_id":m["problem_id"],"norb":m["norb"],"nelec":[m["nalpha"],m["nbeta"]],
        "operator_sha256":construction["operator_sha256"],"requests":requests,"CPU_parity_reference":None,
        "pinned_CPU_ffsim_platform_previously_qualified":True,"worker_sha256":sha256(worker.read_bytes())}
    config["config_id"]=digest(config);save_json(final/"request.json",config);send(final/"request.json",destination+"/request.json")
    script="\n".join(["#!/bin/bash","set -euo pipefail","export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1",
        "export PYTHONPATH="+REMOTE+"/cpu-runtime-v1/packages",
        shlex.join([PYTHON,destination+"/worker.py",destination+"/request.json",operator_remote,destination+"/output"])])+"\n"
    atomic_bytes(final/"job.sbatch",script.encode());send(final/"job.sbatch",destination+"/job.sbatch")
    command=shlex.join(["sbatch","--parsable","--partition=gpu-short","--cpus-per-task=1","--mem=24G","--time=00:20:00",
        "--job-name=paper-TN-final-"+case[:4],"--output="+destination+"/slurm.out","--error="+destination+"/slurm.err",destination+"/job.sbatch"])
    intent={"case_id":case,"remote":destination,"command":command,"config_id":config["config_id"],"operator_sha256":construction["operator_sha256"]}
    save_json(final/"submission-intent.json",intent);reply=ssh(command).strip().split(";")[0]
    if not reply.isdigit():raise RuntimeError("Uncertain final readout Slurm reply")
    job={**intent,"job_id":reply};save_json(final/"job.json",job);print("TN_FINAL_SUBMITTED",case,reply,flush=True);return job


def collect(case,job):
    root=ROOT/"TN-controls-v1"/case/"final-readout"
    if (root/"completed.json").exists():return True
    state=ssh("sacct -X -n -j "+job["job_id"]+" --format=JobID,State,Elapsed,ExitCode -P")
    save_json(root/"slurm-observed.json",{"accounting":state})
    if any(s in state for s in ("FAILED","TIMEOUT","OUT_OF_MEMORY","CANCELLED")):
        save_json(root/"terminal-attention.json",{"actual_slurm_accounting":state,"reserved_work_preserved":True})
        raise RuntimeError("TN exact final readout requires reconciliation")
    if "COMPLETED" not in state:return False
    raw=json.loads(ssh("cat "+job["remote"]+"/output/result.json"))
    if raw["config_id"]!=job["config_id"] or raw["status"]!="EXACT_REMOTE_CPU_LUCJ_COMPLETED":raise RuntimeError("TN final result mismatch")
    save_json(root/"remote-result.json",raw);construction=json.loads((root/"construction.json").read_text())
    m,h,g=load_physical(case);ledger=BudgetLedger(ROOT/"common-budget.sqlite")
    for record in raw["sampling_results"]:
        seed=record["seed"];key=run_identity(case,METHOD,100000,seed,"S_TN_INDEPENDENT_FINAL")
        if key!=record["run_id"]:raise RuntimeError("Wrong TN final seed")
        output=ROOT/"common-runs/S_TN_INDEPENDENT_FINAL"/case/METHOD/("100000-"+str(seed));output.mkdir(parents=True,exist_ok=True)
        if (output/"result.json").exists():continue
        target=output/"final-counts.npz"
        subprocess.run(["rtk","proxy","scp","-P","2222",*OPTIONS,HOST+":"+job["remote"]+"/output/"+key+".npz",str(target)],check=True,timeout=120)
        if sha256(target.read_bytes())!=record["counts_sha256"]:raise RuntimeError("TN final counts mismatch")
        with np.load(target,allow_pickle=False) as a:ids,counts=a["blocked_ids"].copy(),a["counts"].copy()
        ledger.complete(key,"independent-final",100000,record["counts_sha256"])
        tick,cpu=perf_counter(),process_time();readouts=final_readouts(output,m,h,g,ids,counts)
        save_json(output/"result.json",{"status":"COMPLETED","lane":"S_TN_INDEPENDENT_FINAL","run_id":key,"case_id":case,
            "method_id":METHOD,"budget":100000,"seed":seed,"problem_id":m["problem_id"],"final_CI":readouts,
            "budget_ledger":ledger.snapshot(key),"independent_final_model_shots":100000,"pilot_model_shots":0,"actual_QPU_shots":0,
            "sampling_resources":construction["resources"]["Z_sampling_equivalent"],"preparation_resources":construction["resources"]["preparation"],
            "coefficient_slots":construction["coefficient_slots"],"nonredundant_independent_parameters":None,"VQE_optimized_variables":0,
            "surrogate_optimized_coordinate_slots":construction["surrogate_optimized_coordinate_slots"],"shared_optimization_result":construction["optimization_source"],
            "total_model_cnot_applications":100000*construction["resources"]["Z_sampling_equivalent"]["cnot_count"],
            "this_attempt_wall_seconds":perf_counter()-tick,"this_attempt_CPU_seconds":process_time()-cpu,"local_attempt_time_is_CI_only":True,
            "remote_shared_simulation_result":str((root/"remote-result.json").relative_to(ROOT)),"FCI_used_for_construction":False,
            "classical_surrogate_optimization_free":False,"paper_native_TN_Dice_optimization_reproduced":False})
        print("TN_FINAL_COLLECTED",case,seed,flush=True)
    save_json(root/"completed.json",{"status":"FIVE_INDEPENDENT_EXACT_FINAL_READOUTS_COMPLETED","readout_seeds":list(SEEDS),
        "shared_optimizer_once":True,"independent_optimization_replicates":1,"model_final_shots":500000})
    return True


def main():
    import fcntl
    with (ROOT/"TN-final-controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            complete=0;attention={}
            for case in CASES:
                try:
                    job=submit(case)
                    if job is not None and collect(case,job):complete+=1
                except RuntimeError as exc:
                    # A missing final state in one bounded optimization must
                    # neither be silently replaced nor stop other cases.
                    attention[case]=repr(exc)
            save_json(ROOT/"TN-controls-v1/status.json",{"completed_case_final_readouts":complete,"planned_cases":4,
                "attention_cases":attention,"no_automatic_reoptimization_or_resampling":True,"whole_experiment_complete":False})
            if complete+len(attention)==len(CASES):
                repair_pending=any((ROOT/"TN-successor-v2/contract.json").exists()
                    and not(ROOT/"TN-controls-v2"/case/"no-final-state.json").exists()
                    and not(ROOT/"TN-controls-v1"/case/"final-readout/terminal-attention.json").exists()
                    for case in attention)
                if not repair_pending:return
            sleep(45)


if __name__=="__main__":main()
