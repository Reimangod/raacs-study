"""High-RAM exact Fe40 baseline fallback; central paid shots, immutable jobs.

The GPU qualification remains queued independently. CPU fallback does not
claim GPU acceleration and does not replace the physical ansatz/readout.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
from time import perf_counter, process_time

import numpy as np

from benchmark.gpu_remote import ssh, send, OPTIONS, HOST
from .budget import BudgetLedger
from .common_execution import load_physical, prepare_lucj, final_readouts, run_identity
from .gpu_qualification import REMOTE, PYTHON
from .io import ROOT, atomic_bytes, digest, save_json, sha256


def submit(budget=10000):
    with (ROOT/"campaign-controller.lock").open("a") as lock:
        # Do not race the sequential controller while it is assigning cells.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = Path(__file__).with_name("remote_cpu_lucj_worker.py")
        remote_root = REMOTE+"/CPU-LUCJ-v1"
        ssh("mkdir -p "+remote_root)
        send(source, remote_root+"/worker.py")
        send(ROOT/"prepared/N2-32-R1p2-P2/B_LUCJ_NAIVE_L1-L1/operator.npz", remote_root+"/qualification-operator.npz")
        runtime = json.loads(ssh("cat "+REMOTE+"/cpu-runtime-v1/runtime.json"))
        if runtime["runtime"]["ffsim"] != "0.0.84":
            raise RuntimeError("Qualified source CPU library version differs")
        case = "FE2S2-40-P2"
        meta, _, _ = load_physical(case)
        jobs = []
        ledger = BudgetLedger(ROOT/"common-budget.sqlite")
        for method in ("B_LUCJ_NAIVE_L1", "C_LUCJ_COMPRESSED_L1"):
            directory = ROOT/"remote-CPU-LUCJ-v1"/str(budget)/method
            directory.mkdir(parents=True, exist_ok=True)
            if (directory/"job.json").exists():
                jobs.append(json.loads((directory/"job.json").read_text()))
                continue
            if (directory/"submission-intent.json").exists():
                raise RuntimeError("Prior unknown CPU submission: reconcile actual Slurm state")
            _, construction = prepare_lucj(case, method)
            operator = ROOT/"prepared"/case/(method+"-L1")/"operator.npz"
            requests = []
            db = sqlite3.connect(ROOT/"run-queue.sqlite")
            for seed in (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009):
                run_id = run_identity(case, method, budget, seed, "C_PRIMARY")
                ledger.register(run_id, budget)
                request = {"operator_sha256": construction["operator_sha256"], "problem_id": meta["problem_id"],
                           "seed": seed, "shots": budget, "role": "independent-final"}
                status = ledger.reserve(run_id, "independent-final", request, budget)
                if status not in ("NEW_RESERVATION", "RESERVED"):
                    raise RuntimeError("Existing known remote draw needs collection, not resubmission")
                requests.append({"run_id": run_id, "seed": seed, "shots": budget})
                row = db.execute("SELECT status FROM planned_cells WHERE case_id=? AND method_id=? AND budget=? AND seed=?", (case, method, budget, seed)).fetchone()
                if row[0] not in ("WAITING_REMOTE_EXECUTOR", "NOT_STARTED"):
                    raise RuntimeError("Cell already owned/executed: "+str(row))
            config = {"case_id": case, "problem_id": meta["problem_id"], "method_id": method, "norb": meta["norb"],
                      "nelec": [meta["nalpha"], meta["nbeta"]], "operator_sha256": construction["operator_sha256"],
                      "CPU_parity_reference": {"operator_file": remote_root+"/qualification-operator.npz",
                          "expected_file": REMOTE+"/GPU-Givens-v1/B_LUCJ_NAIVE_L1/fixture.npz"},
                      "requests": requests, "worker_sha256": sha256(source.read_bytes()), "runtime": runtime}
            config["config_id"] = digest(config)
            save_json(directory/"request.json", config)
            save_json(directory/"construction.json", construction)
            remote = remote_root+"/"+str(budget)+"/"+method
            ssh("mkdir -p "+remote)
            send(operator, remote+"/operator.npz")
            send(directory/"request.json", remote+"/request.json")
            script = "\n".join(["#!/bin/bash", "set -euo pipefail",
                   "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1",
                   "export PYTHONPATH="+REMOTE+"/cpu-runtime-v1/packages",
                   shlex.join([PYTHON, remote_root+"/worker.py", remote+"/request.json", remote+"/operator.npz", remote+"/output"])])+"\n"
            atomic_bytes(directory/"job.sbatch", script.encode())
            send(directory/"job.sbatch", remote+"/job.sbatch")
            command = shlex.join(["sbatch", "--parsable", "--partition=gpu-short", "--cpus-per-task=1", "--mem=24G",
                                  "--time=00:15:00", "--job-name=paper-Fe40-CPU-"+method[0],
                                  "--output="+remote+"/slurm.out", "--error="+remote+"/slurm.err", remote+"/job.sbatch"])
            intent = {"method": method, "budget": budget, "remote": remote, "command": command,
                      "config_id": config["config_id"], "worker_sha256": sha256(source.read_bytes())}
            save_json(directory/"submission-intent.json", intent)
            reply = ssh(command).strip().split(";")[0]
            if not reply.isdigit():
                raise RuntimeError("Uncertain Slurm submission: "+reply)
            job = {**intent, "job_id": reply}
            save_json(directory/"job.json", job)
            for request in requests:
                db.execute("UPDATE planned_cells SET status='RUNNING',reason=? WHERE case_id=? AND method_id=? AND budget=? AND seed=?",
                           ("Exact remote CPU Slurm job "+reply, case, method, budget, request["seed"]))
            db.commit()
            db.close()
            jobs.append(job)
            print("SUBMITTED", method, reply, "CPU-only", flush=True)
        save_json(ROOT/"remote-CPU-LUCJ-v1"/str(budget)/"jobs.json", {"jobs": jobs, "GPU_used": False})
        return jobs


def collect(budget=10000):
    root = ROOT/"remote-CPU-LUCJ-v1"/str(budget)
    jobs = json.loads((root/"jobs.json").read_text())["jobs"]
    case = "FE2S2-40-P2"
    meta, h1, eri = load_physical(case)
    ledger = BudgetLedger(ROOT/"common-budget.sqlite")
    for job in jobs:
        directory = root/job["method"]
        queue = ssh("squeue -h -u reiwatanabe -o '%i %T %R'")
        live = "\n".join(line for line in queue.splitlines() if line.split() and line.split()[0] == job["job_id"])
        accounting = ssh("sacct -n -j "+job["job_id"]+" --format=JobID,State,Elapsed,ExitCode,TotalCPU").strip()
        save_json(directory/"slurm-observed.json", {"live": live or None, "accounting": accounting})
        if live:
            print("LIVE", live, flush=True)
            continue
        raw = ssh("if test -f "+job["remote"]+"/output/result.json; then cat "+job["remote"]+"/output/result.json; else tail -25 "+job["remote"]+"/slurm.err; fi").strip()
        try:
            remote_result = json.loads(raw)
        except json.JSONDecodeError:
            save_json(directory/"terminal-without-result.json", {"accounting": accounting, "evidence": raw,
                      "no_automatic_reexecution": True})
            continue
        if remote_result["status"] != "EXACT_REMOTE_CPU_LUCJ_COMPLETED" or remote_result["config_id"] != job["config_id"]:
            raise RuntimeError("Remote result binding/status mismatch")
        save_json(directory/"remote-result.json", remote_result)
        construction = json.loads((directory/"construction.json").read_text())
        for request in remote_result["sampling_results"]:
            output = ROOT/"common-runs/C_PRIMARY"/case/job["method"]/(str(budget)+"-"+str(request["seed"]))
            output.mkdir(parents=True, exist_ok=True)
            if (output/"result.json").exists():
                continue
            if run_identity(case, job["method"], budget, request["seed"], "C_PRIMARY") != request["run_id"]:
                raise RuntimeError("Wrong queue run in remote count result")
            count_file = output/"final-counts.npz"
            subprocess.run(["rtk", "proxy", "scp", "-P", "2222", *OPTIONS, HOST+":"+job["remote"]+"/output/"+request["run_id"]+".npz", str(count_file)], check=True, timeout=120)
            if sha256(count_file.read_bytes()) != request["counts_sha256"]:
                raise RuntimeError("Remote count transfer digest mismatch")
            with np.load(count_file, allow_pickle=False) as a:
                ids, counts = a["blocked_ids"].copy(), a["counts"].copy()
            if int(counts.sum()) != budget or request["actual_model_shots"] != budget:
                raise RuntimeError("Remote model shot total mismatch")
            ledger.complete(request["run_id"], "independent-final", budget, request["counts_sha256"])
            tick, cpu = perf_counter(), process_time()
            readouts = final_readouts(output, meta, h1, eri, ids, counts)
            report = {"run_id": request["run_id"], "case_id": case, "problem_id": meta["problem_id"],
                      "method_id": job["method"], "budget": budget, "seed": request["seed"], "lane": "C_PRIMARY",
                      "status": "COMPLETED", "budget_ledger": ledger.snapshot(request["run_id"]),
                      "pilot_model_shots": 0, "independent_final_model_shots": budget, "actual_QPU_shots": 0,
                      "sampling_resources": construction["resources"]["Z_sampling_equivalent"],
                      "preparation_resources": construction["resources"]["preparation"],
                      "total_model_cnot_applications": budget*construction["resources"]["Z_sampling_equivalent"]["cnot_count"],
                      "coefficient_slots": construction["coefficient_slots"], "nonredundant_independent_parameters": None,
                      "VQE_optimized_variables": 0, "final_CI": readouts,
                      "this_attempt_wall_seconds": perf_counter()-tick, "this_attempt_CPU_seconds": process_time()-cpu,
                      "detail": {"construction": construction, "exact_remote_CPU_preparation_shared_once": remote_result["preparation"],
                                 "shared_preparation_group_id": job["config_id"], "sampling_cost": request,
                                 "remote_job_id": job["job_id"], "GPU_used": False,
                                 "remote_result_path": str((directory/"remote-result.json").relative_to(ROOT)),
                                 "local_attempt_time_is_CI_only_not_end_to_end": True},
                      "backend": "Exact ffsim fixed-number sector, Slurm CPU",
                      "source_runtime": remote_result["runtime"], "FCI_used_for_construction": False,
                      "is_primary_run": True, "whole_study_complete": False}
            save_json(output/"result.json", report)
            db = sqlite3.connect(ROOT/"run-queue.sqlite")
            db.execute("UPDATE planned_cells SET status='COMPLETED',reason=NULL,result_path=? WHERE case_id=? AND method_id=? AND budget=? AND seed=?",
                       (str((output/"result.json").relative_to(ROOT)), case, job["method"], budget, request["seed"]))
            db.commit()
            db.close()
            print("COLLECTED", case, job["method"], request["seed"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "collect"])
    p.add_argument("--budget", type=int, default=10000)
    a = p.parse_args()
    submit(a.budget) if a.action == "submit" else collect(a.budget)
