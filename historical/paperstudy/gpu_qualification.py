"""Submit exactly two small, isolated, one-A100 numerical qualification jobs."""
import argparse
import json
import shlex
from pathlib import Path

import ffsim
import numpy as np

from benchmark.gpu_remote import ssh, send
from .input_adapter import array_bytes
from .io import ROOT, atomic_bytes, save_json, sha256
from .lucj import naive

REMOTE = "/share/reiwatanabe/raacs-paper-aligned-20261025-v1"
PYTHON = "/share/reiwatanabe/ceo-qsci-engine-20261022-v2/venv/bin/python"
ENV = "/share/reiwatanabe/ceo-qsci-engine-20261022-v2/venv"


def submit():
    root = ROOT / "qualification/GPU"
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    ssh("mkdir -p "+REMOTE+"/qualification")
    worker = Path(__file__).with_name("gpu_sector_worker.py")
    remote_worker = REMOTE+"/qualification/gpu_sector_worker.py"
    send(worker, remote_worker)
    for seed in (101, 211):
        directory = root / str(seed)
        directory.mkdir(exist_ok=True)
        job_path = directory / "job.json"
        if job_path.exists():
            jobs.append(json.loads(job_path.read_text()))
            continue
        if (directory/"submission-intent.json").exists():
            raise RuntimeError("Submission uncertain: reconcile before retry")
        n, ne = 4, (2, 2)
        rng = np.random.default_rng(seed)
        t1 = rng.normal(0, .02, (2, 2))
        t2 = rng.normal(0, .02, (2, 2, 2, 2))
        t2 = (t2+t2.transpose(1, 0, 3, 2))/2
        op = naive(t1, t2, n)
        expected = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne)
        payload = array_bytes({"norb": np.array(n), "nelec": np.array(ne),
                              "diag_coulomb_mats": op.diag_coulomb_mats, "orbital_rotations": op.orbital_rotations,
                              "final_orbital_rotation": op.final_orbital_rotation, "expected_sector_vector": expected})
        atomic_bytes(directory/"fixture.npz", payload)
        remote_case = REMOTE+"/qualification/"+str(seed)
        ssh("mkdir -p "+remote_case)
        send(directory/"fixture.npz", remote_case+"/fixture.npz")
        script = "\n".join(["#!/bin/bash", "set -euo pipefail",
                  "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                  "export CUDA_PATH="+ENV+"/lib/python3.12/site-packages/nvidia/cuda_runtime",
                  'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
                  'export CUPY_ACCELERATORS=""',
                  shlex.join([PYTHON, remote_worker, remote_case+"/fixture.npz", remote_case+"/result.json"])])+"\n"
        atomic_bytes(directory/"job.sbatch", script.encode())
        send(directory/"job.sbatch", remote_case+"/job.sbatch")
        command = shlex.join(["sbatch", "--parsable", "--partition=gpu-short", "--gres=gpu:a100:1",
                              "--cpus-per-task=1", "--mem=4G", "--time=00:05:00",
                              "--job-name=paper-align-q"+str(seed), "--output="+remote_case+"/slurm.out",
                              "--error="+remote_case+"/slurm.err", remote_case+"/job.sbatch"])
        intent = {"seed": seed, "remote": remote_case, "command": command,
                  "fixture_sha256": sha256(payload), "worker_sha256": sha256(worker.read_bytes()),
                  "one_A100": True, "stage": "S1_SMALL_KERNEL_QUALIFICATION"}
        save_json(directory/"submission-intent.json", intent)
        reply = ssh(command).strip()
        job_id = reply.split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError("Uncertain Slurm response: "+reply)
        job = {**intent, "job_id": job_id}
        save_json(job_path, job)
        jobs.append(job)
        print("SUBMITTED", job_id, seed, flush=True)
    save_json(root/"jobs.json", {"jobs": jobs, "production_campaign_dispatched": False})
    return jobs


def status():
    root = ROOT / "qualification/GPU"
    jobs = json.loads((root/"jobs.json").read_text())["jobs"]
    results = []
    for job in jobs:
        live = ssh("squeue -h -j "+job["job_id"]+" -o '%i %T %j %R'").strip()
        accounting = ssh("sacct -n -j "+job["job_id"]+" --format=JobID,State,Start,End,Elapsed,ExitCode,AllocTRES").strip()
        remote = job["remote"]
        result = ssh("if test -f "+remote+"/result.json; then cat "+remote+"/result.json; else tail -20 "+remote+"/slurm.err 2>/dev/null || true; fi").strip()
        evidence = {"job_id": job["job_id"], "live": live or None, "accounting": accounting, "result_or_stderr": result}
        save_json(root/str(job["seed"])/"observed-status.json", evidence)
        results.append(evidence)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    if a.submit:
        submit()
    else:
        status()
