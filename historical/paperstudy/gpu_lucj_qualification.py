"""Two independent A100 full32q number-sector parity jobs, no legacy edits."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from time import perf_counter

import ffsim
import numpy as np

from benchmark.gpu_remote import ssh, send, HOST, OPTIONS
from .common_execution import load_physical, prepare_lucj
from .gpu_lucj_payload import payload
from .gpu_qualification import ENV, PYTHON, REMOTE
from .input_adapter import array_bytes
from .io import ROOT, atomic_bytes, save_json, sha256


def checked_large_send(path, target, directory):
    """Reconcile completed bytes after a transport timeout before any repeat."""
    expected = sha256(path.read_bytes())
    command = "if test -f "+shlex.quote(target)+"; then sha256sum "+shlex.quote(target)+"; fi"
    before = ssh(command).strip().split()
    if before and before[0] == expected:
        save_json(directory/"transfer-reconciled.json", {"local_sha256": expected, "remote_sha256": before[0],
                  "transferred_again": False, "reason": "Existing remote bytes verified after prior transport termination"})
        return
    args = ["rtk", "proxy", "scp", "-P", "2222", *OPTIONS, str(path), HOST+":"+target]
    tick = perf_counter()
    proc = subprocess.Popen(args, start_new_session=True)
    try:
        code = proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
        raise RuntimeError("Own exact transfer process group timed out and terminated; reconcile bytes before retry")
    after = ssh(command).strip().split()
    record = {"local_sha256": expected, "remote_sha256": after[0] if after else None,
              "scp_exit_code": code, "wall_seconds": perf_counter()-tick, "integrity_verified": bool(after and after[0] == expected)}
    save_json(directory/"transfer-integrity.json", record)
    if not record["integrity_verified"]:
        raise RuntimeError("Transferred qualification vector not byte-identical")


def submit():
    root = ROOT/"qualification/GPU-Givens-v1"
    root.mkdir(parents=True, exist_ok=True)
    remote_root = REMOTE+"/GPU-Givens-v1"
    ssh("mkdir -p "+remote_root)
    worker = Path(__file__).with_name("gpu_lucj_engine.py")
    send(worker, remote_root+"/gpu_lucj_engine.py")
    case = "N2-32-R1p2-P2"
    meta, _, _ = load_physical(case)
    prepared = []
    for method in ("B_LUCJ_NAIVE_L1", "C_LUCJ_COMPRESSED_L1"):
        directory = root/method
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/"job.json").exists():
            prepared.append(json.loads((directory/"job.json").read_text()))
            continue
        op, construction = prepare_lucj(case, method)
        configuration = directory/"fixture.json"
        if not configuration.exists():
            tick = perf_counter()
            config, arrays = payload(op, meta["norb"], (meta["nalpha"], meta["nbeta"]),
                             [{"id": "qualification-readout", "seed": 101, "shots": 1024}])
            arrays["expected_sector_vector"] = ffsim.apply_unitary(ffsim.hartree_fock_state(meta["norb"], (5, 5)),
                                                                 op, meta["norb"], (5, 5), copy=False)
            out = __import__("io").BytesIO()
            np.savez_compressed(out, **arrays)
            atomic_bytes(directory/"fixture.npz", out.getvalue())
            save_json(configuration, config)
            save_json(directory/"CPU-reference.json", {"wall_seconds": perf_counter()-tick, "full_sector_size": int(arrays["expected_sector_vector"].size),
                      "scope": "Qualification-only full number-sector reference, not a large full-Fock vector",
                      "operator_sha256": construction["operator_sha256"], "fixture_sha256": sha256(out.getvalue())})
        remote = remote_root+"/"+method
        ssh("mkdir -p "+remote)
        send(configuration, remote+"/fixture.json")
        checked_large_send(directory/"fixture.npz", remote+"/fixture.npz", directory)
        script = "\n".join(["#!/bin/bash", "set -euo pipefail",
                  "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                  "export CUDA_PATH="+ENV+"/lib/python3.12/site-packages/nvidia/cuda_runtime",
                  'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
                  'export CUPY_ACCELERATORS=""',
                  shlex.join([PYTHON, remote_root+"/gpu_lucj_engine.py", remote+"/fixture.json", remote+"/fixture.npz", remote+"/output"])])+"\n"
        atomic_bytes(directory/"job.sbatch", script.encode())
        send(directory/"job.sbatch", remote+"/job.sbatch")
        prepared.append({"method": method, "remote": remote, "worker_sha256": sha256(worker.read_bytes()),
                         "fixture_sha256": sha256((directory/"fixture.npz").read_bytes()), "stage": "S1_FULL32Q_SECTOR_GPU_PARITY"})
    jobs = []
    for item in prepared:
        directory = root/item["method"]
        if "job_id" in item:
            jobs.append(item)
            continue
        if (directory/"submission-intent.json").exists():
            raise RuntimeError("Uncertain prior Slurm submission requires actual reconciliation")
        remote = item["remote"]
        command = shlex.join(["sbatch", "--parsable", "--partition=gpu-short", "--gres=gpu:a100:1", "--cpus-per-task=1",
                              "--mem=8G", "--time=00:10:00", "--job-name=paper-lucj32-"+item["method"][0],
                              "--output="+remote+"/slurm.out", "--error="+remote+"/slurm.err", remote+"/job.sbatch"])
        save_json(directory/"submission-intent.json", {**item, "command": command})
        reply = ssh(command).strip().split(";")[0]
        if not reply.isdigit():
            raise RuntimeError("Uncertain submission response: "+reply)
        job = {**item, "job_id": reply}
        save_json(directory/"job.json", job)
        jobs.append(job)
        print("SUBMITTED", item["method"], reply, flush=True)
    save_json(root/"jobs.json", {"jobs": jobs, "primary_sampling_dispatched": False})
    return jobs


def status():
    root = ROOT/"qualification/GPU-Givens-v1"
    jobs = json.loads((root/"jobs.json").read_text())["jobs"]
    result = []
    for job in jobs:
        # Completed IDs may be purged from squeue (which then exits nonzero).
        # Query the account, then select the exact ID; sacct remains authoritative.
        queue = ssh("squeue -h -u reiwatanabe -o '%i %T %j %R'")
        live = "\n".join(line for line in queue.splitlines() if line.split() and line.split()[0] == job["job_id"])
        accounting = ssh("sacct -n -j "+job["job_id"]+" --format=JobID,State,Start,End,Elapsed,ExitCode,AllocTRES").strip()
        raw = ssh("if test -f "+job["remote"]+"/output/result.json; then cat "+job["remote"]+"/output/result.json; else tail -20 "+job["remote"]+"/slurm.err 2>/dev/null || true; fi").strip()
        evidence = {"job": job, "live": live or None, "accounting": accounting, "result_or_stderr": raw}
        save_json(root/job["method"]/"observed-status.json", evidence)
        result.append(evidence)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    submit() if a.submit else status()
