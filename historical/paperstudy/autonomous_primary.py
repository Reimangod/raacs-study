"""Finish all480 primary cells, preserving frozen tier order and remote work.

This is the S4 coordinator, not a declaration that S5/S6 are complete.
"""
import fcntl
import json
import os
import shutil
from pathlib import Path
import sqlite3
from time import sleep, time

from benchmark.gpu_remote import ssh
from .campaign import dispatch
from .io import ROOT, save_json
from .remote_cpu_dispatch import submit, collect
from .research_report import report


def counts(budget=None):
    db = sqlite3.connect(ROOT/"run-queue.sqlite")
    rows = db.execute("SELECT status,COUNT(*) FROM planned_cells"+(" WHERE budget=?" if budget else "")+" GROUP BY status",
                      (budget,) if budget else ()).fetchall()
    db.close()
    return dict(rows)


def local_pending(budget):
    # The only registered over-local-limit primary executor is Fe40 LUCJ.
    # Do not mistake a disk boundary in an earlier case for remote-only work.
    db=sqlite3.connect(ROOT/'run-queue.sqlite')
    n=db.execute("SELECT COUNT(*) FROM planned_cells WHERE budget=? AND status IN ('NOT_STARTED','RUNNING','WAITING_REMOTE_EXECUTOR') AND NOT(case_id='FE2S2-40-P2' AND method_id IN('B_LUCJ_NAIVE_L1','C_LUCJ_COMPRESSED_L1'))",(budget,)).fetchone()[0]
    db.close();return n


def observe_remote(jobs):
    observations=[]
    for job in jobs:
        state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode').strip()
        observations.append({'job_id':job['job_id'],'state':state})
    return observations


def main():
    with (ROOT/"autonomous-primary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for budget in (10000, 100000, 1000000):
            if counts(budget).get("COMPLETED", 0) == 160:
                continue
            job_file = ROOT/"remote-CPU-LUCJ-v1"/str(budget)/"jobs.json"
            while counts(budget).get('COMPLETED',0)<160:
                if counts(budget).get('FAILED_ENGINEERING_PRESERVED',0)or counts(budget).get('RESOURCE_CAPPED',0):
                    save_json(ROOT/'autonomous-primary-status.json',{'pid':os.getpid(),'phase':'NEEDS_ENGINEERING_ATTENTION',
                        'budget':budget,'status_counts':counts(),'whole_experiment_complete':False});return
                free=shutil.disk_usage(ROOT).free
                if free<3*2**30:
                    save_json(ROOT/'autonomous-primary-status.json',{'pid':os.getpid(),'phase':'S4_STORAGE_WAIT_NOT_REMOTE_ONLY',
                        'budget':budget,'status_counts':counts(),'free_bytes':free,'local_pending':local_pending(budget),
                        'existing_remote_jobs_preserved':job_file.exists(),'observed_at_unix':time(),'whole_experiment_complete':False})
                    sleep(30);continue
                if job_file.exists():
                    jobs=json.loads(job_file.read_text())['jobs'];observations=observe_remote(jobs)
                    save_json(ROOT/"autonomous-primary-status.json", {"pid": os.getpid(), "phase": "S4_REMOTE_EXACT_CPU",
                              "budget": budget, "actual_Slurm_observations": observations, "status_counts": counts(),
                              "observed_at_unix": time(), "whole_experiment_complete": False})
                    # Observation failure is not interpreted as job termination.
                    if any(any(word in observation["state"] for word in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")) for observation in observations):
                        save_json(ROOT/"autonomous-primary-status.json", {"phase": "NEEDS_ENGINEERING_ATTENTION",
                                  "actual_Slurm_observations": observations, "budget": budget, "whole_experiment_complete": False})
                        return
                    collect(budget)
                if local_pending(budget):
                    save_json(ROOT/'autonomous-primary-status.json',{'pid':os.getpid(),'phase':'S4_LOCAL_TIER',
                        'budget':budget,'status_counts':counts(),'observed_at_unix':time(),'whole_experiment_complete':False})
                    dispatch()
                    if counts(budget).get('FAILED_ENGINEERING_PRESERVED',0)or counts(budget).get('RESOURCE_CAPPED',0):
                        save_json(ROOT/'autonomous-primary-status.json',{'pid':os.getpid(),'phase':'NEEDS_ENGINEERING_ATTENTION',
                            'budget':budget,'status_counts':counts(),'whole_experiment_complete':False});return
                    boundary=ROOT/'controller/resource-boundary.json'
                    if boundary.exists()and json.loads(boundary.read_text()).get('kind')=='CPU_ALLOCATION_EXHAUSTED':
                        save_json(ROOT/'autonomous-primary-status.json',{'phase':'NEEDS_ENGINEERING_ATTENTION','budget':budget,
                            'reason':'Frozen CPU allocation reached; no automatic cap extension','status_counts':counts(),'whole_experiment_complete':False});return
                # A partial local dispatch (e.g. disk floor) must return to the
                # local queue, NOT enter an irreversible remote-only wait.
                if not local_pending(budget)and counts(budget).get('COMPLETED',0)<160 and not job_file.exists():
                    submit(budget)
                report()
                if counts(budget).get('COMPLETED',0)<160:sleep(20)
            report()
        save_json(ROOT/"autonomous-primary-status.json", {"phase": "S4_PRIMARY_480_COMPLETE_PENDING_S5_S6",
                  "status_counts": counts(), "whole_experiment_complete": False})


if __name__ == "__main__":
    main()
