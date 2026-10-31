"""Predefined capacity sensitivity from existing final counts; zero new shots."""
import fcntl
import json
import os
import sqlite3
from time import sleep,perf_counter,process_time
import numpy as np
from .ci import solve
from .common_execution import load_physical
from .io import ROOT,save_json,sha256


def sweep():
    db=sqlite3.connect(ROOT/"run-queue.sqlite")
    rows=db.execute("SELECT case_id,method_id,budget,seed,result_path FROM planned_cells WHERE status='COMPLETED'").fetchall()
    triggers={}
    for case,method,B,seed,path in rows:
        if B!=1000000:continue
        source=json.loads((ROOT/path).read_text())
        if any(ci["r_cap"]==32 and ci["spin_string_cap_reached"] for ci in source["final_CI"]):
            triggers[case]={"method_id":method,"seed":seed,"source_result":path,"criterion":"High-budget r32 spin-string cap reached; energies not used"}
    for case,trigger in triggers.items():
        folder=ROOT/"CI-extension-v1"/case
        if not (folder/"trigger.json").exists():save_json(folder/"trigger.json",trigger)
    completed=0
    for case,method,B,seed,path in rows:
        if case not in triggers:continue
        output=ROOT/"CI-extension-v1"/case/method/(str(B)+"-"+str(seed))/"CI-r64.json"
        if output.exists():completed+=1;continue
        source=ROOT/path;result=json.loads(source.read_text());m,h,g=load_physical(case)
        counts_file=source.parent/"final-counts.npz"
        with np.load(counts_file,allow_pickle=False) as a:ids,counts=a["blocked_ids"].copy(),a["counts"].copy()
        tick,cpu=perf_counter(),process_time()
        ci=solve(h,g,m["core_energy"],(m["nalpha"],m["nbeta"]),ids,counts,64)
        smaller=next(c for c in result["final_CI"] if c["r_cap"]==32)
        if ci["converged"] and smaller["converged"] and ci["energy_Ha"]>smaller["energy_Ha"]+1e-8:
            raise RuntimeError("Nested common CI energy monotonicity failed")
        save_json(output,{"status":"COMPLETED","case_id":case,"method_id":method,"budget":B,"seed":seed,
            "problem_id":m["problem_id"],"source_result":path,"source_result_sha256":sha256(source.read_bytes()),
            "source_counts_sha256":sha256(counts_file.read_bytes()),"new_model_shots":0,"actual_QPU_shots":0,
            "trigger":triggers[case],"CI":ci,"CPU_seconds":process_time()-cpu,"wall_seconds":perf_counter()-tick,
            "selection_by_energy_or_winner":False,"all_budgets_all_methods_all_seeds_in_triggered_case":True})
        completed+=1;print("CI4096",case,method,B,seed,ci["energy_Ha"],flush=True)
    db.close();save_json(ROOT/"CI-extension-v1/status.json",{"triggered_cases":list(triggers),"completed_readouts":completed,"new_model_shots":0,"whole_experiment_complete":False})


def main(watch=False):
    with (ROOT/"CI-extension-controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            sweep()
            if not watch:return
            db=sqlite3.connect(ROOT/"run-queue.sqlite")
            pending=db.execute("SELECT COUNT(*) FROM planned_cells WHERE status IN ('NOT_STARTED','RUNNING','WAITING_REMOTE_EXECUTOR')").fetchone()[0]
            db.close()
            if not pending:return
            sleep(45)


if __name__=="__main__":
    import argparse;p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");main(p.parse_args().watch)
