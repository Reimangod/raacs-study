"""Execute the pre-registered S5 grid, without changing primary settings."""
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
from time import sleep,time,perf_counter

from benchmark.gpu_remote import ssh
from .io import ROOT,save_json,digest
from .sensitivity_lucj import CASES,SEEDS,method_id,freeze,submit,collect

PROJECT=ROOT.parents[2]/"ceo-sampler-benchmark"


def queue():
    db=sqlite3.connect(ROOT/"run-queue.sqlite",timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS sensitivity_cells(id TEXT PRIMARY KEY,case_id TEXT,method_id TEXT,seed INTEGER,status TEXT,reason TEXT,result_path TEXT)")
    for case in CASES:
        for mid in [method_id(c,L) for L in (2,4) for c in (False,True)]+["D_SQDRIFT_N5","D_SQDRIFT_N50"]:
            for seed in SEEDS:
                key=digest(["S5",case,mid,seed,100000])
                db.execute("INSERT OR IGNORE INTO sensitivity_cells VALUES(?,?,?,?,?,?,?)",(key,case,mid,seed,"NOT_STARTED",None,None))
    for seed in SEEDS:
        key=digest(["S5","NAPH20-P1","D_SQDRIFT_N20-M512",seed,100000])
        db.execute("INSERT OR IGNORE INTO sensitivity_cells VALUES(?,?,?,?,?,?,?)",(key,"NAPH20-P1","D_SQDRIFT_N20-M512",seed,"NOT_STARTED",None,None))
    db.commit()
    return db


def status(db,phase):
    counts=dict(db.execute("SELECT status,COUNT(*) FROM sensitivity_cells GROUP BY status").fetchall())
    save_json(ROOT/"sensitivity-status.json",{"phase":phase,"pid":os.getpid(),"status_counts":counts,"planned":125,"observed_at_unix":time(),"whole_experiment_complete":False})


def child(module,args,log,seconds):
    before=resource.getrusage(resource.RUSAGE_CHILDREN);tick=perf_counter()
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",RAYON_NUM_THREADS="1",QISKIT_NUM_PROCS="1")
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as stream:
        proc=subprocess.Popen(["rtk","proxy",sys.executable,"-u","-m",module,*args],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM);code=proc.wait(timeout=30)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    save_json(log.with_suffix(".cost.json"),{"exit_code":code,"wall_seconds":perf_counter()-tick,"child_CPU_seconds":after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,"log":str(log.relative_to(ROOT))})
    return code


def update_lucj(db,case,compressed,L,value):
    mid=method_id(compressed,L)
    for seed in SEEDS:
        path=ROOT/"common-runs/S5_SENSITIVITY"/case/mid/("100000-"+str(seed))/"result.json"
        if path.exists():
            db.execute("UPDATE sensitivity_cells SET status='COMPLETED',reason=NULL,result_path=? WHERE case_id=? AND method_id=? AND seed=?",(str(path.relative_to(ROOT)),case,mid,seed))
        else:
            db.execute("UPDATE sensitivity_cells SET status=?,reason=? WHERE case_id=? AND method_id=? AND seed=?",(value,None if value=="RUNNING" else value,case,mid,seed))
    db.commit()


def main():
    with (ROOT/"sensitivity-controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze();db=queue();status(db,"S5_LUCJ")
        pending=[]
        for case in CASES:
            for L in (2,4):
                for compressed in (False,True):
                    mid=method_id(compressed,L)
                    if db.execute("SELECT COUNT(*) FROM sensitivity_cells WHERE case_id=? AND method_id=? AND status='COMPLETED'",(case,mid)).fetchone()[0]==5:
                        continue
                    if shutil.disk_usage(ROOT).free<3*2**30:
                        status(db,"STORAGE_BOUNDARY_NO_NEW_WORK");return
                    if compressed:
                        init=ROOT/"sensitivity-initialization"/case/("compressed-L"+str(L))/"result.json"
                        if not init.exists():
                            code=child("paperstudy.sensitivity_initialization",["--case",case,"--n-reps",str(L)],ROOT/"sensitivity-controller"/(case+"-"+mid+"-init.log"),1260)
                            if code!=0 or not init.exists():
                                update_lucj(db,case,compressed,L,"INITIALIZATION_FAILED_PRESERVED");continue
                        evidence=json.loads(init.read_text())
                        if evidence["status"]!="AUTHOR_INITIALIZER_RETURNED":
                            update_lucj(db,case,compressed,L,"INITIALIZATION_FAILED_PRESERVED");continue
                    submit(case,compressed,L)
                    update_lucj(db,case,compressed,L,"RUNNING")
                    pending.append((case,compressed,L))
                    while len(pending)>=2:
                        remaining=[]
                        for config in pending:
                            outcome=collect(*config)
                            update_lucj(db,*config,"RUNNING" if outcome=="LIVE" else outcome)
                            if outcome=="LIVE":remaining.append(config)
                            elif outcome!="COMPLETED":
                                status(db,"NEEDS_ENGINEERING_ATTENTION");return
                        pending=remaining;status(db,"S5_LUCJ")
                        if len(pending)>=2:sleep(20)
        while pending:
            remaining=[]
            for config in pending:
                outcome=collect(*config);update_lucj(db,*config,"RUNNING" if outcome=="LIVE" else outcome)
                if outcome=="LIVE":remaining.append(config)
                elif outcome!="COMPLETED":status(db,"NEEDS_ENGINEERING_ATTENTION");return
            pending=remaining;status(db,"S5_LUCJ")
            if pending:sleep(20)
        for key,case,mid,seed,state,_,_ in db.execute("SELECT * FROM sensitivity_cells WHERE method_id LIKE 'D_%' ORDER BY case_id,method_id,seed").fetchall():
            if state in ("COMPLETED","RESOURCE_CAPPED","FAILED_ENGINEERING_PRESERVED"):continue
            if shutil.disk_usage(ROOT).free<3*2**30:
                status(db,"STORAGE_BOUNDARY_NO_NEW_WORK");return
            length=20 if "N20" in mid else (5 if mid.endswith("N5") else 50)
            M=512 if "M512" in mid else 128
            db.execute("UPDATE sensitivity_cells SET status='RUNNING' WHERE id=?",(key,));db.commit()
            path=ROOT/"common-runs/S5_SENSITIVITY"/case/mid/("100000-"+str(seed))/"result.json"
            log=ROOT/"sensitivity-controller"/(key+".log")
            code=child("paperstudy.sensitivity_sqdrift",["--case",case,"--seed",str(seed),"--length",str(length),"--shots-per-member",str(M)],log,1800)
            state="COMPLETED" if code==0 and path.exists() else "FAILED_ENGINEERING_PRESERVED"
            reason=None if state=="COMPLETED" else "Worker/cap boundary: preserve log and primitive ledger"
            db.execute("UPDATE sensitivity_cells SET status=?,reason=?,result_path=? WHERE id=?",(state,reason,str(path.relative_to(ROOT)) if path.exists() else None,key));db.commit()
            status(db,"S5_SQDRIFT")
        status(db,"S5_GRID_TERMINAL_PENDING_TN_AND_S6")


if __name__=="__main__":main()
