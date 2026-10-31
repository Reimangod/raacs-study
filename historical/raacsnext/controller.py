"""Single-owner bounded development execution; no scientific tuning in controller."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
from time import time
from .paths import ROOT,PROJECT
from .io import save,load,sha256,space_check
from .run import science_hashes


def freeze_code():
    frozen=ROOT/'qualification'/'scientific-code.json'
    current=science_hashes()
    if frozen.exists()and load(frozen)['code_SHA256']!=current:
        raise RuntimeError('Scientific code changed after qualification; retain affected evidence')
    if not frozen.exists():
        save(frozen,{'code_SHA256':current,'small_tests_passed':13,'new_molecular_runs_before_tests':0,
            'tests_path':str(ROOT/'test_qualification.py'),'test_SHA256':sha256((ROOT/'test_qualification.py').read_bytes()),
            'frozen_unix':time(),'actual_QPU_shots':0})
        from .io import write_bytes
        for name in current:write_bytes(ROOT/'qualification'/'source'/name,(ROOT/'raacsnext'/name).read_bytes())


def limits():
    start=load(ROOT/'campaign-start.json')
    elapsed=time()-start['started_unix']
    used=sum(load(p)['child_CPU_seconds']for p in(ROOT/'dispatch').glob('*/terminal.json'))
    if elapsed>=86400:raise RuntimeError('Registered pilot wall limit reached')
    if used>=64*3600:raise RuntimeError('Registered pilot CPU limit reached')
    space_check(2*1024**2)
    return elapsed,used


def run_cell(db,row,cap=None):
    key,case,method,budget,seed=row
    result=ROOT/'runs'/'DEVELOPMENT'/case/method/(str(budget)+'-'+str(seed))/'result.json'
    if result.exists():
        record=load(result)
        if record['status']!='COMPLETED':raise RuntimeError('Unexpected result status')
        db.execute("UPDATE cells SET status='COMPLETED',result_path=? WHERE id=?",(str(result.relative_to(ROOT)),key));db.commit()
        return record
    limits();dispatch=ROOT/'dispatch'/key;dispatch.mkdir(parents=True,exist_ok=True)
    if(dispatch/'started.json').exists():
        raise RuntimeError('Previously launched child requires process/evidence reconciliation, not blind resubmit')
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
    env.update({k:'1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')})
    env['PYTHONWARNINGS']='ignore'
    command=['rtk','proxy',sys.executable,'-u','-m','raacsnext.run','--case',case,'--variant',method,
        '--budget',str(budget),'--seed',str(seed)]
    if cap is not None:command+=['--CPU-cap',str(cap)]
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time()
    with(dispatch/'worker.log').open('a')as log:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(dispatch/'started.json',{'pid':child.pid,'owner_pid':os.getpid(),'command':command,'started_unix':started,
            'case_id':case,'method':method,'seed':seed})
        db.execute("UPDATE cells SET status='RUNNING' WHERE id=?",(key,));db.commit()
        code=child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(dispatch/'terminal.json',{'exit_code':code,'child_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'wall_seconds':time()-started,'result_exists':result.exists(),'finished_unix':time()})
    status='COMPLETED'if code==0 and result.exists()else'FAILED_ENGINEERING_PRESERVED'
    db.execute('UPDATE cells SET status=?,reason=?,result_path=? WHERE id=?',(status,None if status=='COMPLETED'else'Preserved child failure',
        str(result.relative_to(ROOT))if result.exists()else None,key));db.commit()
    save(ROOT/'progress.json',{'updated_unix':time(),'status_counts':dict(db.execute('SELECT status,COUNT(*) FROM cells GROUP BY status')),
        'last_case':case,'last_method':method,'last_seed':seed,'last_status':status,'whole_plan_complete':False})
    print(case,method,seed,status,flush=True)
    if status!='COMPLETED':raise RuntimeError('Child failed; no scientific failure substituted')
    return load(result)


def main():
    with(ROOT/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze_code();protocol=load(ROOT/'protocol.json')
        save(ROOT/'controller-start.json',{'pid':os.getpid(),'started_unix':time(),'command':'python -m raacsnext.controller'})
        db=sqlite3.connect(ROOT/'queue.sqlite',timeout=30)
        try:
            caps={}
            for case in protocol['cases']:
                rows=db.execute("SELECT id,case_id,method,budget,seed FROM cells WHERE case_id=? AND method='D0' ORDER BY seed",(case,)).fetchall()
                results=[run_cell(db,row)for row in rows]
                caps[case]=2*max(x['construction_CPU_seconds']for x in results)
            save(ROOT/'D0-derived-CPU-caps.json',{'caps_CPU_seconds':caps,'ratio':2,
                'source_result_SHA256':{str(p.relative_to(ROOT)):sha256(p.read_bytes())for p in(ROOT/'runs'/'DEVELOPMENT').glob('*/D0/*/result.json')},
                'scope':'native construction including pilot simulation/compilation; excludes final sampling and final reporting CI'})
            for seed in protocol['pilot_seeds']:
                for case in protocol['cases']:
                    for method in('D1','D2','D3'):
                        row=db.execute('SELECT id,case_id,method,budget,seed FROM cells WHERE case_id=? AND method=? AND seed=?',
                            (case,method,seed)).fetchone()
                        run_cell(db,row,caps[case])
            save(ROOT/'RAACS-pilot-complete.json',{'finished_unix':time(),'RAACS_completed':48,
                'controls_remaining':120,'whole_plan_complete':False})
        except Exception as exc:
            save(ROOT/'controller-attention.json',{'error':repr(exc),'unix':time(),'whole_plan_complete':False})
            raise
        finally:db.close()


if __name__=='__main__':main()
