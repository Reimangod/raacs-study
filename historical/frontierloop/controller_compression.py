"""Durable D2 progress: overlap one qualified A100 with one M4 CI worker.

Never resubmit a job merely because its observer timed out. Existing job.json
is collected, completed draws/readouts are reused, unknown outcomes preserved.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from time import sleep,time,perf_counter
from .paths import ROOT
from .io import load,save,sha256,space
from .compression_trial import DIRECTORY,guard
from .compression_gpu_dispatch import submit,collect


def wait_for_local_space():
    """Do not start new local numeric work below the registered disk floor.

    Existing remote jobs and their preserved evidence are not cancelled.
    Changing execution pacing does not change any frozen scientific setting.
    """
    while True:
        try:
            return space()
        except OSError:
            save(ROOT/'progress-D2-COMPRESSION-SCREEN.json',{
                'phase':'WAITING_FOR_REGISTERED_LOCAL_FREE_SPACE','parent_pid':os.getpid(),
                'unix':time(),'new_local_numeric_child_started':False,'remote_jobs_not_cancelled':True,
                'no_automatic_deletion':True,'goal_complete':False})
            sleep(20)


def command(action,case,point,mode=None):
    spec=guard();d=DIRECTORY/case/point
    target=d/'draw-result.json' if action=='local' else d/mode/'result.json'
    failed=d/'controller-draw-failure.json' if action=='local' else d/mode/'failure.json'
    if target.exists():return 'COMPLETED'
    if failed.exists():return 'FAILED_ENGINEERING_PRESERVED'
    wait_for_local_space()
    args=['rtk','proxy',sys.executable,'-u','-m','frontierloop.compression_trial',action,'--case',case,'--point',point]
    if mode:args+=['--mode',mode]
    log=d/('local.log' if action=='local' else mode+'.log');tick=perf_counter()
    with log.open('a') as f:
        child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        save(ROOT/'progress-D2-COMPRESSION-SCREEN.json',{'phase':action,'case':case,'point':point,
            'mode':mode,'child_pid':child.pid,'parent_pid':os.getpid(),'unix':time(),
            'command':args,'log':str(log),'goal_complete':False})
        code=child.wait()
    if code or not target.exists():
        save(failed,{'status':'FAILED_ENGINEERING_PRESERVED','returncode':code,
            'elapsed_wall_seconds':perf_counter()-tick,'log':str(log),'no_automatic_remeasurement':True})
        print('D2_ENGINEERING_FAILURE',case,point,mode,flush=True);return 'FAILED_ENGINEERING_PRESERVED'
    print('D2_CELL_PHASE_COMPLETE',case,point,mode or action,flush=True);return 'COMPLETED'


def readouts(case):
    spec=guard()
    for point in spec['points']:
        if not (DIRECTORY/case/point/'draw-result.json').exists():continue
        for mode in spec['readouts']:command('readout',case,point,mode)


def main():
    spec=guard()
    if not (DIRECTORY/'manifest.json').exists():raise RuntimeError('D2 manifest must precede execution')
    with (ROOT/'controller-D2-COMPRESSION-SCREEN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(ROOT/'owner-D2-COMPRESSION-SCREEN.json',{'pid':os.getpid(),'unix':time(),
            'controller_SHA256':sha256(Path(__file__).read_bytes()),'scientific_freeze':load(ROOT/'freeze-D2-COMPRESSION-SCREEN.json')})
        gpu_cases=spec['cases'][1:];jobs={};finished=set()
        # One free GPU is sufficient; leave unrelated jobs untouched.
        job=submit(gpu_cases[0])
        if job:jobs[gpu_cases[0]]=job
        case=spec['cases'][0]
        for point in spec['points']:command('local',case,point)
        readouts(case)
        while len(finished)<len(gpu_cases):
            for case in list(jobs):
                if case in finished:continue
                result=collect(case)
                if result is not None:
                    finished.add(case)
                    # Start the next case before M4 works on these readouts.
                    next_case=next((c for c in gpu_cases if c not in jobs and c not in finished),None)
                    if next_case:
                        job=submit(next_case)
                        if job:jobs[next_case]=job
                    readouts(case)
            if not any(c not in finished for c in jobs):
                next_case=next((c for c in gpu_cases if c not in finished),None)
                if next_case:
                    job=submit(next_case)
                    if job:jobs[next_case]=job
            if len(finished)<len(gpu_cases):sleep(15)
        states=[]
        for case in spec['cases']:
            for point in spec['points']:
                d=DIRECTORY/case/point
                states.append({'case':case,'point':point,'draw_completed':(d/'draw-result.json').exists(),
                    'readouts':{mode:('COMPLETED' if (d/mode/'result.json').exists() else
                        'FAILED_ENGINEERING_PRESERVED' if (d/mode/'failure.json').exists() else 'NOT_COMPLETED') for mode in spec['readouts']}})
        save(ROOT/'cycle-D2-COMPRESSION-SCREEN-runs-finished.json',{'cells':states,
            'all_16_draws_and_48_readouts_completed':all(x['draw_completed'] and all(s=='COMPLETED' for s in x['readouts'].values()) for x in states),
            'goal_complete':False})
        print('D2_SCREEN_TERMINAL_NOT_GOAL_COMPLETION',flush=True)


if __name__=='__main__':main()
