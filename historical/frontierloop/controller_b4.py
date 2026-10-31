"""Run frozen B4 after the live C5 worker has actually finished.

Never restarts C5. Resolves and checks the exact process fingerprint, not a
lock file. One molecular CPU worker at a time. This is a job dependency,
not authorization to alter a frozen protocol or infer scientific success.
"""
import fcntl
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import time,sleep
from .paths import ROOT,PROJECT
from .io import load,save,space
from .run_b4 import freeze


def fingerprint(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else None


def c5_process():
    lines=subprocess.check_output(['ps','-ax','-o','pid=,command='],text=True).splitlines()
    prefix=str(PROJECT/'.venv/bin/python')
    matches=[]
    for line in lines:
        pid,command=line.strip().split(None,1)
        if not command.endswith('-m frontierloop.controller_controls'):
            continue
        executable=command.split()[0]
        if executable==prefix:
            matches.append(int(pid))
        elif executable=='.venv/bin/python':
            cwd=subprocess.run(['lsof','-a','-p',pid,'-d','cwd','-Fn'],capture_output=True,text=True)
            directories=[s[1:] for s in cwd.stdout.splitlines() if s.startswith('n')]
            if cwd.returncode==0 and directories==[str(PROJECT)]:
                matches.append(int(pid))
    if len(matches)>1:
        raise RuntimeError('Multiple live C5 controllers; do not guess an owner')
    return matches[0] if matches else None


def main():
    with (ROOT/'controller-B4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if freeze()!=load(ROOT/'freeze-B4.json'):
            raise RuntimeError('B4 frozen protocol changed')
        terminal=ROOT/'cycle-C5-CONTROLS-runs-finished.json'
        pid=c5_process();identity=fingerprint(pid) if pid else None
        if not terminal.exists() and pid is None:
            raise RuntimeError('No live C5 process and no terminal evidence; inspect, never restart')
        save(ROOT/'pipeline-C5-to-B4.json',{'controller_pid':os.getpid(),'started_unix':time(),
            'phase':'WAITING_ON_VERIFIED_C5_PROCESS' if pid else 'CHECKING_C5_TERMINAL',
            'C5_pid':pid,'C5_process_fingerprint':identity,'B4_frozen':freeze(),
            'C5_restarted':False,'goal_complete':False})
        while pid is not None and fingerprint(pid)==identity:
            sleep(5)
        if not terminal.exists():
            raise RuntimeError('C5 process ended without terminal manifest; preserve all evidence')
        completed=load(terminal)
        if len(completed['rows'])!=120 or any(r['status'] not in
            ('COMPLETED','MISSING_SAME_PROBLEM_SOURCE') for r in completed['rows']):
            raise RuntimeError('C5 has unresolved engineering results; inspect before B4')
        # Terminal C5 results change the next comparison, never the already
        # frozen B4 policy or its seeds. Every planned physical case is kept.
        spec=load(ROOT/'protocol-B4.json');rows=[];started=time()
        for case in spec['cases']:
            for seed in spec['seeds']:
                space();cpu0=resource.getrusage(resource.RUSAGE_CHILDREN);tick=time()
                task=subprocess.run([sys.executable,'-u','-m','frontierloop.run_b4',
                    '--case',case,'--seed',str(seed)],timeout=1000)
                cpu1=resource.getrusage(resource.RUSAGE_CHILDREN)
                row={'case':case,'seed':seed,'returncode':task.returncode,
                    'worker_whole_process_CPU_seconds':cpu1.ru_utime+cpu1.ru_stime-cpu0.ru_utime-cpu0.ru_stime,
                    'worker_whole_process_wall_seconds':time()-tick}
                rows.append(row)
                save(ROOT/'progress-B4.json',{'started_unix':started,'updated_unix':time(),
                    'rows':rows,'goal_complete':False})
                if task.returncode!=0:
                    raise RuntimeError('B4 worker failed; no blind retry')
        save(ROOT/'cycle-B4-runs-finished.json',{'started_unix':started,'finished_unix':time(),
            'rows':rows,'goal_complete':False})
        save(ROOT/'pipeline-C5-to-B4-terminal.json',{'status':'C5_AND_B4_EXECUTION_TERMINAL',
            'finished_unix':time(),'C5_restarted':False,'goal_complete':False})


if __name__=='__main__':
    main()
