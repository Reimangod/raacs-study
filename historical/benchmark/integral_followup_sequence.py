"""Use the freed E actor slot for actual HCI qualification and postprocessing.

The existing E coordinator is verified by OS command before waiting. A JSON
RUNNING flag is never liveness evidence. No concurrent third numerical actor.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1]


def command(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None


def save(path,record):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(wait_pid):
    output=ROOT/'results/integral-followup-sequence-v1';output.mkdir(exist_ok=True)
    E=ROOT/'results/molecular-strong-controls-standard-v2'
    state=output/'execution-status.json'
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observed=command(wait_pid)
        if observed is not None and ' -m benchmark.molecular_strong_sequence_v2' not in observed:
            raise RuntimeError('Refuse foreign/reused PID; no scheduling or process interference')
        save(output/'dependency-observation.json',{'pid':wait_pid,'OS_command':observed,
            'liveness_observed':observed is not None,'E_manifest_sha256':sha(E/'manifest.json')})
        while observed is not None:
            save(state,{'status':'WAITING_ON_VERIFIED_E_COORDINATOR','pid':wait_pid,'OS_command':observed,
                'numerical_child_started':False,'whole_goal_complete':False})
            time.sleep(15)
            observed=command(wait_pid)
            if observed is not None and ' -m benchmark.molecular_strong_sequence_v2' not in observed:
                raise RuntimeError('Dependency PID reused; inspect terminal evidence rather than trusting reuse')
        # Also prove no orphaned E numerical child still owns the data. Holding
        # this lock prevents a competing E numerical resume during followup.
        with (E/'run.lock').open('a') as reservation:
            fcntl.flock(reservation,fcntl.LOCK_EX|fcntl.LOCK_NB)
            terminal=json.loads((E/'execution-status.json').read_text())
            if terminal['status']!='MOLECULAR_STRONG_CONTROLS_EXECUTED_PENDING_AUDIT' or not (E/'validation.json').exists():
                save(state,{'status':'DEPENDENCY_TERMINAL_INCOMPLETE','E_status':terminal['status']});return
            if shutil.disk_usage(ROOT).free<int(2.75*2**30):
                save(state,{'status':'PENDING_CAPACITY_BOUNDARY','numerical_child_started':False});return
            files=['benchmark/threshold_hci.py','benchmark/integral_threshold_hci_study.py',
                'benchmark/threshold_hci_analysis.py','benchmark/molecular_strong_postprocess.py',
                'benchmark/integral_followup_sequence.py','study-v1/INTEGRAL_THRESHOLD_HCI_V1.md',
                'study-v1/MOLECULAR_STRONG_POSTPROCESS_V1.md']
            binding={'files':{p:sha(ROOT/p) for p in files},'dependency_manifest_sha256':sha(E/'manifest.json')}
            bindpath=output/'code-binding.json'
            if bindpath.exists():assert json.loads(bindpath.read_text())==binding
            else:save(bindpath,binding)
            def child(args):
                print('RUN',args,flush=True)
                save(state,{'status':'ONE_NUMERICAL_CHILD_RUNNING','command':args,'whole_goal_complete':False})
                subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
            child(['-m','pytest','benchmark/tests/test_threshold_hci.py','benchmark/tests/test_molecular_strong_postprocess.py',
                '-q','--junitxml=results/integral-followup-preflight-tests-v1.xml'])
            for mode in ('quick','standard'):
                base=ROOT/f'results/integral-threshold-HCI-{mode}-v1'
                child(['-m','benchmark.integral_threshold_hci_study','--mode',mode])
                if json.loads((base/'execution-status.json').read_text())['status'].startswith('PENDING_'):
                    save(state,{'status':'PENDING_HCI_BOUNDARY','mode':mode});return
                child(['-m','benchmark.threshold_hci_analysis','--root',str(base)])
                child(['-m','benchmark.threshold_hci_analysis','--root',str(base),'--component','report'])
                if mode=='quick' and json.loads((base/'validation.json').read_text())['status']!='THRESHOLD_HCI_QUICK_INDEPENDENTLY_VALIDATED':
                    raise RuntimeError('HCI actual quick qualification failed; no standard dispatch')
            for mode in ('quick','standard'):
                source=ROOT/f'results/molecular-strong-controls-{mode}-v2'
                target=ROOT/f'results/molecular-strong-postprocess-{mode}-v1'
                child(['-m','benchmark.molecular_strong_postprocess','--source',str(source),'--output',str(target)])
                if json.loads((target/'execution-status.json').read_text())['status'].startswith('PENDING_'):
                    save(state,{'status':'PENDING_POSTPROCESS_BOUNDARY','mode':mode});return
            save(state,{'status':'CLASSICAL_HCI_AND_NATIVE_BOND_FOLLOWUP_EXECUTED','whole_goal_complete':False,
                'hardware_shots':0,'new_quantum_acquisitions':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args();run(a.wait_pid)
