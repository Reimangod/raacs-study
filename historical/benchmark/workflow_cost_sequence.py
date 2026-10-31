"""Wait for exact verified existing E/F/HCI actors, then serial cost qualification.

Sleeping coordinators are not numerical workers. Never infer death from a JSON
state, and never restart earlier experiments or retry their terminal censors.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .workflow_cost_study import ROOT, FREE_FLOOR, prepare, save, frozen, sha


def process_command(pid):
    p = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    if p.returncode != 0 and p.returncode != 1:
        raise RuntimeError('Transient OS process observation failed; do not infer terminal')
    return p.stdout.strip() if p.returncode == 0 else None


def dependency_snapshot(dependencies):
    out = []
    for pid, module in dependencies:
        command = process_command(pid)
        if command is not None and (' -m ' + module) not in command:
            raise RuntimeError('Foreign/reused dependency PID; no process interference')
        out.append({'PID': pid, 'module': module, 'command': command, 'OS_live': command is not None})
    return out


def run(f_pid, followup_pid):
    output = ROOT / 'results/total-workflow-sequence-v1'
    output.mkdir(exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dependencies = [(f_pid, 'benchmark.gate_noise_sequence'),
                        (followup_pid, 'benchmark.integral_followup_sequence')]
        snapshot = dependency_snapshot(dependencies)
        frozen(output / 'dependency-binding.json', {'dependencies': [{'PID': p, 'module': m} for p, m in dependencies],
               'files': {str(p): sha(ROOT / p) for p in
                         ('benchmark/workflow_cost_study.py', 'benchmark/workflow_cost_analysis.py',
                          'benchmark/workflow_cost_sequence.py', 'study-v1/TOTAL_WORKFLOW_COST_V1.md')}})
        # Freeze the entire quick+standard numerical cohort before observing any
        # outcome of either. This prepares only physical integrals, no solver.
        for mode in ('quick', 'standard'):
            prepare(ROOT / f'results/total-workflow-{mode}-v1', mode)
        while any(s['OS_live'] for s in snapshot):
            save(output / 'execution-status.json', {'status': 'WAITING_ON_VERIFIED_NUMERICAL_DEPENDENCIES',
                 'dependencies': snapshot, 'new_numerical_child_started': False, 'whole_goal_complete': False})
            time.sleep(20)
            snapshot = dependency_snapshot(dependencies)
        follow = json.loads((ROOT / 'results/integral-followup-sequence-v1/execution-status.json').read_text())
        f = ROOT / 'results/gate-noise-full-standard-v1'
        e = ROOT / 'results/molecular-strong-controls-standard-v2'
        ready = (follow['status'] == 'CLASSICAL_HCI_AND_NATIVE_BOND_FOLLOWUP_EXECUTED' and
                 (f / 'completion-sequence-v1.json').exists() and (e / 'validation.json').exists())
        if not ready:
            save(output / 'execution-status.json', {'status': 'DEPENDENCY_TERMINAL_INCOMPLETE',
                 'followup_status': follow['status'], 'no_new_numerical_dispatch': True})
            return
        if shutil.disk_usage(ROOT).free < FREE_FLOOR:
            save(output / 'execution-status.json', {'status': 'PENDING_CAPACITY_BOUNDARY', 'no_new_numerical_dispatch': True})
            return
        environment = dict(__import__('os').environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        def child(arguments):
            print('RUN', arguments, flush=True)
            save(output / 'execution-status.json', {'status': 'RUNNING_ONE_NUMERICAL_CHILD',
                 'arguments': arguments, 'whole_goal_complete': False})
            subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)
        child(['-m', 'pytest', 'benchmark/tests/test_workflow_cost.py', '-q',
               '--junitxml=results/total-workflow-preflight-tests-v1.xml'])
        for mode in ('quick', 'standard'):
            root = ROOT / f'results/total-workflow-{mode}-v1'
            child(['-m', 'benchmark.workflow_cost_study', '--mode', mode])
            status = json.loads((root / 'execution-status.json').read_text())
            if status['status'] != 'ALL_WORKFLOW_TASKS_TERMINAL_PENDING_AUDIT':
                save(output / 'execution-status.json', {'status': 'WORKFLOW_EXECUTION_PENDING', 'mode': mode,
                     'actual_status': status['status']})
                return
            child(['-m', 'benchmark.workflow_cost_analysis', '--root', str(root)])
            if mode == 'quick':
                assert json.loads((root / 'validation.json').read_text())['status'] == 'WORKFLOW_QUICK_INDEPENDENTLY_VALIDATED'
            child(['-m', 'benchmark.workflow_cost_analysis', '--root', str(root), '--component', 'report'])
        save(output / 'execution-status.json', {'status': 'WORKFLOW_TOTAL_COST_COHORTS_AUDITED_REPORTED',
             'whole_goal_complete': False, 'hardware_shots': 0, 'new_FCI_solves': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait-f-pid', type=int, required=True)
    p.add_argument('--wait-followup-pid', type=int, required=True)
    a = p.parse_args()
    run(a.wait_f_pid, a.wait_followup_pid)
