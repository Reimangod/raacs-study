"""One owned numerical child: successor quick audit → standard → audit/report.

No automatic rerun of a failed draw, no resource-floor lowering, no remote job.
Pending capacity is reported, not repeatedly retried. Full goal stays active.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def sequence():
    root=Path(__file__).resolve().parents[1]
    quick=root/'results/molecular-strong-controls-quick-v2'
    standard=root/'results/molecular-strong-controls-standard-v2'
    standard.mkdir(exist_ok=True)
    def run(args):
        command=[sys.executable,'-m',*args]
        print('RUN',command,flush=True)
        return subprocess.run(command,cwd=root,check=True).returncode
    def status(folder):
        p=folder/'execution-status.json'
        return json.loads(p.read_text()).get('status') if p.exists() else 'NOT_STARTED'
    # Own OS lock lives beside, not within the immutable numerical manifests.
    import fcntl
    with (standard/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mode,folder in [('quick',quick),('standard',standard)]:
            if mode=='standard':
                gate=json.loads((quick/'physical-layout-validation-v2.json').read_text())
                if gate['status']!='SPIN_BLOCKED_NATIVE_QUICK_QUALIFIED':
                    raise RuntimeError('Actual quick physical/layout qualification incomplete')
            while status(folder)!='MOLECULAR_STRONG_CONTROLS_EXECUTED_PENDING_AUDIT':
                run(['benchmark.molecular_strong_controls_v2','--mode',mode])
                s=status(folder)
                if s=='PENDING_CASE_SLICE':continue
                if s.startswith('PENDING_'):
                    print('PRESERVED',s,'; no automatic capacity retry or data deletion',flush=True);return
                if s!='MOLECULAR_STRONG_CONTROLS_EXECUTED_PENDING_AUDIT':
                    raise RuntimeError('Unexpected execution state '+s)
            run(['benchmark.molecular_strong_v2_audit','--root',str(folder)]+(['--cached'] if (folder/'physical-layout-validation-v2.json').exists() else []))
            run(['benchmark.molecular_strong_report','--root',str(folder)])
        print('E_NATIVE_SUCCESSOR_EXECUTED_AUDITED_REPORTED; whole goal remains incomplete',flush=True)


if __name__=='__main__':
    sequence()
