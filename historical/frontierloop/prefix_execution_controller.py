"""One bounded N17 coordinator, not a recurring automation or new experiment.

Observe ONLY existing sampling jobs. Once their actual terminal evidence is
present, finish the already specified A100 readout batch without re-submission.
Transport observation failure is not scientific failure or a reason to rerun.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic,sleep
from .paths import ROOT
from .io import load,save,sha256,write
from .prefix_performance_sampling import LOCAL as SAMPLES,verified_protocol,capacity,db
from . import prefix_performance_ci as ci
from benchmark.gpu_remote import ssh

LOCAL=ROOT/'runs/N17-AUTONOMOUS-CONTROLLER-v1'
TERMINAL_BAD={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def terminal_state(accounting,job):
    rows=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]==job]
    if len(rows)!=1:return 'UNRESOLVED'
    state=rows[0][1].split()[0].split('+')[0]
    if state=='COMPLETED' and len(rows[0])>3 and rows[0][3]=='0:0':return 'COMPLETED'
    if state in TERMINAL_BAD or state=='COMPLETED':return 'TERMINAL_ENGINEERING_FAILURE'
    return 'LIVE_OR_PENDING'


def command(module,*args):
    return ['rtk','proxy',sys.executable,'-u','-m','frontierloop.'+module,*args]


def run():
    # Exclusive creation means a second controller cannot silently take over.
    # If this process dies, reconcile its PID/command/journal before a successor.
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'owner.claim').open('x') as f:f.write(str(os.getpid())+'\n')
    cpu_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}
    if set(cpu_env.values())!={'1'}:raise RuntimeError('Explicit one-thread coordinator environment required')
    spec=verified_protocol();ci.configure('A100')
    jobs=[load(SAMPLES/f'lane{i}-job.json') for i in range(2)]
    if {j['job_id'] for j in jobs}!={'6987','6988'}:raise RuntimeError('Unexpected frozen sampling job ownership')
    m4=load(ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-M4-v1/protocol.json')
    if sha256(Path(ci.__file__).read_bytes())!=m4['preparer_SHA256'] or sha256(ci.generate().encode())!=m4['worker_SHA256']:
        raise RuntimeError('Pending A100 readout implementation differs from actual frozen M4 execution')
    save(LOCAL/'owner.json',dict(PID=os.getpid(),command=sys.argv,python=sys.executable,
        sampling_jobs=[j['job_id'] for j in jobs],sampling_protocol_SHA256=sha256((SAMPLES/'protocol.json').read_bytes()),
        M4_CI_protocol_SHA256=sha256((ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-M4-v1/protocol.json').read_bytes()),
        coordinator_SHA256=sha256(Path(__file__).read_bytes()),maximum_wall_seconds=18000,
        tests_SHA256=sha256((ROOT/'qualification-N17-execution-controller-v2-tests.xml').read_bytes()),
        minimum_poll_interval_seconds=60,no_other_jobs_modified=True,no_scientific_settings_changed=True))
    started=monotonic();step=0
    def progress(stage,**extra):
        record=dict(stage=stage,PID=os.getpid(),elapsed_wall_seconds=monotonic()-started,**extra)
        save(LOCAL/'progress.json',record);print(json.dumps(record),flush=True)
    def execute(module,*args):
        nonlocal step
        step+=1;cmd=command(module,*args);base=LOCAL/f'step{step:03d}'
        save(base.with_suffix('.intent.json'),dict(command=cmd,stage='STARTED',no_automatic_retry=True))
        with base.with_suffix('.out').open('xb') as out,base.with_suffix('.err').open('xb') as err:
            process=subprocess.Popen(cmd,stdout=out,stderr=err)
            save(base.with_suffix('.process.json'),dict(PID=process.pid,command=cmd))
            code=process.wait()
        save(base.with_suffix('.result.json'),dict(returncode=code))
        if code:raise RuntimeError('Bounded step failed; evidence/possible external work preserved: '+str(base))
    def observe(ids):
        q=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
        acc=ssh('sacct -X -n -j '+','.join(ids)+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30)
        live={line.split('|')[0] for line in q.splitlines()}
        states={job:('LIVE_OR_PENDING' if job in live else terminal_state(acc,job)) for job in ids}
        return states,dict(queue=q,accounting=acc)
    def completion_count():
        with db() as c:
            return c.execute("SELECT count(*) FROM draws WHERE status='COMPLETED'").fetchone()[0]
    try:
        while monotonic()-started<18000:
            try:
                states,evidence=observe([j['job_id'] for j in jobs])
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
                progress('OBSERVATION_UNAVAILABLE_NOT_TERMINAL_NO_RESTART',error=repr(exc));sleep(60);continue
            progress('OBSERVING_EXISTING_SAMPLING_JOBS',states=states,**evidence)
            if any(state=='TERMINAL_ENGINEERING_FAILURE' for state in states.values()):
                raise RuntimeError('Actual sampling job terminal engineering failure; no scientific retry authorized')
            eligible=[j for j in jobs if states[j['job_id']]=='COMPLETED' and not (SAMPLES/f"lane{j['lane']}-collection.json").exists()]
            if eligible:
                try:capacity()
                except OSError as exc:
                    progress('WAIT_LOCAL_CAPACITY_NO_DELETION',error=repr(exc));sleep(60);continue
                # Collector itself checks actual per-lane terminal/input hashes.
                execute('collect_prefix_performance')
            if completion_count()==sum(len(c['sampling_requests']) for c in spec['circuits']):break
            sleep(60)
        else:
            progress('BOUNDED_MONITOR_WINDOW_ENDED_JOBS_NOT_CANCELLED')
            save(LOCAL/'terminal.json',dict(status='MONITOR_WINDOW_ENDED_NOT_SCIENTIFIC_FAILURE',jobs_not_cancelled=True,goal_complete=False));return
        execute('audit_prefix_sampling')
        if not (ci.LOCAL/'protocol.json').exists():execute('prefix_performance_ci','prepare','--batch','A100')
        pending=load(ci.LOCAL/'protocol.json')
        if pending['worker_SHA256']!=m4['worker_SHA256'] or pending['pipeline_SHA256']!=m4['pipeline_SHA256']:
            raise RuntimeError('A100/M4 common readout parity broken; no dispatch')
        if not (ci.LOCAL/'job.json').exists():
            if (ci.LOCAL/'submission-intent.json').exists():raise RuntimeError('Unresolved CI dispatch intent; no blind re-submit')
            execute('prefix_performance_ci','submit','--batch','A100')
        job=load(ci.LOCAL/'job.json')['job_id']
        while monotonic()-started<18000:
            try:states,evidence=observe([job])
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
                progress('CI_OBSERVATION_UNAVAILABLE_NO_RESTART',job_id=job,error=repr(exc));sleep(60);continue
            progress('OBSERVING_FROZEN_A100_CI_BATCH_CPU_ONLY',job_id=job,states=states,**evidence)
            if states[job]=='TERMINAL_ENGINEERING_FAILURE':raise RuntimeError('Actual CI job failed; preserve without resubmission')
            if states[job]=='COMPLETED':break
            sleep(60)
        else:raise TimeoutError('Bounded coordinator window ended; CI job left untouched')
        if not (ci.LOCAL/'bundle-collection-v1.json').exists():execute('collect_prefix_ci','collect','--batch','A100')
        execute('analyze_prefix_performance_ci','--batch','A100')
        save(LOCAL/'terminal.json',dict(status='N17_ALL_EXECUTION_BATCHES_FINISHED_ANALYSIS_READY_FOR_WHOLE_STUDY_REVIEW',
            CPU_CI_job=job,elapsed_wall_seconds=monotonic()-started,new_settings_selected_from_energy=False,
            full_paper_superiority=False,goal_complete=False))
        progress('ALL_N17_BATCHES_AUDITED_FULL_GOAL_STILL_UNPROVEN')
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='COORDINATOR_STOPPED_WITH_EVIDENCE_NOT_SCIENTIFIC_NEGATIVE_RESULT',
            error=repr(exc),elapsed_wall_seconds=monotonic()-started,other_jobs_untouched=True,goal_complete=False))
        raise


if __name__=='__main__':run()
