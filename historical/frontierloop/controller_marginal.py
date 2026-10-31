"""Bounded collection of the already submitted F2 job; never submit or retry draws."""
import fcntl
import os
from time import monotonic,sleep,time
from .paths import ROOT
from .io import load,save
from .marginal_remote import DIRECTORY,collect
from .analyze_marginal import main as analyze


def main():
    job=load(DIRECTORY/'job.json')
    with (DIRECTORY/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(DIRECTORY/'collector-owner.json',{'pid':os.getpid(),'job_id':job['job_id'],'started_unix':time(),
            'authority':'COLLECT_AND_AUDIT_EXISTING_JOB_ONLY_NO_SUBMIT_NO_RERUN','max_wall_seconds':2700})
        deadline=monotonic()+2700;transient=0
        while monotonic()<deadline:
            try:r=collect();transient=0
            except OSError as exc:
                save(DIRECTORY/'collector-pause.json',{'error':repr(exc),'local_capacity_or_IO_pause':True,'no_reexecution':True})
                sleep(30);continue
            except Exception as exc:
                transient+=1
                save(DIRECTORY/'collector-incident.json',{'error':repr(exc),'consecutive_failures':transient,
                    'no_job_cancelled_or_resubmitted':True})
                if transient>=3:raise
                sleep(30);continue
            if r is not None:
                analyze();save(DIRECTORY/'collector-finished.json',{'job_id':job['job_id'],'completed_unix':time(),
                    'status':'EVIDENCE_COLLECTED_AND_AUDITED_NOT_GOAL_COMPLETION','goal_complete':False})
                print('F2_COLLECTION_AND_RAW_AUDIT_FINISHED',flush=True);return
            sleep(30)
        save(DIRECTORY/'collector-time-limit.json',{'status':'COLLECTOR_STOPPED_PRESERVING_REMOTE_JOB','job_id':job['job_id'],
            'no_job_cancelled_or_resubmitted':True,'goal_complete':False})


if __name__=='__main__':main()
