"""Resolve N33 lane1 pre-auth SSH failure using actual Slurm history first."""
import json,shlex,subprocess
from pathlib import Path
from benchmark.gpu_remote import ssh
from . import analytical_source_sampling as s
from .io import load,save,sha256


def run():
    s.verified();folder=s.SAMPLING;target=s.REMOTE+'/lane1.sbatch'
    if (folder/'lane1-job.json').exists():raise RuntimeError('Already known submission: no retry')
    evidence=folder/'lane1-reconciliation-v1.json'
    if evidence.exists():raise RuntimeError('Reconciliation already attempted: inspect evidence, no blind restart')
    code="import subprocess,json; out={}; out['queue']=subprocess.check_output(['squeue','-h','-u','reiwatanabe','-o','%i|%T|%j|%o'],text=True); out['history']=subprocess.check_output(['sacct','-X','-n','-u','reiwatanabe','-S','now-2hours','--format=JobID,JobName%200,State,SubmitLine%2000','-P'],text=True); print(json.dumps(out))"
    result=json.loads(ssh(shlex.join(['python3','-c',code]),timeout=40))
    result.update(exact_script=target,script_SHA256=sha256((folder/'lane1.sbatch').read_bytes()),
        prior_lane0=load(folder/'lane0-job.json'),no_other_chat_modified=True)
    matches=[line for kind in ('queue','history') for line in result[kind].splitlines() if target in line]
    result['exact_target_matches']=matches
    # A complete pre-auth failure plus no target command in both scheduler and
    # accounting allows ONE reconciled submit. Any existing match requires audit.
    if matches:
        result['status']='EXISTING_TARGET_FOUND_NO_RESUBMISSION';save(evidence,result);raise RuntimeError('Existing target: bind exact job ID after inspection')
    if not result['history'].strip() or '7036|' not in result['history'] or 'SubmitLine' in result['history']:
        raise RuntimeError('Incomplete actual scheduler history; no resubmit')
    result['status']='EXACT_LANE1_ABSENT_IN_LIVE_QUEUE_AND_SUBMITLINE_HISTORY';save(evidence,result)
    intent=load(folder/'lane1-submission-intent.json')
    if target not in intent['command'] or '--gres=gpu:a100:1' not in intent['command']:raise RuntimeError('Exact frozen command required')
    save(folder/'lane1-reconciled-submission-intent-v1.json',dict(command=intent['command'],reconciliation_SHA256=sha256(evidence.read_bytes())))
    answer=ssh(intent['command'],timeout=30).strip();job=answer.split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain response; inspect, never auto-repeat')
    record=dict(job_id=job,lane=1,circuit_ids=intent['circuit_ids'],remote=s.REMOTE)
    save(folder/'lane1-job.json',record)
    save(folder/'jobs.json',[load(folder/'lane0-job.json'),record])
    print('N33_RECONCILED_EXACT_LANE1_SUBMITTED_ONCE',job,flush=True)


if __name__=='__main__':run()
