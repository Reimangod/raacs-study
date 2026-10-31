"""One-connection, durable-receipt N33 lane1 reconciliation, no blind retry."""
import json,shlex
from benchmark.gpu_remote import ssh
from . import analytical_source_sampling as s
from .io import load,save,sha256


def run():
    s.verified();folder=s.SAMPLING;target=s.REMOTE+'/lane1.sbatch'
    if (folder/'lane1-job.json').exists():raise RuntimeError('Known job; no resubmit')
    local=folder/'lane1-reconciliation-v2.json'
    if local.exists():raise RuntimeError('Existing v2: inspect response rather than rerun')
    intent=load(folder/'lane1-submission-intent.json');expected=sha256((folder/'lane1.sbatch').read_bytes())
    # The actual status audit and at-most-once submission share one SSH session.
    # Receipt is written on remote before the response. A partial intent is NOT
    # automatically retried if sbatch/connection failed at an uncertain point.
    code='''import os,json,pathlib,hashlib,subprocess
r=pathlib.Path(REMOTE); target=str(r/'lane1.sbatch'); receipt=r/'lane1-reconciled-receipt-v2.json'; marker=r/'lane1-reconciled-intent-v2.json'
assert hashlib.sha256(pathlib.Path(target).read_bytes()).hexdigest()==EXPECTED
queue=subprocess.check_output(['squeue','-h','-u','reiwatanabe','-o','%i|%T|%j|%o'],text=True)
history=subprocess.check_output(['sacct','-X','-n','-u','reiwatanabe','-S','now-2hours','--format=JobID,JobName%200,State,SubmitLine%2000','-P'],text=True)
matches=[x for value in (queue,history) for x in value.splitlines() if target in x]
proof=dict(queue=queue,history=history,exact_target_matches=matches,script_SHA256=EXPECTED)
if receipt.exists():
 ans=json.loads(receipt.read_text()); assert ans['script_SHA256']==EXPECTED; ans['reused_existing_receipt']=True
elif matches:
 ids={x.split('|')[0].strip() for x in matches}; assert len(ids)==1
 ans=dict(status='RECOVERED_EXISTING_EXACT_TARGET',job_id=ids.pop(),script_SHA256=EXPECTED,audit=proof)
elif marker.exists():
 raise RuntimeError('Partial remote submission intent; no automatic resubmit')
else:
 assert history.strip() and '7036|' in history
 fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f: json.dump(dict(audit=proof,command=COMMAND),f); f.flush();os.fsync(f.fileno())
 answer=subprocess.check_output(shlex.split(COMMAND),text=True).strip(); job=answer.split(';')[0]; assert job.isdigit()
 ans=dict(status='SUBMITTED_ONCE_AFTER_ACTUAL_ABSENCE',job_id=job,script_SHA256=EXPECTED,audit=proof)
 with receipt.open('x') as f: json.dump(ans,f);f.flush();os.fsync(f.fileno())
print(json.dumps(ans))
'''
    code='import shlex\nREMOTE='+repr(s.REMOTE)+'\nEXPECTED='+repr(expected)+'\nCOMMAND='+repr(intent['command'])+'\n'+code
    answer=json.loads(ssh(shlex.join(['python3','-c',code]),timeout=45));save(local,answer)
    if not answer['job_id'].isdigit() or answer['script_SHA256']!=expected:raise RuntimeError('Actual job receipt invalid')
    record=dict(job_id=answer['job_id'],lane=1,circuit_ids=intent['circuit_ids'],remote=s.REMOTE)
    save(folder/'lane1-job.json',record);save(folder/'jobs.json',[load(folder/'lane0-job.json'),record])
    print('N33_LANE1_ACTUAL_RECONCILED_JOB',answer['job_id'],answer['status'],flush=True)


if __name__=='__main__':run()
