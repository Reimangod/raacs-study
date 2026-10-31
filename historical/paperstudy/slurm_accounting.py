"""Read only this study's submitted job accounting; no external job mutation."""
import csv,json
from datetime import datetime,timedelta,timezone
from io import StringIO
from time import time
from benchmark.gpu_remote import ssh
from .gpu_qualification import REMOTE
from .io import ROOT,save_json,atomic_bytes
from .analysis_products import write_csv


def jobs():
    found={}
    def visit(value,path):
        if isinstance(value,dict):
            if str(value.get('job_id','')).isdigit()and str(value.get('remote','')).startswith(REMOTE+'/'):
                found[str(value['job_id'])]={'source':str(path.relative_to(ROOT)),'remote':value['remote']}
            for v in value.values():visit(v,path)
        elif isinstance(value,list):
            for v in value:visit(v,path)
    for pattern in('qualification/**/job*.json','remote-CPU-LUCJ-v1/**/jobs.json','remote-S5-LUCJ-v1/**/job.json',
        'TN-controls-v1/**/job.json','TN-controls-v2/**/job.json','TN-high-final-v1/*/job.json','TN-successor-v2/qualification/job.json',
        'TN-runtime-v1/job.json','TN-MPS-diagnostic-v1/job.json','hardware-postprocessing-v*/*/job.json',
        'paper-anchors/P2-Fe-remote-v1/*/job.json'):
        for p in ROOT.glob(pattern):visit(json.loads(p.read_text()),p)
    return found


def gpu_count(text):
    values=dict(piece.split('=',1)for piece in text.split(',')if'='in piece)
    if'gres/gpu'in values:return int(values['gres/gpu'])
    return sum(int(v)for k,v in values.items()if k.startswith('gres/gpu:'))


def seconds(text):
    if not text:return None
    days=0
    if'-'in text:day,text=text.split('-',1);days=int(day)
    parts=list(map(float,text.split(':')))
    return days*86400+sum(v*60**i for i,v in enumerate(reversed(parts)))


def measured_CPU_time(raw,elapsed):
    value=seconds(raw)
    # This site's TotalCPU is 00:00:00 even for long, demonstrably CPU-bound
    # jobs. Preserve the reported field, but do not turn missing accounting
    # instrumentation into a claim that the computation was free.
    if value==0 and int(elapsed or 0)>0:return None
    return value


def unclosed_or_unclassified_jobs(rows):
    # The states actually observed in this study's terminal job evidence.
    # Unrecognized future states are preserved, not guessed to be finished.
    return [r['job_id']for r in rows if r['state'].split()[0]not in('COMPLETED','TIMEOUT','CANCELLED')]


def overlap(intervals):
    proof=[]
    for i,a in enumerate(intervals):
        for b in intervals[i+1:]:
            start,end=max(a['start'],b['start']),min(a['end'],b['end'])
            if end>start:proof.append({'job_ids':[a['job_id'],b['job_id']],'start_unix':start,'end_unix':end,
                'overlap_seconds':end-start,'allocated_GPU_count_sum':a['GPUs']+b['GPUs']})
    return proof


def run():
    sources=jobs()
    if not sources:return
    fmt='JobIDRaw,JobName%80,State,Partition,Start,End,Submit,ElapsedRaw,AllocCPUS,TotalCPU,CPUTimeRAW,ReqTRES%150,AllocTRES%150,MaxRSS,ExitCode'
    raw=ssh('sacct -n -j '+','.join(sorted(sources,key=int))+' --format='+fmt+' -P')
    out=ROOT/'analysis-v1';atomic_bytes(out/'slurm-accounting-source.psv',raw.encode())
    offset=ssh('date +%z').strip();sign=1 if offset[0]=='+'else-1
    zone=timezone(sign*timedelta(hours=int(offset[1:3]),minutes=int(offset[3:5])))
    def epoch(s):
        try:return datetime.fromisoformat(s).replace(tzinfo=zone).timestamp()
        except ValueError:return None
    columns=['job_id','name','state','partition','start','end','submit','elapsed_seconds','allocated_CPUs','actual_CPU_time',
        'allocated_CPU_seconds','requested_TRES','allocated_TRES','MaxRSS_source','exit_code']
    rows=[];now=time();intervals=[]
    for fields in csv.reader(StringIO(raw),delimiter='|'):
        if not fields:continue
        d=dict(zip(columns,fields));job=d['job_id']
        if job not in sources:continue # Steps are retained in raw for RSS; not summed twice.
        d.update(sources[job]);d['Slurm_reported_TotalCPU_seconds']=seconds(d['actual_CPU_time'])
        d['actual_CPU_seconds']=measured_CPU_time(d['actual_CPU_time'],d['elapsed_seconds'])
        d['zero_Slurm_TotalCPU_not_assumed_free']=d['actual_CPU_seconds']is None
        d['allocated_GPUs']=gpu_count(d['allocated_TRES'])
        d['allocated_GPU_seconds']=d['allocated_GPUs']*int(d['elapsed_seconds'] or 0)
        d['queue_wait_seconds']=max(0,epoch(d['start'])-epoch(d['submit']))if epoch(d['start'])and epoch(d['submit'])else None
        d['allocation_is_not_GPU_kernel_utilization']=True;rows.append(d)
        if d['allocated_GPUs']and d['state'].split()[0]in('RUNNING','COMPLETED'):
            s,e=epoch(d['start']),epoch(d['end'])
            if s and s<now:
                intervals.append({'job_id':job,'start':s,'end':min(e or now,now),'GPUs':d['allocated_GPUs']})
    proof=overlap(intervals);write_csv(out/'slurm-job-costs-counted-once.csv',rows)
    save_json(out/'GPU-concurrency-evidence.json',{'observed_at_unix':now,'remote_timezone_offset':offset,
        'study_job_sources':sources,'GPU_intervals':intervals,'overlaps':proof,
        'two_GPU_simultaneous_allocation_observed':any(p['allocated_GPU_count_sum']>=2 for p in proof),
        'actual_utilization_or_speedup_claimed':False,'allocated_GPU_seconds_observed':sum(r['allocated_GPU_seconds']for r in rows),
        'allocated_CPU_seconds_observed':sum(int(r['allocated_CPU_seconds'] or 0)for r in rows),
        'actual_CPU_seconds_observed':sum(r['actual_CPU_seconds']for r in rows)if all(r['actual_CPU_seconds']is not None for r in rows)else None,
        'known_CPU_seconds_subtotal':sum(r['actual_CPU_seconds']or 0 for r in rows),
        'jobs_with_unresolved_actual_CPU_time':sum(r['actual_CPU_seconds']is None for r in rows),
        'worker_process_CPU_evidence_is_separate_not_additive_to_same_job_actual_CPU':True,
        'live_jobs_have_incomplete_accounting':bool(unclosed_or_unclassified_jobs(rows)),
        'jobs_with_nonterminal_or_unclassified_state':unclosed_or_unclassified_jobs(rows),
        'running_job_accounting_would_be_provisional':True,'steps_not_double_added':True})
    print(json.dumps({'jobs':len(rows),'two_GPU_overlap_intervals':len(proof),'actual_speedup_proved':False}),flush=True)


if __name__=='__main__':run()
