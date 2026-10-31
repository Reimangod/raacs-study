"""Bounded collect/re-read of fixed D2/F2/H1 counts, no shot execution."""
import fcntl
import os
from pathlib import Path
from time import monotonic,sleep,time,process_time,perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256,arrays,space,digest
from .lowest_orbital import optimize
from .marginal_remote_worker import select_half_pool
from .half_remap_remote import collect as collect_h1,DIRECTORY as H1
from .marginal_remote import DIRECTORY as F2
from paperstudy.ci import common_strings
from paperstudy.budget import BudgetLedger

DIRECTORY=ROOT/'runs/C7-LOWEST-ROOT-v1'


def binding():
    names=['lowest_penalized.py','lowest_orbital.py','lowest_readouts.py','energy_cache.py','hole_rdm.py','marginal_remote_worker.py']
    return {'protocol_SHA256':sha256((ROOT/'protocol-C7-LOWEST-ROOT.json').read_bytes()),
        'code_SHA256':{n:sha256((ROOT/'frontierloop'/n).read_bytes()) for n in names},
        'settings_SHA256':sha256((ROOT/'protocol-C3-SPIN-OO.json').read_bytes()),
        'qualification_SHA256':sha256((ROOT/'qualification-lowest-orbital-tests.xml').read_bytes()),
        'low_level_qualification_SHA256':sha256((ROOT/'qualification-lowest-penalized-v2-tests.xml').read_bytes())}


def freeze():
    DIRECTORY.mkdir(exist_ok=True);path=DIRECTORY/'freeze.json';b=binding()
    if path.exists():
        if load(path)!=b:raise RuntimeError('C7 frozen code changed')
    else:
        for name,count in [('qualification-lowest-orbital-tests.xml',2),('qualification-lowest-penalized-v2-tests.xml',1)]:
            suites=list(ET.parse(ROOT/name).getroot().iter('testsuite'))
            if sum(int(s.get('tests','0')) for s in suites)!=count or any(int(s.get(k,'0')) for s in suites for k in ['failures','errors']):raise RuntimeError('Lowest-root qualification missing')
        save(path,b)
    return b


def source(case,method):
    if method=='F2':return F2/'remote-output/outputs'/case,True
    if method=='H1':return H1/'remote-output/outputs'/case,True
    return ROOT/'runs/D2-COMPRESSION-SCREEN'/case/('F25' if method=='D2_F25_INCIDENT' else 'EXACT'),False


def run(case,method,b):
    src,half=source(case,method);d=DIRECTORY/case/method
    if (d/'result.json').exists():return 'COMPLETED'
    if (d/'failure.json').exists():return 'FAILED_ENGINEERING_PRESERVED'
    if (d/'started.json').exists():return 'PARTIAL_PRESERVED_NO_AUTORETRY'
    if not (src/'draw-result.json').exists():return 'WAITING_FOR_FIXED_SOURCE'
    if binding()!=b:raise RuntimeError('C7 source drift')
    space();draw=load(src/'draw-result.json');path=src/('half-counts.npz' if half else 'final-counts.npz')
    if sha256(path.read_bytes())!=draw['counts_SHA256']:raise RuntimeError('C7 raw count binding mismatch')
    physical=OLD/'data/physical'/case;m=load(physical/'problem.json');ne=m['nalpha'],m['nbeta']
    with np.load(path,allow_pickle=False) as a:
        strings=select_half_pool(a['half_strings'],a['counts'],m['norb'],ne[0],64) if half else common_strings(a['blocked_ids'],a['counts'],m['norb'],ne,64)[0]
    if half:
        assert draw['physical_problem_id']==m['problem_id'];runid=draw['run_id']
    else:
        req=load(src/'request.json');assert req['problem_id']==m['problem_id'];runid=req['run_id']
    for mode in ['C2-OO','C3-SPIN-OO','C4-SINGLET-OO']:
        if (src/mode/'result.json').exists():assert load(src/mode/'result.json')['CI_strings']==strings.tolist()
    request={'case':case,'method':method,'source_run_id':runid,'source_counts_SHA256':draw['counts_SHA256'],
        'source_draw_SHA256':sha256((src/'draw-result.json').read_bytes()),'physical_problem_id':m['problem_id'],
        'integrals_SHA256':sha256((physical/'integrals.npz').read_bytes()),'CI_strings':strings.tolist(),
        'freeze_digest':digest(b),'engineering_incident_not_primary':method=='D2_F25_INCIDENT'}
    d.mkdir(parents=True,exist_ok=True);save(d/'request.json',request)
    BudgetLedger(ROOT/'production-shots.sqlite').record_readout(runid,'C7-LOWEST-ROOT-v1','final',request)
    save(d/'started.json',{'pid':os.getpid(),'request_SHA256':sha256((d/'request.json').read_bytes())})
    cpu,tick=process_time(),perf_counter()
    try:
        with np.load(physical/'integrals.npz',allow_pickle=False) as a:h,g=a['h1'],a['eri']
        def checkpoint(history,rotations):save(d/'checkpoint.json',{'history':history,'new_model_shots':0})
        r=optimize(h,g,m['core_energy'],ne,strings,load(ROOT/'protocol-C3-SPIN-OO.json'),checkpoint)
        rotations,hh,gg=(r.pop(k) for k in ['rotations','final_h1','final_eri'])
        payload=arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings);write(d/'optimized-representation.npz',payload)
        r.update(status='COMPLETED',case=case,method=method,readout='C7-LOWEST-ROOT-v1',source_counts_SHA256=draw['counts_SHA256'],
            physical_problem_id=m['problem_id'],CI_strings=strings.tolist(),actual_CI_dimension=len(strings)**2,r_cap=64,
            representation_SHA256=sha256(payload),whole_body_CPU_seconds=process_time()-cpu,whole_body_wall_seconds=perf_counter()-tick,
            host='Mac_M4_NOT_SLURM_RUNTIME_COMPARABLE',new_model_shots=0,actual_QPU_shots=0,goal_complete=False)
        save(d/'result.json',r);print('C7_READOUT',case,method,r['final_energy_Ha'],flush=True);return 'COMPLETED'
    except Exception as exc:
        save(d/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),'new_model_shots':0,
            'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick});print('C7_FAILED',case,method,repr(exc),flush=True);return 'FAILED_ENGINEERING_PRESERVED'


def main():
    b=freeze();spec=load(ROOT/'protocol-C7-LOWEST-ROOT.json');queue=[(c,m) for m in spec['methods'] for c in spec['cases']]
    queue.append(('N2-32-R2p4-P2','D2_F25_INCIDENT'))
    with (DIRECTORY/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(DIRECTORY/'owner.json',{'pid':os.getpid(),'started_unix':time(),'authority':'EXISTING_COUNTS_READOUT_ONLY_NO_SAMPLER_SUBMISSION'})
        deadline=monotonic()+2700
        while monotonic()<deadline:
            if (H1/'job.json').exists() and not (H1/'collected.json').exists():
                try:collect_h1()
                except Exception as exc:save(DIRECTORY/'H1-collection-observation-error.json',{'error':repr(exc),'source_not_restarted':True})
            states=[]
            for case,method in queue:
                try:status=run(case,method,b)
                except OSError as exc:status='LOCAL_CAPACITY_PAUSE';save(DIRECTORY/'capacity-pause.json',{'error':repr(exc)})
                states.append({'case':case,'method':method,'status':status})
            summary={'states':states,'completed':sum(x['status']=='COMPLETED' for x in states),'planned':len(queue),
                'new_model_shots':0,'goal_complete':False};save(DIRECTORY/'status.json',summary)
            if all(x['status'] in ['COMPLETED','FAILED_ENGINEERING_PRESERVED','PARTIAL_PRESERVED_NO_AUTORETRY'] for x in states):
                save(DIRECTORY/'terminal.json',summary);return
            sleep(30)
        save(DIRECTORY/'controller-time-limit.json',{'status':'BOUNDED_COLLECTOR_PREFIX_PRESERVED_NO_SAMPLER_CANCELLED','goal_complete':False})


if __name__=='__main__':main()
