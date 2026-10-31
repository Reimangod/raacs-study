"""Complete the original plan's high-budget TN readouts, not new training.

The existing20 medium-budget readouts remain immutable supplementary data.
All four already selected operators receive five1M independent readouts. This
post-start scope-completion supplement is not a new confirmatory endpoint.
"""
import argparse,fcntl,json,shlex,subprocess,shutil
from pathlib import Path
from time import sleep,perf_counter,process_time
import numpy as np
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from .budget import BudgetLedger
from .common_execution import load_physical,run_identity,final_readouts
from .gpu_qualification import REMOTE,PYTHON
from .io import ROOT,save_json,atomic_bytes,digest,sha256
from .tn_final_readout import CASES,SEEDS,METHOD

BUDGET=1000000
LANE='S_TN_HIGH_BUDGET_COMPLETION'


def spec():
    sources={}
    for case in CASES:
        old=ROOT/'TN-controls-v1'/case/'final-readout'
        if not(old/'completed.json').exists():raise RuntimeError('All original TN sources must be terminal first')
        construction=json.loads((old/'construction.json').read_text())
        effective_path=ROOT/'TN-controls-v1'/case/'effective-source-v2.json'
        effective=json.loads(effective_path.read_text())if effective_path.exists()else None
        source=ROOT/effective['local_source_directory']if effective else ROOT/'TN-controls-v1'/case
        operator=source/('selected-operator.npz'if effective else'optimized-operator.npz')
        if sha256(operator.read_bytes())!=construction['operator_sha256']:raise RuntimeError('Selected TN operator changed')
        sources[case]={'operator':str(operator.relative_to(ROOT)),'operator_sha256':construction['operator_sha256'],
            'construction':str((old/'construction.json').relative_to(ROOT)),
            'construction_sha256':sha256((old/'construction.json').read_bytes()),
            'problem_id':json.loads((old/'request.json').read_text())['problem_id']}
    value={'version':1,'lane':LANE,'cases':list(CASES),'seeds':list(SEEDS),'method_id':METHOD,
        'model_shots_per_readout':BUDGET,'planned_readouts':20,'sources':sources,
        'reason':'Science plan section5.2 says high-budget TN; existing20 final readouts are100k, not1M',
        'old_medium_readouts_preserved':True,'new_optimizers':0,'new_operator_selection':False,
        'all_cases_not_only_favorable_cases':True,'post_start_scope_completion_not_confirmatory':True,
        'same_numeric_seed_across_budgets_may_share_sampling_random_stream_prefix':True,
        'five_independent_readouts_within_each_budget_not_independent_optimizer_repeats':True,
        'actual_QPU_shots':0,'controller_sha256':sha256(Path(__file__).read_bytes()),
        'worker_sha256':sha256(Path(__file__).with_name('remote_cpu_lucj_worker.py').read_bytes())}
    path=ROOT/'TN-high-final-v1/spec.json'
    if path.exists()and json.loads(path.read_text())!=value:raise RuntimeError('High TN readout specification changed')
    save_json(path,value);return value


def submit(case,contract):
    out=ROOT/'TN-high-final-v1'/case;out.mkdir(parents=True,exist_ok=True)
    if(out/'job.json').exists():return json.loads((out/'job.json').read_text())
    if(out/'submission-intent.json').exists():raise RuntimeError('Uncertain high TN submission; reconcile before retry')
    source=contract['sources'][case];m,_,_=load_physical(case)
    if source['problem_id']!=m['problem_id']:raise RuntimeError('High TN physical problem mismatch')
    construction=json.loads((ROOT/source['construction']).read_text());ledger=BudgetLedger(ROOT/'common-budget.sqlite');requests=[]
    for seed in SEEDS:
        key=run_identity(case,METHOD,BUDGET,seed,LANE);ledger.register(key,BUDGET)
        request={'operator_sha256':source['operator_sha256'],'problem_id':m['problem_id'],
            'seed':seed,'shots':BUDGET,'role':'independent-final'}
        status=ledger.reserve(key,'independent-final',request,BUDGET)
        if status not in('NEW_RESERVATION','RESERVED'):raise RuntimeError('High TN draw already executed; do not duplicate')
        requests.append({'run_id':key,'seed':seed,'shots':BUDGET})
    remote=REMOTE+'/TN-high-final-v1/'+case;ssh('mkdir -p '+remote)
    worker=Path(__file__).with_name('remote_cpu_lucj_worker.py')
    if sha256(worker.read_bytes())!=contract['worker_sha256']:raise RuntimeError('Qualified final worker changed')
    config={'case_id':case,'problem_id':m['problem_id'],'norb':m['norb'],'nelec':[m['nalpha'],m['nbeta']],
        'operator_sha256':source['operator_sha256'],'requests':requests,'CPU_parity_reference':None,
        'pinned_CPU_ffsim_platform_previously_qualified':True,'worker_sha256':contract['worker_sha256'],
        'lane':LANE,'scope_completion_spec_digest':digest(contract),'new_optimizer':False}
    config['config_id']=digest(config);save_json(out/'request.json',config);save_json(out/'construction.json',construction)
    send(worker,remote+'/worker.py');send(ROOT/source['operator'],remote+'/operator.npz');send(out/'request.json',remote+'/request.json')
    script='\n'.join(['#!/bin/bash','set -euo pipefail',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
        'export PYTHONPATH='+REMOTE+'/cpu-runtime-v1/packages',
        shlex.join([PYTHON,remote+'/worker.py',remote+'/request.json',remote+'/operator.npz',remote+'/output'])])+'\n'
    atomic_bytes(out/'job.sbatch',script.encode());send(out/'job.sbatch',remote+'/job.sbatch')
    command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=24G','--time=00:20:00',
        '--job-name=paper-TN-high-'+case[:4],'--output='+remote+'/slurm.out','--error='+remote+'/slurm.err',remote+'/job.sbatch'])
    intent={'case_id':case,'remote':remote,'command':command,'config_id':config['config_id'],
        'operator_sha256':source['operator_sha256'],'worker_sha256':contract['worker_sha256'],'GPU_requested':False}
    save_json(out/'submission-intent.json',intent);reply=ssh(command).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Unknown high TN Slurm submission')
    job={**intent,'job_id':reply};save_json(out/'job.json',job);print('TN_HIGH_SUBMITTED',case,reply,flush=True);return job


def collect(case,job):
    out=ROOT/'TN-high-final-v1'/case
    if(out/'completed.json').exists():return True
    state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
    save_json(out/'slurm-observed.json',{'accounting':state})
    if any(s in state for s in('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY')):
        raise RuntimeError('High TN job terminal without automatic resampling: '+state)
    if 'COMPLETED'not in state:return False
    raw=json.loads(ssh('cat '+job['remote']+'/output/result.json'))
    if raw.get('config_id')!=job['config_id']or raw.get('status')!='EXACT_REMOTE_CPU_LUCJ_COMPLETED':raise RuntimeError('High TN output binding mismatch')
    save_json(out/'remote-result.json',raw);construction=json.loads((out/'construction.json').read_text())
    m,h,g=load_physical(case);ledger=BudgetLedger(ROOT/'common-budget.sqlite')
    if sorted(v['seed']for v in raw['sampling_results'])!=sorted(SEEDS):raise RuntimeError('Wrong high TN seed coverage')
    for record in raw['sampling_results']:
        seed=record['seed'];key=run_identity(case,METHOD,BUDGET,seed,LANE)
        if key!=record['run_id']or record['actual_model_shots']!=BUDGET:raise RuntimeError('High TN run/shot binding mismatch')
        output=ROOT/'common-runs'/LANE/case/METHOD/(str(BUDGET)+'-'+str(seed));output.mkdir(parents=True,exist_ok=True)
        if(output/'result.json').exists():continue
        target=output/'final-counts.npz'
        subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,HOST+':'+job['remote']+'/output/'+key+'.npz',str(target)],check=True,timeout=120)
        if sha256(target.read_bytes())!=record['counts_sha256']:raise RuntimeError('High TN counts digest mismatch')
        with np.load(target,allow_pickle=False)as a:ids,counts=a['blocked_ids'].copy(),a['counts'].copy()
        if int(counts.sum())!=BUDGET:raise RuntimeError('High TN actual count total mismatch')
        ledger.complete(key,'independent-final',BUDGET,record['counts_sha256'])
        tick,cpu=perf_counter(),process_time();readouts=final_readouts(output,m,h,g,ids,counts)
        save_json(output/'result.json',{'status':'COMPLETED','lane':LANE,'run_id':key,'case_id':case,'method_id':METHOD,
            'budget':BUDGET,'seed':seed,'problem_id':m['problem_id'],'final_CI':readouts,'budget_ledger':ledger.snapshot(key),
            'independent_final_model_shots':BUDGET,'pilot_model_shots':0,'actual_QPU_shots':0,
            'sampling_resources':construction['resources']['Z_sampling_equivalent'],'preparation_resources':construction['resources']['preparation'],
            'coefficient_slots':construction['coefficient_slots'],'nonredundant_independent_parameters':None,'VQE_optimized_variables':0,
            'shared_optimization_result':construction['optimization_source'],'shared_operator_sha256':construction['operator_sha256'],
            'total_model_cnot_applications':BUDGET*construction['resources']['Z_sampling_equivalent']['cnot_count'],
            'this_attempt_wall_seconds':perf_counter()-tick,'this_attempt_CPU_seconds':process_time()-cpu,'local_attempt_time_is_CI_only':True,
            'remote_shared_simulation_result':str((out/'remote-result.json').relative_to(ROOT)),
            'post_start_scope_completion_not_confirmatory':True,'new_optimizer':False,'FCI_used_for_construction':False})
        print('TN_HIGH_COLLECTED',case,seed,flush=True)
    save_json(out/'completed.json',{'status':'FIVE_HIGH_BUDGET_READOUTS_COMPLETED','model_final_shots':5*BUDGET,'new_optimizers':0})
    return True


def main():
    with(ROOT/'TN-high-final-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);contract=spec()
        while True:
            if shutil.disk_usage(ROOT).free<3*2**30:sleep(30);continue
            done=0;attention={}
            for case in CASES:
                terminal=ROOT/'TN-high-final-v1'/case/'terminal-attention.json'
                if terminal.exists():attention[case]=json.loads(terminal.read_text());continue
                try:
                    if collect(case,submit(case,contract)):done+=1
                except RuntimeError as exc:
                    attention[case]={'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),'no_automatic_resampling':True}
                    save_json(terminal,attention[case])
            save_json(ROOT/'TN-high-final-v1/status.json',{'completed_cases':done,'planned_cases':4,'planned_readouts':20,
                'model_shots_per_readout':BUDGET,'attention_cases':attention,'new_optimizers':0,'whole_experiment_complete':False})
            if done+len(attention)==4:return
            sleep(45)


if __name__=='__main__':main()
