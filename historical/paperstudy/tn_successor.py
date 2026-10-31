"""One bounded, outcome-independent engineering successor for unusable TN attempts.

Original jobs/evidence are immutable. A live original is never restarted. An
original with a usable final state is never superseded to seek a better score.
The successful attempt's caps are unchanged; failed-attempt costs stay payable.
"""
import json,shlex,subprocess
from pathlib import Path
from time import sleep,time
import ffsim,numpy as np
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from .gpu_qualification import REMOTE,PYTHON
from .io import ROOT,atomic_bytes,save_json,digest,sha256
from .tn_checkpoint import load_best

CASES=('FE2S2-40-P2','NAPH20-P1','N2-32-R1p2-P2','N2-32-R2p4-P2')


def freeze():
    files=[Path(__file__).with_name(n)for n in('tn_checkpoint.py','tn_checkpoint_worker.py','ci.py')]
    value={'version':'TN-engineering-successor-v2','cases':list(CASES),'maximum_successor_attempts_per_case':1,
        'trigger':'Actual original terminal and no usable final operator, never source surrogate energy',
        'scientific_invariants':{'MPS_max_bond':50,'MPS_cutoff':1e-10,'surrogate_seed':31001,'optimizer_seed':31001,
            'surrogate_samples_per_call':10000,'common_CI_r_cap':32,'maximum_started_objective_calls':500,
            'objective_start_wall_cap_seconds':3000,'NOMAD_MAX_TIME':3000,'subproblems':3,'parallel_evaluations':4,
            'initial_operator':'Same byte-identical input as original case','final_shots':100000,'final_seeds':[21001,21011,21019,21023,21031]},
        'changes':['Atomic per-trial theta and best-state checkpoints','NOMAD native MAX_TIME, not only callback rejection',
            'Completed-objective-backed best may be recovered after external termination; no optimizer convergence claim'],
        'not_changed':['Primary480 or S5 frozen125 scientific code','MPS objective and final exact readout','Method score or angles'],
        'failed_original_cost_included':True,'equal_total_CPU_work_not_claimed':True,'paper_native_TN_Dice_reproduction_claimed':False,
        'code_sha256':{p.name:sha256(p.read_bytes())for p in files},
        'NOMAD_MAX_TIME_source':'https://github.com/bbopt/nomad/blob/master/doc/user_guide/source/HowToUseNomad.rst'}
    path=ROOT/'TN-successor-v2/contract.json'
    if path.exists()and json.loads(path.read_text())!=value:raise RuntimeError('Frozen engineering successor changed')
    save_json(path,value);return value


def script(root,remote,worker,seconds=3300,name='paper-TN-v2',mem='16G'):
    body='\n'.join(['#!/bin/bash','set -euo pipefail','export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
        'export PYTHONPATH='+remote+':'+REMOTE+'/tn-runtime-v1/packages:'+REMOTE+'/cpu-runtime-v1/packages',
        shlex.join([PYTHON,'-u',remote+'/'+worker,remote+'/request.json'])])+'\n'
    atomic_bytes(root/'job.sbatch',body.encode());send(root/'job.sbatch',remote+'/job.sbatch')
    return shlex.join(['sbatch','--parsable','--partition=gpu-standard','--cpus-per-task=4','--mem='+mem,
        '--time=00:'+str(seconds//60).zfill(2)+':'+str(seconds%60).zfill(2),'--job-name='+name,
        '--output='+remote+'/slurm.out','--error='+remote+'/slurm.err',remote+'/job.sbatch'])


def qualification():
    root=ROOT/'TN-successor-v2/qualification';root.mkdir(parents=True,exist_ok=True)
    if(root/'result.json').exists():
        result=json.loads((root/'result.json').read_text())
        if result['status']!='MAX_TIME_AND_DURABLE_CHECKPOINT_QUALIFIED':raise RuntimeError('Unqualified TN successor')
        return True
    if not(root/'job.json').exists():
        if(root/'submission-intent.json').exists():raise RuntimeError('Uncertain TN qualification submission')
        remote=REMOTE+'/TN-successor-v2/qualification';ssh('mkdir -p '+remote)
        code='''import json,time,threading
from pathlib import Path
import numpy as np,PyNomad
from tn_checkpoint import sha,save,publish_best,load_best
root=Path(__file__).parent;lock=threading.Lock();records=[];best=[None];config={'config_id':'engineering-fixture','problem_id':'quadratic-not-molecule'}
def fun(point):
    x=np.array([point.get_coord(i)for i in range(point.size())]);time.sleep(.03);value=float(np.sum((x-.1)**2))
    with lock:
        i=len(records);record={'index':i,'status':'COMPLETED','theta_sha256':sha(x.astype('<f8').tobytes()),'CI':{'energy_Ha':value}}
        records.append(record);save(root/'surrogate-evaluations'/(str(i)+'.json'),record)
        if best[0]is None or value<best[0]:publish_best(root,config,record,x,{'structural_test_theta':x});best[0]=value
    point.setBBO(str(value).encode());return 1
start=time.perf_counter()
options=['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 100000','MAX_TIME 1','DISPLAY_DEGREE 0','PSD_MADS_OPTIMIZATION true','PSD_MADS_NB_VAR_IN_SUBPROBLEM 5','PSD_MADS_SUBPROBLEM_MAX_BB_EVAL 4','PSD_MADS_NB_SUBPROBLEM 3','NB_THREADS_PARALLEL_EVAL 4','SEED 31001']
returned=PyNomad.optimize(fun,[0.]*24,[],[],options);pointer=load_best(root,config);elapsed=time.perf_counter()-start
if not records or elapsed>90:raise RuntimeError('Native MAX_TIME did not provide bounded return')
save(root/'result.json',{'status':'MAX_TIME_AND_DURABLE_CHECKPOINT_QUALIFIED','wall_seconds':elapsed,'objective_calls':len(records),'best_checkpoint':pointer,'optimizer_result':returned,'molecular_outcomes_evaluated':False,'native_MAX_TIME_accepted':True})
'''
        atomic_bytes(root/'qualification.py',code.encode());send(root/'qualification.py',remote+'/qualification.py')
        send(Path(__file__).with_name('tn_checkpoint.py'),remote+'/tn_checkpoint.py')
        save_json(root/'request.json',{'engineering_fixture_only':True});send(root/'request.json',remote+'/request.json')
        command=script(root,remote,'qualification.py',120,'paper-TN-v2-qual','2G')
        intent={'remote':remote,'command':command,'source_sha256':sha256(code.encode()),'molecular_outcomes':False}
        save_json(root/'submission-intent.json',intent);reply=ssh(command).strip().split(';')[0]
        if not reply.isdigit():raise RuntimeError('Uncertain qualification sbatch response')
        save_json(root/'job.json',{**intent,'job_id':reply});print('TN_V2_QUAL_SUBMITTED',reply,flush=True)
        return False
    job=json.loads((root/'job.json').read_text());state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
    if'COMPLETED'not in state:
        if any(s in state for s in('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED')):raise RuntimeError('TN successor qualification failed: '+state)
        return False
    result=json.loads(ssh('cat '+job['remote']+'/result.json'));save_json(root/'result.json',result)
    return qualification()


def original_state(case):
    root=ROOT/'TN-controls-v1'/case;job=json.loads((root/'job.json').read_text())
    state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
    terminal=any(s in state for s in('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','COMPLETED'))
    available=ssh('if test -f '+job['remote']+'/optimized-operator.npz && test -f '+job['remote']+'/optimization-result.json; then echo yes; fi').strip()=='yes'
    return job,state,terminal,available


def submit(case):
    root=ROOT/'TN-controls-v2'/case;root.mkdir(parents=True,exist_ok=True)
    if(root/'job.json').exists():return json.loads((root/'job.json').read_text())
    old,state,terminal,available=original_state(case)
    if available:return 'ORIGINAL_FINAL_AVAILABLE'
    if not terminal:return None
    if(root/'submission-intent.json').exists():raise RuntimeError('Uncertain TN successor submission; do not duplicate')
    previous=json.loads((ROOT/'TN-controls-v1'/case/'request.json').read_text())
    contract=freeze();config={k:v for k,v in previous.items()if k not in('config_id','worker_sha256')}
    worker=Path(__file__).with_name('tn_checkpoint_worker.py');helper=Path(__file__).with_name('tn_checkpoint.py')
    physical=ROOT/'data/physical'/case/'integrals.npz';operator=ROOT/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/operator.npz'
    for p,name in((physical,'integrals.npz'),(operator,'initial-operator.npz')):
        if sha256(p.read_bytes())!=previous['input_sha256'][name]:raise RuntimeError('Cannot restart from a different physical input or initial state')
    config.update(worker_sha256=sha256(worker.read_bytes()),checkpoint_code_sha256=sha256(helper.read_bytes()),
        engineering_successor_contract=digest(contract),original_job=old,original_terminal_accounting=state,
        maximum_successor_attempts_per_case=1,original_attempt_cost_retained=True,optimizer_convergence_not_claimed=True)
    config['config_id']=digest(config);save_json(root/'request.json',config)
    remote=REMOTE+'/TN-controls-v2/'+case;ssh('mkdir -p '+remote+'/paperstudy')
    for p,name in((worker,'worker.py'),(helper,'tn_checkpoint.py'),(physical,'integrals.npz'),(operator,'initial-operator.npz'),
        (root/'request.json','request.json'),(Path(__file__).with_name('ci.py'),'paperstudy/ci.py'),(Path(__file__).with_name('__init__.py'),'paperstudy/__init__.py')):send(p,remote+'/'+name)
    command=script(root,remote,'worker.py',3300,'paper-TN-v2-'+case[:5])
    intent={'case_id':case,'remote':remote,'config_id':config['config_id'],'command':command,'worker_sha256':config['worker_sha256'],
        'original_job_id':old['job_id'],'original_attempt_cost_retained':True,'no_original_live_job_restarted':True}
    save_json(root/'submission-intent.json',intent);reply=ssh(command).strip().split(';')[0]
    if not reply.isdigit():raise RuntimeError('Uncertain TN successor submission')
    job={**intent,'job_id':reply};save_json(root/'job.json',job);print('TN_V2_SUBMITTED',case,reply,flush=True);return job


def collect(case,job):
    root=ROOT/'TN-controls-v2'/case
    if(root/'effective-source-published.json').exists():return True
    state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
    if not any(s in state for s in('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','COMPLETED')):return False
    # Preserve the entire bounded successor; every trial theta/count is small.
    destination=root/'terminal-evidence';destination.mkdir(exist_ok=True)
    if not(root/'terminal-copy.json').exists():
        size=int(ssh('du -sb '+job['remote']).split()[0])
        if size>100*2**20:raise MemoryError('Inspect unexpectedly large TN successor output')
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+job['remote']+'/.',str(destination/'remote')],check=True,timeout=120)
        save_json(root/'terminal-copy.json',{'state':state,'files_sha256':{str(p.relative_to(destination)):sha256(p.read_bytes())for p in destination.rglob('*')if p.is_file()}})
    config=json.loads((root/'request.json').read_text());saved=destination/'remote'
    if not(saved/'best-state.json').exists():
        save_json(root/'no-final-state.json',{'state':state,'no_automatic_third_attempt':True});return True
    pointer=load_best(saved,config)
    # Independent reconstruction checks that serialized coefficients really
    # describe the checkpoint operator, rather than merely agreeing on hashes.
    pairs=tuple([tuple(p)for p in v]for v in config['pairs'])
    reconstructed=ffsim.UCJOpSpinBalanced.from_parameters(np.array(pointer['theta']),norb=config['norb'],n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
    with np.load(saved/pointer['operator'],allow_pickle=False)as a:
        differences={name:float(np.max(abs(a[name]-getattr(reconstructed,name))))for name in('diag_coulomb_mats','orbital_rotations','final_orbital_rotation')}
    if max(differences.values())>1e-8:raise RuntimeError('Checkpoint operator is not described by saved theta')
    normal=saved/'optimization-result.json'
    if normal.exists():
        result=json.loads(normal.read_text())
        if result['configuration']['config_id']!=job['config_id']or result['operator_sha256']!=pointer['operator_sha256']:raise RuntimeError('TN successor result binding mismatch')
        metadata='optimization-result.json';operator='optimized-operator.npz'
    else:
        result={'status':'TN_COMMON_READOUT_BOUNDED_CHECKPOINT_RECOVERED','configuration':config,'problem_id':config['problem_id'],
            'selected_coefficients':pointer['theta'],'operator_sha256':pointer['operator_sha256'],'checkpoint':pointer,
            'original_terminal_accounting':state,'optimizer_returned':False,'optimizer_convergence_not_claimed':True,
            'independent_final_readout_still_required':True,'failed_original_attempt_cost_retained':True}
        metadata='bounded-checkpoint-reconciliation.json';operator=pointer['operator']
        save_json(root/metadata,result);send(root/metadata,job['remote']+'/'+metadata)
    save_json(root/'selected-source-result.json',result)
    pointer_value={'job':job,'local_source_directory':str(root.relative_to(ROOT)),'metadata_file':metadata,'remote_operator_file':operator,
        'source_result_sha256':digest(result),'independent_coefficient_to_operator_max_differences':differences,
        'original_attempt_not_erased':True,'same_final_shots_and_seeds':True}
    save_json(ROOT/'TN-controls-v1'/case/'effective-source-v2.json',pointer_value)
    save_json(root/'effective-source-published.json',pointer_value);print('TN_V2_FINAL_SOURCE_READY',case,flush=True);return True


def main():
    import fcntl
    with(ROOT/'TN-successor-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze()
        while not qualification():sleep(20)
        while True:
            terminal=0
            for case in CASES:
                result=submit(case)
                if result=='ORIGINAL_FINAL_AVAILABLE':terminal+=1
                elif isinstance(result,dict)and collect(case,result):terminal+=1
            save_json(ROOT/'TN-successor-v2/status.json',{'resolved_cases':terminal,'planned_cases':4,'no_automatic_third_attempt':True,'whole_experiment_complete':False})
            if terminal==4:return
            sleep(45)


if __name__=='__main__':main()
