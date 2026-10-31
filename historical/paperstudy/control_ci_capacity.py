"""Apply the already registered CI4096 diagnostic to all S5/TN controls too.

Triggered solely by the primary case's r32 capacity event. All control settings
and seeds in that case receive the same extra readout; zero new observations,
no winner-conditioned setting selection, no original-result changes.
"""
import argparse,fcntl,json,sqlite3
from pathlib import Path
from time import sleep,perf_counter,process_time
import numpy as np
from .io import ROOT,save_json,sha256
from .common_execution import load_physical
from .ci import solve


def sources():
    triggers={p.parent.name for p in(ROOT/'CI-extension-v1').glob('*/trigger.json')}
    db=sqlite3.connect(ROOT/'run-queue.sqlite')
    result=[ROOT/p for case,p in db.execute('SELECT case_id,result_path FROM sensitivity_cells WHERE status="COMPLETED"')if case in triggers]
    db.close()
    for lane in('S_TN_INDEPENDENT_FINAL','S_TN_HIGH_BUDGET_COMPLETION'):
        result += [p for p in ROOT.glob('common-runs/'+lane+'/*/*/*/result.json')if p.parents[2].name in triggers]
    return result,triggers


def sweep():
    root=ROOT/'control-CI-extension-v1'
    spec={'version':1,'trigger':'Persisted primary high-budget r32 capacity event, never which method wins',
        'cases':'Every triggered primary case','controls':'ALL completed S5 settings and TN independent final seeds of that case',
        'r_cap':64,'D_cap':4096,'new_model_shots':0,'source_results_mutated':False,
        'primary_or_S5_scientific_freeze_changed':False,'CI_code_sha256':sha256(Path(__file__).with_name('ci.py').read_bytes())}
    specpath=root/'spec.json'
    if specpath.exists()and json.loads(specpath.read_text())!=spec:raise RuntimeError('CI sensitivity settings changed')
    save_json(specpath,spec)
    save_json(root/'provenance-addendum.json',{'control_coverage_extension_documented_after_primary_started':True,
        'not_an_independently_preregistered_confirmatory_endpoint':True,'original_spec_not_changed':True})
    paths,triggers=sources();done=0
    for source in paths:
        r=json.loads(source.read_text());case=r['case_id'];lane=r['lane'];method=r['method_id'];B=r['budget'];seed=r['seed']
        path=root/lane/case/method/(str(B)+'-'+str(seed))/'CI-r64.json'
        if path.exists():
            if json.loads(path.read_text())['source_result_sha256']!=sha256(source.read_bytes()):raise RuntimeError('Control source changed')
            done+=1;continue
        m,h,g=load_physical(case);countsfile=source.parent/'final-counts.npz'
        with np.load(countsfile,allow_pickle=False)as a:ids,counts=a['blocked_ids'].copy(),a['counts'].copy()
        tick,cpu=perf_counter(),process_time();ci=solve(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),ids,counts,64)
        old=next(c for c in r['final_CI']if c['r_cap']==32)
        if old['converged']and ci['converged']and ci['energy_Ha']>old['energy_Ha']+1e-8:raise RuntimeError('Nonmonotone nested control CI')
        save_json(path,{'status':'COMPLETED','lane':lane,'case_id':case,'method_id':method,'budget':B,'seed':seed,
            'source_result':str(source.relative_to(ROOT)),'source_result_sha256':sha256(source.read_bytes()),
            'source_counts_sha256':sha256(countsfile.read_bytes()),'primary_capacity_trigger':str((ROOT/'CI-extension-v1'/case/'trigger.json').relative_to(ROOT)),
            'CI':ci,'new_model_shots':0,'actual_QPU_shots':0,'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
            'all_settings_not_only_favorable_controls':True})
        done+=1;print('CONTROL_CI4096',case,method,seed,flush=True)
    save_json(root/'status.json',{'triggered_cases':sorted(triggers),'currently_due_control_readouts':len(paths),
        'completed_control_readouts':done,'new_model_shots':0,'whole_experiment_complete':False})


def settled():
    db=sqlite3.connect(ROOT/'run-queue.sqlite')
    active=sum(db.execute('SELECT COUNT(*) FROM '+table+' WHERE status IN("NOT_STARTED","RUNNING","WAITING_REMOTE_EXECUTOR")').fetchone()[0]
        for table in('planned_cells','sensitivity_cells'));db.close()
    tn=all((ROOT/'TN-controls-v1'/case/'final-readout/completed.json').exists()
        or(ROOT/'TN-controls-v1'/case/'final-readout/terminal-attention.json').exists()
        or(ROOT/'TN-controls-v2'/case/'no-final-state.json').exists()
        for case in('NAPH20-P1','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'))
    high=not(ROOT/'TN-high-final-v1/spec.json').exists()or all((ROOT/'TN-high-final-v1'/case/'completed.json').exists()
        or(ROOT/'TN-high-final-v1'/case/'terminal-attention.json').exists()
        for case in('NAPH20-P1','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'))
    return active==0 and tn and high


def main(watch=False):
    with(ROOT/'control-CI-extension-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            sweep()
            if not watch or settled():return
            sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');main(p.parse_args().watch)
