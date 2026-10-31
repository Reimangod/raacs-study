"""All-method occupied/virtual-block Boys sensitivity required by the plan.

Forty additional cells, not replacements for any of the canonical480 or S5125.
Frozen methods, 100k total shots, ten new seeds, same physical HF reference.
"""
import json,sqlite3,shutil
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from .orbital_basis import CASE,SOURCE,prepare,evaluation_crosswalk
from .autonomous_sensitivity import child
from .common_execution import load_physical,run_identity
from .ci import solve
from .io import ROOT,save_json,digest,sha256

SEEDS=(41001,41011,41017,41023,41039,41047,41051,41059,41077,41081)
METHODS=('A_RAACS_CURRENT','B_LUCJ_NAIVE_L1','C_LUCJ_COMPRESSED_L1','D_SQDRIFT_N20')
LANE='S_ORBITAL_BASIS';B=100000


def spec():
    source=json.loads((ROOT/'data/physical'/SOURCE/'problem.json').read_text())
    names=('orbital_basis.py','common_execution.py','sqdrift_execution.py','sqdrift_native_worker.py','ci.py','lucj.py')
    value={'case_id':CASE,'source_case':SOURCE,'source_problem_id':source['problem_id'],
        'basis_construction':'Boys separately in occupied5 and virtual5; no occupied/virtual mixing',
        'reference_state':'Identical physical HF up to global phase; occupied subspace checked',
        'source_molecular_orbitals_sha256':sha256((ROOT/'data/physical'/SOURCE/'orbital-basis.npz').read_bytes()),
        'methods':list(METHODS),'total_model_shots':B,'seeds':list(SEEDS),'planned_cells':40,
        'CI_r_caps':[8,16,32,64],'r64_reason':'Same naphthalene physical problem already triggered the predeclared r32 capacity diagnostic in canonical high-budget runs; applied to all new stratum methods/seeds',
        'original_P1_full_Boys_LO_reproduction_claimed':False,'primary480_changed':False,'new_independent_molecule_claimed':False,
        'main_outcomes_not_used_for_method_tuning':True,'code_sha256':{n:sha256(Path(__file__).with_name(n).read_bytes())for n in names}}
    path=ROOT/'basis-sensitivity-v1/spec.json'
    if path.exists()and json.loads(path.read_text())!=value:raise RuntimeError('Basis sensitivity scientific settings changed')
    save_json(path,value);return value


def capacity_readout(path,result):
    out=ROOT/'basis-sensitivity-v1/readout-r64'/result['method_id']/(str(result['seed'])+'.json')
    if out.exists():return
    m,h,g=load_physical(CASE);filename=path.parent/'final-counts.npz'
    with np.load(filename,allow_pickle=False)as a:ids,counts=a['blocked_ids'],a['counts']
    tick,cpu=perf_counter(),process_time();ci=solve(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),ids,counts,64)
    save_json(out,{'status':'COMPLETED','CI':ci,'source_result_sha256':sha256(path.read_bytes()),
        'source_counts_sha256':sha256(filename.read_bytes()),'new_model_shots':0,'actual_QPU_shots':0,
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick})


def main():
    import fcntl
    with(ROOT/'basis-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);spec();prepare();evaluation_crosswalk()
        root=ROOT/'basis-sensitivity-v1';log=root/'controller'
        init=ROOT/'qualification'/(CASE+'-author-compressed-fresh')/'result.json'
        if not init.exists():
            code=child('paperstudy.author_compressed',['--case',CASE,'--fresh','--seconds','1200'],log/'compressed-initialization.log',1260)
            if code or not init.exists()or json.loads(init.read_text())['status']!='AUTHOR_INITIALIZER_RETURNED':
                save_json(root/'initialization-attention.json',{'exit_code':code,'no_substitute_initializer':True});return
        db=sqlite3.connect(ROOT/'run-queue.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS basis_cells(id TEXT PRIMARY KEY,case_id TEXT,method_id TEXT,budget INTEGER,seed INTEGER,status TEXT,reason TEXT,result_path TEXT)')
        for seed in SEEDS:
            for method in METHODS:db.execute('INSERT OR IGNORE INTO basis_cells VALUES(?,?,?,?,?,?,?,?)',
                (run_identity(CASE,method,B,seed,LANE),CASE,method,B,seed,'NOT_STARTED',None,None))
        db.commit()
        for seed in SEEDS:
            for method in METHODS:
                key=run_identity(CASE,method,B,seed,LANE);old=db.execute('SELECT status,result_path FROM basis_cells WHERE id=?',(key,)).fetchone()
                path=ROOT/'common-runs'/LANE/CASE/method/(str(B)+'-'+str(seed))/'result.json'
                if old[0]=='COMPLETED':capacity_readout(path,json.loads(path.read_text()));continue
                if old[0]in('FAILED_ENGINEERING_PRESERVED','RESOURCE_CAPPED'):continue
                if shutil.disk_usage(ROOT).free<3*2**30:
                    save_json(root/'resource-attention.json',{'free_bytes':shutil.disk_usage(ROOT).free,'remaining_unstarted_preserved':True});return
                db.execute("UPDATE basis_cells SET status='RUNNING' WHERE id=?",(key,));db.commit()
                code=child('paperstudy.cell_worker',['--case',CASE,'--method',method,'--budget',str(B),'--seed',str(seed),'--lane',LANE],log/(key+'.log'),1800)
                status='COMPLETED'if code==0 and path.exists()else'FAILED_ENGINEERING_PRESERVED'
                reason=None if status=='COMPLETED'else'Bounded method worker failed; raw work/ledger preserved'
                db.execute('UPDATE basis_cells SET status=?,reason=?,result_path=? WHERE id=?',(status,reason,str(path.relative_to(ROOT))if path.exists()else None,key));db.commit()
                if status=='COMPLETED':capacity_readout(path,json.loads(path.read_text()))
                counts=dict(db.execute('SELECT status,COUNT(*) FROM basis_cells GROUP BY status').fetchall())
                save_json(root/'status.json',{'planned_cells':40,'status_counts':counts,'canonical_primary_unchanged':True,'whole_experiment_complete':False})
                print('BLOCK_BOYS',method,seed,status,counts,flush=True)
        db.close()


if __name__=='__main__':main()
