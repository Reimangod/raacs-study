"""G5 whole N20 bank, unchanged G4 science, bounded batched verifier."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET
from .paths import ROOT
from .io import load,save,sha256,space
from .strong_projection import source_plans,binding as g4_binding,FREEZE as G4_FREEZE,RUN as G4_RUN,total_cpu
from .frozen_projection_screen import read_selected
from .strong_bank_prepare import prepare,verify_batch

PROTOCOL=ROOT/'protocol-G5-WHOLE-N20.json'
FREEZE=ROOT/'freeze-G5-WHOLE-N20.json'
RUN=ROOT/'runs/G5-WHOLE-N20-v1'
QUALIFIER=ROOT/'qualification/G5-batched-prefix-v1/result.json'


def binding():
    g4=load(G4_FREEZE)
    if g4_binding()!=g4: raise RuntimeError('G4 inputs/kernels changed')
    q=load(QUALIFIER)
    if q['members']!=64 or q['actual_circuit_checks']!=192 or q['status']!='ALL64_G4_PREPARATIONS_AND192_ACTUAL_CIRCUITS_BATCH_PARITY_PASSED':
        raise RuntimeError('All G4 batched parity required')
    for name,sha in q['implementation_SHA256'].items():
        if sha256(Path(name).read_bytes())!=sha: raise RuntimeError('Qualified batch implementation changed')
    xml=ROOT/'qualification-projection-batch-v1-tests.xml'
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    if sum(int(x.get('tests',0)) for x in suites)!=3 or any(int(x.get(k,0)) for x in suites for k in ['failures','errors','skipped']):
        raise RuntimeError('Batch namespace/failure qualification required')
    sources=source_plans(load(PROTOCOL)); c5=load(ROOT/'manifest-C5-CONTROLS.json')
    for source in sources:
        if source['status']!='AVAILABLE' or len(source['members'])!=783: raise RuntimeError('All4 complete N20 banks required')
        frozen=next(x for x in c5['cells'] if x['case']==source['case'] and x['method']=='D_SQDRIFT_N20')
        counts=Path(source['source_result']).parent/'final-counts.npz'
        if sha256(counts.read_bytes())!=frozen['source_counts_SHA256']: raise RuntimeError('Old observed counts changed')
        source.update(source_counts_path=str(counts),source_counts_SHA256=frozen['source_counts_SHA256'])
    return dict(protocol_SHA256=sha256(PROTOCOL.read_bytes()),G4_freeze_SHA256=sha256(G4_FREEZE.read_bytes()),
        G4_database_SHA256=load(G4_RUN/'terminal.json')['database_SHA256'],
        qualifier_SHA256=sha256(QUALIFIER.read_bytes()),tests_SHA256=sha256(xml.read_bytes()),
        implementation_SHA256={str(p):sha256(p.read_bytes()) for p in [Path(__file__),
            ROOT/'frontierloop/strong_bank_prepare.py',ROOT/'frontierloop/mps_batch_worker.py',
            ROOT/'test_projection_batch.py']},sources=sources,new_model_shots=0)


def insert(db,row,certificate,circuits):
    db.execute('INSERT INTO members VALUES(?,?,?,?,?,?,?,?)',(row['case'],row['index'],row['member_id'],
        json.dumps(row,sort_keys=True,allow_nan=False),certificate,circuits['NATIVE'],circuits['RUSTIQ'],circuits['PROJECTED']))


def run():
    frozen=load(FREEZE)
    if binding()!=frozen: raise RuntimeError('G5 preexecution freeze drift')
    space(); spec=load(PROTOCOL)
    # Exclusive directory creation gives this one execution durable ownership.
    RUN.mkdir(exist_ok=False)
    cpu,wall=total_cpu(),perf_counter()
    save(RUN/'started.json',dict(freeze_SHA256=sha256(FREEZE.read_bytes()),new_model_shots=0))
    db=sqlite3.connect(RUN/'compiled-members.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,member_id TEXT,record TEXT,certificate_gz BLOB,native_gz BLOB,rustiq_gz BLOB,projected_gz BLOB,PRIMARY KEY(case_id,idx))')
    db.commit()
    oldpath=G4_RUN/'compiled-members.sqlite'
    if sha256(oldpath.read_bytes())!=frozen['G4_database_SHA256']: raise RuntimeError('G4 evidence changed')
    old=sqlite3.connect('file:'+str(oldpath)+'?mode=ro',uri=True)
    statuses=Counter(); reused,new_checks,new_preparations=0,0,0; cases=[]

    def budget():
        space()
        if total_cpu()-cpu>spec['max_CPU_seconds'] or perf_counter()-wall>spec['max_wall_seconds']:
            raise TimeoutError('G5 frozen global work cap')
        if sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())>spec['max_new_directory_bytes']:
            raise OSError('G5 compact storage cap')

    def verify_and_commit(items):
        nonlocal new_checks
        if not items: return
        budget()
        try:
            result=verify_batch(items,spec['max_batch_verifier_wall_seconds'])
            values=result['results']
        except Exception as exc:
            values={x['request']['key']:dict(status='FAILED_ENGINEERING_PRESERVED',error=repr(exc),
                traceback=traceback.format_exc()) for x in items}
        failed=False
        for item in items:
            row=item['row']; value=values[item['request']['key']]
            if value['status']=='PASSED':
                row.update(status='CERTIFIED_RESOURCE_POINT',independent_check=value['check'],
                    independent_evaluation_origin='NEW_G5_BATCH_MPS_CHECK',G5_prefix_evidence_reused=False)
                new_checks+=3
            else:
                row.update(status='FAILED_ENGINEERING_PRESERVED',failure=value); failed=True
            db.execute('UPDATE members SET record=? WHERE case_id=? AND idx=?',
                (json.dumps(row,sort_keys=True,allow_nan=False),row['case'],row['index']))
            statuses[row['status']]+=1
        db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        save(RUN/'progress.json',dict(statuses=dict(statuses),reused_G4_members=reused,
            new_MPS_circuit_checks=new_checks,CPU_seconds=total_cpu()-cpu,wall_seconds=perf_counter()-wall))
        if failed: raise RuntimeError('Preserved G5 batch engineering failure; no unchecked expansion')

    try:
        for source in frozen['sources']:
            budget(); case=source['case']; meta=dict(load(source['meta_path']),case_id=case)
            entries=[e for m in source['members'][16:] for e in m['artifacts'].values()]
            blobs=read_selected(entries); batch=[]
            for plan in source['members']:
                budget()
                if plan['index']<16:
                    prior=old.execute('SELECT record,certificate_gz,native_gz,rustiq_gz,projected_gz FROM members WHERE case_id=? AND idx=?',
                        (case,plan['index'])).fetchone()
                    r=json.loads(prior[0])
                    if r['status']!='CERTIFIED_RESOURCE_POINT' or r['member_id']!=plan['member_id'] or r['spec']!=plan['spec'] or r['source_artifacts']!=plan['artifacts']:
                        raise RuntimeError('Exact G4 prefix evidence binding failed')
                    r.update(G5_prefix_evidence_reused=True,independent_evaluation_origin='G4_DATABASE:'+frozen['G4_database_SHA256'],
                        G5_new_MPS_circuit_checks=0)
                    insert(db,r,prior[1],dict(zip(['NATIVE','RUSTIQ','PROJECTED'],prior[2:])))
                    db.commit(); reused+=1; statuses[r['status']]+=1
                    continue
                item=prepare(meta,plan,blobs); new_preparations+=1
                insert(db,item['row'],item['certificate'],item['qpy_blobs']); db.commit(); batch.append(item)
                if len(batch)==spec['batch_members']:
                    verify_and_commit(batch); batch=[]
                    if (plan['index']+1)%64==0: print(case,plan['index']+1,dict(statuses),flush=True)
            verify_and_commit(batch)
            cases.append(dict(case=case,members=783,shot_weight=sum(m['historical_member_shots'] for m in source['members'])))
            print(case,'WHOLE783_TERMINAL',flush=True)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]: raise RuntimeError('G5 DB integrity')
        if statuses!={'CERTIFIED_RESOURCE_POINT':3132} or reused!=64 or new_checks!=9204:
            raise RuntimeError('G5 full coverage failed')
    except Exception as exc:
        save(RUN/'incomplete.json',dict(status='ENGINEERING_PARTIAL_PRESERVED',error=repr(exc),
            traceback=traceback.format_exc(),statuses=dict(statuses),CPU_seconds=total_cpu()-cpu,
            wall_seconds=perf_counter()-wall,new_model_shots=0,whole_bank_energy_join=False))
        raise
    finally:
        old.close(); db.close()
    save(RUN/'terminal.json',dict(status='ALL4_WHOLE_N20_BANKS_CERTIFIED_RESOURCE_TERMINAL',cases=cases,
        statuses=dict(statuses),reused_G4_members=reused,new_prepared_members=new_preparations,
        new_MPS_circuit_checks=new_checks,total_independent_circuit_coverage=9396,
        CPU_seconds_including_children=total_cpu()-cpu,wall_seconds=perf_counter()-wall,
        database_SHA256=sha256((RUN/'compiled-members.sqlite').read_bytes()),freeze_SHA256=sha256(FREEZE.read_bytes()),
        new_model_shots=0,new_QPU_shots=0,new_energy_evaluations=0,new_CI_evaluations=0,
        energy_reuse_requires_source_count_and_T2_context_audit=True,independent_confirmation=False,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:
        b=binding()
        if FREEZE.exists() and load(FREEZE)!=b: raise RuntimeError('Never replace G5 freeze')
        save(FREEZE,b); print('G5_FROZEN_3132_MEMBERS_64_PREFIX_REFERENCES')
    else: run()
