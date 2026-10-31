"""Engineering parity on ALL64 G4 points; no new molecular outcomes."""
import json
import sqlite3
from time import perf_counter
from .paths import ROOT
from .io import load,save,sha256,space
from .strong_projection import binding,FREEZE,RUN,total_cpu
from .frozen_projection_screen import read_selected
from .strong_bank_prepare import prepare,semantics,verify_batch

TARGET=ROOT/'qualification/G5-batched-prefix-v1/result.json'


def run():
    if TARGET.exists(): raise RuntimeError('Preserve completed/partial qualification')
    frozen=load(FREEZE)
    if binding()!=frozen: raise RuntimeError('G4 frozen kernels/inputs changed')
    terminal=load(RUN/'terminal.json')
    dbpath=RUN/'compiled-members.sqlite'
    if sha256(dbpath.read_bytes())!=terminal['database_SHA256']: raise RuntimeError('G4 DB changed')
    db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
    cpu,wall=total_cpu(),perf_counter(); rows=[]
    for source in frozen['sources']:
        space()
        blobs=read_selected([e for m in source['members'] for e in m['artifacts'].values()])
        meta=dict(load(source['meta_path']),case_id=source['case'])
        items=[]
        for plan in source['members']:
            old=db.execute('SELECT record,certificate_gz,native_gz,rustiq_gz,projected_gz FROM members WHERE case_id=? AND idx=?',
                (source['case'],plan['index'])).fetchone()
            previous=json.loads(old[0]); item=prepare(meta,plan,blobs); current=item['row']
            for field in ['problem_id','member_id','spec','source_artifacts','resources','policy','source_sparse_support',
                    'active_qubits','coefficient_slots_original','coefficient_slots_restricted','certificate_SHA256']:
                if current[field]!=previous[field]: raise RuntimeError('G4 preparation parity: '+field)
            for name,blob in zip(['NATIVE','RUSTIQ','PROJECTED'],old[2:]):
                if semantics(item['qpy_blobs'][name])!=semantics(blob):
                    raise RuntimeError('Ordered actual circuit semantics changed: '+name)
            items.append(item)
        result=verify_batch(items)
        for item in items:
            r=result['results'][item['request']['key']]
            if r['status']!='PASSED': raise RuntimeError('Batched verification failed: '+str(r))
            rows.append(dict(case=source['case'],index=item['row']['index'],member_id=item['row']['member_id'],
                preparation_parity=True,ordered_three_circuit_semantics_parity=True,check=r['check']))
        print(source['case'],'ALL16_G4_PARITIES_PASSED',flush=True)
    db.close()
    save(TARGET,dict(status='ALL64_G4_PREPARATIONS_AND192_ACTUAL_CIRCUITS_BATCH_PARITY_PASSED',
        members=len(rows),actual_circuit_checks=3*len(rows),rows=rows,
        G4_freeze_SHA256=sha256(FREEZE.read_bytes()),G4_database_SHA256=terminal['database_SHA256'],
        implementation_SHA256={str(p):sha256(p.read_bytes()) for p in [ROOT/'frontierloop/strong_bank_prepare.py',
            ROOT/'frontierloop/mps_batch_worker.py',ROOT/'frontierloop/qualify_projection_batch.py']},
        CPU_seconds_including_children=total_cpu()-cpu,wall_seconds=perf_counter()-wall,
        new_model_shots=0,new_QPU_shots=0,new_energy_evaluations=0,new_CI_evaluations=0,
        scope='ENGINEERING_PARITY_NOT_INDEPENDENT_QUANTUM_CONFIRMATION'))


if __name__=='__main__': run()
