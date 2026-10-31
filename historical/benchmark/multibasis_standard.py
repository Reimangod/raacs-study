"""Accounting successor: inherit qualified quick acquisitions, explicit unknowns.

Numeric constructor/selection/rank/caps remain the frozen v1 implementation.
Only metadata interpretation and linked-acquisition import differ. This adapter
is hash-bound as executed code; it never changes the old qualification dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from .common import ROOT,file_digest,save_json
from . import multibasis_study as study

ORIGINAL_CONFIG=study.configuration
ORIGINAL_RESOURCES=study.resources
ORIGINAL_SAMPLER=study.CircuitCountSampler


def unknown_parameter_resources(qc,n_parameters,n_operators):
    out=ORIGINAL_RESOURCES(qc,n_parameters,n_operators)
    out['parameter_count']=None;out['selected_operator_count']=None
    out['parameter_count_scope']='NOT_RECONSTRUCTED_COMBINED_BINDING; use original source and orbital-parameter metadata'
    return out


class ExplicitCostSampler(ORIGINAL_SAMPLER):
    def draw(self,*args,**kwargs):
        ids,counts,meta=super().draw(*args,**kwargs)
        for key in ('sampling_resources','preparation_resources'):
            meta[key]['parameter_count']=None;meta[key]['selected_operator_count']=None
            meta[key]['parameter_count_scope']='NOT_RECONSTRUCTED_COMBINED_BINDING'
        meta['new_quantum_acquisition_in_this_run']=True
        return ids,counts,meta


def settings():
    cfg=ORIGINAL_CONFIG('standard');quick=ROOT/'results/multibasis-NOCI-quick-v1'
    validation=json.loads((quick/'validation.json').read_text())
    if validation['status']!='ALL_QUICK_SPACES_INDEPENDENTLY_VALIDATED':raise RuntimeError('Independent quick audit')
    if file_digest(quick/'events.sqlite')!=validation['database_sha256']:raise RuntimeError('Sealed quick database changed')
    cfg.update(schema='multiple-orbital-measurement-NOCI-v2-accounting',
        numeric_namespace_compatibility=study.NAMESPACE,
        inherited_quick={'path':str(quick),'database_sha256':validation['database_sha256'],
                         'validation_sha256':file_digest(quick/'validation.json'),'quantum_shots':validation['unique_committed_new_model_shots'],
                         'classical_uniform_samples':validation['unique_classical_uniform_samples']},
        accounting_spec_sha256=file_digest(ROOT/'study-v1/MULTIBASIS_ACCOUNTING_V2.md'))
    cfg['implementation']['multibasis_standard.py']=file_digest(__file__)
    return cfg


def import_quick(root,cfg):
    marker=root/'inherited-quick-acquisitions-v1.json'
    if marker.exists():
        if json.loads(marker.read_text())['source']!=cfg['inherited_quick']:raise RuntimeError('Inherited source drift')
        return
    db=study.connect(root)
    if db.execute('SELECT count(*) FROM acquisitions').fetchone()[0]:raise RuntimeError('Do not import over an existing study')
    quick=Path(cfg['inherited_quick']['path'])
    original=sqlite3.connect(f"file:{quick/'events.sqlite'}?mode=ro",uri=True)
    for case,blob in original.execute('SELECT * FROM bases').fetchall():
        db.execute('INSERT INTO bases VALUES(?,?)',(case,blob))
    for case,method,b,qpy,blob in original.execute('SELECT * FROM circuits').fetchall():
        payload=study.unpack(blob)
        payload['resources']['parameter_count']=None;payload['resources']['selected_operator_count']=None
        payload['inherited_quick_circuit']=cfg['inherited_quick']
        db.execute('INSERT INTO circuits VALUES(?,?,?,?,?)',(case,method,b,qpy,study.pack(payload)))
    acquisitions=original.execute('SELECT * FROM acquisitions').fetchall()
    if any(x[-1]!='COMPLETED' for x in acquisitions):raise RuntimeError('Partial quick acquisition')
    db.executemany('INSERT INTO acquisitions VALUES(?,?,?,?,?,?,?)',acquisitions)
    for did,aid,b,status,blob in original.execute('SELECT * FROM draws').fetchall():
        payload=study.unpack(blob);payload['inherited_quick_raw_draw_id']=did
        payload['inherited_quick_database_sha256']=cfg['inherited_quick']['database_sha256']
        payload['metadata']['new_quantum_acquisition_in_this_run']=False
        for key in ('sampling_resources','preparation_resources'):
            if key in payload['metadata']:
                payload['metadata'][key]['parameter_count']=None;payload['metadata'][key]['selected_operator_count']=None
        db.execute('INSERT INTO draws VALUES(?,?,?,?,?)',(did,aid,b,status,study.pack(payload)))
    db.commit();db.close();original.close()
    save_json(marker,{'status':'QUALIFIED_COUNTS_LINKED_NOT_NEW_ACQUISITIONS','source':cfg['inherited_quick'],
                      'acquisitions':len(acquisitions),'tasks_or_R_readouts_copied':0})


def run(root):
    root=Path(root).resolve();cfg=settings()
    source=Path(cfg['source'])
    if root==source or source in root.parents or root in source.parents:raise ValueError('Protect source')
    root.mkdir(parents=True,exist_ok=True)
    # Match the primary runner's one-owner protection during import. Release it
    # before entering run(), which acquires the same lock for its whole lifetime.
    import fcntl
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=cfg:raise RuntimeError('Frozen standard successor drift')
        save_json(root/'manifest.json',cfg);import_quick(root,cfg)
    study.configuration=lambda mode:cfg
    study.resources=unknown_parameter_resources;study.CircuitCountSampler=ExplicitCostSampler
    try:return study.run(root,'standard')
    finally:
        study.configuration=ORIGINAL_CONFIG;study.resources=ORIGINAL_RESOURCES;study.CircuitCountSampler=ORIGINAL_SAMPLER


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/multibasis-NOCI-standard-v2'))
    print(json.dumps(run(p.parse_args().output)))
