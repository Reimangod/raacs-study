"""Outcome-blind deterministic selection of existing comparator counts."""
import csv
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,sha256
from raacsnext.run import physical


def select_row(rows,case,method,budget):
    eligible=[x for x in rows if x['case_id']==case and x['method_id']==method
        and int(x['budget'])==budget and int(x['r_cap'])==32]
    if not eligible:
        return None
    return min(eligible,key=lambda x:(int(x['seed']),x['lane'],x['result_path']))


def validate_record(record,meta,budget):
    if record['problem_id']!=meta['problem_id'] or record['case_id']!=meta['case_id']:
        raise ValueError('Different problem/orbital identity; source substitution prohibited')
    if record['budget']!=budget or record['status']!='COMPLETED':
        raise ValueError('Source budget or terminal state mismatch')


def build_manifest():
    spec=load(ROOT/'protocol-C5-CONTROLS.json')
    index=OLD/'analysis-v1/all-observed-readouts.csv'
    rows=list(csv.DictReader(index.open(encoding='utf-8-sig')))
    cells=[]
    for case in spec['cases']:
        meta,_,_=physical(case)
        meta=dict(meta,case_id=case)
        for method in spec['methods']:
            row=select_row(rows,case,method,spec['source_total_model_shot_budget'])
            cell={'case':case,'method':method,'expected_problem_id':meta['problem_id']}
            if row is None:
                cells.append(dict(cell,status='MISSING_SAME_PROBLEM_SOURCE'))
                continue
            path=OLD/row['result_path'];record=load(path)
            validate_record(record,meta,spec['source_total_model_shot_budget'])
            counts_path=path.parent/'final-counts.npz'
            with np.load(counts_path,allow_pickle=False) as z:
                n=int(z['counts'].sum())
                assert len(z['blocked_ids'])==len(z['counts']) and np.all(z['counts']>0)
            if n!=record['independent_final_model_shots']:
                raise ValueError('Source final counts do not reconcile')
            cell.update(status='AVAILABLE',source_result=str(path),source_seed=int(row['seed']),
                source_result_SHA256=sha256(path.read_bytes()),source_counts_SHA256=sha256(counts_path.read_bytes()),
                source_request_SHA256=sha256((path.parent/'request.json').read_bytes())
                    if (path.parent/'request.json').exists() else None,
                source_standalone_request_available=(path.parent/'request.json').exists(),
                source_total_budget=record['budget'],source_final_model_shots=n,
                sampler_resources={'cnot_count':float(row['CNOT']),'cnot_depth':float(row['CNOT_depth']),
                    'total_unitary_depth':float(row['total_depth'])},
                sampler_coefficient_slots_mean=float(row['coefficient_slots_or_member_mean']),
                Rustiq_resources={k:float(row[k]) if row.get(k) else None for k in
                    ('Rustiq_cnot_count','Rustiq_cnot_depth','Rustiq_total_unitary_depth')},
                member_count=record.get('member_count',1),
                source_setup_cost_not_included_in_additional_OO_cost=True,
                original_paper_driver_reproduced=record.get('paper_author_driver_reproduced',False))
            cells.append(cell)
    return {'protocol_SHA256':sha256((ROOT/'protocol-C5-CONTROLS.json').read_bytes()),
        'index_SHA256':sha256(index.read_bytes()),'cells':cells,
        'available_sources':sum(x['status']=='AVAILABLE' for x in cells),
        'missing_sources':sum(x['status']!='AVAILABLE' for x in cells),
        'new_model_shots':0,'selection_uses_energy':False}


def load_source(cell):
    if cell['status']!='AVAILABLE':
        raise ValueError('Missing source')
    path=Path(cell['source_result']);record=load(path)
    if sha256(path.read_bytes())!=cell['source_result_SHA256']:
        raise RuntimeError('Control source result changed')
    counts_path=path.parent/'final-counts.npz'
    if sha256(counts_path.read_bytes())!=cell['source_counts_SHA256']:
        raise RuntimeError('Control source counts changed')
    with np.load(counts_path,allow_pickle=False) as z:
        ids,counts=z['blocked_ids'].copy(),z['counts'].copy()
    if int(counts.sum())!=cell['source_final_model_shots'] or record['problem_id']!=cell['expected_problem_id']:
        raise RuntimeError('Control source binding failed')
    return ids,counts
