"""Independent read-only reconciliation of the current synthesis and figure index.

No producer aggregation helper is imported. This checks all bound input/output
files, all planned current rows, the actual F ledger and independent CSV
arithmetic. It does not repeat molecular experiments or every CI eigensolve.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(test,reason):
    if not test:raise ValueError(reason)


def json_file(path):return json.loads(Path(path).read_text())
def csv_file(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def close(a,b,tol=1e-10):
    require(math.isfinite(float(a)) and math.isfinite(float(b)),'Nonfinite aggregation')
    require(abs(float(a)-float(b))<=tol*max(1,abs(float(a)),abs(float(b))),'Aggregation mismatch')


def check_hci(rows,summary):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['method'],int(row['R_cap']),float(row['epsilon1_Ha']))].append(row)
    require(len(summary)==len(groups),'Dropped HCI group')
    for row in summary:
        g=groups.pop((row['method'],int(row['R_cap']),float(row['epsilon1_Ha'])))
        available=[v for v in g if v['status']=='PAIRED_DIMENSION_DIAGNOSTIC']
        nontrivial=[v for v in available if int(v['matched_d'])>1]
        values=sorted(float(v['sampler_minus_truncated_HCI_mHa']) for v in nontrivial)
        for key,expected in [('planned',len(g)),('available',len(available)),('nontrivial',len(values)),
                             ('trivial_HF',len(available)-len(values)),('missing',len(g)-len(available))]:
            require(int(row[key])==expected,'HCI availability aggregation mismatch')
        if values:
            n=len(values);median=(values[(n-1)//2]+values[n//2])/2
            close(row['median_sampler_minus_truncated_HCI_mHa'],median)
        else:require(row['median_sampler_minus_truncated_HCI_mHa']=='','Missing HCI median became a value')
        for key,expected in [('negative_favors_sampler',sum(v < -1e-8 for v in values)),
                             ('within_numerical_1e_8_mHa',sum(abs(v)<=1e-8 for v in values)),
                             ('positive_favors_truncated_HCI',sum(v > 1e-8 for v in values))]:
            require(int(row[key])==expected,'HCI sign summary mismatch')
        require(row['ground_state_error']=='' and row['work_matched']=='False','False HCI cost/ground claim')
    require(not groups,'Unreported HCI groups')


def check_cost(rows,summary):
    require(len(summary)==len({r['method'] for r in rows}),'Dropped workflow method')
    for s in summary:
        g=[r for r in rows if r['method']==s['method']]
        good=[r for r in g if r['status']=='COMPLETED']
        mol=defaultdict(list)
        for row in good:mol[row['molecule']].append(row)
        require(int(s['planned'])==len(g) and int(s['completed'])==len(good) and
                int(s['missing_or_failed'])==len(g)-len(good),'Workflow missingness hidden')
        require(s['historical_CEO_training_shot_cost']=='','Unknown historical quantum cost became known')
        for field,col in [('total_workflow_wall_seconds','equal_molecule_observed_total_wall_seconds'),
                          ('all_draw_CNOT_times_shots_proxy','equal_molecule_all_paid_CNOT_times_model_samples')]:
            means=[sum(float(r[field]) for r in group)/len(group) for group in mol.values()]
            close(s[col],sum(means)/len(means))


def validate_links(path):
    checked=0
    for value in re.findall(r'\]\(([^)]+)\)',Path(path).read_text()):
        if '://' in value or value.startswith('#'):continue
        dest=(Path(path).parent/value.split('#')[0]).resolve()
        require(dest.exists(),'Broken report link: '+str(dest));checked+=1
    return checked


def review(snapshot,figures):
    snapshot=Path(snapshot).resolve();figures=Path(figures).resolve()
    record=json_file(snapshot/'source-evidence.json');index=json_file(figures/'index.json')
    require(record['schema']=='current-source-scalable-sampler-synthesis-v3','Unknown current schema')
    require(index['schema']=='current18figure-semantic-source-index-v1','Unknown figure schema')
    inputs={}
    for path,expected in record['source_sha256'].items():
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        require(sha(p)==expected,'Current source drift: '+path);inputs[str(p)]=expected
    for path,expected in record['output_sha256'].items():
        require(sha(snapshot/path)==expected,'Current output drift: '+path)
    for collection in ('input_sha256','figure_sha256'):
        for path,expected in index[collection].items():
            require(sha(ROOT/path)==expected,'Figure/source drift: '+path);inputs[str(ROOT/path)]=expected
    questions=json_file(snapshot/'questions.json');metrics=json_file(snapshot/'primary-metrics.json')
    require([q['number'] for q in questions]==list(range(1,16)),'15-question coverage')
    require([m['number'] for m in metrics]==list(range(1,11)),'10-metric coverage')
    objective=next(Path(p) for p in record['source_sha256'] if p.endswith('/goal-objective.md'))
    body=objective.read_text().split('# 11. 最終レポートで答える研究上の質問',1)[1].split('# 12.',1)[0]
    titles={int(i):t for i,t in re.findall(r'^(\d+)\. (.+)$',body,re.M)}
    require(all(titles[q['number']]==q['title'] for q in questions),'Objective title drift')
    e=json_file(ROOT/record['sources']['E']);f=json_file(ROOT/record['sources']['F'])
    cost=json_file(ROOT/record['sources']['cost']);h=json_file(ROOT/record['sources']['HCI_review'])
    erows=e['rows'];require(len(erows)==e['planned']==234,'E planned coverage')
    require(dict(Counter(r['status'] for r in erows))==record['runtime']['E']['statuses'],'E current status mismatch')
    with sqlite3.connect(f'file:{ROOT}/results/gate-noise-full-standard-v1/events.sqlite?mode=ro',uri=True) as db:
        fc=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
        chunks,shots=db.execute('SELECT count(*),sum(shots) FROM chunks').fetchone()
    require(not any(s in fc for s in ('RUNNING','STARTED','PENDING','QUEUED','NOT_STARTED')),'F has nonterminal work')
    require(sum(fc.values())==3600 and fc==record['runtime']['F']['statuses'],'F terminal coverage')
    require(shots==f['known_committed_new_model_shots']==record['runtime']['F']['known_model_samples'] and
            chunks==record['runtime']['F']['committed_chunks'],'F work count mismatch')
    require(len(cost['rows'])==cost['terminal']==cost['planned']==90 and not cost['unexpected_failures'],'Cost coverage')
    check_cost(cost['rows'],csv_file(snapshot/'workflow-method-summary.csv'))
    require(csv_file(snapshot/'all-planned-workflows.csv')==csv_file(ROOT/'results/total-workflow-standard-v1/report-v1/all-planned-workflows.csv'),
            'Planned workflow CSV changed')
    paired=csv_file(snapshot/'HCI-all-planned-same-d.csv')
    require(paired==csv_file(ROOT/'results/HCI-paired-diagnostic-standard-v1-review-v2/all-planned-pairs.csv'),
            'HCI planned CSV changed')
    require(len(paired)==h['planned_pairs']==1872,'HCI pair coverage')
    check_hci(paired,csv_file(snapshot/'HCI-conditional-summary.csv'))
    direct=csv_file(snapshot/'direct-configuration-comparisons.csv')
    require(len(direct)==9 and sum(r['identical_selected_space']=='True' for r in direct)==4,'Direct diagnostic claim mismatch')
    require(all(r['work_matched']=='False' and r['ground_state_error_Ha']=='' for r in direct),'Direct scope overclaim')
    require([r['figure'] for r in index['rows']]==list(range(1,19)),'18-figure coverage')
    for row in index['rows']:
        require(row['source_audits_reused'] and not row['all_export_files_visually_reviewed'] and
                not row['all_plotted_values_recomputed_by_this_index'],'Figure index overstates review')
        for ext in ('png','svg','pdf'):
            path=str(Path(row['representative']).with_suffix('.'+ext))
            require(path in index['figure_sha256'],'Representative format not bound')
    require(not record['whole_goal_complete'] and not index['whole_goal_complete'],'Premature overall closure')
    require(record['new_experiments']==record['new_solver_calls']==record['hardware_shots']==0,'Spurious new work claim')
    links=validate_links(snapshot/'REPORT_JA.md')+validate_links(figures/'INDEX.md')
    result={'status':'CURRENT_COUNTS_AGGREGATIONS_LINKS_AND_18_SCOPED_FIGURE_BINDINGS_REVIEWED',
       'source_snapshot_sha256':sha(snapshot/'source-evidence.json'),'figure_index_sha256':sha(figures/'index.json'),
       'independently_checked_source_files':len(inputs),'current_output_files':len(record['output_sha256']),
       'indexed_figure_files':len(index['figure_sha256']),'checked_links':links,
       'E_cells':234,'F_cells':3600,'workflow_cells':90,'HCI_planned_pairs':1872,
       'HCI_summary_groups':len(csv_file(snapshot/'HCI-conditional-summary.csv')),
       'questions':15,'metrics':10,'figure_types':18,
       'new_experiments':0,'new_eigensolver_calls':0,'hardware_shots':0,
       'reviewer_sha256':sha(__file__),'whole_goal_complete':False,
       'scope':'Independent aggregation/source/link reconciliation; not a new numerical, physical or full-objective completion certificate'}
    out=snapshot/'current-review-v1';out.mkdir(exist_ok=True)
    path=out/'validation.json';body=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if path.exists():require(path.read_text()==body,'Existing review differs; use successor')
    else:path.write_text(body)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--figures',required=True)
    a=p.parse_args();print(json.dumps(review(a.snapshot,a.figures),sort_keys=True))


if __name__=='__main__':main()
