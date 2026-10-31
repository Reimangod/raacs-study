"""Small stdlib-only final inventory; does not recompute quantum/CI outcomes."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    t3=ROOT/'runs/T3-READOUT-SEEDS-v1';tail=ROOT/'runs/T3-FREQUENCY-TAIL-v1'
    frozen=load(ROOT/'freeze-T3-READOUT-SEEDS.json');terminal=load(t3/'terminal.json')
    summary=load(ROOT/'analysis/T3-readout-seed-summary.json');tp=load(tail/'protocol-freeze.json');tt=load(tail/'terminal.json')
    assert sha(t3/'evaluations.sqlite')==terminal['catalog_SHA256']==tp['T3_catalog_SHA256']
    assert sha(t3/'terminal.json')==tp['T3_terminal_SHA256']
    assert sha(ROOT/'freeze-T3-READOUT-SEEDS.json')==tp['T3_freeze_SHA256']
    assert sha(ROOT/'frontierloop/readout_frequency_tail.py')==tp['implementation_SHA256']
    with (ROOT/'analysis/T3-readout-seed-all.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    keys=[(r['case'],r['method'],int(r['seed']),r['arm']) for r in rows]
    assert len(keys)==len(set(keys))==144
    assert sum(r['status']=='COMPLETED' for r in rows)==summary['completed_endpoint_rows']==137
    assert len(terminal['states'])==48 and summary['paired_cells']==46
    assert sum(summary['Global_vs_frequency'].values())==46
    by_id={p['cell_id']:p for p in frozen['plans']}
    assert {p['cell_id'] for p in tp['plans']}=={r['cell_id'] for r in tt['rows']}
    for row in tt['rows']:
        p=by_id[row['cell_id']]
        assert row['arm']=='SAME_UNION_FREQUENCY' and not row['Global_or_native_screen_retried']
        if row['status']!='COMPLETED':continue
        root=load(tail/p['cell_id']/'root.json');strings=p['request']['same_union_frequency'];ne=p['context']['nelec']
        assert root['CI_strings']==strings and all(s.bit_count()==ne[0]==ne[1] for s in strings)
        assert row['energy_Ha']==root['energy_Ha'] and row['S2']==root['spin_square']
        assert abs(row['energy_Ha']+.1*row['S2']-row['penalized_energy_Ha'])<1e-8
        assert row['actual_CI_dimension']==len(strings)**2<=128**2
    assert not tt['original_T3_states_changed'] and tt['new_model_shots']==tt['new_QPU_shots']==0
    g5=load(ROOT/'freeze-G5-WHOLE-N20.json')
    for source in g5['sources']:
        p=next(p for p in frozen['plans'] if p['source']['case']==source['case'] and p['source']['method_id']=='C5_D_SQDRIFT_N20')
        assert p['source']['payload_SHA256']==source['source_counts_SHA256']
        assert p['context']['problem_id']==source['problem_id']
    result=dict(status='PASSED_T3_144_ROWS_AND_ADDITIVE_TAIL_IDENTITY_COST_SCOPE_CHECK',
        original_cells=48,original_statuses=dict(Counter(s['status'] for s in terminal['states'])),
        original_numeric_endpoints=137,original_primary_pairs=46,
        tail_statuses=dict(Counter(r['status'] for r in tt['rows'])),
        new_completed_CI_roots=terminal['new_completed_CI_roots']+sum(r.get('new_CI_solves',0) for r in tt['rows']),
        numeric_campaign_CPU_seconds=terminal['CPU_seconds']+tt['CPU_seconds'],
        cost_excludes_freeze_tests_analysis_old_sources=True,new_model_shots=0,new_QPU_shots=0,
        T3_catalog_SHA256=sha(t3/'evaluations.sqlite'),T3_terminal_SHA256=sha(t3/'terminal.json'),
        tail_terminal_SHA256=sha(tail/'terminal.json'),independent_confirmation=False,goal_complete=False)
    from .io import save
    save(ROOT/'analysis/T3-final-inventory-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
