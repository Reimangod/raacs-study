"""Check the derived report against frozen sources without repeating compilation."""
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from .paths import ROOT
from .io import load,save,sha256
from .strong_projection_bank import binding,FREEZE,RUN


def run():
    frozen=load(FREEZE); summary=load(ROOT/'analysis/G5-whole-N20-summary.json')
    assert binding()==frozen
    assert sha256((ROOT/'analysis/R2-common-CI-all.csv').read_bytes())==load(ROOT/'freeze-G3-FULLBANK-PROJECTION.json')['R2_comparison_csv_SHA256']
    terminal=load(RUN/'terminal.json')
    assert sha256((RUN/'compiled-members.sqlite').read_bytes())==terminal['database_SHA256']
    assert summary['members']==3132 and summary['actual_QPY_recounts']==9396
    assert summary['new_model_shots']==summary['new_QPU_shots']==summary['new_energy_CI_evaluations']==0
    table=list(csv.DictReader((ROOT/'analysis/G5-energy-resource-joined.csv').open(encoding='utf-8-sig')))
    assert len(table)==16
    for c in summary['cases']:
        rows={r['method']:r for r in table if r['case']==c['case']}
        assert len(rows)==4 and c['members']==783 and c['model_shot_weight']==100000
        s=next(x for x in frozen['sources'] if x['case']==c['case'])
        assert sha256(Path(s['source_counts_path']).read_bytes())==s['source_counts_SHA256']==c['counts_SHA256']
        for row in rows.values():
            assert int(row['R_cap'])==128 and int(row['final_model_shots'])==100000
            assert row['independent_parameter_rank']=='' and row['independent_quantum_confirmation']=='False'
            assert row['compiler_cost_field_is_shared_campaign_not_per_row']=='True'
        original=rows['N20_ORIGINAL_NATIVE_OLD_R2_FREQUENCY'];compressed=rows['N20_COMPRESSED_SAME_OLD_R2_FREQUENCY']
        assert original['energy_Ha']==compressed['energy_Ha'] and original['recovery_pct']==compressed['recovery_pct']
        control=rows['N20_STRONG_COMPILER_SAME_GLOBAL'];new=rows['N20_COMPRESSED_SAME_GLOBAL']
        assert control['energy_Ha']==new['energy_Ha'] and control['CI_numeric_CPU_seconds_not_cold_total']==new['CI_numeric_CPU_seconds_not_cold_total']
        assert float(new['energy_Ha'])==c['Global_T2_energy_Ha']
        for k in ['CNOT','CNOT_depth','total_depth']:
            assert float(new[k])<=float(control[k])<=float(original[k])
    suites=list(ET.parse(ROOT/'qualification-G5-preexecution-full-tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==135
    assert not any(int(s.get(k,0)) for s in suites for k in ['errors','failures','skipped'])
    save(ROOT/'analysis/G5-final-report-audit.json',dict(status='PASS_CURRENT_DATA_CONTEXT_AND_REPORT_NOT_GLOBAL_GOAL_COMPLETION',
        frozen_inputs_and_old_R2_csv_unchanged=True,joined_rows=16,cases=4,
        actual_saved_QPY_recounts_from_completed_auditor=9396,
        same_evaluation_reuse_not_new_precision_observations=True,parameter_unknowns_preserved=True,
        full_suite_tests=135,goal_complete=False,
        evidence_SHA256={str(p.relative_to(ROOT)):sha256(p.read_bytes()) for p in [FREEZE,RUN/'terminal.json',
            ROOT/'analysis/G5-whole-N20-summary.json',ROOT/'analysis/G5-energy-resource-joined.csv',
            ROOT/'analysis/G5-all-members.csv',ROOT/'analysis/G5_WHOLE_N20_RESULTS_JA.md',
            ROOT/'frontierloop/analyze_strong_bank.py',ROOT/'frontierloop/audit_strong_bank_report.py']}))
    print('G5_FROZEN_INPUT_AND16_JOINED_ROWS_AUDIT_PASSED')


if __name__=='__main__':run()
