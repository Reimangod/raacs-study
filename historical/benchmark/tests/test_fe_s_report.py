import csv
import json
from benchmark.common import ROOT,file_digest


def test_feasibility_report_matches_saved_source_and_does_not_claim_ground_accuracy():
    root=ROOT/'results/FeS-feasibility-pilot-v1'
    manifest=json.loads((root/'report-v1/manifest.json').read_text())
    for p,expected in manifest['sources'].items():assert file_digest(root/p)==expected
    for p,expected in manifest['outputs'].items():assert file_digest(root/'report-v1'/p)==expected
    source=json.loads((root/'RAACS-full-cost/result.json').read_text())
    row=next(csv.DictReader((root/'report-v1/result.csv').open()))
    assert float(row['final_QSCI_energy_Ha'])==source['final_QSCI_energy_Ha']
    assert row['ground_state_error_Ha']==''
    assert int(row['total_model_samples'])==int(row['pilot_model_samples'])+int(row['independent_final_model_samples'])==2000
    assert row['S_squared_certified']=='False'
    assert manifest['new_experiments']==manifest['new_eigensolver_calls']==0


def test_feasibility_producer_keeps_frozen_code_and_public_input_identity():
    root=ROOT/'results/FeS-feasibility-pilot-v1'
    cfg=json.loads((root/'manifest.json').read_text())
    for p,expected in cfg['implementation'].items():assert file_digest(ROOT/'benchmark'/p)==expected
    physical=json.loads((root/'physical-problem/problem.json').read_text())
    assert physical['n_qubits']==40 and physical['FCI_computed'] is False
    assert physical['reference_semantics'].endswith('NOT_VERIFIED_RHF')
    assert physical['ground_state_reference'] is None
    source=json.loads((root/'public-inputs/provenance.json').read_text())
    assert source['Fe4']['header']['n_qubits']==72
    assert source['Fe4']['full_file_downloaded'] is False
