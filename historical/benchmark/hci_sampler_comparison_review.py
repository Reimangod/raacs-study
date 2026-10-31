"""Independent saved-endpoint/CSV/hash review. No solver or producer imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert all(sha(folder / name) == digest for name, digest in manifest['files'].items())
    assert sha(folder / 'comparison.json') == folder.name == manifest['comparison_content_sha256']
    body = json.loads((folder / 'comparison.json').read_text())
    assert all(sha(path) == digest for path, digest in body['inputs'].items())
    source_code = Path(__file__).with_name('hci_sampler_comparison.py')
    assert sha(source_code) == body['analysis_code_sha256']
    validations = [json.loads(Path(p).read_text()) for p in body['inputs'] if Path(p).name == 'validation.json']
    ev, = [v for v in validations if v['status'] == 'MOLECULAR_STRONG_ALL_RECORDS_AUDITED']
    hv, = [v for v in validations if v['status'] == 'THRESHOLD_HCI_STANDARD_INDEPENDENTLY_AUDITED']
    quantum = {(r['case'], r['method']): r for r in ev['rows']}
    hci = {(r['case'][2:], r['epsilon1_Ha']): r for r in hv['rows'] if r['case'].startswith('E-')}
    expected = {(case, method, r, epsilon) for case, method in quantum
                for r in (8, 32, 64) for c, epsilon in hci if c == case}
    with (folder / 'all-planned-pairs.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    observed = set(); differences = equal = missing = 0
    for row in rows:
        case, method = row['case'], row['method']
        rcap, epsilon = int(row['R_cap']), float(row['epsilon1_Ha'])
        key = case, method, rcap, epsilon
        assert key not in observed; observed.add(key)
        q, h = quantum[(case, method)], hci[(case, epsilon)]
        assert row['sampler_status'] == q['status'] and row['HCI_status'] == h['status']
        qr = next((r for r in q.get('readouts', []) if r['R'] == rcap), {})
        qe, he = qr.get('energy_Ha'), h.get('final_energy_Ha')
        available = q['status'] == 'COMPLETED' and qe is not None and he is not None
        assert row['work_matched_superiority_established'] == 'False'
        assert row['HCI_CNOT'] == row['sampler_ground_state_error_Ha'] == row['HCI_ground_state_error_Ha'] == ''
        if available:
            assert row['comparison_status'] == 'AVAILABLE_DESCRIPTIVE_PAIR'
            assert math.isclose(float(row['sampler_minus_HCI_mHa']), (qe - he) * 1000, rel_tol=0, abs_tol=1e-10)
            assert float(row['sampler_energy_Ha']) == qe and float(row['HCI_last_solved_energy_Ha']) == he
            same = qr['actual_dimension'] == h['final_dimension']
            assert row['equal_actual_dimension'] == str(same)
            assert int(row['sampler_actual_dimension']) == qr['actual_dimension']
            assert int(row['HCI_actual_dimension']) == h['final_dimension']
            differences += 1; equal += same
        else:
            assert row['comparison_status'] == 'MISSING_PAIR_PRESERVED'
            assert row['sampler_minus_HCI_mHa'] == ''
            missing += 1
    assert observed == expected and len(rows) == len(expected) == body['planned_pairs']
    assert differences == body['counts']['AVAILABLE_DESCRIPTIVE_PAIR']
    assert missing == body['counts']['MISSING_PAIR_PRESERVED']
    assert equal == body['equal_actual_dimension_available_pairs']
    out = {'status': 'ALL_PLANNED_SAVED_ENDPOINT_DIFFERENCES_DIMENSIONS_AND_HASHES_CHECKED',
           'planned_pairs': len(rows), 'available_pairs': differences, 'missing_pairs': missing,
           'equal_actual_dimension_available_pairs': equal,
           'source_input_files_checked': len(body['inputs']), 'snapshot_sha256': folder.name,
           'review_code_sha256': sha(__file__), 'new_CI_solves': 0, 'new_model_shots': 0,
           'scope': 'Saved audited endpoint arithmetic only, NOT fresh CI validation or matched-work superiority.',
           'whole_goal_complete': False}
    target = folder / 'review-v1.json'
    serialized = json.dumps(out, sort_keys=True, indent=2) + '\n'
    if target.exists():
        assert target.read_text() == serialized
    else:
        target.write_text(serialized)
    print(serialized)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--snapshot', type=Path, required=True)
    review(parser.parse_args().snapshot)
