"""Descriptive join of immutable audited E/HCI endpoints; no numerical backend.

Different dimensions and timing boundaries remain visible. An energy difference
is not a work-matched win, a ground-state error, or a hardware advantage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R_VALUES = (8, 32, 64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unique_index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError('Duplicate endpoint identity: ' + str(key))
        result[key] = row
    return result


def require_same_problem(a, b):
    for key in ('problem_id', 'physical_integral_binding', 'n_qubits', 'nalpha', 'nbeta', 'hf_id'):
        if a.get(key) is None or a[key] != b.get(key):
            raise ValueError('Physical problem mismatch: ' + key)
    if not finite(a.get('HF_energy_Ha')) or not finite(b.get('HF_energy_Ha')):
        raise ValueError('Missing physical HF energy')
    if abs(a['HF_energy_Ha'] - b['HF_energy_Ha']) > 1e-9:
        raise ValueError('Physical HF energy mismatch')


def endpoint_row(case, method, rcap, q, h):
    readouts = unique_index(q.get('readouts') or [], ('R',))
    qr = readouts.get((rcap,), {})
    qe, he = qr.get('energy_Ha'), h.get('final_energy_Ha')
    qd, hd = qr.get('actual_dimension'), h.get('final_dimension')
    available = q['status'] == 'COMPLETED' and finite(qe) and finite(he)
    equal_d = available and type(qd) is int and type(hd) is int and qd == hd
    resources = q.get('sampling_resources') or {}
    return {
        'case': case, 'method': method, 'R_cap': rcap, 'epsilon1_Ha': h['epsilon1_Ha'],
        'method_kind': 'CLASSICAL_CARDINALITY_CONTROL' if method == 'classical-SCI64' else 'SAMPLER',
        'comparison_status': 'AVAILABLE_DESCRIPTIVE_PAIR' if available else 'MISSING_PAIR_PRESERVED',
        'sampler_status': q['status'], 'HCI_status': h['status'],
        'HCI_threshold_set_closed': h['status'] == 'HCI_VARIATIONAL_SELECTION_NO_NEW_IDS',
        'sampler_energy_Ha': qe if finite(qe) else None,
        'HCI_last_solved_energy_Ha': he if finite(he) else None,
        'sampler_minus_HCI_mHa': 1000 * (qe - he) if available else None,
        'sampler_actual_dimension': qd, 'HCI_actual_dimension': hd,
        'equal_actual_dimension': equal_d,
        'dimension_comparison': ('EXACTLY_EQUAL_SAVED_ENDPOINT_DIMENSIONS' if equal_d else
                                 'NOT_MATCHED' if available else 'UNAVAILABLE'),
        'sampler_CNOT': resources.get('cnot_count'), 'sampler_CNOT_depth': resources.get('cnot_depth'),
        'HCI_CNOT': None, 'HCI_quantum_shots': h.get('quantum_shots'),
        'sampler_known_committed_model_shots': q.get('known_committed_model_shots'),
        'sampler_unknown_inflight_draws': q.get('unknown_inflight_draws'),
        'sampler_recorded_invocation_seconds': q.get('wall_seconds_this_invocation'),
        'HCI_criterion_seconds_setup_separate': h.get('criterion_numeric_seconds'),
        'timing_scope': 'UNMATCHED_BOUNDARIES_SHARED_SETUP_NOT_RECHARGED',
        'sampler_ground_state_error_Ha': None, 'HCI_ground_state_error_Ha': None,
        'work_matched_superiority_established': False,
    }


def build(e_root, hci_root, output):
    e_root, hci_root, output = map(lambda p: Path(p).resolve(), (e_root, hci_root, output))
    for source in (e_root, hci_root):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('Output must be separate from all source trees')
    inputs = {}

    def read(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Source digest mismatch: ' + str(path))
        inputs[str(path)] = digest
        return json.loads(path.read_text())

    em, hm = read(e_root / 'manifest.json'), read(hci_root / 'manifest.json')
    ev = read(e_root / 'validation.json')
    hv = read(hci_root / 'validation.json')
    if ev['status'] != 'MOLECULAR_STRONG_ALL_RECORDS_AUDITED':
        raise ValueError('E independent record audit required')
    if hv['status'] != 'THRESHOLD_HCI_STANDARD_INDEPENDENTLY_AUDITED':
        raise ValueError('HCI independent standard audit required')
    if ev['manifest_sha256'] != sha(e_root / 'manifest.json') or hv['manifest_sha256'] != sha(hci_root / 'manifest.json'):
        raise ValueError('Audit/manifest binding mismatch')
    eq = unique_index(ev['rows'], ('case', 'method'))
    hh = unique_index(hv['rows'], ('case', 'epsilon1_Ha'))
    selected = [item for item in hm['queue'] if item['source_cohort'] == 'E26']
    if {item['id'][2:] for item in selected} != {item['id'] for item in em['queue']}:
        raise ValueError('Require every E case, not only successful/matching cases')
    rows = []
    for item in selected:
        case = item['id'][2:]
        es = read(e_root / 'cases' / case / 'physical-source.json')
        hs = read(hci_root / 'cases' / item['id'] / 'source.json')
        if es['meta_sha256'] != hs['metadata_sha256'] or es['integrals_sha256'] != hs['archive_sha256']:
            raise ValueError('Physical file identity mismatch')
        physical = Path(es['path'])
        read(physical / 'problem.json', es['meta_sha256'])
        archive = physical / 'integrals.npz'
        if sha(archive) != es['integrals_sha256']:
            raise ValueError('Integral archive changed')
        inputs[str(archive)] = es['integrals_sha256']
        for method in em['methods']:
            q = eq[(case, method)]
            raw = read(e_root / 'cases' / case / method / 'result.json', q['result_sha256'])
            if raw.get('problem') is not None:
                require_same_problem(raw['problem'], hs['problem'])
            for epsilon in hm['thresholds']:
                h = hh[(item['id'], epsilon)]
                read(hci_root / 'cases' / item['id'] / ('epsilon-' + str(epsilon)) / 'result.json', h['source_result_sha256'])
                for rcap in R_VALUES:
                    rows.append(endpoint_row(case, method, rcap, q, h))
    expected = len(em['queue']) * len(em['methods']) * len(hm['thresholds']) * len(R_VALUES)
    if len(rows) != expected:
        raise ValueError('Planned Cartesian denominator lost')
    unique_index(rows, ('case', 'method', 'R_cap', 'epsilon1_Ha'))
    if any(sha(p) != digest for p, digest in inputs.items()):
        raise ValueError('Source changed during read-only comparison')
    body = {'schema': 'E-HCI-audited-endpoint-comparison-v1', 'rows': rows,
            'planned_pairs': expected, 'counts': dict(Counter(r['comparison_status'] for r in rows)),
            'equal_actual_dimension_available_pairs': sum(r['equal_actual_dimension'] for r in rows),
            'inputs': inputs, 'analysis_code_sha256': sha(__file__),
            'new_model_shots': 0, 'new_hardware_shots': 0, 'new_CI_solves': 0,
            'whole_goal_complete': False}
    serialized = canonical(body)
    folder = output / hashlib.sha256(serialized.encode()).hexdigest()
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = '''# HCI と既存サンプラー：監査済みendpointの比較

26物理問題の全予定手法、R=8/32/64、固定された3つのepsilonを全て保持。
`sampler_minus_HCI_mHa` が正なら、保存済みHCIの方が低いenergy。
負なら当該サンプラーの方が低い。ただし、CI次元・費用を揃えた勝利ではない。
欠測・MPS拒否・HCI次元打ち切りは削除しない。best epsilon/seedは選ばない。

物理入力のSHAとproblem/generator-independent integral identityを確認して結合。
同じ分子名でも活性空間が異なるデータは混ぜない。
equal_actual_dimensionは、保存endpointの次元が偶然一致したかの記述属性。
一致subsetだけで全条件の性能を推定しない。新たな次元制限・再対角化はしない。

時間の境界が異なり、HCI setupは共有。秒数比からspeedupを主張しない。
量子側の未知in-flight費用を0にしない。古典SCIのCNOT/depthはN/A。
HCIはvariational threshold段階で、optimized SHCI・PT2・正解参照ではない。
信頼できるground referenceがなく、絶対誤差・chemical accuracy・相関回収率は未算出。
これは事後の記述的比較であり、matched work優越性・量子優位性・全研究完了ではない。
元のHCI/E結果、policy、queue、capは変更していない。追加sampling/CI solveは0。
'''
    exports = {'comparison.json': serialized, 'all-planned-pairs.csv': buffer.getvalue(), 'REPORT_JA.md': report}
    exports['manifest.json'] = canonical({'comparison_content_sha256': folder.name,
        'files': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in exports.items()},
        'whole_goal_complete': False})
    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in exports.items():
        path = folder / name
        if path.exists() and path.read_bytes() != contents.encode():
            raise ValueError('Refuse changed snapshot output')
        if not path.exists():
            path.write_bytes(contents.encode())
    print(canonical({'snapshot': str(folder), 'planned_pairs': expected, 'counts': body['counts'],
                     'equal_actual_dimension_available_pairs': body['equal_actual_dimension_available_pairs']}))
    return body, folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--e-root', type=Path, default=ROOT / 'results/molecular-strong-controls-standard-v2')
    parser.add_argument('--hci-root', type=Path, default=ROOT / 'results/integral-threshold-HCI-standard-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/HCI-native-sampler-comparison-v1')
    args = parser.parse_args()
    build(args.e_root, args.hci_root, args.output)
