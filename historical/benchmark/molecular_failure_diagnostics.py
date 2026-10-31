"""Read-only classification of committed molecular failures, not a new gate.

Produces a historical, content-addressed snapshot while the native cohort may
continue. Never changes status, fills an energy, retries a task or calls a solver.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def classify(result, minimum_norm):
    status = result['status']
    base = {'category': 'OTHER_TERMINAL_STATUS', 'whole_cell_physical_method_loss': False,
            'original_status': status,
            'original_new_primary_model_shots': result.get('new_primary_model_shots'),
            'original_unreported_inflight_model_shots': result.get('unreported_inflight_model_shots'),
            'structured_rejected_operation_pre_sampling': False,
            'rejected_operation_new_model_shots': None,
            'whole_cell_quantum_cost_reconciled_by_this_report': False}
    if status == 'COMPLETED':
        return dict(base, category='COMPLETED_NOT_AUTOMATICALLY_ACCURATE')
    error = result.get('error', '')
    if error.startswith("BlockedApproximationUnqualified("):
        try:
            record, _ = json.JSONDecoder().raw_decode(error[error.index('{'):])
            norm = record['build']['retained_norm_squared']
            zero = record['confirmed_new_model_shots']
            if not (isinstance(norm, (int, float)) and math.isfinite(norm) and 0 <= norm < minimum_norm
                    and type(zero) is int and zero == 0):
                raise ValueError('Inconsistent pre-sampling rejection certificate')
            return dict(base, category='MPS_APPROXIMATION_UNQUALIFIED',
                        structured_rejected_operation_pre_sampling=True,
                        rejected_operation_new_model_shots=0,
                        retained_norm_squared=norm,
                        minimum_registered_retained_norm_squared=minimum_norm)
        except (ValueError, KeyError, TypeError):
            return dict(base, category='UNCLASSIFIED_ENGINEERING_FAILURE',
                        classification_issue='Invalid or incomplete pre-sampling rejection evidence')
    if error == "RuntimeError('MPS construction resource cap at gate boundary')":
        return dict(base, category='MPS_CONSTRUCTION_RESOURCE_CAP',
                    classification_issue='Generic exception does not separate wall/RSS or certify whole-cell shot cost')
    if error.startswith("RuntimeError('CCSD did not converge;"):
        return dict(base, category='CCSD_INITIALIZATION_REJECTED')
    if error.startswith("RuntimeError('Preserved amplitude initialization failure:"):
        return dict(base, category='PRESERVED_INITIALIZATION_DEPENDENCY_FAILURE')
    if status == 'ENGINEERING_OR_INITIALIZATION_FAILURE_PRESERVED':
        return dict(base, category='UNCLASSIFIED_ENGINEERING_FAILURE')
    return base


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Write outside the frozen/live source tree')
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'molecular-strong-controls-v2-spin-blocked':
        raise ValueError('This diagnostic applies to the exact spin-blocked v2 schema')
    inputs = {str(manifest_path): sha(manifest_path)}
    for name, expected in manifest['implementation'].items():
        path = ROOT / 'benchmark' / name
        if sha(path) != expected:
            raise RuntimeError('Native numerical source identity drift: ' + name)
    rows = []
    for item in manifest['queue']:
        for method in manifest['methods']:
            folder = source / 'cases' / item['id'] / method
            path = folder / 'result.json'
            row = {'case': item['id'], 'method': method, 'n_qubits': None,
                   'status': 'NOT_TERMINAL_AT_SNAPSHOT', 'category': 'NOT_TERMINAL_AT_SNAPSHOT',
                   'physical_method_loss': False, 'new_primary_model_shots': None,
                   'unreported_inflight_model_shots': None,
                   'wall_seconds_this_invocation': None, 'peak_RSS': None,
                   'retained_norm_squared': None, 'rejected_operation_new_model_shots': None,
                   'whole_cell_quantum_cost_reconciled_by_this_report': False}
            physical_failure = folder.parent / 'physical-failure.json'
            if not path.exists() and physical_failure.exists():
                raw = physical_failure.read_bytes()
                inputs[str(physical_failure)] = hashlib.sha256(raw).hexdigest()
                row.update(status='PHYSICAL_INPUT_FAILURE_PRESERVED',
                           category='PHYSICAL_INPUT_FAILURE_PRESERVED',
                           source_failure_sha256=inputs[str(physical_failure)])
            if path.exists():
                raw = path.read_bytes()
                inputs[str(path)] = hashlib.sha256(raw).hexdigest()
                record = json.loads(raw)
                evidence = classify(record, manifest['minimum_retained_norm_squared'])
                row.update(status=record['status'], category=evidence['category'],
                           n_qubits=record.get('problem', {}).get('n_qubits'),
                           new_primary_model_shots=record.get('new_primary_model_shots'),
                           unreported_inflight_model_shots=record.get('unreported_inflight_model_shots'),
                           wall_seconds_this_invocation=record.get('wall_seconds_this_invocation'),
                           peak_RSS=record.get('peak_RSS'),
                           retained_norm_squared=evidence.get('retained_norm_squared'),
                           rejected_operation_new_model_shots=evidence['rejected_operation_new_model_shots'])
                row['source_result_sha256'] = inputs[str(path)]
                row['classification'] = evidence
                dependency = folder.parent / 'amplitude-dependencies' / 'ccsd.json'
                if evidence['category'] in ('CCSD_INITIALIZATION_REJECTED', 'PRESERVED_INITIALIZATION_DEPENDENCY_FAILURE'):
                    if dependency.exists():
                        depraw = dependency.read_bytes()
                        inputs[str(dependency)] = hashlib.sha256(depraw).hexdigest()
                        dep = json.loads(depraw)
                        row['dependency_status'] = dep['status']
                        row['dependency_error'] = dep.get('error')
                        row['dependency_standalone_seconds_shared_not_additive'] = dep.get('wall_seconds')
                        row['dependency_sha256'] = inputs[str(dependency)]
            rows.append(row)
    if len(rows) != len(manifest['queue']) * len(manifest['methods']):
        raise RuntimeError('Keep every planned cell')
    # Existing terminal records must still equal the bytes we classified.
    # Later creation of previously unstarted cells does not change a snapshot.
    if not all(sha(path) == expected for path, expected in inputs.items()):
        raise RuntimeError('A committed source changed during read-only classification')
    categories = dict(Counter(r['category'] for r in rows))
    bymethod = defaultdict(Counter)
    for row in rows:
        bymethod[row['method']][row['category']] += 1
    body = {'schema': 'molecular-committed-failure-diagnostics-v1',
            'observed_UTC': datetime.now(timezone.utc).isoformat(),
            'source': str(source), 'planned': len(rows),
            'terminal_at_snapshot': sum(r['status'] != 'NOT_TERMINAL_AT_SNAPSHOT' for r in rows),
            'category_counts': categories, 'by_method': {m: dict(c) for m, c in bymethod.items()},
            'rows': rows, 'input_hashes': inputs, 'analysis_code_sha256': sha(__file__),
            'new_model_shots': 0, 'new_hardware_shots': 0, 'new_ground_state_solves': 0,
            'source_status_or_cost_modified': False, 'whole_goal_complete': False,
            'scope': 'Committed historical snapshot, not full-cohort comparison or final independent audit. '
                     'Pending cells may finish later. Resource/initialization failures are missingness, not physical method defeats.'}
    serialized = canonical(body)
    snapshot = output / 'snapshots' / hashlib.sha256(serialized.encode()).hexdigest()
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'diagnostics.json').write_text(serialized)
    simple = [{k: v for k, v in r.items() if k != 'classification'} for r in rows]
    fields = sorted({k for r in simple for k in r})
    with (snapshot / 'all-planned-cells.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(simple)
    lines = ['# 大規模系：失敗原因の読取り専用診断', '',
             f'{body["terminal_at_snapshot"]}/{len(rows)}件が、このsnapshotの読取り時点で終端。', '',
             '| 区分 | 件数 |', '|---|---:|']
    lines += [f'| {key} | {n} |' for key, n in sorted(categories.items())]
    lines += ['', 'この表は成功した回路だけの比較ではなく、未終端も含む全予定セルの台帳です。', '',
              '- `COMPLETED`は必要精度を満たしたことや他手法への勝利を意味しません。',
              '- MPS近似の未認定は、登録bond/cutoffでのシミュレーターの限界です。LUCJそのものの精度不良ではありません。',
              '- MPS構築resource capは時間/RSS等を区別できない汎用例外です。元の不明なin-flight費用を0へ補正しません。',
              '- CCSD初期化失敗と、その同じ共有依存を読む後続手法の失敗を分けます。共有初期化秒数を二重計上しません。',
              '- 構造化証拠がある近似拒否で0と確認できるのは、その拒否操作の新規samplesのみです。セル全体の費用ではありません。',
              '- 失敗・欠測を除外してRAACSが一般的に優れると判断できません。成功条件の異なる利用可能subsetも明示が必要です。',
              '- 元結果・status・費用・凍結条件は変更せず、再試行・最適化・sampling・FCIを行っていません。',
              '- 実行中cohortの途中snapshotであり、全セル終端後の既存独立監査に置き換わりません。', '']
    (snapshot / 'REPORT_JA.md').write_text('\n'.join(lines))
    exports = {p.name: sha(p) for p in snapshot.iterdir() if p.is_file()}
    (snapshot / 'manifest.json').write_text(canonical({'files': exports, 'snapshot': snapshot.name,
                                                     'read_only_source': True, 'whole_goal_complete': False}))
    print(json.dumps({'snapshot': str(snapshot), 'planned': len(rows),
                      'terminal_at_snapshot': body['terminal_at_snapshot'], 'category_counts': categories}))
    return body, snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'results/molecular-strong-controls-standard-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/molecular-failure-diagnostics-v1')
    args = parser.parse_args()
    build(args.source, args.output)
