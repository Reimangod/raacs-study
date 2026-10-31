"""Additive Fig14: saved QSCI / full-SQD energies, not a recovery-only ablation.

Reads all planned cells directly from the independently audited immutable
selected-ground database. No solver, quantum acquisition, optimizer or FCI.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/full-fermionic-SQD-projected-standard-v1'
METHODS = {
    'B1:source_final_ceo_ovp': ('CEO OVP', '#3465a4', 'o'),
    'B1:source_final_ceo_mvp': ('CEO MVP', '#9c4a93', 's'),
    'B1:lucj_ccsd_2_vqe': ('CCSD2-VQE LUCJ', '#d17a22', '^'),
    'RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small': ('RAACS', '#2c8263', 'D'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def key(task):
    return tuple(task[name] for name in ('case', 'method', 'rep', 'readout_flip_probability'))


def project(config, records, references):
    """Pure projection; all planned cells, both energies and failures stay explicit."""
    expected = set(itertools.product(config['cases'], config['methods'], config['reps'],
                                     config['readout_probabilities']))
    require(len(expected) == len(config['cases'])*len(config['methods'])*len(config['reps'])*len(config['readout_probabilities']),
            'Duplicate planned identities')
    seen = set(); rows = []
    for tid, task, status, result in records:
        identity = key(task)
        require(identity in expected and identity not in seen, 'Unexpected or duplicate cell')
        seen.add(identity)
        require(result['task'] == task and result['status'] == status, 'Cell/result identity drift')
        meta = references[task['case']]
        require(meta['case']['molecule'] == task['case'].split('_R')[0], 'Molecule mismatch')
        row = dict(id=tid, **task, status=status, molecule=meta['case']['molecule'],
                   distance_A=meta['case']['distance'], M=config['M'], R_cap=config['R'],
                   QSCI_energy_Ha=None, SQD_energy_Ha=None,
                   QSCI_error_mHa=None, SQD_error_mHa=None, SQD_minus_QSCI_mHa=None,
                   QSCI_d=None, SQD_d=None, matched_d=None,
                   matched_QSCI_energy_Ha=None, matched_SQD_energy_Ha=None,
                   matched_QSCI_error_mHa=None, matched_SQD_error_mHa=None,
                   matched_SQD_minus_QSCI_mHa=None, trivial_matched_HF_space=None,
                   reference_FCI_energy_Ha=meta['fci_energy'],
                   reference_role='OUTER_REPORTING_ONLY',
                   comparison_scope='QSCI_VS_FULL_SQD_NOT_RECOVERY_ONLY_CAUSAL_EFFECT',
                   work_matched=False, new_samples=0, new_CI_solves=0,
                   new_FCI_solves=0, hardware_shots=0)
        if status != 'COMPLETED':
            require(status in ('SOURCE_INITIALIZATION_MISSING', 'ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED'),
                    'Nonterminal or unknown status')
            row['missing_or_failure_reason'] = result.get('reason', result.get('error', status))
            rows.append(row)
            continue
        require(result['same_raw_counts'] is True, 'Unpaired source samples')
        require(result['new_quantum_sample_acquisitions'] == result['hardware_shots'] == 0,
                'Unexpected new acquisition')
        q, s, d = result['QSCI'], result['SQD'], result['same_actual_d']
        require(type(q['actual_dimension']) is int and type(s['dimension']) is int and
                1 <= q['actual_dimension'] <= config['R'] and 1 <= s['dimension'] <= config['R'],
                'Invalid common-cap dimensions')
        require(type(d['d']) is int and d['d'] == min(q['actual_dimension'], s['dimension']),
                'Invalid matched dimension')
        energies = [q['energy_Ha'], s['energy_Ha'], d['QSCI_energy_Ha'], d['SQD_energy_Ha'], meta['fci_energy']]
        require(all(finite(x) for x in energies), 'Nonfinite energy')
        qe, se, qd, sd, ref = energies
        require(min(qe, se, qd, sd) >= ref - 1e-8, 'Saved variational inequality failed')
        require(qd >= qe - 1e-8 and sd >= se - 1e-8, 'Restriction beats source space')
        row.update(QSCI_energy_Ha=qe, SQD_energy_Ha=se,
                   QSCI_error_mHa=1000*(qe-ref), SQD_error_mHa=1000*(se-ref),
                   SQD_minus_QSCI_mHa=1000*(se-qe),
                   QSCI_d=q['actual_dimension'], SQD_d=s['dimension'], matched_d=d['d'],
                   matched_QSCI_energy_Ha=qd, matched_SQD_energy_Ha=sd,
                   matched_QSCI_error_mHa=1000*(qd-ref), matched_SQD_error_mHa=1000*(sd-ref),
                   matched_SQD_minus_QSCI_mHa=1000*(sd-qd), trivial_matched_HF_space=d['d'] == 1,
                   missing_or_failure_reason=None)
        rows.append(row)
    require(seen == expected, 'Missing planned cells')
    return sorted(rows, key=lambda row: key(row))


def load_rows(source):
    """Hash-qualified source read; no producer/numerical module imports."""
    inputs = {}
    def bind(path):
        inputs[str(path)] = sha(path)
        return path
    def read(path):
        return json.loads(bind(path).read_text())
    config = read(source / 'manifest.json')
    validation = read(source / 'selected-ground-validation.json')
    parent = read(source / 'validation.json')
    require(validation['status'] == 'BOUNDED_SELECTED_GROUND_ALL_EVENTS_VALIDATED', 'Unqualified source')
    require(validation['parent_validation_sha256'] == sha(source/'validation.json'), 'Parent audit drift')
    database = bind(source / 'events.sqlite')
    require(inputs[str(database)] == validation['database_sha256'] == parent['database_sha256'], 'Database drift')
    references = {case: read(Path(config['source'])/'references'/case/'reference.json') for case in config['cases']}
    records = []
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        require(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'SQLite integrity')
        for tid, task, status, raw in db.execute('SELECT id,task,status,result FROM cells ORDER BY id'):
            require(raw is not None, 'Unfinished source cell')
            records.append((tid, json.loads(task), status, json.loads(zlib.decompress(raw))))
    rows = project(config, records, references)
    require(sum(row['status'] == 'COMPLETED' for row in rows) == validation['completed'], 'Completed count drift')
    # Independent arithmetic cross-check against the existing report, not just
    # importing that reporter's helper functions or trusting plot filenames.
    manifest = read(source/'report-v1/manifest.json')
    report_path = bind(source/'report-v1/all-planned-cells.csv')
    require(inputs[str(report_path)] == manifest['artifacts'][report_path.name], 'Report drift')
    with report_path.open(newline='') as stream:
        report = {r['id']: r for r in csv.DictReader(stream)}
    require(set(report) == {r['id'] for r in rows}, 'Report omitted planned cells')
    for row in rows:
        old = report[row['id']]
        require(old['status'] == row['status'], 'Report status mismatch')
        for field in ('QSCI_error_mHa', 'SQD_error_mHa', 'SQD_minus_QSCI_mHa',
                      'matched_d', 'matched_SQD_minus_QSCI_mHa', 'QSCI_d', 'SQD_d'):
            if row['status'] == 'COMPLETED':
                require(abs(float(old[field])-row[field]) <= 1e-8, 'Report arithmetic mismatch: '+field)
            else:
                require(old[field] == '', 'Missing result was filled')
    require(all(sha(p) == value for p, value in inputs.items()), 'Source changed during read')
    return rows, inputs


def build(output):
    source = SOURCE.resolve(); output = Path(output).resolve()
    require(output != source and source not in output.parents and output not in source.parents, 'Protect source')
    output.mkdir(parents=True, exist_ok=True)
    existing = output/'manifest.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        require(saved['code_sha256'] == sha(__file__), 'Changed code requires a successor')
        require(all(sha(p) == value for p, value in saved['inputs'].items()), 'Saved input drift')
        require(all(sha(output/p) == value for p, value in saved['outputs'].items()), 'Saved output drift')
        return saved
    require(not any(output.iterdir()), 'Partial output preserved; choose a successor')
    require(shutil.disk_usage(output).free > 128*2**20, 'Insufficient figure headroom')
    rows, inputs = load_rows(source)
    with (output/'all-planned-energy-pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'Fig14-raw-energy-pairs-v1'
    figures = []
    for molecule in sorted({r['molecule'] for r in rows}):
        group = [r for r in rows if r['molecule'] == molecule]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        handles = {}
        for noise, axisrow in zip((0., .01), axes):
            for matched, ax in zip((False, True), axisrow):
                prefix = 'matched_' if matched else ''
                eligible = [r for r in group if r['readout_flip_probability'] == noise]
                good = [r for r in eligible if r['status'] == 'COMPLETED']
                values = [r[prefix+arm+'_error_mHa'] for r in good for arm in ('QSCI', 'SQD')]
                lo, hi = min([-.01]+values), max([.01]+values)
                margin = .04*(hi-lo); limits = (lo-margin, hi+margin)
                ax.plot(limits, limits, '--', color='gray', lw=.8, label='Equal energy')
                for method, (label, color, marker) in METHODS.items():
                    selected = [r for r in good if r['method'] == method]
                    artist = ax.scatter([r[prefix+'QSCI_error_mHa'] for r in selected],
                                        [r[prefix+'SQD_error_mHa'] for r in selected],
                                        s=23, marker=marker, color=color, alpha=.65, label=label)
                    handles[label] = artist
                ax.set(xscale='symlog', yscale='symlog', xlim=limits, ylim=limits,
                       xlabel='Explicit QSCI error (mHa; reporting-only FCI)',
                       ylabel='Full SQD error (mHa; reporting-only FCI)',
                       title=('Ideal counts' if noise == 0 else 'Synthetic 1% readout flips')+' / '+
                             ('same realized d' if matched else 'common R cap 64'))
                ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.15)
                ax.text(.04, .96, f'{len(good)}/{len(eligible)} saved cells; lower than diagonal = SQD better',
                        transform=ax.transAxes, va='top', fontsize=8)
        fig.suptitle(f'{molecule}: QSCI / full SQD from the same M=1000 acquisition; all three repeats')
        fig.legend(handles.values(), handles.keys(), loc='lower center', bbox_to_anchor=(.5,.071), ncol=4, fontsize=9)
        fig.text(.025,.019, 'Full SQD changes recovery, spin-product space and batch/carryover selection together; NOT a recovery-only causal effect.\n'
                 'Same d does not match classical work. Distances/repeats are not independent molecules. Missing cells remain in CSV.\n'
                 'No new samples, CI solves or FCI. Negative rounding residuals are retained; symlog linear region = 0.01 mHa.', fontsize=8)
        fig.tight_layout(rect=(0,.13,1,.95))
        stem = molecule+'_Fig14_QSCI_fullSQD_energy_pairs'; figures.append(stem)
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(output/(stem+'.'+ext), dpi=180, facecolor='white')
        plt.close(fig)
    require(all(sha(p) == value for p, value in inputs.items()), 'Input drift while plotting')
    readme = '# Fig14: same-acquisition QSCI / full-SQD energy pairs\n\n'
    readme += ('全720予定セルを元SQLiteから読み、708完了と12欠測を保全した派生図。'
               '各点は保存済みのcase/method/repeat/noiseセルで、新しい実験ではない。'
               '斜線より下ではSQDのenergyが低い。左は同じR64上限、右は各セルの小さい実dへ制限した診断。'
               '同実dでも基底選択・古典作業量は同じでなく、d=1は自明なHF一致になり得る。\n\n'
               'Full SQDではRecoveryだけでなくspin-product空間・batch選択・carryoverも変わる。'
               'したがって差をRecovery単独の因果効果とは呼ばない。合成readoutノイズはgate noiseでも実機でもない。'
               'FCI参照は外側の誤差表示だけ。元の独立selected-space検証を再利用し、新CI/FCI計算は行わない。'
               '表の値・状態を既存all-planned CSVと照合したが、この描画処理は固有値の独立再計算ではない。\n\n')
    for stem in figures:
        readme += f'- {stem}: [PNG]({stem}.png) / [SVG]({stem}.svg) / [PDF]({stem}.pdf)\n'
    readme += '\n研究全体の完了や、全費用込みの優位性を意味しない。旧図・凍結データは変更していない。\n'
    (output/'README_JA.md').write_text(readme)
    outputs = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    saved = dict(schema='same-acquisition-QSCI-full-SQD-Fig14-v1', inputs=inputs, outputs=outputs,
                 code_sha256=sha(__file__), planned_cells=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
                 figures=figures, formats=['png','svg','pdf'],
                 values_from_audited_raw_database=True, crosschecked_existing_report=True,
                 new_matrix_or_eigenvalue_recomputation=False, new_samples=0, new_CI_solves=0, new_FCI_solves=0,
                 hardware_shots=0, whole_goal_complete=False)
    existing.write_text(json.dumps(saved, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return saved


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'results/Recovery-energy-Fig14-supplement-v1')
    args = p.parse_args()
    result = build(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','outputs')}, sort_keys=True))
