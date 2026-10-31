"""Additive Fig18 layout repair from audited results; never rerun a workflow.

The v1 colorbar overlaps its last panel. V2 reserves a separate column, uses
shared method labels, and keeps every failed cell visible. Original numerical
and presentation artifacts remain immutable. Existing v2 exports are verified,
not rerendered, on a repeated call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'RAACS-full-cost': ('RAACS', 'o'),
    'LUCJ-CCSD-2-fixed': ('Fixed LUCJ2', 's'),
    'LUCJ-CCSD-2-QSCI4': ('QSCI-guided LUCJ2', '^'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_once(path, body):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError('Existing derivative differs; use a successor: ' + str(path))
    else:
        path.write_bytes(body)


def valid_row(row):
    """Null cost and failed construction must not become a free frontier point."""
    if row['status'] != 'COMPLETED' or row.get('unknown_inflight') or not row['readouts']:
        return False
    import math
    for value in (row['total_workflow_wall_seconds'], row['all_draw_CNOT_times_shots_proxy'],
                  row['readouts'][-1]['ground_error_mHa']):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Nonfinite completed workflow value')
    if row['total_workflow_wall_seconds'] <= 0 or row['all_draw_CNOT_times_shots_proxy'] < 0:
        raise ValueError('Invalid recorded cost')
    if row['readouts'][-1]['R'] != 64:
        raise ValueError('Unregistered final R')
    return True


def validate_cached(out, inputs):
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['input_sha256'] != inputs:
        raise ValueError('Presentation input drift')
    for name, expected in manifest['output_sha256'].items():
        if sha(out / name) != expected:
            raise ValueError('Presentation output drift: ' + name)
    return manifest


def build(root=None):
    root = Path(root or ROOT / 'results/total-workflow-standard-v1').resolve()
    original = root / 'report-v1'
    out = root / 'report-presentation-v2'
    inputs = {str(root / 'validation.json'): sha(root / 'validation.json'),
              str(original / 'manifest.json'): sha(original / 'manifest.json'),
              str(original / 'observed-frontiers.json'): sha(original / 'observed-frontiers.json'),
              str(original / 'all-planned-workflows.csv'): sha(original / 'all-planned-workflows.csv'),
              str(Path(__file__)): sha(__file__)}
    old = json.loads((original / 'manifest.json').read_text())
    for item in old['figures']:
        path = original / item['file']
        if sha(path) != item['sha256']:
            raise ValueError('Original presentation drift: ' + str(path))
        inputs[str(path)] = item['sha256']
    if (out / 'manifest.json').exists():
        manifest = validate_cached(out, inputs)
        return {'status': 'VERIFIED_WITHOUT_RERENDER', 'output': str(out),
                'figures': len(manifest['figure_files']), 'new_experiments': 0}
    data = json.loads((root / 'validation.json').read_text())
    if data['status'] != 'WORKFLOW_ALL_RECORDS_AUDITED' or data['unexpected_failures']:
        raise ValueError('Unaudited standard source')
    if data['planned'] != data['terminal'] or len(data['rows']) != data['planned']:
        raise ValueError('Incomplete planned-cell coverage')
    points = [r for r in data['rows'] if valid_row(r)]
    if len(points) != data['completed']:
        raise ValueError('Completed source contains an unaccounted row')
    for row in data['rows']:
        if row['method'] not in METHODS:
            raise ValueError('Unexpected method')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    out.mkdir(exist_ok=True)
    figure_files = []
    nondominated = {(r['case'], r['method']) for r in
                    json.loads((original / 'observed-frontiers.json').read_text())
                    if r['observed_nondominated']}
    for molecule in sorted({r['molecule'] for r in data['rows']}):
        selected = [r for r in data['rows'] if r['molecule'] == molecule]
        cases = sorted({r['case'] for r in selected},
                       key=lambda c: next(r['distance_A'] for r in selected if r['case'] == c))
        if len(cases) != 5:
            raise ValueError('Expected all five planned geometries')
        errors = [r['readouts'][-1]['ground_error_mHa'] for r in selected if valid_row(r)]
        lo, hi = (min(errors), max(errors)) if errors else (0., 1.)
        norm = Normalize(lo, hi if hi > lo else lo + 1.)
        fig = plt.figure(figsize=(14, 8.5), layout='constrained')
        grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .055])
        for i, case in enumerate(cases):
            ax = fig.add_subplot(grid[i // 3, i % 3])
            missing = []
            for r in [v for v in selected if v['case'] == case]:
                label, marker = METHODS[r['method']]
                if not valid_row(r):
                    missing.append(label + ': unavailable (CCSD/init)')
                    continue
                x, y = r['total_workflow_wall_seconds'], r['all_draw_CNOT_times_shots_proxy']
                ax.scatter(x, y, c=[r['readouts'][-1]['ground_error_mHa']], norm=norm,
                           cmap='viridis', marker=marker, s=65, edgecolors='black', linewidths=.6)
                if (case, r['method']) in nondominated:
                    ax.scatter(x, y, facecolors='none', edgecolors='black', s=145, linewidths=.8)
            ax.set(title=case, xlabel='Given-integrals total workflow wall (s)',
                   ylabel='All paid logical CNOT × model samples')
            ax.margins(x=.15, y=.15)
            ax.grid(alpha=.15)
            if missing:
                ax.text(.02, .98, '\n'.join(missing), transform=ax.transAxes,
                        va='top', fontsize=8, wrap=True)
        note = fig.add_subplot(grid[1, 2]); note.axis('off')
        legend = [Line2D([], [], color='black', linestyle='none', marker=marker,
                         markerfacecolor='gray', label=label, markersize=7)
                  for label, marker in METHODS.values()]
        legend.append(Line2D([], [], color='black', linestyle='none', marker='o',
                             markerfacecolor='none', label='Observed nondominated', markersize=10))
        note.legend(handles=legend, loc='upper left', frameon=False, fontsize=10)
        note.text(0, .55,
                  'One realization; three-method subset.\n'
                  '2,000 total model samples / method,\nincluding all construction pilots.\n'
                  'R cap64; actual d may differ.\n'
                  'Color: reporting-only FCI error.\n'
                  'Missing cells retained, not method defeats.\n'
                  'Hardware shots0; not QPU-time cost.\n'
                  'Prior RHF/integrals, export, audits and\nplots excluded from timed boundary.\n'
                  'Host/JIT caches uncontrolled; no speedup\nor general superiority claim.',
                  transform=note.transAxes, va='top', fontsize=9, linespacing=1.4)
        cax = fig.add_subplot(grid[:, 3])
        fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), cax=cax,
                     label='R64 QSCI error (mHa; outer FCI reference)')
        fig.suptitle('Fig18 | ' + molecule + ' | natural workflow cost and accuracy', fontsize=14)
        for ext in ('png', 'svg', 'pdf'):
            path = out / f'Fig18_given_integrals_total_workflow_{molecule}.{ext}'
            fig.savefig(path, dpi=180)
            figure_files.append(path.name)
        plt.close(fig)
    save_once(out / 'all-planned-workflows.csv', (original / 'all-planned-workflows.csv').read_bytes())
    save_once(out / 'README.md', (
        '# Fig18 presentation v2\n\n'
        'Additive layout-only successor: dedicated colorbar column, shared method legend, '
        'all five distances, and explicit unavailable cells. No point/energy/resource/time '
        'or frontier membership changed; CSV bytes equal v1. Original files retained.\n\n'
        'This is a given-integrals, one-realization, three-method observed frontier, '
        'not historical CEO end-to-end cost, QPU runtime or inferential superiority.\n').encode())
    if any(sha(path) != value for path, value in inputs.items()):
        raise ValueError('Source changed during export')
    outputs = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    manifest = {'schema': 'given-integrals-Fig18-presentation-v2', 'input_sha256': inputs,
                'output_sha256': outputs, 'figure_files': figure_files,
                'planned': data['planned'], 'completed': data['completed'],
                'source_numerical_values_changed': False, 'new_experiments': 0,
                'new_CI_solves': 0, 'new_FCI_solves': 0, 'hardware_shots': 0,
                'whole_goal_complete': False}
    save_once(out / 'manifest.json', (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode())
    return {'status': 'ADDITIVE_LAYOUT_EXPORTED', 'output': str(out),
            'figures': len(figure_files), 'new_experiments': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path)
    print(json.dumps(build(parser.parse_args().root), sort_keys=True))
