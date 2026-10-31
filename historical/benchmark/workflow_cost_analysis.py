"""Post-terminal independent count/circuit/Ritz audit and total-cost frontier.

No parent ground-state reference is read until ALL mode tasks are terminal.
Small-system references are outer diagnostic labels, never selection features.
"""
from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import zlib

from .workflow_cost_study import ROOT, configuration, task_id, sha, save, digest


def named_selection(counts, info, cap):
    from .local_sampler_primitives import valid_sector
    ranked = [info['hf_id']] + [x for x, n in sorted(counts.items(), key=lambda a: (-a[1], a[0]))
                              if x != info['hf_id'] and valid_sector(x, info['n_qubits'], info['nalpha'], info['nbeta'])][:cap-1]
    return sorted(ranked)


def named_energy(backend, ids):
    import numpy as np
    matrix = backend.matrix(ids, ids)
    values, vectors = np.linalg.eigh(matrix)
    residual = float(np.linalg.norm(matrix @ vectors[:, 0] - values[0] * vectors[:, 0]))
    assert residual < 1e-8
    return float(values[0])


def independent_evidence(folder, method, result):
    """Read raw SQLite directly and independently count actual QPY CX gates."""
    from qiskit import qpy
    from .local_sampler_primitives import unpack_counts64
    adaptive = method == 'RAACS-full-cost'
    base = folder / 'native-RAACS' if adaptive else folder
    if not (base / 'events.sqlite').exists():
        return None
    native = json.loads((base / 'result.json').read_text()) if adaptive and (base / 'result.json').exists() else {}
    final_ids = set(native.get('final_draw_ids', [])) if adaptive else {result.get('final_draw_id')}
    final, circuits, finalcid = {}, {}, None
    shots, gate, depth, requested, started = 0, 0, 0, 0, 0
    with sqlite3.connect(f'file:{base / "events.sqlite"}?mode=ro', uri=True) as db:
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        column = 'resources' if adaptive else 'payload'
        for cid, sample, payload in db.execute(f'SELECT id,sampling_qpy,{column} FROM circuits'):
            qc = qpy.load(BytesIO(zlib.decompress(sample)))[0]
            resource = json.loads(payload)
            if not adaptive:
                resource = resource['resources']
            cx = int(qc.count_ops().get('cx', 0))
            cxdepth = int(qc.depth(filter_function=lambda inst: inst.operation.name == 'cx') or 0)
            assert cx == resource['cnot_count'] and cxdepth == resource['cnot_depth']
            circuits[cid] = resource
        column = 'counts64' if adaptive else 'counts'
        for did, cid, nrequest, blob, rawmeta in db.execute(f'SELECT id,circuit,requested,{column},metadata FROM draws'):
            requested += nrequest
            if blob is None:
                started += 1
                continue
            ids, nn = unpack_counts64(blob)
            counts = dict(zip(map(int, ids), map(int, nn)))
            meta = json.loads(rawmeta)
            n = sum(counts.values())
            assert n == meta['actual_raw_model_shots'] <= nrequest
            assert all(value > 0 for value in counts.values())
            assert meta['hardware_shots'] == 0 and meta['FCI_used'] is False
            # The internal MPS permutation must map counts back to logical IDs.
            from .mps_layout_qualification import logical_id
            layout = meta['layout_evidence']
            logical = {logical_id(x, layout['logical_to_simulator_qubit']): k
                       for x, k in zip(layout['backend_ids'], layout['backend_counts'])}
            assert logical == counts
            shots += n
            gate += n * circuits[cid]['cnot_count']
            depth += n * circuits[cid]['cnot_depth']
            if did in final_ids:
                if finalcid is not None:
                    assert cid == finalcid
                finalcid = cid
                for x, k in counts.items():
                    final[x] = final.get(x, 0) + k
    return {'shots': shots, 'requested': requested, 'started': started, 'gate': gate,
            'depth': depth, 'final': final, 'resources': circuits.get(finalcid),
            'database_sha256': sha(base / 'events.sqlite')}


def audit(root):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .resource_adaptive_sampler import integral_problem
    from .molecular_mps_scaling import no_full_state_guards
    root = Path(root).resolve()
    cfg = json.loads((root / 'manifest.json').read_text())
    assert cfg == configuration(cfg['mode']), 'Frozen runtime/config source drift'
    status = json.loads((root / 'execution-status.json').read_text())
    assert status['status'] == 'ALL_WORKFLOW_TASKS_TERMINAL_PENDING_AUDIT'
    assert status['terminal'] == status['planned'] == len(cfg['queue'])
    # Check completeness and identities BEFORE reading any reference energy.
    receipts = []
    inputs = {'manifest.json': sha(root / 'manifest.json'),
              'input-export.json': sha(root / 'input-export.json')}
    for index, item in enumerate(cfg['queue']):
        folder = root / 'cases' / item['case'] / item['method']
        receipt = json.loads((folder / 'process-receipt.json').read_text())
        assert receipt['index'] == index and receipt['task_id'] == task_id(item, cfg)
        assert receipt['total_workflow_wall_seconds'] > 0
        assert receipt['timing_scope'] == cfg['timing_scope']
        for name, value in receipt['files'].items():
            assert sha(folder / name) == value
            inputs[str((folder / name).relative_to(root))] = value
        inputs[str((folder / 'process-receipt.json').relative_to(root))] = sha(folder / 'process-receipt.json')
        receipts.append(receipt)
    rows, comparisons, unexpected = [], [], []
    maximum, solves = 0., 0
    with threadpool_limits(limits=1), no_full_state_guards():
        for item, receipt in zip(cfg['queue'], receipts):
            case, method = item['case'], item['method']
            folder = root / 'cases' / case / method
            physical = root / 'physical-inputs' / case
            backend, info = integral_problem(physical)
            source = ROOT / 'results/standard-v1/references' / case
            for name, expected in cfg['physical_source_hashes'][case].items():
                assert sha(source / name) == expected
            metadata = json.loads((physical / 'problem.json').read_text())
            assert set(metadata) == {'n_qubits','nalpha','nbeta','hf_id','hf_energy_Ha','schema'}
            with np.load(physical / 'integrals.npz') as z, np.load(source / 'reference.npz') as original:
                assert set(z.files) == {'h1', 'eri'}
                assert np.array_equal(z['h1'], original['h1']) and np.array_equal(z['eri'], original['eri'])
            reference = json.loads((source / 'reference.json').read_text())
            result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').exists() else {}
            evidence = independent_evidence(folder, method, result)
            summary = json.loads((folder / 'workflow-summary.json').read_text()) if (folder / 'workflow-summary.json').exists() else {}
            if evidence:
                assert evidence['shots'] <= evidence['requested'] <= cfg['total_model_shots']
                assert evidence['shots'] == summary.get('known_model_shots', evidence['shots'])
                assert evidence['gate'] == summary.get('gate_shot_proxy', evidence['gate'])
                assert evidence['depth'] == summary.get('depth_shot_proxy', evidence['depth'])
                if summary.get('database_sha256'):
                    assert evidence['database_sha256'] == summary['database_sha256']
            values = []
            if receipt['status'] == 'COMPLETED':
                assert receipt['returncode'] == 0 and evidence and evidence['final']
                assert evidence['shots'] == cfg['total_model_shots'] and not evidence['started']
                assert receipt['all_model_work_known'] and receipt['total_model_shots'] == evidence['shots']
                for cap, stored in zip(cfg['R_grid'], summary['readouts']):
                    ids = named_selection(evidence['final'], info, cap)
                    energy = named_energy(backend, ids)
                    assert energy >= reference['fci_energy'] - 1e-8
                    error = abs(energy - stored['energy_Ha'])
                    maximum = max(maximum, error)
                    assert stored['R'] == cap and stored['selected_ids'] == ids and error < 1e-8
                    values.append({'R': cap, 'd': len(ids), 'energy_Ha': energy,
                                   'ground_error_mHa': 1000 * (energy - reference['fci_energy'])})
                    solves += 1
                assert abs(values[-1]['energy_Ha'] - result['final_energy_Ha']) < 1e-8
            elif not any(tag in str(result.get('error', '')) + receipt['status'] for tag in
                         ('CCSD', 'Preserved amplitude initialization failure', 'resource cap',
                          'cap at gate boundary', 'RESOURCE_CAPPED', 'PROCESS_', 'PENDING_')):
                unexpected.append({'case': case, 'method': method, 'status': receipt['status'], 'error': result.get('error')})
            row = {'case': case, 'molecule': reference['case']['molecule'], 'distance_A': reference['case']['distance'],
                   'method': method, 'status': receipt['status'], 'n_qubits': info['n_qubits'],
                   'total_workflow_wall_seconds': receipt['total_workflow_wall_seconds'],
                   'total_wall_scope': cfg['timing_scope'], 'peak_RSS_bytes': summary.get('peak_RSS_bytes'),
                   'known_committed_model_shots': evidence['shots'] if evidence else None,
                   'total_model_shots': receipt['total_model_shots'], 'unknown_inflight': not receipt['all_model_work_known'],
                   'all_draw_CNOT_times_shots_proxy': evidence['gate'] if evidence else None,
                   'all_draw_Depth_times_shots_proxy': evidence['depth'] if evidence else None,
                   'final_sampling_resources': evidence['resources'] if evidence else None,
                   'readouts': values, 'hardware_shots': 0, 'new_FCI_solves': 0}
            rows.append(row)
        # Paired realized-d comparison independently reselects from frequency
        # order, never truncates numerically sorted selected determinant IDs.
        for case in sorted({r['case'] for r in rows}):
            available = {r['method']: r for r in rows if r['case'] == case and r['status'] == 'COMPLETED'}
            if 'RAACS-full-cost' not in available:
                continue
            backend, info = integral_problem(root / 'physical-inputs' / case)
            evidence = {}
            for method in available:
                folder = root / 'cases' / case / method
                evidence[method] = independent_evidence(folder, method, json.loads((folder / 'result.json').read_text()))
            a = available['RAACS-full-cost']
            for method, b in available.items():
                if method == 'RAACS-full-cost':
                    continue
                for va, vb in zip(a['readouts'], b['readouts']):
                    d = min(va['d'], vb['d'])
                    ea = named_energy(backend, named_selection(evidence['RAACS-full-cost']['final'], info, d))
                    eb = named_energy(backend, named_selection(evidence[method]['final'], info, d))
                    solves += 2
                    comparisons.append({'case': case, 'baseline': method, 'R': va['R'], 'matched_d': d,
                                        'RAACS_minus_baseline_common_R_mHa': 1000 * (va['energy_Ha'] - vb['energy_Ha']),
                                        'RAACS_minus_baseline_actual_d_mHa': 1000 * (ea - eb)})
    missing_quick = ([{'case': r['case'], 'method': r['method'], 'status': r['status']}
                      for r in rows if r['status'] != 'COMPLETED'] if cfg['mode'] == 'quick' else [])
    record = {'status': 'WORKFLOW_QUICK_INDEPENDENTLY_VALIDATED' if cfg['mode'] == 'quick' and not unexpected and not missing_quick
              else 'WORKFLOW_QUICK_QUALIFICATION_INCOMPLETE' if cfg['mode'] == 'quick' and not unexpected
              else 'WORKFLOW_ALL_RECORDS_AUDITED' if not unexpected else 'WORKFLOW_AUDIT_UNEXPECTED_FAILURE',
              'planned': len(cfg['queue']), 'terminal': len(rows), 'completed': sum(r['status'] == 'COMPLETED' for r in rows),
              'unexpected_failures': unexpected, 'quick_qualification_noncompleted_tasks': missing_quick,
              'independent_projected_solves': solves,
              'maximum_Ritz_parity_error_Ha': maximum, 'rows': rows, 'comparisons': comparisons,
              'input_hashes': inputs, 'FCI_reference_outer_reporting_only': True,
              'new_FCI_solves': 0, 'hardware_shots': 0, 'whole_goal_complete': False}
    save(root / 'validation.json', record)
    print(record['status'], 'terminal', len(rows), 'solves', solves, flush=True)
    if unexpected:
        raise RuntimeError('Preserved unexpected engineering failures; do not dispatch next cohort')
    return record


def frontier(rows):
    """Three-dimensional observed (energy, wall, gate-shot) frontier per problem."""
    good = [r for r in rows if r['status'] == 'COMPLETED' and r['readouts'] and not r['unknown_inflight']]
    def vector(r):
        return (r['readouts'][-1]['energy_Ha'], r['total_workflow_wall_seconds'], r['all_draw_CNOT_times_shots_proxy'])
    result = []
    for a in good:
        av = vector(a)
        dominated = any(b['case'] == a['case'] and b['method'] != a['method'] and
                        all(x <= y for x, y in zip(vector(b), av)) and
                        any(x < y for x, y in zip(vector(b), av)) for b in good)
        result.append({'case': a['case'], 'method': a['method'], 'observed_nondominated': not dominated,
                       'scope': 'single realization, R64, given-integrals workflow; not inference or hardware cost'})
    return result


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    root = Path(root)
    data = json.loads((root / 'validation.json').read_text())
    assert not data['unexpected_failures']
    for path, expected in data['input_hashes'].items():
        assert sha(root / path) == expected
    out = root / 'report-v1'
    out.mkdir(exist_ok=True)
    full = []
    for r in data['rows']:
        row = {k: v for k, v in r.items() if k not in ('readouts', 'final_sampling_resources')}
        row.update({f'final_{k}': v for k, v in (r['final_sampling_resources'] or {}).items()})
        for value in r['readouts']:
            row.update({f'R{value["R"]}_{key}': item for key, item in value.items() if key != 'R'})
        full.append(row)
    keys = sorted({k for row in full for k in row})
    with (out / 'all-planned-workflows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(full)
    save(out / 'observed-frontiers.json', frontier(data['rows']))
    figures = []
    for molecule in sorted({r['molecule'] for r in data['rows']}):
        selected = [r for r in data['rows'] if r['molecule'] == molecule]
        cases = sorted({r['case'] for r in selected}, key=lambda c: next(r['distance_A'] for r in selected if r['case'] == c))
        fig, axes = plt.subplots(1, len(cases), figsize=(max(6, 3.8 * len(cases)), 4.8), squeeze=False)
        errors = [r['readouts'][-1]['ground_error_mHa'] for r in selected if r['readouts']]
        norm = Normalize(vmin=min(errors) if errors else 0, vmax=max(errors) if errors and max(errors) > min(errors) else (min(errors) + 1 if errors else 1))
        markers = {'RAACS-full-cost': 'o', 'LUCJ-CCSD-2-fixed': 's', 'LUCJ-CCSD-2-QSCI4': '^'}
        scatter = None
        for ax, case in zip(axes[0], cases):
            subset = [r for r in selected if r['case'] == case]
            missing = []
            for r in subset:
                if r['status'] != 'COMPLETED' or not r['readouts']:
                    missing.append(r['method'] + ': ' + r['status'])
                    continue
                scatter = ax.scatter(r['total_workflow_wall_seconds'], r['all_draw_CNOT_times_shots_proxy'],
                    c=[r['readouts'][-1]['ground_error_mHa']], norm=norm, cmap='viridis', s=55,
                    marker=markers[r['method']], edgecolors='black', linewidths=.5)
                ax.annotate(r['method'].replace('LUCJ-CCSD-2-', 'LUCJ-'),
                    (r['total_workflow_wall_seconds'], r['all_draw_CNOT_times_shots_proxy']), xytext=(4, 5),
                    textcoords='offset points', fontsize=7)
            ax.set(title=case, xlabel='Observed total workflow wall (s)', ylabel='All paid logical CNOT × model samples')
            ax.grid(alpha=.15)
            if missing:
                ax.text(.02, .98, '\n'.join(missing), transform=ax.transAxes, va='top', fontsize=6.5, wrap=True)
        if scatter is not None:
            fig.colorbar(scatter, ax=list(axes[0]), label='R64 QSCI error (mHa; outer FCI reference)', shrink=.7, pad=.02)
        fig.suptitle('Fig18 candidate | ' + molecule + ' | 2,000 total model samples / method', fontsize=12)
        fig.text(.02, .025, 'Given-integrals process boundary only. All pilots included. Hardware shots0. Not QPU time; no whole-study superiority.\n'
                 'Fresh process/private CCSD directory, uncontrolled host/JIT caches; one realization. Missing cells retained. Three-method subset.', fontsize=8)
        fig.subplots_adjust(left=.08, right=.86, bottom=.24, top=.86, wspace=.4)
        for ext in ('png', 'svg', 'pdf'):
            path = out / f'Fig18_given_integrals_total_workflow_{molecule}.{ext}'
            fig.savefig(path, dpi=180)
            figures.append({'file': path.name, 'sha256': sha(path)})
        plt.close(fig)
    save(out / 'manifest.json', {'validation_sha256': sha(root / 'validation.json'), 'figures': figures,
         'all_planned_rows': len(full), 'scope': 'observed given-integrals total wall and all paid gate-shot proxy',
         'hardware_shots': 0, 'new_FCI_solves': 0, 'whole_goal_complete': False})
    (out / 'REPORT_JA.md').write_text('# 総workflow費用の実測\n\n'
        f'{len(full)}予定セル、{data["completed"]}完了。失敗・欠測を全予定表に保持。\n\n'
        '総時間はPopen直前から子process終了まで。保存済み物理積分を与えた後の構築・sampling・CI・保存を含む。'
        '積分作成・共有入力変換・外部監査・図作成を含まない。既存CEO/VQE全費用の代用ではない。\n\n'
        '量子費用は全pilot・最終読出しの実消費model samples。CNOT×samplesは補助proxyでQPU時間ではない。'
        'ハードウェアshots0、FCIは全セル終端後の誤差ラベルのみ。1実現なので90%成功保証なし。\n\n'
        'Fig18は3手法の限定された観測frontier候補であり、研究全体の費用優越性や未知分子一般化は主張しない。\n')
    print('WORKFLOW_REPORT_SAVED', len(figures), 'figures', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--component', choices=['audit', 'report'], default='audit')
    args = parser.parse_args()
    audit(args.root) if args.component == 'audit' else report(args.root)
