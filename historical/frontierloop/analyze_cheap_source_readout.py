"""N42 scope-explicit audit. Never rewrite N33 or turn pending cases into failures."""
from pathlib import Path
from io import BytesIO
from collections import Counter
from time import process_time, perf_counter
import sys
import numpy as np
from qiskit import qpy
from .paths import ROOT, OLD
from .io import load, save, write, sha256
from . import cheap_source_readout as r
from . import cheap_source_sampling as s
from . import analytical_source_sampling as prior
from . import optimized_control_sampling as tn
from .analyze_finite_source_readout import audit_root
from .analyze import reference


def resource_inventory(p, cases):
    inherited = load(ROOT/'analysis/N33-ALL-summary.json'); result = {}
    n15 = ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1'
    ad = load(prior.LOCAL/'protocol.json'); tp = load(tn.LOCAL/'protocol.json')
    for case, method in sorted({(x['case'], x['method']) for x in p['control_inventory'] if x['available'] and x['case'] in cases}):
        oldrow = next(x for x in inherited['rows'] if x['case'] == case and x['method'] == method and x['status'] == 'COMPLETED')
        if method.startswith('C5_'):
            file = n15/case/method/'EXACT/opt1-sampling.qpy'; expected_sha = None
        elif method == 'ANALYTICAL_OPT_L1_8EVAL':
            z = next(z for z in ad['sources'] if z['case'] == case)['resources'][0]['files']['sampling']
            file, expected_sha = Path(z['file']), z['SHA256']
        elif method == 'EXISTING_TN_COMMON_L1':
            z = next(z for z in tp['resources'] if z['case'] == case and z['method'] == method)['compiler_points'][0]['circuits']['sampling']
            file, expected_sha = Path(z['path']), z['SHA256']
        else:
            raise ValueError('Unknown strong control')
        blob = file.read_bytes()
        if expected_sha is not None and sha256(blob) != expected_sha:
            raise ValueError('Strong control QPY changed')
        qc = qpy.load(BytesIO(blob))[0]
        actual = dict(CNOT=int(qc.count_ops().get('cx', 0)), CNOT_Depth=int(qc.depth(filter_function=lambda x: x.operation.name == 'cx')), Total_Depth=int(qc.depth()))
        if any(actual[k] != oldrow[k] for k in actual):
            raise ValueError('Actual control QPY disagrees with prior audited source resources')
        result[case, method] = dict(actual, reported_coordinate_slots=oldrow['declared_parameters'], independent_parameter_rank=None,
                                   QPY=str(file), QPY_SHA256=sha256(blob), coordinate_count_inherited_not_independent_rank=True)
    for source in p['sources']:
        if source['case'] not in cases: continue
        z = source['files']['sampling.qpy']; blob = Path(z['path']).read_bytes()
        if sha256(blob) != z['SHA256']: raise ValueError('New source QPY changed')
        qc = qpy.load(BytesIO(blob))[0]
        actual = dict(CNOT=int(qc.count_ops().get('cx', 0)), CNOT_Depth=int(qc.depth(filter_function=lambda x: x.operation.name == 'cx')), Total_Depth=int(qc.depth()))
        expected = source['resources']['Z_sampling_equivalent']
        if (actual['CNOT'], actual['CNOT_Depth'], actual['Total_Depth']) != (expected['cnot_count'], expected['cnot_depth'], expected['total_unitary_depth']):
            raise ValueError('New actual resources disagree')
        result[source['case'], s.METHOD] = dict(actual, reported_coordinate_slots=source['coefficient_slots'],
                   independent_parameter_rank=None, QPY=z['path'], QPY_SHA256=z['SHA256'])
    return result


def run(partition):
    if partition not in ('M4', 'A100_INPUTS', 'ALL'): raise ValueError('Explicit scope required')
    cpu, wall = process_time(), perf_counter(); p, q, sub = r.verified()
    parts = ['M4', 'A100_INPUTS'] if partition == 'ALL' else [partition]
    terminals = [load(r.LOCAL/(part+'-terminal.json')) for part in parts]
    states = [row for t in terminals for row in t['states']]; cases = {row['case'] for row in states}
    tasks = [t for t in q['tasks'] if t['case'] in cases]
    identity = lambda x: (x['case'], x['method'], x['replicate'], x['arm'])
    byid = {identity(x): x for x in states}
    if len(states) != len(tasks) or len(byid) != len(tasks): raise ValueError('Coverage or duplicate endpoint')
    if any(t['protocol_SHA256'] != sha256((r.LOCAL/'protocol.json').read_bytes()) for t in terminals): raise ValueError('Terminal protocol binding')
    inv = [x for x in r.inventory(p, sub) if x['case'] in cases]; counts = r.records(p, sub, inv)
    if any(x['status'] == 'WAITING_FOR_FROZEN_INPUTS' for x in counts.values()): raise ValueError('Terminal without completed counts')
    resources = resource_inventory(p, cases); roots = {}; rows = []
    for task in tasks:
        state = byid[identity(task)]
        if any(state[k] != v for k, v in task.items()): raise ValueError('Task identity mismatch')
        out = {k: state.get(k) for k in ('case', 'method', 'replicate', 'arm', 'status', 'R', 'actual_D', 'draw_id', 'basis_id', 'new_CI_attempt', 'cache_origin', 'root_path', 'root_SHA256')}
        if state['status'] == 'COMPLETED':
            actual = counts[state['draw_id']]; d = p['physical'][state['case']]
            R = actual['R32'] if state['arm'] == 'R32_CAP' else r.common_R(task, inv, counts)
            strings = sorted(actual['rank'][:R]); pairs = [(a, b) for a in strings for b in strings]
            if actual != state['source_evidence'] or R != state['R'] or strings != state['half_strings'] or r.parent.basis_key(d, pairs, q['library_SHA256']) != state['basis_id']:
                raise ValueError('Count/ranking/actual CI space differs')
            if state['basis_id'] not in roots: roots[state['basis_id']] = audit_root(state, d, p['library'])
            out.update(roots[state['basis_id']]); hf, ref = reference(state['case'])
            meta = load(OLD/'data/evaluation-only'/(state['case']+'.json'))
            if meta['problem_id'] != d['problem_id']: raise ValueError('Reference physical problem differs')
            out.update(correlation_energy_Recovery_pct=100*(hf-out['energy_Ha'])/(hf-ref),
                       reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
                       reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED',
                       raw_model_shots_per_source=1000, **resources[state['case'], state['method']])
        else: out['energy_Ha'] = None
        rows.append(out)
    groups = []
    for case, method, arm in sorted({(x['case'], x['method'], x['arm']) for x in rows}):
        local = [x for x in rows if (x['case'], x['method'], x['arm']) == (case, method, arm)]
        valid = [x for x in local if x['status'] == 'COMPLETED']
        groups.append(dict(case=case, method=method, arm=arm, completed=len(valid), planned=len(local),
             energy_mean_Ha=float(np.mean([x['energy_Ha'] for x in valid])) if valid else None,
             SD_mHa=float(np.std([x['energy_Ha'] for x in valid], ddof=1)*1000) if len(valid) > 1 else None,
             correlation_energy_Recovery_mean_pct=float(np.mean([x['correlation_energy_Recovery_pct'] for x in valid])) if valid else None,
             actual_R=[x['R'] for x in valid], actual_D=[x['actual_D'] for x in valid],
             statistical_superiority_certified=False))
    comparisons = []
    for case in sorted(cases):
        for arm in r.ARMS:
            new = next(g for g in groups if (g['case'], g['method'], g['arm']) == (case, s.METHOD, arm))
            controls = [g for g in groups if g['case'] == case and g['arm'] == arm and g['method'] in s.STRONG and g['completed'] == 3]
            best = min(controls, key=lambda g: g['energy_mean_Ha']) if controls else None
            original = next((g for g in controls if g['method'] == 'C5_C_LUCJ_COMPRESSED_L1'), None)
            diff = lambda g: 1000*(new['energy_mean_Ha']-g['energy_mean_Ha']) if g and new['completed'] == 3 else None
            comparisons.append(dict(case=case, arm=arm, retrospective_best_control=best['method'] if best else None,
                 cheap_minus_best_control_mHa=diff(best), cheap_minus_initial_L1_mHa=diff(original),
                 common_R_equals_one=bool(new['actual_R'] and all(R == 1 for R in new['actual_R'])),
                 statistical_superiority_certified=False))
    runtime = []; accounts = []
    for c in sub['circuits']:
        if c['case'] not in cases: continue
        folder = s.SAMPLING/('collected/circuits' if c['backend'] == 'A100' else 'circuits')/c['circuit_id']/'output'
        z = load(folder/'result.json')
        runtime.append(dict(case=c['case'], backend=c['backend'], CPU_seconds=z['whole_body_CPU_seconds'],
             wall_seconds=z['whole_body_wall_seconds'], norm_squared=z['norm_squared'],
             actual_new_model_samples=sum(x['actual_model_shots'] for x in z['sampling_results']),
             result_SHA256=sha256((folder/'result.json').read_bytes())))
    for f in sorted(s.SAMPLING.glob('lane*-collection.json')):
        v = load(f)
        parts_acct = next(line.split('|') for line in v['accounting'].splitlines() if line.split('|')[0] == v['job']['job_id'])
        if parts_acct[1] != 'COMPLETED' or parts_acct[3] != '0:0': raise ValueError('Slurm not successful')
        h, m, sec = map(int, parts_acct[2].split(':'))
        accounts.append(dict(job_id=v['job']['job_id'], GPU_allocation_seconds=h*3600+m*60+sec, accounting=v['accounting']))
    result = dict(status='N42_SCOPED_ACTUAL_READOUTS_AUDITED', scope=partition, cases=sorted(cases), rows=rows, groups=groups, comparisons=comparisons,
         endpoint_status_counts=dict(Counter(x['status'] for x in rows)), distinct_saved_roots_audited=len(roots),
         actual_new_CI_roots=sum(a['status'] == 'COMPLETED' for t in terminals for a in t['attempts']),
         failed_CI_attempts=sum(a['status'] != 'COMPLETED' for t in terminals for a in t['attempts']),
         actual_partition_CPU_seconds=sum(t['CPU_seconds'] for t in terminals), source_runtime=runtime, GPU_accounting=accounts,
         new_model_samples=sum(x['actual_new_model_samples'] for x in runtime), new_QPU_shots=0,
         old_draws_reused_not_independent_new_confirmation=True, coefficient_slots_not_independent_parameter_rank=True,
         old_initialization_costs=[dict(case=x['case'], cost=x['old_initialization_cost']) for x in p['sources']],
         new_initialization_cost_evidence='analysis/N41-cheap-mp2-summary.json',
         old_source_cost_evidence='analysis/SOURCE-cost-components-v1.json', cold_total_CPU_unknown_not_zero=True,
         missing_other_case_results_not_negative=True, native_best_paper_reproduction=False, cold_matched_work=False,
         source_protocol_SHA256=sha256((s.LOCAL/'protocol.json').read_bytes()),
         previous_audited_resource_summary_SHA256=sha256((ROOT/'analysis/N33-ALL-summary.json').read_bytes()),
         terminal_SHA256={part: sha256((r.LOCAL/(part+'-terminal.json')).read_bytes()) for part in parts},
         audit_CPU_seconds=process_time()-cpu, audit_wall_seconds=perf_counter()-wall, goal_complete=False)
    save(ROOT/'analysis'/f'N42-{partition}-summary.json', result)
    lines = [f'# N42 {partition}：安いMP2初期化の有限Shot試験', '',
             f"{len(rows)} endpoints／{result['endpoint_status_counts']}、新model samples {result['new_model_samples']}、実QPU0。新CI根{result['actual_new_CI_roots']}、失敗{result['failed_CI_attempts']}。他partitionはこの表の集計範囲に含めない。", '',
             '| 条件 | 方法 | CI規約 | 数値／予定 | energy平均 Ha | SD mHa | 相関energy Recovery % | actual R | actual D |',
             '|---|---|---|---:|---:|---:|---:|---|---|']
    fmt = lambda value: '欠測' if value is None else f'{value:.6f}'
    for g in groups:
        lines.append(f"| {g['case']} | {g['method']} | {g['arm']} | {g['completed']}/{g['planned']} | {fmt(g['energy_mean_Ha'])} | {fmt(g['SD_mHa'])} | {fmt(g['correlation_energy_Recovery_mean_pct'])} | {g['actual_R']} | {g['actual_D']} |")
    lines += ['', '## 回路資源', '', '| 条件 | 方法 | CNOT | CNOT Depth | Total Depth | 報告された係数座標slot | 独立rank |', '|---|---|---:|---:|---:|---:|---|']
    for (case, method), z in sorted(resources.items()):
        lines.append(f"| {case} | {method} | {z['CNOT']} | {z['CNOT_Depth']} | {z['Total_Depth']} | {z['reported_coordinate_slots']} | unknown |")
    lines += ['', '## 解釈の境界', '',
              'R32_CAPが主たる固定上限比較。COMMON_Rは新sourceの小さいsupportにも合わせるため、R=1へ落ちた場合の同点はHFだけの空間での同点であり、精度を維持した勝利ではない。3反復から統計的優越を認定しない。', '',
              '旧対照counts・CI根を再利用しており新しい独立確認ではない。係数座標slotと独立parameter rankは別。noise-free logical all-to-allで、native hardwareや最高予算論文の完全再現ではない。', '',
              '新初期化・旧初期化・旧調整・source sampling・CI・audit・Slurm割当の費用を別々に保持する。全cold費用の不明部分を0にしない。参照精度が未確認のN₂、SCI参照のFeをFCIと呼ばない。', '',
              f"実partition CPU {result['actual_partition_CPU_seconds']:.6f}秒、今回audit {result['audit_CPU_seconds']:.6f}秒。", '']
    write(ROOT/'analysis'/f'N42_{partition}_RESULTS_JA.md', '\n'.join(lines).encode())
    print('N42_AUDITED', partition, len(rows), result['endpoint_status_counts'], 'newroots', result['actual_new_CI_roots'])
    for c in comparisons: print(c['case'], c['arm'], 'cheap-minus-best mHa', c['cheap_minus_best_control_mHa'])


if __name__ == '__main__': run(sys.argv[1])
