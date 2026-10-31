"""All sixteen T1 cells, matched screens, real CI dimensions and full costs."""
import csv
from io import StringIO
from collections import Counter
from pathlib import Path
import numpy as np
from .paths import ROOT, OLD
from .io import load, save, write, sha256
from .native_trim_screen import DIRECTORY, FREEZE, binding
from .analyze import reference

NAMES = {'C5_C_LUCJ_COMPRESSED_L1': 'Compressed LUCJ L1', 'C5_D_SQDRIFT_N20': 'SqDRIFT20',
    'RAACS_B4': 'RAACS B4', 'CLASSICAL_CISD_100K_DRAWS': '古典CISD draw'}


def main():
    terminal = load(DIRECTORY/'terminal.json'); frozen = load(FREEZE)
    if binding() != frozen or len(terminal['states']) != 16:
        raise RuntimeError('T1 not fully terminal or frozen inputs changed')
    rows = []
    references = {}
    for cell in frozen['sources']:
        case, method = cell['case'], cell['method_id']; p = DIRECTORY/case/method
        old = load(cell['R2_baseline']); new = load(p/'result.json') if (p/'result.json').exists() else None
        failure = load(p/'failure.json') if (p/'failure.json').exists() else None
        hf, ref = reference(case)
        refpath = OLD/'data/evaluation-only'/(case+'.json')
        ev = load(refpath)
        references[case] = {'path': str(refpath), 'SHA256': sha256(refpath.read_bytes()),
            'kind': 'FCI' if ev['energies_Ha'].get('fci_energy') is not None else 'SCI',
            'used_for_reporting_after_freeze_only': True}
        history = new['calls'] if new else load(p/'CI-history.json')['calls'] if (p/'CI-history.json').exists() else []
        if new:
            if new['source_payload_SHA256'] != cell['payload_SHA256'] or new['physical_problem_id'] != cell['expected_problem_id']:
                raise RuntimeError('Actual T1 source/context binding mismatch')
            saved = p/'final-selected-CI-state.npz'
            if sha256(saved.read_bytes()) != new['final_state_SHA256']:
                raise RuntimeError('Saved final selected eigenvector changed')
            with np.load(saved, allow_pickle=False) as a:
                strings = list(map(int, a['strings_a']))
                if strings != new['calls'][-1]['CI_strings'] or not np.array_equal(a['strings_a'], a['strings_b']):
                    raise RuntimeError('Saved state/string indexing mismatch')
                if abs(np.sum(a['amplitudes']**2)-1) > 1e-9:
                    raise RuntimeError('Saved selected state norm changed')
            for call in history:
                if call['actual_CI_dimension'] != call['actual_r']**2 or call['actual_r'] > 128:
                    raise RuntimeError('Actual classical space cap changed')
                if abs(call['energy_Ha']+.1*call['spin_square']-call['penalized_energy_Ha']) > 1e-8:
                    raise RuntimeError('Physical/penalized energy mismatch')
                if abs(sum(call['electron_marginal_weights_alpha'])-1) > 1e-8 or abs(sum(call['electron_marginal_weights_beta'])-1) > 1e-8:
                    raise RuntimeError('Trim marginal weights not normalized')
            nested = set(strings) <= set(old['CI_strings'])
            monotonic = not nested or new['native_Trim_penalized_energy_Ha'] >= old['penalized_energy_Ha']-1e-8
        else:
            nested, monotonic = None, None
        resources = cell.get('sampler_resources') or {}
        rows.append({'case': case, 'method': method, 'source_kind': cell['kind'],
            'status': 'COMPLETED' if new else failure['status'] if failure else 'NOT_COMPLETED',
            'original_total_model_budget': cell['source_total_budget'],
            'original_final_model_shots': cell['source_final_model_shots'],
            'original_classical_draws': cell.get('classical_draws', 0),
            'CNOT_unchanged': resources.get('cnot_count'), 'CNOT_depth_unchanged': resources.get('cnot_depth'),
            'total_depth_unchanged': resources.get('total_depth', resources.get('total_unitary_depth')),
            'independent_parameter_rank': None, 'source_parameter_slot_proxy_not_rank': cell.get('sampler_coefficient_slots_mean'),
            'R_cap': 128, 'old_frequency_dimension': old['actual_CI_dimension'],
            'old_frequency_energy_Ha': old['energy_Ha'], 'old_frequency_S2': old['spin_square'],
            'old_frequency_recovery_pct': 100*(hf-old['energy_Ha'])/(hf-ref),
            'old_frequency_CPU_seconds': old['incremental_CPU_seconds'],
            'native_Trim_energy_Ha': new['native_Trim_energy_Ha'] if new else None,
            'native_Trim_recovery_pct': 100*(hf-new['native_Trim_energy_Ha'])/(hf-ref) if new else None,
            'native_Trim_S2': new['native_Trim_S2'] if new else None,
            'native_Trim_dimension': new['native_Trim_actual_CI_dimension'] if new else None,
            'Trim_minus_frequency_mHa': 1000*(new['native_Trim_energy_Ha']-old['energy_Ha']) if new else None,
            'matched_screen_energy_Ha': new['matched_disjoint_screen_control_energy_Ha'] if new else None,
            'matched_screen_S2': new['matched_disjoint_screen_control_S2'] if new else None,
            'matched_screen_kernel_CPU_seconds': new['screen_CPU_seconds_attributed_to_both_arms'] if new else None,
            'merge_additional_kernel_CPU_seconds': new['merge_additional_CPU_seconds'] if new else None,
            'Trim_full_CPU_seconds': new['full_attempt_CPU_seconds'] if new else failure['full_attempt_CPU_seconds'] if failure else None,
            'Trim_full_wall_seconds': new['full_attempt_wall_seconds'] if new else failure['full_attempt_wall_seconds'] if failure else None,
            'completed_CI_calls': len(history), 'CI_matvecs_completed_calls_only': sum(x['matvec_calls'] for x in history),
            'unfinished_CI_work_unknown': failure is not None,
            'merged_basis_subset_of_frequency': nested, 'penalized_nested_space_consistency': monotonic,
            'source_payload_SHA256': cell['payload_SHA256'], 'new_model_shots': 0,
            'reference_kind': references[case]['kind'], 'exact_singlet_certificate': False})
    complete = [x for x in rows if x['status'] == 'COMPLETED']
    summary = {'statuses': dict(Counter(x['status'] for x in rows)), 'planned_cells': 16,
        'numerically_lower_Trim_than_frequency': sum(x['Trim_minus_frequency_mHa'] < -1e-5 for x in complete),
        'numerically_higher_Trim_than_frequency': sum(x['Trim_minus_frequency_mHa'] > 1e-5 for x in complete),
        'numerically_lower_Trim_than_matched_screen': sum(x['native_Trim_energy_Ha'] < x['matched_screen_energy_Ha']-1e-8 for x in complete),
        'penalized_nested_space_anomalies': [x for x in complete if not x['penalized_nested_space_consistency']],
        'CPU_seconds_entire_T1': terminal['CPU_seconds'], 'wall_seconds_entire_T1': terminal['wall_seconds'],
        'new_model_shots': 0, 'new_QPU_shots': 0,
        'CI_solves_completed': sum(x['completed_CI_calls'] for x in rows),
        'all_references': references, 'not_matched_actual_CPU_despite_same_max_space': True,
        'goal_complete': False}
    save(ROOT/'analysis/T1-native-Trim-summary.json', summary)
    stream = StringIO(); w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(ROOT/'analysis/T1-native-Trim-all.csv', stream.getvalue().encode('utf-8-sig'))
    f = lambda x, d=3: '—' if x is None else f'{x:.{d}f}'
    lines = ['# T1：公式Trimを同条件で試した結果', '',
        f"全16セルの状態：{summary['statuses']}。完了CIは{summary['CI_solves_completed']}回。追加Shotは0。",
        '同じ古いsource観測を使ったpostprocessing比較で、新しい量子samplerや独立seedの確認試験ではない。', '',
        '## 何を比較したか', '',
        '元のR2はHFと頻度順で選んだR128のCIを1回。公式Trimは同じ観測から4つのdisjoint batchを作ってCIし、'
        '各spin listの上位25%（最大32）を統合してもう1回CIする。全CIのR上限128、元軌道・有限0.1Ha S²罰則は同じ。'
        'Rの上限が同じでも実際の大きさ・CPUは異なる。原著のfull workflowの再現とも呼ばない。', '',
        '## Energyと古典計算費用：正の差はTrimで悪化', '']
    for case in load(ROOT/'protocol-T1-NATIVE-TRIM.json')['cases']:
        items = [x for x in rows if x['case'] == case]
        lines += ['### '+case, '', '| source | 元のEnergy Ha | Trim Energy Ha | 差 mHa | 実CI次元 元→Trim | CPU秒 元→Trim全体 | 状態 |',
            '|---|---:|---:|---:|---:|---:|---|']
        for x in items:
            lines.append('| '+' | '.join([NAMES[x['method']], f(x['old_frequency_energy_Ha'], 8),
                f(x['native_Trim_energy_Ha'], 8), f(x['Trim_minus_frequency_mHa']),
                str(x['old_frequency_dimension'])+'→'+str(x['native_Trim_dimension'] or '—'),
                f(x['old_frequency_CPU_seconds'])+'→'+f(x['Trim_full_CPU_seconds']), x['status']])+' |')
        successes = [x for x in items if x['status'] == 'COMPLETED']
        if successes:
            low = min(successes, key=lambda x: x['native_Trim_energy_Ha'])
            improved = sum(x['Trim_minus_frequency_mHa'] < -1e-5 for x in successes)
            lines += ['', f"この条件では{len(successes)}件が数値完了し、頻度順より低いenergyは{improved}件。"
                f"Trim内の観測最小は{NAMES[low['method']]}。これは追加費用・spin・旧の全対照を無視した総合勝者ではない。", '']
    lines += ['## 同じ4つのscreenとの比較とspin', '',
        '| 条件 | source | 最良screen Ha | merge後 Ha | S² 元頻度→merge | Recovery % 元→merge |',
        '|---|---|---:|---:|---:|---:|']
    for x in rows:
        lines.append('| '+' | '.join([x['case'], NAMES[x['method']], f(x['matched_screen_energy_Ha'], 8),
            f(x['native_Trim_energy_Ha'], 8), f(x['old_frequency_S2'], 5)+'→'+f(x['native_Trim_S2'], 5),
            f(x['old_frequency_recovery_pct'], 2)+'→'+f(x['native_Trim_recovery_pct'], 2)])+' |')
    lines += ['', 'screen対照は同じ4計算の最良値だが、費用は選ばれた1回分ではなく4回分を課金した。'
        'Trimにはさらにmerge費用を課金。CSVのscreen kernel費用にはdriver/partition overheadを含めず、'
        'Trim全体にはそれらを含める。元R2費用は元の実測で、時刻をそろえたtiming replicateではない。', '',
        'Recoveryは相関エネルギー回復率で、ノイズによるconfiguration recoveryの回数/成功率ではない。'
        '新しいconfiguration recoveryは行っていない。S²は有限罰則であり、厳密singletではない。', '',
        '## この実験で変わらない回路資源', '',
        '| 条件 | source | CNOT | CNOT Depth | Total Depth | 元総model予算 / final | 古典draw |',
        '|---|---|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append('| '+' | '.join([x['case'], NAMES[x['method']], f(x['CNOT_unchanged'], 2),
            f(x['CNOT_depth_unchanged'], 2), f(x['total_depth_unchanged'], 2),
            str(x['original_total_model_budget'])+' / '+str(x['original_final_model_shots']),
            str(x['original_classical_draws'])])+' |')
    lines += ['', 'CNOT/Depth/元Parametersはpostprocessingだけでは変わらない。独立parameter rankは未認証なので、'
        'CSVでも未知を0へしない。CISDは量子samplerなしであり、相関状態を0 CNOTで量子準備した意味ではない。'
        'RAACSのpilotは元総予算に含め、CIには元のfinal観測だけを使った。', '',
        '## 判定と次の行動', '',
        f"この固定設定でTrimが頻度順より数値的に低いセルは{summary['numerically_lower_Trim_than_frequency']}/16。"
        f"計算全体は{terminal['CPU_seconds']:.2f} CPU秒 / {terminal['wall_seconds']:.2f} wall秒。",
        '実CI次元が上限16,384より大幅に小さくなった場合、上位25%を残す設定が重要な配置まで捨てた可能性がある。'
        '結果が悪いことを「Trimという方法自体が無意味」と一般化しない。一方、失敗を消して同じfreezeの閾値を変更もしない。',
        '次の小試験では、保存済み4screenのmarginal weightsを使って、予算を使い切るglobal rankingや、'
        'より緩いnative Trimのmergeを比較できる。screenを再実行しなくてもよいが、全screen費用は各比較armへ帰属させる。'
        '派生basisを先にfreezeし、新しいmergeのenergyを見て選ばない。原著HCIのarbitrary pair選択は依然別途未実施。', '',
        '公式実装への帰属：[Qiskit Trim guide](https://qiskit.github.io/qiskit-addon-sqd/guides/trim_sqd.html)、'
        '[Qiskit source](https://github.com/Qiskit/qiskit-addon-sqd)、[Merz et al.](https://arxiv.org/abs/2605.01138)。'
        '我々の新規手法やprotein/ExtSQD/SBD-Gのfull再現とは主張しない。']
    write(ROOT/'analysis/T1_NATIVE_TRIM_RESULTS_JA.md', ('\n'.join(lines)+'\n').encode())
    print(summary['statuses'], 'CI_SOLVES', summary['CI_solves_completed'], 'NESTED_ANOMALIES', len(summary['penalized_nested_space_anomalies']))


if __name__ == '__main__':
    main()
