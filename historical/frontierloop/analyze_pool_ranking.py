"""Attribution across equal final spaces without concealing inherited work."""
import csv
from io import StringIO
from pathlib import Path
from collections import Counter
from .paths import ROOT
from .io import load, save, write, sha256
from .pooled_selection_screen import DIRECTORY, FREEZE, binding
from .analyze_trim import NAMES
from .analyze import reference


def main():
    frozen = load(FREEZE); terminal = load(DIRECTORY/'terminal.json')
    if binding() != frozen or len(terminal['states']) != 16:
        raise RuntimeError('T2 not complete or fixed inputs changed')
    rows = []; per_source = []
    for p in frozen['plans']:
        source = p['source']; case, method = source['case'], source['method_id']
        old = load(p['T1_result_path']); freq = load(source['R2_baseline'])
        path = DIRECTORY/case/method/'terminal.json'; r = load(path)
        if r['frozen_selection_SHA256'] != p['selection_SHA256']:
            raise RuntimeError('T2 selected bases drift')
        hf, ref = reference(case)
        for arm in r['rows']:
            if arm['status'] == 'COMPLETED':
                if arm['CI_strings'] != p['selection']['arms'][arm['arm']] or arm['actual_CI_dimension'] != p['selection']['all_arms_have_same_actual_CI_dimension']:
                    raise RuntimeError('Matched candidate universe/final space check failed')
                if abs(arm['energy_Ha']+.1*arm['S2']-arm['penalized_energy_Ha']) > 1e-8:
                    raise RuntimeError('Physical/penalized root consistency failed')
                expect_screen = sum(x['CPU_seconds'] for x in old['calls'] if x['round'] == 0) if arm['arm'] != 'SAME_UNION_FREQUENCY' and p['selection']['screening_CI_required_operationally'] else 0.
                if abs(expect_screen-arm['required_screen_CI_CPU_seconds_attributed']) > 1e-8:
                    raise RuntimeError('Operationally unnecessary screening cost misattributed')
                if arm['evaluation_reused'] and not Path(arm['evaluation_origin'].split('#')[0]).exists():
                    raise RuntimeError('Reused numerical observation origin missing')
            rows.append({'case': case, 'method': method, 'arm': arm['arm'], 'status': arm['status'],
                'source_kind': source['kind'], 'candidate_universe_size': len(p['selection']['universe']),
                'actual_CI_dimension': arm.get('actual_CI_dimension'),
                'energy_Ha': arm.get('energy_Ha'), 'S2': arm.get('S2'),
                'recovery_pct': 100*(hf-arm['energy_Ha'])/(hf-ref) if arm['status'] == 'COMPLETED' else None,
                'old_R2_frequency_energy_Ha': freq['energy_Ha'], 'old_R2_dimension': freq['actual_CI_dimension'],
                'T1_quarter_Trim_energy_Ha': old['native_Trim_energy_Ha'],
                'accounted_numeric_CPU_seconds_not_cold_total': arm.get('accounted_operational_CI_CPU_seconds_not_cold_total'),
                'required_screen_CPU_seconds': arm.get('required_screen_CI_CPU_seconds_attributed'),
                'evaluation_reused': arm.get('evaluation_reused'), 'evaluation_origin': arm.get('evaluation_origin'),
                'source_final_model_shots': source['source_final_model_shots'],
                'source_total_budget_model_shots': source['source_total_budget'],
                'source_classical_draws': source.get('classical_draws', 0), 'new_model_shots': 0,
                'result_SHA256': sha256(path.read_bytes())})
        a = {x['arm']: x for x in r['rows']}
        g = a['GLOBAL_CI_MARGINAL']; f = a['SAME_UNION_FREQUENCY']; n = a['NATIVE_TRIM_RATIO1']
        per_source.append({'case': case, 'method': method, 'source_kind': source['kind'],
            'universe_size': len(p['selection']['universe']), 'all_statuses': [x['status'] for x in r['rows']],
            'all_arms_identical_basis': len({tuple(x) for x in p['selection']['arms'].values()}) == 1,
            'global_minus_same_union_frequency_mHa': 1000*(g['energy_Ha']-f['energy_Ha']) if g['status'] == f['status'] == 'COMPLETED' else None,
            'global_minus_native_ratio1_mHa': 1000*(g['energy_Ha']-n['energy_Ha']) if g['status'] == n['status'] == 'COMPLETED' else None,
            'global_minus_old_R2_frequency_mHa': 1000*(g['energy_Ha']-freq['energy_Ha']) if g['status'] == 'COMPLETED' else None,
            'new_roots': r['new_CI_roots'], 'new_CPU_seconds': r['new_CPU_seconds']})
    comparable = [x for x in per_source if x['global_minus_same_union_frequency_mHa'] is not None]
    def tally(field):
        return dict(Counter('LOWER' if x[field] < -1e-5 else 'HIGHER' if x[field] > 1e-5 else 'NUMERICALLY_EQUAL'
                           for x in per_source if x[field] is not None))
    summary = {'status_counts': dict(Counter(x['status'] for x in rows)), 'planned_arms': 48,
        'global_vs_same_union_frequency': tally('global_minus_same_union_frequency_mHa'),
        'global_vs_native_ratio1': tally('global_minus_native_ratio1_mHa'),
        'global_vs_old_R2_frequency': tally('global_minus_old_R2_frequency_mHa'),
        'identical_basis_sources_not_independent_wins': sum(x['all_arms_identical_basis'] for x in per_source),
        'new_completed_CI_roots': terminal['new_CI_roots'],
        'numerical_evaluation_reuse_rows': sum(bool(x['evaluation_reused']) for x in rows),
        'new_CPU_seconds': terminal['new_CPU_seconds'], 'new_wall_seconds': terminal['new_wall_seconds'],
        'previous_T1_CPU_seconds_not_erased': load(ROOT/'runs/T1-NATIVE-TRIM-v1/terminal.json')['CPU_seconds'],
        'per_source': per_source, 'new_model_shots': 0, 'actual_QPU_shots': 0, 'goal_complete': False}
    save(ROOT/'analysis/T2-pooled-ranking-summary.json', summary)
    stream = StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(ROOT/'analysis/T2-pooled-ranking-all.csv', stream.getvalue().encode('utf-8-sig'))
    fnum = lambda x, k=3: '—' if x is None else f'{x:.{k}f}'
    lines = ['# T2：同じ配置数で、何を使って順位付けすると良いか', '',
        f"48比較armの状態：{summary['status_counts']}。新たに解いたCIは{terminal['new_CI_roots']}件。"
        f"既存/同一basis評価の再利用は{summary['numerical_evaluation_reuse_rows']}行。追加量子Shotは0。", '',
        '## 今回切り分けたこと', '',
        'T1の上位25% trimでは全16セルが悪化した。今回は同じ4screenのinput配置unionを使い、'
        '3方式ともmin(128,unionの数)のspin stringを必ず残した。最終CI次元を同じにして、'
        '「単に配置を多く残したから」なのか「CI重要度の順位が効いたから」なのかを分ける。', '',
        '| 方式 | 配置の順位 | 運用時に4screenのCIが必要か |', '|---|---|---|',
        '| Global CI marginal | 4screenのα/β CI確率を等重みで平均し、全候補を一括順位付け | unionが128を超える場合のみ必要 |',
        '| 同じunionの頻度順 | 元の実観測回数で順位付け | 不要。余計なscreen費用を負担させない |',
        '| Native Trim ratio1 | 公式Trimの重み順位とbatch順merge、最終上限でcut | unionが128を超える場合のみ必要 |', '',
        '全候補を残せる場合は3方式が同じbasisになるので、追加screenを計算する理由もない。'
        '同じbasisの旧R2/T1 rootも再計算せず、元evaluationと費用を結び付けた。'
        'この再利用を独立replicateや新手法の無料CIとは扱わない。', '',
        '## 最終energy：同じCI次元の3方式', '']
    for case in load(ROOT/'protocol-T1-NATIVE-TRIM.json')['cases']:
        lines += ['### '+case, '', '| source | Global Ha | 頻度順 Ha | Native ratio1 Ha | 実CI次元 | 元R2との差 mHa（Global） |',
            '|---|---:|---:|---:|---:|---:|']
        for src in [x for x in per_source if x['case'] == case]:
            rr = {x['arm']: x for x in rows if x['case'] == case and x['method'] == src['method']}
            g, f, n = [rr[k] for k in ['GLOBAL_CI_MARGINAL', 'SAME_UNION_FREQUENCY', 'NATIVE_TRIM_RATIO1']]
            lines.append('| '+' | '.join([NAMES[src['method']], fnum(g['energy_Ha'], 8), fnum(f['energy_Ha'], 8),
                fnum(n['energy_Ha'], 8), str(g['actual_CI_dimension'] or '—'), fnum(src['global_minus_old_R2_frequency_mHa'])])+' |')
        lines += ['', '同じsource内では回路・CNOT・Depth・元の観測数は同じ。energyの差は配置選択側の差で、'
            '新しい量子samplerによる改善ではない。元R2は全観測からの頻度順なので、今回の小さい候補unionより広い場合がある。', '']
    lines += ['## 追加費用・spinも含む全比較', '',
        '| 条件 | source | 方式 | Recovery % | S² | 計上する数値CPU秒 | 評価の再利用 |',
        '|---|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append('| '+' | '.join([r['case'], NAMES[r['method']], r['arm'], fnum(r['recovery_pct'], 2),
            fnum(r['S2'], 5), fnum(r['accounted_numeric_CPU_seconds_not_cold_total']), str(r['evaluation_reused'])])+' |')
    lines += ['', '数値CPUには、必要な方式だけ過去4screenの実測CI費用を戻して計上した。'
        '頻度順controlはinput unionを元のrequest生成だけで作れるため、screenの対角化費用は不要。'
        'driver/partition/初期化・元source構築の費用を含むcold-start総費用ではない。'
        '実験campaignとしてはT1全80CIとT2の全新規root/失敗費用を保存し、勝った1armだけの費用にしない。', '',
        f"T2新規処理は{terminal['new_CPU_seconds']:.2f} CPU秒 / {terminal['new_wall_seconds']:.2f} wall秒。"
        f"以前のT1に{summary['previous_T1_CPU_seconds_not_erased']:.2f} CPU秒を使った事実も残す。", '',
        '## 判断の範囲', '',
        f"Globalと同じunion頻度順の数値比較：{summary['global_vs_same_union_frequency']}。"
        f"元R2頻度順との比較：{summary['global_vs_old_R2_frequency']}。",
        'これらは4物理条件の探索的診断で、16独立分子や新しい量子試行とは数えない。'
        '同じ候補を全て使うケースの同点は、CI情報の優位性を示す成功ではない。'
        'energyが良くても追加screen費用が大きければtrade-off。S²の増加だけでphysical energyが下がった可能性も別に見る。'
        '有限0.1Ha罰則であり、厳密singlet・global root数学証明・原著HCI/DICEの再現ではない。', '',
        '回路資源は[T1のsource表](T1_NATIVE_TRIM_RESULTS_JA.md)と同じ。G3のSqDRIFT5削減を、'
        '今回のSqDRIFT20に検証せず結び付けない。新規性や最強論文への総合優越はまだ未達。']
    write(ROOT/'analysis/T2_POOLED_SELECTION_RESULTS_JA.md', ('\n'.join(lines)+'\n').encode())
    print(summary['status_counts'], 'NEW_ROOTS', terminal['new_CI_roots'], 'GLOBAL_VS_FREQ', summary['global_vs_same_union_frequency'])


if __name__ == '__main__':
    main()
