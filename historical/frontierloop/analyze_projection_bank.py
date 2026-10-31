"""Whole-bank weighted resource report and explicitly certified observation reuse."""
import csv
import gzip
from io import BytesIO, StringIO
import json
import sqlite3
from time import process_time, perf_counter
from pathlib import Path
from collections import Counter
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load, save, write, sha256
from .projection_compile import choose, FIELDS
from paperstudy.lucj import circuit_views


def run():
    cpu, wall = process_time(), perf_counter()
    directory = ROOT/'runs/G3-FULLBANK-PROJECTION-v1'
    terminal = load(directory/'terminal.json')
    freeze = load(ROOT/'freeze-G3-FULLBANK-PROJECTION.json')
    database = directory/'compiled-members.sqlite'
    if sha256(database.read_bytes()) != terminal['database_SHA256']:
        raise RuntimeError('Terminal database changed')
    csv_path = ROOT/'analysis/R2-common-CI-all.csv'
    if sha256(csv_path.read_bytes()) != freeze['R2_comparison_csv_SHA256']:
        raise RuntimeError('Frozen old R2 comparison changed')
    old_ci = list(csv.DictReader(csv_path.open(encoding='utf-8-sig')))
    db = sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True)
    records = []
    validation = Counter()
    try:
        for case, index, mid, payload, certificate, compiled in db.execute('SELECT * FROM members ORDER BY case_id,idx'):
            r = json.loads(payload)
            if r['status'] != 'CERTIFIED_RESOURCE_POINT':
                raise RuntimeError('Do not reuse outcomes for a partially certified bank')
            if (r['case'], r['index'], r['member_id']) != (case, index, mid):
                raise RuntimeError('Database identity mismatch')
            if sha256(certificate) != r['certificate_SHA256'] or sha256(compiled) != r['selected_QPY_gz_SHA256']:
                raise RuntimeError('Stored certificate/QPY bytes changed')
            cert = json.loads(gzip.decompress(certificate))
            if cert['original_member_id'] != mid or r['policy'] != choose(
                    r['resources']['NATIVE'], r['resources']['RUSTIQ'], r['resources']['PROJECTED']):
                raise RuntimeError('Certificate/policy binding mismatch')
            qc = qpy.load(BytesIO(gzip.decompress(compiled)))[0]
            _, _, views = circuit_views(qc)
            if views['Z_sampling_equivalent'] != r['policy']['selected_resources']:
                raise RuntimeError('Saved actual circuit resource recount mismatch')
            records.append(r); validation['saved_QPY_recount'] += 1
            validation[r['independent_probability']['status']] += 1
    finally:
        db.close()
    tail_path = ROOT/'qualification/G3-projection-independent-tail-v1/result.json'
    tail = load(tail_path) if tail_path.exists() else None
    tail_rows = {(x['case'], x['index']): x for x in (tail or {}).get('rows', [])}
    if tail and (not tail['original_G3_records_unchanged'] or tail['status'] != 'ALL_PREVIOUSLY_SKIPPED_TAIL_CHECKS_PASSED'):
        raise RuntimeError('Additional tail qualification failed')
    summary = []
    flat = []
    for source in freeze['sources']:
        case = source['case']
        if source['status'] != 'AVAILABLE':
            summary.append({'case': case, 'status': source['status'], 'energy_Ha': None})
            continue
        group = [r for r in records if r['case'] == case]
        if len(group) != len(source['members']) or len(group) != 783:
            raise RuntimeError('Whole source bank not certified')
        if sha256(Path(source['source_result']).read_bytes()) != source['source_result_SHA256'] or sha256(
                Path(source['source_counts_path']).read_bytes()) != source['source_counts_SHA256']:
            raise RuntimeError('Original source outcomes changed')
        with np.load(source['source_counts_path'], allow_pickle=False) as a:
            final_shots = int(a['counts'].sum())
        if final_shots != 100000 or sum(r['original_member_shot_weight'] for r in group) != final_shots:
            raise RuntimeError('Old counts/whole-member shot weights mismatch')
        for r, member in zip(sorted(group, key=lambda x: x['index']), source['members']):
            if (r['member_id'], r['original_member_shot_weight'], r['source_artifacts']) != (
                    member['member_id'], member['historical_member_shots'], member['artifacts']):
                raise RuntimeError('Per-member source binding mismatch')
        weights = [r['original_member_shot_weight'] for r in group]
        def resource(r, name):
            return r['resources'][name] if name in r['resources'] else r['policy'][
                'baseline_resources' if name == 'BASELINE' else 'selected_resources']
        means = {name: {f: sum(w*resource(r, name)[f] for r, w in zip(group, weights))/final_shots
                 for f in FIELDS} for name in ['NATIVE', 'RUSTIQ', 'PROJECTED', 'BASELINE', 'POLICY']}
        old = next(x for x in old_ci if x['case'] == case and x['method'] == 'C5_D_SQDRIFT_N5')
        if old['status'] != 'COMPLETED' or int(old['R_cap']) != 128 or int(old['final_model_shots']) != final_shots:
            raise RuntimeError('Old R2 endpoint/context mismatch')
        for f, k in zip(FIELDS, ['CNOT', 'CNOT_depth', 'total_depth']):
            if abs(means['NATIVE'][f]-float(old[k])) > 1e-9:
                raise RuntimeError('Whole-bank native recount does not reproduce original resources')
        numerical_checks = [r['independent_probability'] if r['independent_probability']['status'].endswith('_PASSED')
             else tail_rows.get((case, r['index']), {}).get('independent_probability') for r in group]
        errors = [x['probability_L1'] for x in numerical_checks if x is not None]
        supports = [x['original_state_numerics']['physical_support'] if x is not None else None for x in numerical_checks]
        if any(x is None for x in supports):
            raise RuntimeError('Do not claim full independent coverage without the additive tail check')
        source_sparse_summary = {'maximum_member_nonzero_configuration_support': max(supports),
            'single_configuration_member_shot_weight_fraction': sum(w for w, k in zip(weights, supports) if k == 1)/final_shots,
            'single_configuration_is_not_assumed_to_be_HF': True}
        increases = {f: sum(r['policy']['selected_resources'][f] > r['resources']['NATIVE'][f] for r in group) for f in FIELDS}
        row = {'case': case, 'status': 'WHOLE_BANK_CERTIFIED_RESOURCE_AND_OLD_OBSERVATION_REUSE',
            'members': len(group), 'model_shot_weight': final_shots, 'new_model_shots': 0,
            'mean_resources': means, 'selected_member_counts': dict(Counter(r['policy']['selected'] for r in group)),
            'reductions_vs_native_pct': {f: 100*(1-means['POLICY'][f]/means['NATIVE'][f]) for f in FIELDS},
            'additional_reductions_vs_strong_compiler_pct': {f: 100*(1-means['POLICY'][f]/means['BASELINE'][f]) for f in FIELDS},
            'old_R2_energy_Ha_reused_not_reevaluated': float(old['energy_Ha']),
            'old_R2_recovery_pct_reused_not_reevaluated': float(old['recovery_pct']),
            'old_R2_S2_reused_not_recertified': float(old['S2']),
            'old_R2_actual_CI_dimension': int(old['actual_CI_dimension']),
            'old_R2_result_SHA256': old['result_SHA256'],
            'old_R2_incremental_CI_CPU_seconds_not_new_work': float(old['incremental_CI_CPU_seconds']),
            'source_final_counts_SHA256': source['source_counts_SHA256'],
            'same_finite_shot_observations_not_independent_confirmation': True,
            'independent_probability_checked_members_including_tail': len(errors),
            'max_probability_L1': max(errors) if errors else None,
            'projected_active_register_max': max(r['projected_active_qubits'] for r in group),
            'selected_register_mean': sum(w*r['selected_qubits'] for r, w in zip(group, weights))/final_shots,
            'source_sparse_simulability_diagnostic': source_sparse_summary,
            'observed_selected_member_increases_vs_native': increases,
            'optimized_variables_before_and_after': 0, 'independent_parameter_reduction_claimed': False}
        summary.append(row)
        for r in group:
            item = {k: r[k] for k in ['case', 'index', 'member_id', 'original_member_shot_weight', 'selected_qubits']}
            item['policy_selected'] = r['policy']['selected']
            for name in means:
                item.update({name+'_'+f: resource(r, name)[f] for f in FIELDS})
            flat.append(item)
    save(ROOT/'analysis/G3-fullbank-summary.json', {'cases': summary, 'validation': dict(validation),
        'additional_independent_tail': None if tail is None else tail['status'],
        'compiled_members': len(records), 'database_bytes': database.stat().st_size,
        'fullbank_CPU_seconds': terminal['CPU_seconds'], 'fullbank_wall_seconds': terminal['wall_seconds'],
        'recount_analysis_CPU_seconds': process_time()-cpu, 'recount_analysis_wall_seconds': perf_counter()-wall,
        'new_model_shots': 0, 'new_energy_or_CI_evaluations': 0, 'goal_complete': False})
    output = StringIO(); writer = csv.DictWriter(output, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write(ROOT/'analysis/G3-fullbank-all-members.csv', output.getvalue().encode())
    lines = ['# G3：SqDRIFT5全bankの資源削減', '',
        '## 判断', '',
        '**3条件×783 member specs＝2,349件すべてを点検し、元の理想測定分布を変えないまま、'
        'CNOT・CNOT Depth・Total Depthを下げる固定compiler policyを構築できた。** '
        'ただし、これは今回の共通条件SqDRIFT adaptationのcompiler改善。'
        '最強論文全体や古典CIに勝った証拠、独立した分子精度改善、新しい状態準備アルゴリズムとはしない。', '',
        '## 元native実装との比較：同じ100k Shotと同じR128読出し', '',
        '| 条件 | CNOT 元→削減 | CNOT Depth 元→削減 | Total Depth 元→削減 | 旧energy Ha（再利用） | 旧Recovery %（再利用） |',
        '|---|---:|---:|---:|---:|---:|']
    available = [x for x in summary if 'mean_resources' in x]
    for x in available:
        v = x['mean_resources']; cells = [f"{v['NATIVE'][f]:.2f}→{v['POLICY'][f]:.2f}（−{x['reductions_vs_native_pct'][f]:.1f}%）" for f in FIELDS]
        lines.append('| '+x['case']+' | '+' | '.join(cells)+f" | {x['old_R2_energy_Ha_reused_not_reevaluated']:.10f} | {x['old_R2_recovery_pct_reused_not_reevaluated']:.3f} |")
    lines += ['', '小数は元の各memberのShot配分で重み付けした平均。回路1個に小数個のゲートがある意味ではない。'
        '旧countsと同じ測定分布であることに基づき、既存R2の値を再利用した。新しいenergy計算・CI・Shotは0。'
        '新たな独立seedで同じenergyを観測したという意味ではない。', '',
        '## 強い既存compilerにも同じ機会を与えた比較', '',
        '| 条件 | 対照compiler CNOT→policy | CNOT Depth | Total Depth | 採用された削減回路 / 783 |',
        '|---|---:|---:|---:|---:|']
    for x in available:
        v = x['mean_resources']; cells = [f"{v['BASELINE'][f]:.2f}→{v['POLICY'][f]:.2f}（追加−{x['additional_reductions_vs_strong_compiler_pct'][f]:.1f}%）" for f in FIELDS]
        lines.append('| '+x['case']+' | '+' | '.join(cells)+f" | {x['selected_member_counts'].get('PROJECTED', 0)} |")
    lines += ['', '対照は各memberでnative SDKと同じlayoutのRustiqを比較し、CNOT→CNOT Depth→Total Depthの順に固定選択。'
        'projectedが3指標のどれも増やさず少なくとも一つ減らす場合だけ採用。それ以外はその対照回路を維持した。'
        '回路ごとのenergyやFCI参照は選択に使っていない。'
        'この対照選択自体も追加compiler作業であり、固定された元native実装とは別の強化対照として示した。', '',
        '最初の48件ではCNOTを増やしたもの1件、Depthを増やしたもの2件もあった。'
        '都合の悪いものを削除せず、全bankでは固定guardのfallbackとして保存した。', '',
        '## 正しさと保存', '',
        f"- 全2,349件のsymbolic証明、旧source/QPY digest、Shot重みを検査。実際に保存した選択QPYも2,349件すべて再計数した。",
        '- 元G3ではactive16q以下の2,348件を独立Qiskitと照合。残った18qの1件は元の上限・記録を変えず、別の追加資格試験へ分離。',
        f"- 追加資格試験を含む最大確率L1差：{max(x['max_probability_L1'] for x in available):.3g}。",
        '- 元nativeの全bank資源平均も旧R2表に一致。同じ問題・軌道・100k Shot・R128・spin penaltyの旧結果だけを結び付けた。',
        f"- 全bank処理：{terminal['CPU_seconds']:.2f} CPU秒 / {terminal['wall_seconds']:.2f} wall秒。実装・事前資格試験・旧source作成・既存CI費用は別で、この値を総費用とは呼ばない。",
        f"- 選択QPYと証明は単一SQLite、{database.stat().st_size/2**20:.2f} MiB。statevectorは保存していない。",
        '- 追加の再計数監査CPUはG3-fullbank-summary.jsonに分けた。A100は使っていない。小さなcompiler処理はM4で約数十秒のため。', '',
        '## 大事な限界', '',
        '1. NAPH20-BLOCK-BOYSのN5 sourceは欠測。別問題・別距離・別軌道の成功例へ置き換えていない。',
        '2. 固定した理想HF/占有sectorでの測定分布保存であり、任意入力のunitary同値やnoise下のchannel同値ではない。',
        '3. N5は短いmember ensemble。最大projected registerは18qで、小さいsparse状態が多い。40qの一般波動関数をシミュレートできた、または量子優位性を示したとはしない。',
        '4. 最適化変数は元から0。Pauli係数slotの削減を独立パラメータ数の削減とは言わない。',
        '5. 初期状態を利用する[Quantum Constant Propagation](https://doi.org/10.1007/978-3-031-44245-2_9)や、'
        '測定結果だけを保つ[PCOAST](https://arxiv.org/abs/2305.10966)は既知。今回の実装だけで新規性を主張できない。前者は公開abstract/intro、後者はabstractを確認した段階で、両toolの再現比較は未実施。',
        '6. [IBM SqDRIFT tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/sqdrift)の対角項filterはHamiltonian samplingの分布を変える。'
        '今回の各固定member内の定数占有射影とは別で、同じ実験として混ぜない。', '',
        '## 次の意思決定', '',
        'この工程は小費用で実際の資源を減らせたので、今後のSqDRIFT強化対照へ組み込める。'
        '一方、R2で残った精度差は解消していない。次はHCI/Trim型の配置選択をsampler間で対称に比較するか、'
        'より精度の高いN20/50にも同じ固定compiler処理が残るかを小さく調べる。'
        'native原著driver・十分なTN/HCI・独立確認を欠いたまま「最強論文に勝った」としない。']
    lines += ['', '## 数字を大きく見せないための追加点検', '',
        '| 条件 | 1 memberの最大nonzero配置数 | 単一配置memberに割り当てられたShot比率 |',
        '|---|---:|---:|']
    for x in available:
        d = x['source_sparse_simulability_diagnostic']
        lines.append(f"| {x['case']} | {d['maximum_member_nonzero_configuration_support']} | {100*d['single_configuration_member_shot_weight_fraction']:.2f}% |")
    lines += ['', 'ここで単一配置をすべてHFと決め付けてはいない。この短いN5の個々の状態は非常にsparseであり、'
        '今回の3条件は従来の古典sparse kernelでも容易に扱える。'
        '元の40qという物理ラベルや50%超のCNOT削減だけを根拠に、難しい量子状態での優位性を主張しない。',
        'また、全2,349件の保存出力で元nativeよりCNOT・CNOT Depth・Total Depthが増えたmemberは、'
        '実際の再計数ではいずれも0件だった。これはこの固定bankで観測した事実であり、'
        '任意の将来bankでもnativeに対して常に非増加となる一般保証とは分ける。']
    write(ROOT/'analysis/G3_FULLBANK_PROJECTION_RESULTS_JA.md', ('\n'.join(lines)+'\n').encode())
    print('G3_REPORT_VERIFIED', len(records), 'saved QPYs; no new quantum/CI evaluations')


if __name__ == '__main__':
    run()
