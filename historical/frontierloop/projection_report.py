"""Report the whole fixed prefix, including adverse resource trade-offs."""
import csv
from io import StringIO
from collections import Counter
from .paths import ROOT
from .io import load, save, write


def run():
    result = load(ROOT/'runs/G2-FROZEN-OCCUPATION-v1/terminal.json')
    rows = [x for x in result['rows'] if x['status'] == 'RESOURCE_PREFLIGHT_COMPLETE']
    fields = ['cnot_count', 'cnot_depth', 'total_unitary_depth']
    summary = []
    flat = []
    for case in sorted({x['case'] for x in rows}):
        items = [x for x in rows if x['case'] == case]
        record = {'case': case, 'members': len(items), 'active_qubits_min': min(x['active_qubits'] for x in items),
            'active_qubits_max': max(x['active_qubits'] for x in items)}
        for key in ['native_recount_resources', 'same_layout_Rustiq_resources',
                    'projected_resources', 'best_control_per_metric_envelope']:
            record[key] = {f: sum(x[key][f] for x in items)/len(items) for f in fields}
        record['per_metric_vs_envelope'] = {f: dict(Counter(
            'better' if x['projected_resources'][f] < x['best_control_per_metric_envelope'][f]
            else 'worse' if x['projected_resources'][f] > x['best_control_per_metric_envelope'][f]
            else 'equal' for x in items)) for f in fields}
        summary.append(record)
        for x in items:
            row = {k: x[k] for k in ['case', 'index', 'member_id', 'active_qubits', 'frozen_qubits']}
            for key in ['native_recount_resources', 'same_layout_Rustiq_resources', 'projected_resources']:
                row.update({key+'_'+f: x[key][f] for f in fields})
            row['probability_L1'] = x['independent_probability'].get('probability_L1')
            flat.append(row)
    save(ROOT/'analysis/G2-prefix-summary.json', {'cases': summary,
        'all_terminal_statuses': dict(Counter(x['status'] for x in result['rows'])),
        'independent_probability_checks': dict(Counter(x['independent_probability']['status'] for x in rows)),
        'max_probability_L1': max(x['independent_probability'].get('probability_L1', 0.) for x in rows),
        'new_model_shots': 0, 'new_energy_evaluations': 0,
        'CPU_seconds': result['incremental_CPU_seconds'], 'wall_seconds': result['incremental_wall_seconds'],
        'goal_complete': False})
    out = StringIO(); writer = csv.DictWriter(out, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    write(ROOT/'analysis/G2-prefix-all.csv', out.getvalue().encode())
    lines = ['# G2：測定分布を変えない固定占有mode削減の小試験', '',
        '追加Shot・energy・CIはすべて0。固定した先頭16 member specs×3条件、48/48件が完了。',
        'NAPH20-BLOCK-BOYSの同一問題N5 sourceは欠測として保持し、別軌道・別長さへ代用していない。', '',
        '## まず結論', '',
        '回路の動かない軌道を消す処理は実装できた。48件すべてで、独立したQiskit回路の確率と、'
        '元のsource Pauli列を使うsparse計算を照合し、最大L1誤差は'
        f"{max(x['independent_probability'].get('probability_L1', 0.) for x in rows):.3g}だった。"
        'ただし、resourceが悪くなる個別memberもある。これは全bankの結果ではない。', '',
        '| 条件 | 元native CNOT / CNOT Depth / Total Depth | 同じlayoutのRustiq | 固定占有削減 |',
        '|---|---:|---:|---:|']
    for x in summary:
        cells = [' / '.join(f"{x[k][f]:.3f}" for f in fields) for k in
            ['native_recount_resources', 'same_layout_Rustiq_resources', 'projected_resources']]
        lines.append('| '+x['case']+' | '+' | '.join(cells)+' |')
    lines += ['', '表は固定prefixの算術平均。各memberの旧Shot割当は128で同じだが、今回は測定していない。'
        'この平均を全100k-shot ensembleの平均に流用しない。', '',
        'nativeからの大幅削減の相当部分は、既存Rustiqへの変更でも実現した。'
        '提案の追加効果を示すにはRustiqを対照に残す必要がある。', '',
        '## 都合の悪い結果も含めた内訳', '',
        '| 指標 | 両対照の良い方より減少 | 同値 | 増加 |', '|---|---:|---:|---:|']
    for f in fields:
        c = Counter('better' if x['projected_resources'][f] < x['best_control_per_metric_envelope'][f]
                    else 'worse' if x['projected_resources'][f] > x['best_control_per_metric_envelope'][f]
                    else 'equal' for x in rows)
        lines.append(f"| {f} | {c['better']} | {c['equal']} | {c['worse']} |")
    lines += ['', '「各指標で両対照の良い方」は比較用の厳しいenvelopeであり、'
        '一つの実在する対照回路とは限らない。全件の値はG2-prefix-all.csvに保存。', '',
        '## 成立する範囲', '',
        '- 全groupを通してX/Yが作用しないmodeだけを固定。最後に動くmodeを途中で固定しない。',
        '- JWのfermionic relabelは単なるqubit swapではない。FSWAP Cliffordを用い、独立OpenFermionのJWと照合済み。',
        '- 理想HFと固定占有sectorでの同値。任意入力unitary同値・noise下のchannel同値ではない。',
        '- 独立最適化変数は元から0。Pauli角度slotの減少を独立parameter数の削減と主張しない。',
        '- 48件のactive registerは0〜10q。量子優位性や大規模波動関数のシミュレーション成功を意味しない。',
        '- compiler技術の既知部分が多く、科学的新規性・最強原著への総合優越は未認定。', '',
        '## 費用と次の限定試験', '',
        f"実prefix処理は{result['incremental_CPU_seconds']:.3f} CPU秒 / "
        f"{result['incremental_wall_seconds']:.3f} wall秒。準備・実装・資格試験の費用はこの数値に含めない。",
        'full suiteはqualification-G2-current-full-tests.xmlで111 passed。'
        'それだけで全研究の科学的正しさを保証したとは扱わない。', '',
        '次は同じN5 sourceの全783×3 membersを対象に、結果を見ずに固定するresource-only policyで点検する。'
        'まずnative/RustiqからCNOT優先で対照を選び、projectedがCNOT・CNOT Depth・Total Depthの'
        'どれも増やさない場合だけ採用する。それ以外は対照へ戻す。'
        '各memberの元Shot重みで全bankを集計する。新しいenergy・Shot・CIは行わず、旧countsの'
        '再利用や元energyとの結合は全member同値・source bindingを確認できた範囲だけに限定する。',
        '新しいfreeze/費用上限を作り、既存G2 prefix・source・code・failureを変更しない。']
    write(ROOT/'analysis/G2_FROZEN_OCCUPATION_RESULTS_JA.md', ('\n'.join(lines)+'\n').encode())


if __name__ == '__main__':
    run()
