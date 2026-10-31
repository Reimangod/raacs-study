"""Recount every actual G4 QPY; never join prefix resources to bank energy."""
from collections import Counter
import csv
import gzip
from io import BytesIO, StringIO
import json
import sqlite3
from .paths import ROOT
from .io import load, save, write, sha256
from .strong_projection import RUN, FREEZE, FIELDS, binding, choose_safe
from qiskit import qpy
from paperstudy.lucj import circuit_views


def analyze():
    frozen, terminal = load(FREEZE), load(RUN/'terminal.json')
    if binding() != frozen or terminal['freeze_SHA256'] != sha256(FREEZE.read_bytes()):
        raise ValueError('G4 binding drift')
    if terminal['database_SHA256'] != sha256((RUN/'compiled-members.sqlite').read_bytes()):
        raise ValueError('G4 saved database changed')
    db = sqlite3.connect('file:'+str(RUN/'compiled-members.sqlite')+'?mode=ro', uri=True)
    if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]: raise RuntimeError('DB integrity')
    rows = []; flat = []; recounts = 0; failures = []
    for case, idx, mid, record, cert, native, rustiq, projected in db.execute('SELECT * FROM members ORDER BY case_id,idx'):
        r = json.loads(record)
        if r['status'] != 'CERTIFIED_RESOURCE_POINT':
            failures.append(r); continue
        source = next(x for x in frozen['sources'] if x['case'] == case)
        plan = source['members'][idx]
        if mid != plan['member_id'] or r['spec'] != plan['spec'] or r['problem_id'] != source['problem_id']:
            raise ValueError('Member/source context binding failed')
        if sha256(cert) != r['certificate_SHA256']: raise ValueError('Certificate digest')
        if choose_safe(r['resources']) != r['policy']: raise ValueError('Registered guard drift')
        for name, blob in [('NATIVE', native), ('RUSTIQ', rustiq), ('PROJECTED', projected)]:
            if sha256(blob) != r['qpy_gz_SHA256'][name]: raise ValueError('QPY digest')
            qc = qpy.load(BytesIO(gzip.decompress(blob)))[0]
            # Recount the saved sampling circuit directly using the same compiler.
            _, sample, views = circuit_views(qc)
            if views['Z_sampling_equivalent'] != r['resources'][name]:
                raise ValueError('Actual saved QPY recount mismatch')
            if r['independent_check']['circuits'][name]['status'] != 'INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED':
                raise ValueError('Missing independent actual-circuit check')
            recounts += 1
        for k in FIELDS:
            if r['policy']['selected_resources'][k] > r['resources']['NATIVE'][k]:
                raise ValueError('Native metric increased')
        rows.append(r)
        row = dict(case=case,index=idx,member_id=mid,shot_weight=r['original_member_shot_weight'],
            original_qubits=r['original_qubits'], active_qubits=r['active_qubits'],
            source_support=r['source_sparse_support'],selected=r['policy']['selected'],baseline=r['policy']['baseline'],
            coefficient_slots_original=r['coefficient_slots_original'],
            coefficient_slots_restricted=r['coefficient_slots_restricted'],
            optimized_variables_before_and_after=0, independent_parameter_reduction_claimed=False)
        for name in ['NATIVE', 'RUSTIQ', 'PROJECTED']:
            for k in FIELDS: row[name+'_'+k] = r['resources'][name][k]
        for k in FIELDS: row['SELECTED_'+k] = r['policy']['selected_resources'][k]
        flat.append(row)
    db.close()
    result = dict(scope='FIXED_FIRST16_N20_MEMBERS_PER_CASE_NOT_FULL_BANK', terminal=terminal,
        certified_members=len(rows), failed_members=len(failures), failures=failures,
        actual_saved_QPY_recounts=recounts, cases=[], new_model_shots=0, new_QPU_shots=0,
        new_energy_CI_evaluations=0, energy_join=False, strongest_paper_superiority_proven=False)
    for source in frozen['sources']:
        subset = [x for x in rows if x['case'] == source['case']]
        total = sum(x['original_member_shot_weight'] for x in subset)
        means = {name: {k: sum(x['original_member_shot_weight']*(
            x['resources'][name][k] if name in ['NATIVE','RUSTIQ','PROJECTED'] else
            x['policy']['baseline_resources' if name == 'CONTROL_POLICY' else 'selected_resources'][k])
            for x in subset)/total for k in FIELDS} for name in
            ['NATIVE','RUSTIQ','PROJECTED','CONTROL_POLICY','SELECTED_POLICY']} if total else {}
        summary = dict(case=source['case'], completed=len(subset), planned=len(source.get('members', [])),
            original_bank_members=source.get('original_bank_members'), prefix_shot_weight=total,
            total_bank_shots=100000, prefix_time_values=sorted({x['spec']['time'] for x in subset}),
            means_within_prefix_only=means, selection=Counter(x['policy']['selected'] for x in subset),
            baseline=Counter(x['policy']['baseline'] for x in subset),
            excluded_rustiq_tradeoffs=sum(x['policy']['rustiq_count_depth_tradeoff_excluded'] for x in subset))
        if subset:
            summary.update(active_qubits_range=[min(x['active_qubits'] for x in subset),max(x['active_qubits'] for x in subset)],
                source_support_range=[min(x['source_sparse_support'] for x in subset), max(x['source_sparse_support'] for x in subset)],
                max_probability_L1=max(c['probability_L1_with_leakage_bound'] for x in subset
                    for c in x['independent_check']['circuits'].values()),
                max_final_MPS_bond=max(c['max_final_bond_dimension'] for x in subset
                    for c in x['independent_check']['circuits'].values()),
                reduction_percent_vs_native={k: 100*(1-means['SELECTED_POLICY'][k]/means['NATIVE'][k]) for k in FIELDS},
                additional_reduction_percent_vs_control={k: 100*(1-means['SELECTED_POLICY'][k]/means['CONTROL_POLICY'][k]) for k in FIELDS})
        result['cases'].append(summary)
    save(ROOT/'analysis/G4-strong-projection-summary.json', result)
    if flat:
        s = StringIO(); w = csv.DictWriter(s, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
        write(ROOT/'analysis/G4-strong-projection-members.csv', s.getvalue().encode())
    text = ['# G4：SqDRIFT20の固定prefix16回路を圧縮できるか', '',
        '## 結論と対象', '',
        f"全4条件の固定先頭16件、合計64件のうち{len(rows)}件で独立確率検算を通過、失敗{len(failures)}件。保存した実回路QPYを{recounts}件再計数した。これは各783件のうち先頭16件だけであり、全bankの性能結果ではない。新Shot・energy・CIはすべて0。", '',
        '最初から全条件の同じ先頭16件を指定した。良いenergyや削減例を見て選んでいない。ただし先頭はtime=1の一部なので、time=2/3を含む全bankの無作為代表ではない。下の平均はprefix内の元Shot重み平均で、全bank平均ではない。', '',
        '| 条件 | 元native CNOT → 出力 | CNOT削減 | CNOT Depth → 出力 | Depth削減 | Total Depth → 出力 | Total削減 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in result['cases']:
        if not s['completed']: continue
        m=s['means_within_prefix_only']; p=s['reduction_percent_vs_native']
        text.append(f"| {s['case']} | {m['NATIVE']['cnot_count']:.2f} → {m['SELECTED_POLICY']['cnot_count']:.2f} | {p['cnot_count']:.1f}% | {m['NATIVE']['cnot_depth']:.2f} → {m['SELECTED_POLICY']['cnot_depth']:.2f} | {p['cnot_depth']:.1f}% | {m['NATIVE']['total_unitary_depth']:.2f} → {m['SELECTED_POLICY']['total_unitary_depth']:.2f} | {p['total_unitary_depth']:.1f}% |")
    text += ['', '同じ入力・同じ物理layoutで、既存Rustiqにもコンパイルの機会を与えた。元nativeの三資源を増やさないRustiqだけを対照に採用し、その対照より三資源が非増加で一つ以上減る場合だけprojectedへ変更する。したがって、CNOTだけ減らしてDepthを増やす出力は採らない。旧G3のCNOT優先guardは不変で、G4にだけ保守的guardを登録した。', '',
        '| 条件 | 強化対照のCNOT | 強化対照から追加CNOT削減 | 追加CNOT Depth削減 | 追加Total削減 | projected採用/16 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in result['cases']:
        if not s['completed']: continue
        p=s['additional_reduction_percent_vs_control']
        text.append(f"| {s['case']} | {s['means_within_prefix_only']['CONTROL_POLICY']['cnot_count']:.2f} | {p['cnot_count']:.1f}% | {p['cnot_depth']:.1f}% | {p['total_unitary_depth']:.1f}% | {s['selection'].get('PROJECTED',0)} |")
    text += ['', '## 確率検算と状態の複雑さ', '',
        '| 条件 | active qubit範囲 | 元source支持配置数 | 最大MPS bond | 最大確率L1誤差（漏れを含む） |',
        '|---|---:|---:|---:|---:|']
    for s in result['cases']:
        if not s['completed']: continue
        text.append(f"| {s['case']} | {s['active_qubits_range']} | {s['source_support_range']} | {s['max_final_MPS_bond']} | {s['max_probability_L1']:.3g} |")
    text += ['',
        '検算は元Pauli sequenceからの独立sparse状態と、保存したnative/Rustiq/projected各実回路のAer MPSを比較した。MPSの打切り閾値0・bond上限なし・double精度。小系の複素振幅をStatevectorと照合し、実分子では元支持の全配置確率とMPS全ノルムを計算し、支持外の漏れも含めた。MPSや巨大statevectorは保存していない。', '',
        'これは理想的なHF起点の測定分布を数値許容幅内で保存した結果。全入力に対するunitary一致・任意のnoise下の一致・一般的40q状態の効率的古典simulationを保証しない。支持配置数とMPS bondも上表へ出し、40qという見出しだけを強調しない。', '',
        '## 費用・Parameter・energyの境界', '',
        f"G4本実行は子processを含め{terminal['CPU_seconds_including_children']:.2f} CPU秒、{terminal['wall_seconds']:.2f} wall秒。元source構築・以前のShot/CI費用は別に残す。新コード開発・資格試験・freeze/hash解決・下のQPY再計数の費用まで含めた総研究費用ではない。検算はM4上で実行し、A100の新規確保・共有環境変更・研究データ削除は行っていない。", '',
        'SqDRIFTは今回VQE最適化変数0の固定乱数回路。Pauli係数slotの削減と独立Ansatzパラメータの減少は異なる。独立rank低下は主張せず、slot前後値はmember CSV/recordに保存する。CNOT/Depthはall-to-all論理資源であり、実機native2Q/routing/noiseは未評価。', '',
        '**T2のN20 energy/Recoveryとこのprefix削減率を組み合わせた性能表は作らない。** 全783件を証明したものではなく、分布全体の保存はまだ未実証だからである。G3のN5削減率をN20へ流用することも禁止。', '',
        '## 次に進む条件', '',
        '全4条件の固定prefixで追加効果と検算がそろった場合、同じguardの全N20 bankを新しい別freezeで確認する余地がある。先にQPY/証明/数値検算の反復startup費用を見積もり、保存はcompact DBを維持する。全bankがそろってからのみ、旧counts/CIのcertificate-derived reuseを明記して精度と資源を同じ行へ結び付ける。これは独立した新測定の確認とは別。', '',
        'その後も固定Global選択の別subsampling seed（同じcountsのため独立Quantum replicateではない）、より強い本来のTN/HCI対照、古典費用を含む比較が必要。genericな初期状態依存compiler最適化は既知で、新規性・最強論文への総合優越は未証明。RAACSへ固執して小beamを増やす方針には戻さない。', '',
        '## 根拠と再現物', '',
        '- [Aer MPS公式説明](https://qiskit.github.io/qiskit-aer/tutorials/7_matrix_product_state_method.html)：MPSを状態表現に使う既知simulation。今回の実装では近似打切りを無効化し、別backend/sparseと照合。',
        '- [Aer MPS save API](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.library.save_matrix_product_state.html)：MPSをRAMへ取得してノルム/必要振幅を検算。',
        '- [SqDRIFT原著v3](https://arxiv.org/html/2508.02578v3)：Hamiltonian群の回路、電子対称性とfermion-to-qubit layoutを維持した共通setup。原著全driverの完全再現とは呼ばない。',
        '- protocol-G4-STRONG-PROJECTION.json / freeze-G4-STRONG-PROJECTION.json',
        '- runs/G4-STRONG-PROJECTION-v1/terminal.json / compiled-members.sqlite',
        '- analysis/G4-strong-projection-summary.json / G4-strong-projection-members.csv',
        '- diagnostics/G4_PREEXECUTION_INTEGRATION_JA.md：初期のテストimport順/source wrapper不一致の記録。旧失敗証拠は保持。', '']
    write(ROOT/'analysis/G4_STRONG_PROJECTION_RESULTS_JA.md', '\n'.join(text).encode())
    print(json.dumps({k: result[k] for k in ['certified_members','failed_members','actual_saved_QPY_recounts']}, ensure_ascii=False))


if __name__ == '__main__': analyze()
