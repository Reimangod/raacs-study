"""Plain-language current evidence and next investment; never select spin mode."""
import csv
from .paths import ROOT
from .io import load, write
from .analyze import reference


def main():
    cases = load(ROOT/'protocol-B4.json')['cases']
    with (ROOT/'analysis/C5-all-control-readouts.csv').open(encoding='utf-8-sig') as f:
        controls = list(csv.DictReader(f))
    lines = ['# 現在の結果と改善計画：小さい回路から、使える小さい回路へ', '',
        '結論：現時点で「強い論文を総合的に超えた」とは言えない。RAACSは回路の小ささが強みだが、精度と、量子samplingに追加費用を払う理由が不足している。次は強いLUCJの精度を出発点に、そのsampling回路を圧縮する。', '',
        '今回追加した実験は、強い量子対照99読出し、変更方策4run、直接古典CISD 4sourceと24読出し。数値セルは独立分子数や独立optimizer成功数ではない。', '',
        'D1に続きD2は全16 draw・47 readout＋1保存された失敗で終端。単純なnative近似ではN₂/Feの精度悪化が大きい。別候補F2と既知HSQD remap対照H1は各4 draw/12 readoutとraw counts/QPY監査を完了した。同ShotでCNOT約半減だがDepthは小幅改善で、F2とH1の精度優越は条件依存。[F2全結果](F2_POSITIVE_MARGINAL_RESULTS_JA.md)、[H1/F2比較](H1_F2_COMPARISON_JA.md)、[D2全結果](D2_COMPRESSION_RESULTS_JA.md)。', '',
        '旧C3の最低根取り逃しは独立SAで切り分け、C7の12対称比較＋別枠事故回帰を完了した。旧結果は不変。H1は原著の大HCI/hardwareまでの再現ではない。以下はC5/C6時点の歴史表で、新旧solverを混ぜた勝利主張には使わない。', '',
        '**新しい主要発見はCI予算の配分だった。** [R1の全24セル](R1_CI_CAP_RESULTS_JA.md)では、全4条件のF2で、R128/OOなしがR64＋3OOより3.48〜16.35mHa低く、M4数値段階CPUも小さかった。新しいShotや回路改善の効果ではない。この利益を提案法だけに与えず、全33既存強対照・B3/B4・2種の古典CISDを同じR128/OOなしへそろえるR2を完了した（47数値結果、2計算上限、7source欠測）。[全方式のR128比較](R2_COMMON_CI_RESULTS_JA.md)。近距離2条件とFeでは古典CISD、伸長N₂では観測TN/LUCJが有力で、RAACS/F2の全面的優越は無い。次は回路heuristicを増やす前に配置選択と実際の強いSqDRIFT構成を点検する。', '',
        '## 1. 全指標を並べると何が不足するか', '',
        '全行で同じ物理問題、r上限64、同じ3段の軌道最適化を使う。量子samplerは総100,000 model Shot（RAACSは構築pilot込み）。古典CISDは量子Shotを使わず係数から配置を選ぶため、同じ測定情報の比較ではなく、実用性を判断する追加対照である。', '',
        '量子対照の最小energy行は観測済みの比較集合から取った事後envelopeで、無料で最良設定を選べる運用法ではない。全対照・全読出し・欠測21セルは別CSVに残す。', '']
    labels = {'NAPH20-BLOCK-BOYS':'ナフタレン 20q（Boys軌道）',
        'N2-32-R1p2-P2':'N₂ 1.2 Å・32q', 'N2-32-R2p4-P2':'N₂ 2.4 Å・32q',
        'FE2S2-40-P2':'Fe₂S₂ 40q'}
    modes = ['C2-OO','C3-SPIN-OO','C4-SINGLET-OO']
    for case in cases:
        hf, ref = reference(case)
        src = load(ROOT/'runs/B3'/case/'73001/result.json')
        lines += ['### '+labels[case], '',
            '| 読出し | 手法 | Energy Ha | Recovery % | CNOT / Depth / Total Depth | 係数スロット | 実CI配置数 | 追加OO CPU秒 |',
            '|---|---|---:|---:|---|---:|---:|---:|']
        for mode in modes:
            a = load(ROOT/'runs'/mode/case/'RAACS_B3/result.json')
            q = src['sampling_resources']
            lines.append(f"| {mode} | RAACS B3 | {a['final_energy_Ha']:.8f} | {100*(hf-a['final_energy_Ha'])/(hf-ref):.2f} | {q['cnot_count']:.1f} / {q['cnot_depth']:.1f} / {q['total_depth']:.1f} | {src['coefficient_slots_shot_weighted_mean']:.1f} | {len(a['CI_strings'])**2} | {a['CPU_seconds']:.3f} |")
            b = min((r for r in controls if r['case']==case and r['readout']==mode),
                key=lambda r: (float(r['energy_final_Ha']), r['method']))
            lines.append(f"| {mode} | 観測した量子対照の最小値：{b['method']} | {float(b['energy_final_Ha']):.8f} | {float(b['correlation_recovery_pct']):.2f} | {float(b['CNOT_mean']):.1f} / {float(b['CNOT_depth_mean']):.1f} / {float(b['total_depth_mean']):.1f} | {float(b['coefficient_slots_mean']):.1f} | {b['CI_determinant_dimension']} | {float(b['new_OO_CPU_seconds']):.3f} |")
            c = load(ROOT/'runs/C6-CLASSICAL-CISD'/case/'DETERMINISTIC_CISD_MARGINAL'/mode/'result.json')
            lines.append(f"| {mode} | 古典CISD配置選択 | {c['final_energy_Ha']:.8f} | {100*(hf-c['final_energy_Ha'])/(hf-ref):.2f} | 対象外：量子samplerなし | 対象外 | {len(c['CI_strings'])**2} | {c['CPU_seconds']:.3f} |")
        notes = {
            cases[0]: 'RAACSは短いが、強いCompressed LUCJより精度が悪い。さらに安い古典CISD配置選択が低いenergyを出した。まず量子samplingを使う付加価値が必要。',
            cases[1]: '短い回路だけでは精度不足を埋められない。古典CISD配置選択が大きく上回るため、弱い量子対照だけに勝っても実用的な成果にならない。',
            cases[2]: '強相関条件で強いLUCJが有力。RAACSは読出し定義で結果が大きく変わり、無罰則の高Recoveryだけを純粋なsinglet精度と呼べない。精度の高いLUCJを圧縮する試験に意味がある。',
            cases[3]: 'RAACSは観測した量子対照より低いenergyだが、古典対照との勝敗はspin条件で逆転する。全CISD空間の安い計算も有力で、この1条件だけを成功例として独立確認扱いしない。'}
        lines += ['', notes[case], '']
    lines += ['## 2. 数字の正しい意味', '',
        '- C2は無罰則CI、C3は有限0.1 Ha spin罰則、C4は選択配置に含まれる厳密S=0部分空間。異なる問いであり、勝つ定義へ差し替えない。非零S²でも物理energyの変分性が直ちに失われるわけではない。',
        '- Recoveryは参照に対する相関energy回収率。energyの換算なので二つの独立した成功指標には数えない。N₂/Fe参照の独立再認証は未完了。',
        '- CNOT/Depthはsampling回路。最終CI・軌道最適化状態をこの回路が準備したとは言わない。ensembleはShot加重平均で、最大値・全memberも原記録に残る。',
        '- Parameters欄の代わりに係数スロットを表示した。独立parameter rank、最適化変数、古典OO座標、CI係数を混同しない。OO座標は45/120/120/190を別に使用。',
        '- OO CPU秒は追加処理だけ。CCSD/CISD・回路初期化・探索・simulation・CI・失敗attemptまで含む総費用ではない。履歴で不明な費用を0にせず、二重計上もしない。',
        '- 同じRでも実CI次元とS=0次元、matvec数が違う。R64の小screenで原著の大きいCI・最強TNを超えたとは言えない。', '',
        '## 3. 最短の改善ルート', '',
        '| 段階 | 実施内容 | 続行条件・止め方 |',
        '|---|---|---|',
        '| D0：資源点検・完了 | 全4条件×3水準の既知native Givens圧縮。新規Shot・実分子energy評価0 | 合成順序による資源増加を発見。精度保持の成功例には数えない |',
        '| D1：最適化先行表現・完了 | 回転結合・Slater準備最適化を保持。全12資源点を再計数 | 実際の圧縮primitiveとsimulationを結合。資源は減ったが精度は別に評価する |',
        '| D2：全4条件screen・進行 | 無圧縮/3圧縮水準、各100k Shot、R64、3spin読出しを結果前に固定 | 精度と資源の対を比較。全grid費用を計上し、最良層やseedを無料選択しない |',
        '| F2：半register正混合・完了 | 4条件×100k Shot、12読出し。片スピン分布のidentityとraw evidenceを検査 | 半qubit/CNOT約半減だがDepthは数％。同Shotでもfullの半配置本数の半分。全条件での精度優越ではない |',
        '| H1/C7：既知半qubit対照と数値修正・進行 | HSQD spin remapを共通setupへ移植し、全方式へ同じ最低根ターゲティングを適用 | 原著HCIの再現とは別。旧結果へ上書きせず、勝つsolverだけを使わない |',
        '| D3：独立確認 | 固定方策を未使用seedで確認し、Shot/R曲線と失敗率を測定。強いTN・SqDRIFTの原著条件へ接続 | 差の区間、全費用、多重比較を報告。非有意を同等と呼ばない。確認前の点は開発結果のまま |',
        '| 実機向け確認 | 同じ接続図・native gatesでrouting後の資源も比較 | all-to-all論理資源だけで実機優越とは言わない |', '',
        '全指標を一つの恣意的scoreへ足して勝ちを作らない。同精度のresource frontierと、同じ総Shot/CI予算のenergyを併記する。全資源・総費用が同時に良くなった場合だけ総合優位と呼び、それ以外は何を交換した改善か明示する。', '',
        '## 4. 計算環境と費用', '',
        '小さいCI・compiler・集計はM4。A100は数値parityと実測速度利得を確認したstatevector/MPS kernelだけ、独立タスクに最大2台使う。CPUのPySCFをGPUノードで実行しただけではGPU加速ではない。候補ごとの巨大statevector保存を避け、countsと最終回路を圧縮保存する。空き3 GiB以上、この研究rootは1 GiB以内。研究データを自動削除しない。', '',
        '## 5. 比較する一次資料', '',
        '追加検索でSQD向け圧縮と半qubit/HCI-HSQDも既存と確認した。方向D/F2は新規性認定ではなく有用性の小試験である。[既知研究との境界・追加対照](LITERATURE_BOUNDARY_UPDATE_JA.md)を先に読む。H1はHSQDのspin remapのみの共通setup移植で、原著HCI・hardware・大CIまでの再現比較ではない。', '',
        '- [SqDRIFT v3](https://arxiv.org/html/2508.02578v3)：改善合成とsampling/CI条件までそろえる。',
        '- [Compressed/TN LUCJ](https://arxiv.org/html/2511.22476v1)：著者初期化と十分なTN調整を含む強い対照。現在の限定TNは原著最強の再現ではない。',
        '- [公式OO-SQD](https://qiskit.github.io/qiskit-addon-sqd/guides/optimize_orbitals.html)：OO自体は既知の後処理。',
        '- [公式native Givens圧縮](https://qiskit-community.github.io/ffsim/dev/how-to-guides/compress-orbital-rotation.html)：既知generic圧縮との比較が必要。単一粒子目的の最適化は最終sampling精度を保証しない。',
        '- [公式Trim SQD](https://qiskit.github.io/qiskit-addon-sqd/guides/trim_sqd.html)：次の読出し改善候補だが、samplerごとに不公平な後処理をしない。', '',
        '## 詳細資料', '',
        '- [全99量子対照読出しと欠測](C5_CONTROLS_JA.md)',
        '- [変更方策B4と古典対照C6](B4_C6_DECISION_JA.md)',
        '- [native圧縮の全12資源点：精度評価は未実施](D0_NATIVE_RESOURCE_JA.md)',
        '- [D1：合成順序を直した12資源点](D1_MERGED_RESOURCE_JA.md)',
        '- [D2：精度を含む全セルの進行と結果](D2_COMPRESSION_RESULTS_JA.md)',
        '- [代替F2：片スピン正混合の式と検算・限界](F2_CONDITIONAL_MARGINAL_JA.md)',
        '- [F2：全4条件の実結果と費用](F2_POSITIVE_MARGINAL_RESULTS_JA.md)',
        '- [H1/F2：既知手法との共通条件比較、C7数値修正版](H1_F2_COMPARISON_JA.md)',
        '- [R1：新ShotなしのCI上限24セル比較と費用](R1_CI_CAP_RESULTS_JA.md)',
        '- [配置支持を失う原因の診断](SUPPORT_DIAGNOSTIC_JA.md)',
        '- [Agent実行指示](../AGENTS.md)',
        '- [次に行う仕事](../NEXT_ACTIONS_JA.md)', '',
        '「現在の最も強い論文に総合的に勝った」という主張は未達。4条件の開発screenを、未知分子一般化・原著再現・量子優位・採択保証へ拡張しない。']
    write(ROOT/'analysis/CURRENT_RESULTS_AND_PLAN_JA.md', ('\n'.join(lines)+'\n').encode())
    print({'report': str(ROOT/'analysis/CURRENT_RESULTS_AND_PLAN_JA.md'), 'goal_complete': False})


if __name__ == '__main__': main()
