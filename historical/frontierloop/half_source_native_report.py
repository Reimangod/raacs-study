"""Descriptive N18 tables from audited results, without new numerical solves."""
import csv
from io import StringIO
from collections import Counter
from .paths import ROOT
from .io import load, write, sha256, save

METHODS = ('FULL_LUCJ_L1', 'F2', 'H1', 'UNRESTRICTED_HF_HCI')
ARMS = ('SOURCE_FREQUENCY_R128', 'JOINT_TOP_D16384')
LABELS = {'FULL_LUCJ_L1': 'full LUCJ L1', 'F2': 'F2・正混合half',
          'H1': 'H1・spin remap', 'UNRESTRICTED_HF_HCI': '古典HF-HCI'}


def comparisons(rows):
    expected = {(c, m, a) for c in {r['case'] for r in rows} for m in METHODS
                for a in (ARMS if m != 'UNRESTRICTED_HF_HCI' else ARMS[1:])}
    keys = [(r['case'], r['method'], r['arm']) for r in rows]
    if len({r['case'] for r in rows}) != 4 or len(rows) != 28 or len(set(keys)) != 28 or set(keys) != expected:
        raise ValueError('All four cases and all 28 unique endpoints required')
    bykey = dict(zip(keys, rows))
    out = []
    for row in rows:
        if row['method'] not in ('F2', 'H1'):
            continue
        parent = bykey[(row['case'], 'FULL_LUCJ_L1', row['arm'])]
        hf = bykey[(row['case'], 'UNRESTRICTED_HF_HCI', ARMS[1])]
        result = dict(case=row['case'], method=row['method'], arm=row['arm'],
                      status=row['status'], parent_status=parent['status'])
        for key in ('CNOT', 'CNOT_Depth', 'Total_Depth'):
            result[key + '_reduction_pct'] = 100 * (parent[key] - row[key]) / parent[key]
        complete = row['energy_Ha'] is not None and parent['energy_Ha'] is not None
        result['energy_delta_vs_full_mHa'] = 1000 * (row['energy_Ha'] - parent['energy_Ha']) if complete else None
        result['energy_delta_vs_unrestricted_HF_mHa'] = 1000 * (row['energy_Ha'] - hf['energy_Ha']) if row['energy_Ha'] is not None and hf['energy_Ha'] is not None else None
        result['statistical_superiority_or_equivalence_certified'] = False
        out.append(result)
    return out


def fmt(value, places=3):
    return '未取得' if value is None else f'{value:.{places}f}'


def csv_bytes(rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def run():
    path = ROOT / 'analysis/N18-half-source-native-summary.json'
    result = load(path)
    if result['status'] != 'N18_ALL16_TASKS28_READOUTS_AUDITED':
        raise RuntimeError('Actual audited outputs required')
    rows = result['rows']
    contrast = comparisons(rows)
    flat = [{k: v for k, v in row.items() if k != 'all_source_compiler_points'} for row in rows]
    write(ROOT / 'analysis/N18-all28-readouts.csv', csv_bytes(flat))
    write(ROOT / 'analysis/N18-all16-half-full-comparisons.csv', csv_bytes(contrast))
    lines = ['# N18：半qubit方式はCNOTを半減。精度の同等性は未確認', '',
             '全4物理条件、full LUCJ L1・F2・H1と古典HF-HCIを比較。16 task・28 endpointをすべて回収・検証した。', '',
             f"状態：{dict(Counter(r['status'] for r in rows))}。新CI根{result['actual_new_CI_roots']}、失敗CI {result['failed_CI_attempts']}。",
             f"今回のCI campaignは{result['CPU_seconds']:.3f} CPU秒、{result['wall_seconds']:.3f} wall秒。新model Shot・QPU Shot・GPU割当は全て0。", '',
             '保存済み100,000 model Shot/sourceの再解析であり、新しい独立反復ではない。fullは1 Shotからα/βの2本、halfは1本の配置を得る。同Shot数は同情報量を意味しない。', '',
             '## まず結論', '',
             '- F2/H1のCNOTは約50%減るが、CNOT Depthは約2〜5%、Total Depthは約1〜3%しか減らない。2つのspin registerを片方にする既知の構造上の利益であり、深さも半分になるわけではない。',
             '- 強いHCI joint readoutでは、F2は全4条件でfullよりenergyが高い。H1はN₂ 1.2 Åのみfullより低く、他の3条件は高い。1反復の差を統計的な精度同等・優越の証明としない。',
             '- 全4条件で制限なしHF-HCIのenergyが全quantum-sourceより低い。ground-state energy計算として、この強い古典対照を外して勝利を主張しない。古典CI解を0 CNOTで量子準備した意味でもない。',
             '- F2は8〜32 memberの構築を要し、H1は1 member。同じ量子資源でも古典構築費用は異なる。Parameter半減は未確認。',
             '- このscreenだけを根拠にF2/H1のShot・seed sweepを拡大しない。既存結果は保持し、次は強いsource品質・原著対照を扱う別の小検証を先に設計する。']
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case]
        lines += ['', '## ' + case, '',
                  '| 方式 | readout | energy Ha | Recovery % | CNOT | CX Depth | Total Depth | 実CI次元 | native＋CI CPU秒* |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in selected:
            readout = 'frequency R128' if r['arm'] == ARMS[0] else 'HCI joint D≤16384'
            lines.append('| ' + ' | '.join([LABELS[r['method']], readout, fmt(r['energy_Ha'], 9),
                fmt(r['recovery_pct']), fmt(r['CNOT'], 0), fmt(r['CNOT_Depth'], 0),
                fmt(r['Total_Depth'], 0), fmt(r['actual_CI_dimension'], 0), fmt(r['standalone_numeric_CPU_seconds'])]) + ' |')
        lines += ['', '| half方式 | readout | fullとの差 mHa（負なら良い） | CNOT削減 % | CX Depth削減 % | Total Depth削減 % |',
                  '|---|---|---:|---:|---:|---:|']
        for r in [r for r in contrast if r['case'] == case]:
            lines.append('| ' + ' | '.join([LABELS[r['method']], 'frequency' if r['arm'] == ARMS[0] else 'HCI joint',
                fmt(r['energy_delta_vs_full_mHa']), fmt(r['CNOT_reduction_pct']),
                fmt(r['CNOT_Depth_reduction_pct']), fmt(r['Total_Depth_reduction_pct'])]) + ' |')
        lines += ['', 'source構築の記録（readout間で共有。下記を2回加算しない）：', '',
                  '| 方式 | qubits | model Shots | half配置本数 | member数 | pattern/compile CPU秒 | sampling CPU秒 | source入力座標数** |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in METHODS[:-1]:
            r = next(r for r in selected if r['method'] == method)
            lines.append('| ' + ' | '.join([LABELS[method], fmt(r['sampler_qubits'], 0), fmt(r['model_shots_per_source'], 0),
                fmt(r['half_configuration_count'], 0), fmt(r['sampler_member_count'], 0), fmt(r['source_pattern_or_compilation_CPU_seconds']),
                fmt(r['source_sampling_CPU_seconds']), fmt(r['source_model_input_coordinates'], 0)]) + ' |')
    lines += ['', '## 比較の範囲と費用の読み方', '',
              '* 表のCPUはそのreadout単独の数値費用proxy。frequencyにnative HCI費用を加えていない。source初期化、CCSD/CISD、過去のsampling/全member compile、テスト、回収、解析は別。未知のfull source費用は未取得とし、0で埋めない。各readoutのproxyとcampaign CPUを重複して加算しない。', '',
              '** source modelの入力座標を保持した数で、独立量子ansatz Parameter数の比較ではない。全手法の独立Parameter数は未取得。同じJ/U modelからhalf源を構築しており、qubit半減をParameter半減と呼ばない。', '',
              '- R128はhalf配置集合からのCartesian CI。HCIはHFから観測half-pool内で展開し、係数²に基づくjoint D≤16384を別途選択する。同じ最大Dでも実D・古典仕事量は異なる。',
              '- 解いた式は全方式H+0.1S²、表のenergyはphysical H。厳密なS=0や大域最小値を保証するものではない。両energy・S²・実D・費用・失敗は全28 CSV/JSONへ保存。',
              '- QPYに結び付いた共通opt1 logical資源の比較。fullの強いopt1/opt3全対照もJSONに残し、一方だけ新しいcompilerへ変えない。実機routing/noiseの比較ではない。',
              '- F2はone-layer LUCJの片スピンmarginalを正混合で表すもの。H1はspin remapによる別分布であり、元LUCJのexact marginalではない。half countsを架空のαβ joint量子観測へ変換していない。',
              '- Fe₂S₂のRecovery参照はSCI（FCIではなく精度未検証）。N₂のarchive FCIも精度認証は未完。referenceはreportだけに利用し、source生成・順位には使わない。Recoveryとenergyは独立の成功指標ではない。',
              '- ナフタレンはBLOCK-BOYS、N₂の2距離は同じ分子。原著の全Boys軌道・最高TN source・実機HSQDの完全再現や「最強論文への総合優越」ではない。最終CI状態はsampler回路が準備した状態とは別。', '',
              '全28結果：`N18-all28-readouts.csv`。全16 half/full差：`N18-all16-half-full-comparisons.csv`。元数値・全source compiler対照：`N18-half-source-native-summary.json`。']
    write(ROOT / 'analysis/N18_HALF_SOURCE_NATIVE_RESULTS_JA.md', ('\n'.join(lines) + '\n').encode())
    save(ROOT / 'analysis/N18-descriptive-report-binding.json', dict(
        summary_SHA256=sha256(path.read_bytes()), planned_readouts=28, comparisons=16,
        new_numerical_solves=0, independent_confirmation=False, goal_complete=False))
    print('N18_ALL28_TABLES16_CONTRASTS_WRITTEN_NO_NEW_SOLVES', flush=True)


if __name__ == '__main__':
    run()
