"""Planning evidence: actual MPS/known exact-backend work scopes, not a speed claim."""
from .paths import ROOT
from .io import load, save, write, sha256


def run():
    n19=load(ROOT/'analysis/N19-tn-objective-feasibility-summary.json')
    if n19['state_counts']!={'COMPLETED':4}:
        raise RuntimeError('All four MPS attempts must be closed, not winner-only timing')
    base=ROOT/'runs/N17-INDEPENDENT-PREFIX-PERFORMANCE-v1'
    p=load(base/'protocol.json'); audit=load(ROOT/'analysis/N17-sampling-stage-inventory.json')
    if audit['protocol_SHA256']!=sha256((base/'protocol.json').read_bytes()) or audit['verified_unique_draws']!=81:
        raise RuntimeError('Existing exact draw evidence changed or incomplete')
    circuits={c['circuit_id']:c for c in p['circuits']}; rows=[]
    for m in n19['rows']:
        cells=[c for c in p['cells'] if c['case']==m['case'] and c['method']=='ORIGINAL' and c['source'].endswith('L1')]
        if len(cells)!=3 or len({c['circuit_id'] for c in cells})!=1:
            raise RuntimeError('Single actual original circuit, all three registered draws required')
        circuit=circuits[cells[0]['circuit_id']]
        folder=base if circuit['backend']=='M4' else base/'collected'
        path=folder/'circuits'/circuit['circuit_id']/'output/result.json'; result=load(path)
        if result['input_id']!=circuit['input_id'] or result['source_QPY_SHA256']!=circuit['original_QPY_SHA256'] or result['physical_problem_id']!=m['problem_id']:
            raise RuntimeError('Actual circuit/source timing context differs')
        if sum(s['actual_model_shots'] for s in result['sampling_results'])!=300000 or result['actual_QPU_shots']!=0:
            raise RuntimeError('Exact work scope changed')
        rows.append(dict(case=m['case'],MPS_classical_surrogate_samples=10000,
            MPS_body_CPU_seconds=m['CPU_seconds'],MPS_body_wall_seconds=m['wall_seconds'],
            MPS_norm=m['approximate_MPS_norm'],original_batch_joint_dimension_min=m['joint_dimension_min'],
            original_batch_joint_dimension_max=m['joint_dimension_max'],
            exact_backend=circuit['backend'],exact_new_model_shots=300000,
            exact_body_CPU_seconds=result['whole_body_CPU_seconds'],exact_body_wall_seconds=result['whole_body_wall_seconds'],
            exact_result=str(path),exact_result_SHA256=sha256(path.read_bytes()),
            exact_GPU_memory_pool_bytes=result.get('GPU_memory_pool_retained_bytes'),
            same_hardware_or_shot_budget_head_to_head=False,end_to_end_speedup_claimed=False))
    out=dict(status='N19_ALL4_BACKEND_WORK_SCOPES_RECONCILED',rows=rows,
        new_computations=0,no_simulation_speedup_attributed_to_proposed_ansatz=True,
        fast_exact_backend_opportunity_must_be_shared_with_all_controls=True,
        no_exact_52qubit_capacity_claim=True,goal_complete=False,
        N19_summary_SHA256=sha256((ROOT/'analysis/N19-tn-objective-feasibility-summary.json').read_bytes()))
    save(ROOT/'analysis/N19-backend-cost-review.json',out)
    lines=['# N19の判断：まずCI予算を切り分け、今回のサイズではexact backendを活用する','',
           'N19は全4条件で完了。近似MPS surrogate drawsは計40,000、分子CI energy計算は0。',
           'ここではN17の既存exact sampling実測と、作業範囲を分けて照合した。新しいsamplingやCIを追加していない。','',
           '| 条件 | MPS CPU・10k surrogate samples 秒 | MPS body wall 秒 | 既存exact backend | exact body wall・3×100k samples 秒 | 原著型batchのjoint D範囲 |',
           '|---|---:|---:|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['MPS_body_CPU_seconds']:.3f} | {r['MPS_body_wall_seconds']:.3f} | {r['exact_backend']} | {r['exact_body_wall_seconds']:.3f} | {r['original_batch_joint_dimension_min']}〜{r['original_batch_joint_dimension_max']} |")
    lines+=['',
            'CI空間の差は全条件に一様ではない。Feは現上限の約70倍、N₂近距離は約1.6倍だが、N₂遠距離とナフタレンの今回のMPS batchは現上限より小さい。sourceもMPS近似とexactで異なるため、全精度差をCI上限だけに帰属しない。',
            '', '## 今回どこを変えるべきか','',
            '1. **原著のCI空間と現在のcommon D16384の違いを先に測る。** 同じsource counts・同じ物理式で、R128→256→512→1024の入れ子の空間へ広げたときのenergyと実CPUを調べる。大きい空間への改善を新しいansatzの効果と呼ばない。',
            '2. **この4条件では、source探索を全てMPSへ移す理由は弱い。** 既存のM4/A100 exact samplerは実作業時間が小さく、MPS近似誤差も避けられる。今後の探索ではpersistent batch実行を検討する。ただしfast backendは対照にも同じように与える。',
            '3. **原著対照を弱めない。** 既存full/Compressed LUCJ、generic policy、強い古典HF-HCIを残す。最適化中と最終評価で別の未使用drawを用い、source構築費用と量子Shot・古典surrogate drawsを分ける。',
            '4. **500回のTN最適化をまだ自動投入しない。** samplingだけでも費用がかかり、その後に大きいCIがある。CI scalingの実測後に、総CPU/GPU上限内のoptimizer試験を登録する。', '',
            'これは「A100はTNより普遍的に何倍速い」というbenchmarkではない。hardware、sample数、compiler、runtimeの作業単位が異なる。',
            'exact body時間はQPY生成・compile・転送・queue待ち・全job startup・CIを含むcold総時間ではない。N17のA100 allocation合計72秒は別の集計単位であり、各行へ丸ごと加算しない。',
            'MPS CPU値もsourceのCCSD/CISD初期化などを含まない。新しいbackend間のmatched-work比較・原著完全再現・52qubit以上でのexact容量・最強論文への勝利は未証明。', '',
            '## 次の小検証の具体形（未freeze・未投入）','',
            '全4条件の同じfull LUCJ L1保存済み100k countsから、HF必須・frequency順位を変えずにR128/256/512/1024のnested Cartesian空間を作る。',
            '入れ子の空間で単調減少を検査するのは最適化目的H+0.1S²であり、physical H energyとS²も別に報告する。physical H単独の単調減少を数学的に保証したとはしない。',
            '同一の半配置集合になった上限は再計算せずaliasとして残す。現inventoryでは全4条件に16の予定点、unique basisは11の見込み。',
            'R128はN18と同じbasis/physical H+0.1S²の数値parityを先に検査する。大きいbasisはCSRを密に構築せず、既に独立RDM/particle-hole検査済みのmatrix-free routeを起点にする。',
            '新しい実装・大次元での時間/メモリ境界・根の保存方法をfreeze前に点検し、失敗/未収束も残す。参照FCI/SCI係数は選択へ入れない。',
            'この診断で実用的な精度利益が見えた後、他source・古典対照も同じ大きい空間/費用機会へ揃える。full LUCJだけ大きいCIへ変えた比較で、方法間の優越は主張しない。',
            '次のprotocolやjobはまだ作成していない。N17/N18/N19は終端済みで再実行しない。研究全goalはactive・未達。']
    write(ROOT/'analysis/N19_DECISION_AND_NEXT_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],flush=True)


if __name__=='__main__':run()
