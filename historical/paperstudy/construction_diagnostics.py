"""Retrospective diagnostics of the frozen RAACS policy, not policy changes."""
from collections import Counter
import json
from .io import ROOT,save_json,atomic_bytes
from .analysis_products import write_csv


def inverse_pairs(operators):
    stack=[];pairs=[]
    for i,op in enumerate(operators):
        identity=(op['n_qubits'],tuple(op['source']),tuple(op['target']))
        if stack and stack[-1][1]==identity and stack[-1][2]==-op['theta']:
            j,_,_=stack.pop();pairs.append((j,i))
        else:stack.append((i,identity,op['theta']))
    return pairs


def run():
    rows=[];group=Counter()
    for path in ROOT.glob('common-runs/C_PRIMARY/*/A_RAACS_CURRENT/*/result.json'):
        outer=json.loads(path.read_text());v=outer['detail']['native_result'];pairs=inverse_pairs(v['accepted_operators'])
        stop=v['construction_stop_reason'];group[(outer['case_id'],outer['budget'],stop)]+=1
        rows.append({'case_id':outer['case_id'],'budget':outer['budget'],'seed':outer['seed'],
            'accepted_operators':v['accepted_steps'],'construction_stop_reason':stop,
            'pilot_model_shots':outer['pilot_model_shots'],'independent_final_model_shots':outer['independent_final_model_shots'],
            'named_H_requests':v['classical_named_H_requests'],'exact_sparse_state_builds':outer['detail']['exact_sparse_state_builds_this_attempt'],
            'adjacent_inverse_word_reduction_pairs':len(pairs),'inverse_original_indices':json.dumps(pairs),
            'inverse_detection_uses_exact_registered_angles_not_small_angle_pruning':True,
            'no_actual_work_refunded':True,'no_source_operators_or_resources_changed':True,
            'final_compiled_CNOT':outer['sampling_resources']['cnot_count'],
            'final_compiled_CNOT_depth':outer['sampling_resources']['cnot_depth'],'source':str(path.relative_to(ROOT))})
    out=ROOT/'analysis-v1';write_csv(out/'RAACS-full-construction-and-stopping-diagnostics.csv',rows)
    summaries=[{'case_id':c,'budget':B,'stop_reason':s,'observed_runs':n}for(c,B,s),n in sorted(group.items())]
    write_csv(out/'RAACS-stopping-reasons.csv',summaries)
    lines=['# RAACSはどこで、なぜ止まったか','',
        'これは固定した規則の動作を調べる後解析です。結果を良く見せるためのpolicy変更や再実行はしていません。','',
        '| 条件 | 総model Shot | 停止理由 | 件数 |','|---|---:|---|---:|']
    meaning={'NO_PILOT_ENERGY_IMPROVEMENT':'試した候補に、pilotの登録基準を超える改善が見られなかった',
        'SHOT_BUDGET_EXHAUSTED':'構築用の残り測定予算が不足した','MAX_STEPS':'登録した構築段数上限に達した'}
    lines +=[f"| {r['case_id']} | {r['budget']} | {meaning.get(r['stop_reason'],r['stop_reason'])} | {r['observed_runs']} |"for r in summaries]
    total=sum(r['adjacent_inverse_word_reduction_pairs']for r in rows)
    lines +=['','## 有限Shotによる選択の揺らぎ','',
        f'保存済み{len(rows)}runで、同じ生成子の逆向き回転が隣接して打ち消し合う組を延べ{total}組検出しました。単語列で隣接して消した後に露出する逆対も数えています。',
        '同じ候補でもpilotの測定値は揺らぐため、有限Shotの改善判定は「真のエネルギーが必ず下がる」という証明ではありません。逆回転の再追加は、その揺らぎや内部readoutを調べる具体的な診断材料です。',
        'この検出を使って過去の結果を改善したり、使ったShotを取り消したりしていません。論理的な逆対の数と、実際にcompilerが除いたCNOTの数も同一視しません。',
        '最終評価は独立final countsの共通CIです。pilotで良く見えた値をそのまま成果にしていません。係数スロット数と物理的に非冗長な自由度を分けているのも、このような事情があるためです。']
    atomic_bytes(out/'RAACS_STOPPING_JA.md',('\n'.join(lines)+'\n').encode())
    save_json(out/'construction-diagnostics-status.json',{'observed_RAACS_runs':len(rows),'inverse_pairs_observed':total,
        'frozen_policy_unchanged':True,'original_work_not_refunded':True,'whole_experiment_complete':False})
    print(json.dumps({'observed_RAACS_runs':len(rows),'inverse_pairs':total}),flush=True)


if __name__=='__main__':run()
