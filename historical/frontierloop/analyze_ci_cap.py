"""All R1 outcomes and same-device numerical-stage costs, not new shot counts."""
import csv
from io import StringIO
from .paths import ROOT
from .io import load,save,write,sha256
from .ci_cap_screen import DIRECTORY,binding
from .analyze import reference


def main():
    if load(DIRECTORY/'freeze.json')!=binding():raise RuntimeError('R1 freeze drift')
    spec=load(ROOT/'protocol-R1-CI-CAP-SCREEN.json');rows=[]
    for case in spec['cases']:
        hf,ref=reference(case)
        for method in spec['methods']:
            for cap in spec['R_caps']:
                d=DIRECTORY/case/method/str(cap);p=d/'result.json';r=load(p) if p.exists() else None
                if r:
                    request=load(d/'request.json');assert r['source_counts_SHA256']==request['counts_SHA256']
                    assert r['CI_strings']==request['CI_strings'] and r['physical_problem_id']==request['physical_problem_id']
                    if r['C7_initial_reused']:
                        old=ROOT/'runs/C7-LOWEST-ROOT-v1'/case/method/'result.json'
                        assert sha256(old.read_bytes())==r['reused_result_SHA256']
                rows.append({'case':case,'method':method,'R':cap,'status':'COMPLETED' if r else 'FAILED_ENGINEERING_PRESERVED' if (d/'failure.json').exists() else 'NOT_COMPLETED',
                    'energy_Ha':r['energy_Ha'] if r else None,'penalized_energy_Ha':r['penalized_energy_Ha'] if r else None,
                    'recovery_pct':100*(hf-r['energy_Ha'])/(hf-ref) if r else None,
                    'spin_square':r['spin_square'] if r else None,'actual_CI_dimension':r['actual_CI_dimension'] if r else None,
                    'incremental_CI_CPU_seconds':r['incremental_CPU_seconds'] if r else None,
                    'C7_initial_reused':r['C7_initial_reused'] if r else None,
                    'inherited_initial_CI_CPU_seconds':r['inherited_initial_CI_CPU_seconds'] if r else None,
                    'new_model_shots':0,'source_result_SHA256':sha256(p.read_bytes()) if r else None})
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/R1-CI-cap-all.csv',stream.getvalue().encode('utf-8-sig'))
    complete=sum(r['status']=='COMPLETED' for r in rows);f=lambda x:'欠測' if x is None else f'{x:.8f}'
    lines=['# CI上限を先に見直すべきか：新しいShotを使わない24セル試験','',
        f'全24セル中{complete}完了。D2無圧縮、F2正混合、H1 spin-remapの全4条件で、同じ既存countsから頻度上位64/128を選んだ。軌道最適化は**どちらも無し**、元の物理軌道基底を固定した。C3の有限0.1 Ha spin罰則と同じ数値ソルバーを使い、physical energyを別に報告する。','',
        '## Rだけを変えた比較','',
        '| 条件 | Sampler | R64 energy Ha | R128 energy Ha | energy低下 mHa | R128 CI CPU秒 |',
        '|---|---|---:|---:|---:|---:|']
    for case in spec['cases']:
        for method in spec['methods']:
            a,b=[next(r for r in rows if r['case']==case and r['method']==method and r['R']==cap) for cap in spec['R_caps']]
            delta=1000*(a['energy_Ha']-b['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None
            lines.append(f"| {case} | {method} | {f(a['energy_Ha'])} | {f(b['energy_Ha'])} | {f(delta)} | {f(b['incremental_CI_CPU_seconds'])} |")
    lines+=['','R64→128では最大Cartesian CI次元が4,096→16,384と4倍になる。Shot、回路、CNOT、Depth、Parametersは増えず、古典CIの仕事が増える。異なるRの結果を同じCI容量として比較しない。数学的な包含関係で単調性を検査したのは罰則付き目的であり、physical energyの一般的単調性保証ではない。','',
        '## 小さいCIにOOをかけるより、大きいCIを先に解く方がよいか','',
        'ここは同じM4で測った数値段階だけの比較。CCSD初期化・sampling・count選択・転送を含むcold-start総費用の比較ではない。両経路とも同じ保存sourceを使う。','',
        '| 条件 | F2 R64＋3OO energy Ha | F2 R128・OO無し energy Ha | R128側のenergy利益 mHa | R64＋OO CPU秒 | R128 CI CPU秒 |',
        '|---|---:|---:|---:|---:|---:|']
    comparisons=[]
    for case in spec['cases']:
        p=ROOT/'runs/C7-LOWEST-ROOT-v1'/case/'F2/result.json';a=load(p) if p.exists() else None
        b=next(r for r in rows if r['case']==case and r['method']=='F2' and r['R']==128)
        gap=1000*(a['final_energy_Ha']-b['energy_Ha']) if a and b['energy_Ha'] is not None else None
        lines.append(f"| {case} | {f(a['final_energy_Ha'] if a else None)} | {f(b['energy_Ha'])} | {f(gap)} | {f(a['CPU_seconds'] if a else None)} | {f(b['incremental_CI_CPU_seconds'])} |")
        comparisons.append({'case':case,'energy_gain_mHa':gap,'R64_OO_CPU_seconds':a['CPU_seconds'] if a else None,'R128_CI_CPU_seconds':b['incremental_CI_CPU_seconds']})
    lines+=['','R64の初回解は、counts/問題/積分/配置/数値条件の一致を確認できた場合だけC7から再利用した。再利用時の追加仕事は小さいが、元の初回計算費用が0だったとは扱わず、不明な個別時間はnullのまま残した。R128は新しく計算した。','',
        '## 次の投資判断','',
        'この試験でR拡張が効くなら、さらに回路heuristicを増やす前に、全ての強い量子対照・RAACS・古典CISDへ同じR/計算上限を適用する方が効率的である。R64の狭い空間だけで得た方式間の勝敗を一般化しない。','',
        '次は既存countsを使った共通CI-cap比較と、HCI/Trim等の既知の配置選択を検討する。提案法だけ大きいCIを認めず、古典計算量と実配置数をそろえる。なおHCI-HSQD原著の大規模選択まで今回再現したわけではない。','',
        'R拡張の利益は新しい量子ansatzの性能ではなく、後処理制限の診断結果。Recoveryは参照依存のenergy換算で、独立な成功に数えない。参照自体の精度・sectorの未再認証は引き続き限界として残す。全体goalと最強論文への優越は未達。','',
        '[半qubit方式の全比較](H1_F2_COMPARISON_JA.md)、[配置支持の診断](SUPPORT_DIAGNOSTIC_JA.md)。']
    write(ROOT/'analysis/R1_CI_CAP_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    summary={'completed':complete,'planned':24,'F2_R128_vs_R64_OO':comparisons,'new_model_shots':0,'goal_complete':False}
    save(ROOT/'analysis/R1-summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
