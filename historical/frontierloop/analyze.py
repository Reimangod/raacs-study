"""Cumulative exploratory research log, not post-selected confirmation."""
from collections import defaultdict
from io import StringIO
import csv
import numpy as np
from .paths import ROOT,V2,OLD
from .io import load,save,write,sha256


def reference(case):
    r=load(OLD/'data/evaluation-only'/(case+'.json'));e=r['energies_Ha']
    return e['hf_energy'],e['fci_energy']if e.get('fci_energy')is not None else e['sci_energy']


def collect():
    records=[]
    for p in(V2/'runs/DEVELOPMENT').glob('*/*/*/result.json'):
        r=load(p);r['method']=r['variant'];r['phase']='PREVIOUS_DEVELOPMENT';records.append((p,r))
    for p in(ROOT/'runs/B1').glob('*/*/result.json'):
        r=load(p);r['method']='B1_COMPLETE_PASS';r['phase']='CURRENT_DEVELOPMENT';records.append((p,r))
    for p in(ROOT/'runs/B2').glob('*/*/*/result.json'):
        r=load(p);r['method']='B2_'+r['variant'];r['phase']='SINGLE_SEED_SCREEN';records.append((p,r))
    for p in(ROOT/'Fe40-C1-v1').glob('*/result.json'):
        r=load(p);r['method']='LUCJ_COMPRESSED_L1';r['phase']='CURRENT_DEVELOPMENT';records.append((p,r))
    for p in(ROOT/'runs/B3').glob('*/*/result.json'):
        r=load(p);r['method']='B3_RESIDUAL_BEAM';r['phase']='SINGLE_SEED_SCREEN';records.append((p,r))
    rows=[]
    for p,r in records:
        hf,ref=reference(r['case_id']);resources=r['sampling_resources']
        slots=r.get('coefficient_slots_shot_weighted_mean',r.get('coefficient_slots'))
        for c in r['final_CI']:
            rows.append({'phase':r['phase'],'case_id':r['case_id'],'method':r['method'],'seed':r['seed'],'budget':r['budget'],
                'r_cap':c['r_cap'],'actual_d':c['actual_d'],'energy_Ha':c['energy_Ha'],'recovery_pct':100*(hf-c['energy_Ha'])/(hf-ref),
                'CNOT':resources['cnot_count'],'CNOT_Depth':resources['cnot_depth'],
                'Total_Depth':resources.get('total_depth',resources.get('total_unitary_depth')),'coefficient_slots':slots,
                'CPU_construct_seconds':r.get('construction_CPU_seconds'),'pilot_shots':r.get('pilot_model_shots',0),
                'source':str(p),'SHA256':sha256(p.read_bytes()),'CI_converged':c['converged'],'S2':c['spin_square'],
                'all_model_CNOT_applications':r.get('all_model_CNOT_applications',r.get('cost',{}).get('all_model_CNOT_applications')),
                'decision_count':r.get('decision_count'),'bank_size':r.get('bank_size',1),'stop':r.get('construction_stop_reason')})
    return rows


def main():
    rows=collect();groups=defaultdict(list)
    for r in rows:
        if r['r_cap']==64:groups[(r['case_id'],r['method'])].append(r)
    root=ROOT/'analysis';root.mkdir(exist_ok=True)
    out=StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(root/'all-development-readouts.csv',out.getvalue().encode('utf-8-sig'))
    stats=[]
    for(case,method),rs in sorted(groups.items()):
        a={'case_id':case,'method':method,'n':len(rs)}
        for field in('energy_Ha','recovery_pct','CNOT','CNOT_Depth','Total_Depth','coefficient_slots','actual_d','pilot_shots'):
            a[field]=float(np.mean([r[field]for r in rs]))
        costs=[r['CPU_construct_seconds']for r in rs]
        a['CPU_construct_seconds']=float(np.mean(costs))if all(c is not None for c in costs)else None
        stats.append(a)
    save(root/'group-statistics.json',stats)
    lines=['# 自律研究ループ：結果と次の仮説','',
        'これは全て開発データです。目標はまだ未達で、正式な確認試験や原著優越の認定ではありません。旧D0–D3、B1、B2は別試験・別seedとして残しています。','',
        '## B1で確かめたこと','',
        '旧D3の全12停止decisionは新候補0件でした。IDで親を巡回し全bankを試す方式へ修正すると、全12件で探索が延びました。ただし全件が旧CPU上限で停止。bankの構築が進んでも、強い対照の精度へはまだ届きませんでした。','',
        '## B2で確かめたこと','',
        '構築CPUを30秒、総Shotを100,000に固定し、時間だけ延長したbank、最終r64に評価を合わせたbank、同じ評価の単回路を比べました。B2は各1seedであり分散を評価できません。良いseedを選んで旧3seedに足すことはしません。','',
        '## 全4条件のr64数値','',
        'Recoveryは相関energy回収率で、configuration recoveryではありません。N₂/Feの公開参照精度・sector未再認証は継続して明示します。CNOT等は共通all-to-all Z-sampling論理資源、bankではShot重み付きmember平均。係数スロットは非冗長独立自由度の証明ではありません。','']
    for case in load(ROOT/'protocol-B1.json')['cases']:
        lines+=['### '+case,'','| 方法 | n | CI energy Ha | Recovery % | CNOT | CNOT Depth | Total Depth | 係数スロット | 実CI次元 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        selected=[s for s in stats if s['case_id']==case]
        for s in selected:
            lines.append('| '+' | '.join([s['method'],str(s['n']),f"{s['energy_Ha']:.8f}",*[f'{s[f]:.2f}'for f in('recovery_pct','CNOT','CNOT_Depth','Total_Depth','coefficient_slots','actual_d')]])+' |')
        c=next((s for s in selected if s['method']=='LUCJ_COMPRESSED_L1'),None)
        if c:
            candidates=[s for s in selected if s['method'].startswith(('B1','B2','B3'))]
            if candidates:
                best=min(candidates,key=lambda s:s['energy_Ha'])
                lines+=['',f"現時点の新方策の最小観測平均は{best['method']}で、対照との差は{1000*(best['energy_Ha']-c['energy_Ha']):+.3f} mHaです。異なるnの開発比較であり、確認試験の勝者ではありません。"]
        lines.append('')
    lines+=['## 次の検証','',
        'B2でも精度が不足したため、単純な時間延長・rの変更だけでは十分ではありません。次は全paid pilot観測から残差候補を作り、短い4経路を深さ8まで探索するB3を試します。直前のCI利得0だけでは経路を止めず、pilotで選んだbest bankを新しい独立final測定で確認します。候補配置やclassical Ritz vectorを無料で物理状態として準備することはしません。','',
        '各cycleを計画・コードSHA・seedに結合し、旧試験は変更しません。開発探索費用を保持し、正式比較に進むなら別seedの固定policy検証が必要です。','',
        '## 文献との関係','',
        '[SqDRIFT v3](https://arxiv.org/abs/2508.02578v3)と[改善初期化/TN LUCJ](https://arxiv.org/abs/2511.22476v1)を主要な強い比較対象とします。弱い共通予算adaptationに勝っただけでは原著最良点への優越とは呼びません。','',
        '[HF分離ADAPT-QSCIの2026年10月preprint](https://arxiv.org/abs/2610.03114v1)も確認しました。既知HFへの測定集中と選択基準の整合が問題になることを示しています。対象・CI構成が異なるので掲載削減率を本結果へ転用せず、HFをclassically追加する考え自体を新規性とも呼びません。','',
        '新しいquantum-advantage主張には強い古典SCIの費用付き対照も必要です。最終CIが低いことだけで、量子samplingが古典選択より有益だったとは結論できません。']
    write(root/'RESEARCH_LOG_JA.md',('\n'.join(lines)+'\n').encode())
    print({'readouts':len(rows),'groups_r64':len(stats),'goal_achieved':False},flush=True)


if __name__=='__main__':main()
