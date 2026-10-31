"""Plain case-level descriptions of complete seed groups; no tuned winners."""
import csv,json,math
from collections import defaultdict
from .io import ROOT,atomic_bytes,save_json
from .analysis_products import write_csv

NAMES={'A_RAACS_CURRENT':'RAACS','B_LUCJ_NAIVE_L1':'通常初期化LUCJ・1層',
    'C_LUCJ_COMPRESSED_L1':'改善初期化LUCJ・1層','D_SQDRIFT_N20':'SqDRIFT・20励起',
    'E_LUCJ_TN_COMMON_L1':'TN調整LUCJ・1層'}
CASES={'NAPH20-P1':'ナフタレン20q','N2-32-R1p2-P2':'N₂ 32q・1.2 Å',
    'N2-32-R2p4-P2':'N₂ 32q・2.4 Å','FE2S2-40-P2':'Fe₂S₂ cluster40q'}


def resource_minima(group):
    answer=[]
    for field,label in(('mean_CNOT','CNOT'),('mean_CNOT_depth','CNOT Depth'),
        ('mean_total_depth','Total Depth'),('mean_coefficient_slots_or_member_mean','係数座標数')):
        values=[(float(r[field]),r['method_id'])for r in group if r.get(field)not in(None,'')and math.isfinite(float(r[field]))]
        minimum=min(v for v,_ in values)if values else None
        answer.append({'label':label,'value':minimum,'methods':[m for v,m in values if v==minimum],
            'unknown_methods':len(group)-len(values)})
    return answer


def capacity_comparison(index,case,budget):
    """Compare complete primary groups, never just available/favorable seeds."""
    groups={}
    for r in('32','64'):
        group=[index.get(('C_PRIMARY',case,m,str(budget),r))for m in NAMES if m!='E_LUCJ_TN_COMMON_L1']
        if any(row is None or int(row['observed_runs'])!=10 or int(row['planned_runs'])!=10 for row in group):return None
        groups[r]=group
    best={r:min(group,key=lambda row:float(row['mean_energy_Ha']))for r,group in groups.items()}
    raacs=next(row for row in groups['64']if row['method_id']=='A_RAACS_CURRENT')
    return {'r32_lowest_mean_method':best['32']['method_id'],'r64_lowest_mean_method':best['64']['method_id'],
        'r64_lowest_mean_energy_Ha':float(best['64']['mean_energy_Ha']),
        'r64_RAACS_minus_lowest_mean_mHa':1000*(float(raacs['mean_energy_Ha'])-float(best['64']['mean_energy_Ha'])),
        'ranking_changed':best['32']['method_id']!=best['64']['method_id'],
        'statistical_superiority_claimed':False}


def name(method):
    if method in NAMES:return NAMES[method]
    if method.startswith(('B_LUCJ','C_LUCJ')):
        return ('通常初期化LUCJ'if method.startswith('B_')else'改善初期化LUCJ')+'・'+method.rsplit('L',1)[1]+'層'
    if method.startswith('D_SQDRIFT_N'):
        tail=method.split('_N',1)[1];length=tail.split('-')[0]
        return 'SqDRIFT・'+length+'励起'+('・1回路512測定'if'M512'in tail else'')
    return method


def read():
    with(ROOT/'analysis-v1/all-group-statistics.csv').open()as f:all_rows=list(csv.DictReader(f))
    return [r for r in all_rows if int(r['observed_runs'])==int(r['planned_runs'])]


def run():
    rows=read();lines=['# 分子ごとの読み方：精度・小回路・測定予算','',
        '全予定seedがそろった設定の平均だけを以下の説明に使います。平均の大小は記述であり、有意差や同等精度の証明ではありません。未完了の点も全予定CSVには残しています。','']
    for case,label in CASES.items():
        init_path=ROOT/'data/initialization-fresh'/case/'result.json'
        init=json.loads(init_path.read_text())if init_path.exists()else{}
        selected=init.get('selected_method','UNKNOWN')
        note=('この主比較のLUCJは、新規CCSDが登録上限で収束しなかったため、事前登録したCISDへ切り替えたvariantです。元論文の収束CCSDや保存source振幅の結果と同一視しません。'if selected=='CISD'else
              'この主比較のLUCJは、新規CCSD振幅から初期化しています。著者保存振幅とbyte一致したという意味ではありません。'if selected=='CCSD'else
              'この表だけではLUCJの初期振幅の作成経路を確認できません。元の初期化記録を参照し、未確認をCCSD収束へ読み替えません。')
        lines+=['## '+label,'',note,
            '「改善初期化」は著者の圧縮factorizationを登録iteration上限で適用したものです。全stageがoptimizer収束した保証ではなく、TN調整後の強い対照とは別です。','',
            '| 総model Shot | 方法 | 相関energy回収率 % | CNOT | CNOT Depth | Total Depth | 係数スロット | 平均CI実次元 |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
        primary=[r for r in rows if r['lane']=='C_PRIMARY'and r['case_id']==case and int(r['r_cap'])==32]
        for r in sorted(primary,key=lambda x:(int(x['budget']),x['method_id'])):
            def n(k):return f'{float(r[k]):.2f}'if r.get(k)else'不明'
            lines.append('| '+' | '.join([r['budget'],NAMES[r['method_id']],n('mean_correlation_recovery_pct'),n('mean_CNOT'),
                n('mean_CNOT_depth'),n('mean_total_depth'),n('mean_coefficient_slots_or_member_mean'),n('mean_actual_d')])+' |')
        lines.append('')
        for B in(10000,100000,1000000):
            group=[r for r in primary if int(r['budget'])==B]
            if len(group)!=4:
                lines.append(f'{B:,} Shot：4手法すべての10seedがまだそろっていないため、この予算で全体の勝者は決めません。');continue
            best=min(group,key=lambda r:float(r['mean_energy_Ha']));small=min(group,key=lambda r:float(r['mean_CNOT']))
            delta=float(small['mean_energy_Ha'])-float(best['mean_energy_Ha'])
            lines.append(f"{B:,} Shot・主4手法：平均energyが最も低いのは{NAMES[best['method_id']]}、主表の基本合成列でCNOTが最も少ないのは{NAMES[small['method_id']]}です。")
            minima=resource_minima(group);descriptions=[]
            for metric in minima:
                methods='・'.join(NAMES[m]for m in metric['methods'])if metric['methods']else'不明'
                value='不明'if metric['value']is None else f"{metric['value']:.2f}"
                descriptions.append(metric['label']+'：'+methods+'（'+value+'）'+('〔未知値あり・既知の範囲〕'if metric['unknown_methods']else''))
            lines.append('個別の資源最小値は、'+'、'.join(descriptions)+'です。精度を揃えた削減率や、全指標の同時優越ではありません。')
            if set(minima[0]['methods'])!=set(minima[1]['methods']):
                lines.append('この条件では、CNOT数とCNOT Depthの最小法が異なります。「ゲート数が少ない」と「最も浅い回路」を同じ意味にしないでください。')
            if best['method_id']!=small['method_id']:
                lines.append(f'最小CNOTの方法は、平均energyでは最良法より{1000*delta:.3f} mHa高くなっています。したがって「回路は小さいが、今回のCI容量で引き出せる精度とは交換関係」という読み方です。')
            else:lines.append('この予算・CI容量では、小さい回路と低い平均energyが同じ方法にそろっています。ただし他のCI容量や多層対照まで普遍化しません。')
        medium=[r for r in rows if r['case_id']==case and r['lane']in('S5_SENSITIVITY','S_TN_INDEPENDENT_FINAL')and int(r['budget'])==100000 and int(r['r_cap'])==32]
        if medium:
            lines+=['','追加のLUCJ層数・SqDRIFT長さ・TN調整は、以下の全設定を比べます。5回の最終測定の別試験で、主比較の10seedを増やしたものではありません。TNは1個の調整済み回路を5回読んでおり、5個の独立optimizerではありません。TNの前処理費用と時間切れはTN_CONTROL_JA.mdに示します。','',
                '| 感度設定 | 相関energy回収率 % | CNOT | CNOT Depth |','|---|---:|---:|---:|']
            for r in sorted(medium,key=lambda x:x['method_id']):
                pct=f"{float(r['mean_correlation_recovery_pct']):.2f}"if r.get('mean_correlation_recovery_pct')else'不明'
                lines.append(f"| {name(r['method_id'])} | {pct} | {float(r['mean_CNOT']):.2f} | {float(r['mean_CNOT_depth']):.2f} |")
        lines+=['','この表の資源は共通all-to-all論理分解です。SqDRIFTは複数回路のShot重み付き平均、他は最終sampling回路です。探索中の費用は別の総Shot・総CNOT適用回数に含めます。',
            'SqDRIFTのRustiq合成による資源はDECISION_REPORT_JA.mdとfrontier図のD-Rにも示します。同じcounts・energyの資源再計数であり、新しい独立実験ではありません。基本合成だけを使って資源上の優位性を断定しません。']
    index={(r['lane'],r['case_id'],r['method_id'],r['budget'],r['r_cap']):r for r in rows}
    capacity=[]
    for key,r64 in index.items():
        lane,case,method,B,r=key
        if r!='64':continue
        r32=index.get((lane,case,method,B,'32'))
        if r32 is None:continue
        capacity.append({'lane':lane,'case_id':case,'method_id':method,'budget':B,
            'r32_actual_d_mean':float(r32['mean_actual_d']),'r64_actual_d_mean':float(r64['mean_actual_d']),
            'energy_improvement_mHa_readout_only':1000*(float(r32['mean_energy_Ha'])-float(r64['mean_energy_Ha'])),
            'new_model_shots':0,'new_circuit':False})
    write_csv(ROOT/'analysis-v1/CI-capacity-energy-differences-all-complete-groups.csv',capacity)
    lines+=['','## CIを大きくすると何が変わるか','',
        '同じcountsからr32→64へ読み直した全設定の差をCI-capacity-energy-differences-all-complete-groups.csvへ保存しています。energyの改善は追加の古典CI計算によるもので、新しい量子測定や良い回路を発見した成果には加算しません。',
        '逆に、小さいCIでの順位が変わるなら、その容量での運用上の結果と、sampler本来が集めた情報量を分けて考える必要があります。','',
        '以下は主4手法すべての10seedが、r32とr64の両方でそろった条件だけの比較です。「最小」は観測された平均値の大小であり、有意差の判定ではありません。未完了・未triggerの欄から勝者を選びません。','',
        '| 条件 | 総model Shot | r32の平均energy最小法 | r64の平均energy最小法 | r64最小平均energy Ha | r64のRAACSとの差 mHa |',
        '|---|---:|---|---|---:|---:|']
    for case,label in CASES.items():
        for B in(10000,100000,1000000):
            c=capacity_comparison(index,case,B)
            if c is None:
                lines.append(f'| {label} | {B} | 全4手法の両容量比較は未成立 | 未完了または未trigger | — | — |');continue
            lines.append(f"| {label} | {B} | {NAMES[c['r32_lowest_mean_method']]} | {NAMES[c['r64_lowest_mean_method']]} | {c['r64_lowest_mean_energy_Ha']:.8f} | {c['r64_RAACS_minus_lowest_mean_mHa']:.3f} |")
    lines+=['','最後の差はRAACS平均−同条件の最小平均です。正ならRAACSのenergyがその分高く、0なら観測平均の最小法です。同じcountsの再解析なので追加Shotは0ですが、CI計算は追加費用です。r64でも上限に達した場合、さらに大きいCIでの順位は未確認として残します。',
        'r32で小回路と精度が両立して見えても、r64で他法のenergyが下がるなら、その小回路を同精度での無条件な優位とは扱いません。回路のCNOT/Depthはこの読出し変更だけでは変わりません。','',
        '## 強い対照まで含めた中予算の読み方','',
        '以下は100,000 model Shotで、主4手法に全L2/L4・SqDRIFT5/50・TN対照（ナフタレンでは512測定配分も）を加えた、登録gridの観測平均です。基本合成列の比較で、Rustiqは別図・別表に残します。全設定の予定seedがそろった行だけを記載します。',
        '主比較は10seed、追加対照は5seedで、TNは1個のoptimizer-sourceを5回測定しています。古典調整費用も異なるため、同総計算量・統計的優越の判定ではありません。後から最良設定を選んだ、記述的な比較であり、無料で選べる実運用ルールではありません。','',
        '| 条件 | r | 平均energy最小 | CNOT最少 | CNOT Depth最少 | Total Depth最少 | 係数座標数最少 |',
        '|---|---:|---|---|---|---|---|']
    for case,label in CASES.items():
        for r in(32,64):
            pool=[x for x in rows if x['case_id']==case and x['lane']in('C_PRIMARY','S5_SENSITIVITY','S_TN_INDEPENDENT_FINAL')
                and int(x['budget'])==100000 and int(x['r_cap'])==r]
            if len(pool)!=11+int(case=='NAPH20-P1'):
                lines.append(f'| {label} | {r} | 全設定は未完了または未trigger | — | — | — | — |');continue
            best=min(pool,key=lambda x:float(x['mean_energy_Ha']))
            minima=resource_minima(pool)
            labels=['・'.join(name(m)for m in metric['methods'])if metric['methods']else'不明'for metric in minima]
            lines.append('| '+' | '.join([label,str(r),name(best['method_id']),*labels])+' |')
    lines+=['','例えば、短いSqDRIFTの方がRAACSより少ないCNOTで低い平均energyを出す条件があるなら、主4手法だけを見て「RAACSが最も小さいから有利」とは結論しません。各資源の最小法が違う場合も、単一の総合勝者へまとめません。全数値・散らばり・前処理費用と合わせて判断してください。','',
        '## ナフタレンの軌道基底','',
        '局在軌道比較は同じ物理HFとHamiltonian spectrumを保ち、全4手法をそろえています。主比較と別の10seedです。手法ごとに有利な軌道だけ選んで主結果へ混ぜません。','',
        '| 方法 | canonical回収率 % | block-Boys回収率 % | canonical CNOT | block-Boys CNOT |','|---|---:|---:|---:|---:|']
    for method in('A_RAACS_CURRENT','B_LUCJ_NAIVE_L1','C_LUCJ_COMPRESSED_L1','D_SQDRIFT_N20'):
        c=index.get(('C_PRIMARY','NAPH20-P1',method,'100000','32'))
        b=index.get(('S_ORBITAL_BASIS','NAPH20-BLOCK-BOYS',method,'100000','32'))
        def val(row,k):return '未完了'if row is None or not row.get(k)else f'{float(row[k]):.3f}'
        lines.append('| '+' | '.join([NAMES[method],val(c,'mean_correlation_recovery_pct'),val(b,'mean_correlation_recovery_pct'),val(c,'mean_CNOT'),val(b,'mean_CNOT')])+' |')
    lines+=['','これは占有/非占有を別々にBoys局在化した登録variantです。元論文の未報告LO設定の再現ではありません。',
        'Recoveryは適合する公開/評価用参照に対する相関energy回収率です。参照の精度・sector未認証、parameter座標数と独立自由度の違い、古典HCIの容量制限、TN/P/Hの時間・容量上限と原著runtimeとの差は引き続き制約です。']
    atomic_bytes(ROOT/'analysis-v1/CASE_INTERPRETATION_JA.md',('\n'.join(lines)+'\n').encode())
    save_json(ROOT/'analysis-v1/case-interpretation-status.json',{'complete_seed_groups_described':len(rows),
        'all_capacity_differences':len(capacity),'statistical_or_universal_superiority_claimed':False,'whole_experiment_complete':False})
    print(json.dumps({'complete_groups':len(rows),'capacity_comparisons':len(capacity)}),flush=True)


if __name__=='__main__':run()
