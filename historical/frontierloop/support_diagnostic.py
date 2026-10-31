"""Reuse paid counts to inspect probability/CI-support loss, with all missing rows."""
from collections import Counter
from io import StringIO
import csv
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from .marginal_remote_worker import select_half_pool


def half_hist(directory,half,n):
    if not (directory/'draw-result.json').exists():return None
    r=load(directory/'draw-result.json');p=directory/('half-counts.npz' if half else 'final-counts.npz')
    if sha256(p.read_bytes())!=r['counts_SHA256']:raise RuntimeError('Diagnostic raw count binding failed')
    frequency=Counter()
    with np.load(p,allow_pickle=False) as a:
        if half:
            for x,c in zip(a['half_strings'],a['counts']):frequency[int(x)]+=int(c)
        else:
            mask=(1<<n)-1
            for x,c in zip(a['blocked_ids'],a['counts']):
                frequency[int(x)&mask]+=int(c);frequency[int(x)>>n]+=int(c)
    return frequency


def main():
    cases=load(ROOT/'protocol-D2-COMPRESSION-SCREEN.json')['cases'];rows=[]
    for case in cases:
        m=load(OLD/'data/physical'/case/'problem.json');n=m['norb'];na=m['nalpha']
        baseline=ROOT/'runs/D2-COMPRESSION-SCREEN'/case/'EXACT';target=half_hist(baseline,False,n)
        sources=[('D2_'+p,ROOT/'runs/D2-COMPRESSION-SCREEN'/case/p,False) for p in ['EXACT','F25','F50','F75']]
        sources +=[(name,ROOT/'runs'/folder/'remote-output/outputs'/case,True) for name,folder in [('F2','F2-POSITIVE-MARGINAL-v1'),('H1','H1-HALF-REMAP-v1')]]
        for method,d,half in sources:
            actual=half_hist(d,half,n)
            r={'case':case,'method':method,'status':'PAIRED_EMPIRICAL_DIAGNOSTIC' if actual and target else 'REFERENCE_NOT_READY' if not target else 'CANDIDATE_NOT_READY',
                'model_shots':sum(actual.values())//(1 if half else 2) if actual else None,
                'half_configurations':sum(actual.values()) if actual else None,
                'reference_half_configurations':sum(target.values()) if target else None,
                'common_R_cap':64,'pool_overlap_count':None,'pool_union_count':None,
                'selected_pool_probability_under_reference_histogram':None,'reference_top64_probability':None,
                'empirical_Jensen_Shannon_nats_biased_estimator':None,'new_model_shots':0}
            if actual and target:
                def pool(freq):
                    ids=np.array(sorted(freq),np.uint64);counts=np.array([freq[int(x)] for x in ids],np.uint64)
                    return set(map(int,select_half_pool(ids,counts,n,na,64)))
                a,b=pool(actual),pool(target);keys=sorted(set(target)|set(actual));x=np.array([target[k] for k in keys],float);y=np.array([actual[k] for k in keys],float)
                x/=x.sum();y/=y.sum();mid=.5*(x+y)
                kl=lambda p:float(np.sum(p[p>0]*np.log(p[p>0]/mid[p>0])))
                r.update(pool_overlap_count=len(a&b),pool_union_count=len(a|b),
                    selected_pool_probability_under_reference_histogram=sum(target[k] for k in a)/sum(target.values()),
                    reference_top64_probability=sum(target[k] for k in b)/sum(target.values()),
                    empirical_Jensen_Shannon_nats_biased_estimator=.5*(kl(x)+kl(y)))
            rows.append(r)
    out=StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/support-diagnostic-all.csv',out.getvalue().encode('utf-8-sig'))
    save(ROOT/'analysis/support-diagnostic-summary.json',{'rows':rows,'new_model_shots':0,'causal_mechanism_proven':False,'goal_complete':False})
    lines=['# 回路を短くすると、どの配置が失われるか','',
        '既存の支払済みcountsだけを再利用した診断。Hamiltonian/FCIを使って都合のよい配置を選ばず、元のHF入り頻度上位64選択を保持した。全24行を残し、無圧縮Feのcountsが未到着ならその条件は比較不能とする。','',
        'F2は理論上同じ片スピン分布でも独立有限Shotのため完全な同じpoolにはならない。fullは200k本、F2/H1は100k本の配置を使う。下のJS距離は有限標本のbiasedなplug-in推定量で、正確な分布距離や因果効果の証明ではない。','']
    for case in cases:
        lines+=['## '+case,'','| 方法 | 状態 | 参照64配置との共通数 | 選択poolが持つ参照分布の質量 | 経験的JS距離 |','|---|---|---:|---:|---:|']
        for r in rows:
            if r['case']!=case:continue
            f=lambda v:'欠測' if v is None else f'{v:.6f}'
            lines.append(f"| {r['method']} | {r['status']} | {r['pool_overlap_count'] if r['pool_overlap_count'] is not None else '欠測'} | {f(r['selected_pool_probability_under_reference_histogram'])} | {f(r['empirical_Jensen_Shannon_nats_biased_estimator'])} |")
    lines+=['','## 判断に使う範囲','',
        '大きいunitary誤差で支持が変わっていれば、HS距離だけのcompiler最適化がSQD用途と合わない可能性がある。ただし、参照の高確率配置が最終CIで最も重要とは限らず、この表だけでenergy悪化の原因を確定しない。小さいJSや大きいoverlapも精度保証ではない。','',
        '次にsampling-aware圧縮を作る場合、使ったteacher counts・構築CPU・全探索費用を計上する。参照energyやFCI係数を無料教師にしない。改善した候補のfinal評価には未使用sampling seedを使い、ここで再利用したcountsを独立検証と数えない。','',
        '[全energy/資源](H1_F2_COMPARISON_JA.md)、[次の投資判断](F2_H1_NEXT_DECISION_JA.md)。']
    write(ROOT/'analysis/SUPPORT_DIAGNOSTIC_JA.md',('\n'.join(lines)+'\n').encode())
    print({'rows':len(rows),'paired':sum(r['status']=='PAIRED_EMPIRICAL_DIAGNOSTIC' for r in rows),'new_model_shots':0},flush=True)


if __name__=='__main__':main()
