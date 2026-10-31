"""N71 posthoc source-distribution diagnosis; no extra draws or energies."""
from pathlib import Path
from time import process_time,perf_counter
import math
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .correlated_source_qualification import LOCAL as N69
from .pair_group_finite import LOCAL as N70
from .paired_lowshot import paired_sector
from paperstudy.lucj import amplitudes_from_saved
from paperstudy.ci import interleaved_to_blocked

def half_probabilities(prob,n):
    out={};mask=(1<<n)-1
    for b,p in prob.items():
        # Probability a single shot includes x in either spin half, not two independent shots.
        for x in {int(b)&mask,int(b)>>n}:out[x]=out.get(x,0.)+p
    return out

def expected_unique(probabilities,shots):
    universe=set().union(*(p.keys() for p in probabilities));total=0.
    for x in universe:
        log_missing=0.
        for p,s in zip(probabilities,shots):
            mass=min(1.,max(0.,p.get(x,0.)))
            if mass==1:log_missing=-math.inf;break
            log_missing+=s*math.log1p(-mass)
        total+=-math.expm1(log_missing)
    return total

def run():
    p=load(N70/'protocol.json');t=load(N70/'terminal.json');cpu,wall=process_time(),perf_counter();rows=[]
    bindings={str(N70/'protocol.json'):sha256((N70/'protocol.json').read_bytes()),str(N70/'terminal.json'):sha256((N70/'terminal.json').read_bytes())}
    for s in p['sources']:
        meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k);ids,v,_=paired_sector(n,k,t2)
        base={int(b)|(int(b)<<n):float(abs(a)**2) for b,a in zip(ids,v)};probabilities=[];details=[]
        for time,shots in p['time_shots']:
            r=next(r for r in p['states'] if r['case']==s['case'] and r['time']==time);f=Path(r['state_file'])
            assert sha256(f.read_bytes())==r['state_SHA256'];bindings[str(f)]=r['state_SHA256']
            with np.load(f,allow_pickle=False) as a:prob={interleaved_to_blocked(int(b),n):float(abs(v)**2) for b,v in zip(a['source_mode_ids'],a['amplitudes'])}
            tv=.5*sum(abs(base.get(x,0.)-prob.get(x,0.)) for x in base.keys()|prob.keys())
            nonpair=sum(mass for b,mass in prob.items() if (b&((1<<n)-1))!=(b>>n))
            details.append(dict(time=time,total_variation_from_pair=tv,expected_nonpair_shots=shots*nonpair,shots=shots))
            probabilities.append(prob)
        observed={}
        for arm in p['arms']:
            d=next(d for d in t['draws'] if d['case']==s['case'] and d['arm']==arm)
            with np.load(d['path'],allow_pickle=False) as a:
                observed[arm]=dict(observed_nonpair_shots=sum(int(c) for b,c in zip(a['blocked_ids'],a['counts']) if (int(b)&((1<<n)-1))!=(int(b)>>n)),
                    observed_joint_unique=len(a['counts']),observed_half_unique=len(d['rank'])-int((1<<k)-1 not in half_probabilities(dict(zip(map(int,a['blocked_ids']),map(int,a['counts']))),n)))
        shotlist=[s for _,s in p['time_shots']]
        rows.append(dict(case=s['case'],amplitude_kind=kind,per_time=details,
            expected_nonpair_shots=sum(r['expected_nonpair_shots'] for r in details),
            minimal_expected_differing_draws_under_maximal_coupling=sum(r['shots']*r['total_variation_from_pair'] for r in details),
            expected_joint_unique_pair=expected_unique([base],[1000]),expected_joint_unique_hybrid=expected_unique(probabilities,shotlist),
            expected_half_unique_pair=expected_unique([half_probabilities(base,n)],[1000]),
            expected_half_unique_hybrid=expected_unique([half_probabilities(pr,n) for pr in probabilities],shotlist),observed=observed))
    summary=dict(version='N71_POSTHOC_DISTRIBUTION_DIAGNOSTIC_NOT_CAUSAL_CONFIRMATION',rows=rows,bindings=bindings,
        new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,goal_complete=False)
    save(ROOT/'analysis/N71-pair-group-coverage-diagnostic.json',summary)
    lines=['# N71：なぜ一groupの改善が限定的だったのか','',
        'N70後の探索的診断。既存のsource分布と実countsのみを使用し、新Shot・CI・energy評価は0。',
        '非pair配置を増やすこと自体はenergy改善の保証ではない。期待する異なる観測数は最大couplingにおける確率論的な値で、実際に共通乱数実験を行ったという意味ではない。','',
        '| 条件 | 1,000回中の期待非pair観測 | 実非pair観測 | 期待半string種類 pair→group | 期待joint種類 pair→group | 最小期待相違観測数 |',
        '|---|---:|---:|---|---|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['expected_nonpair_shots']:.3f} | {r['observed']['PAIR_PLUS_GROUP']['observed_nonpair_shots']} | {r['expected_half_unique_pair']:.3f}→{r['expected_half_unique_hybrid']:.3f} | {r['expected_joint_unique_pair']:.3f}→{r['expected_joint_unique_hybrid']:.3f} | {r['minimal_expected_differing_draws_under_maximal_coupling']:.3f} |")
    lines+=['','異なる乱数streamを使った一回のN70のenergy差を、追加groupの因果効果や平均効果と断定しない。',
        '期待coverageは参照energyを使わず分布から算出したが、重要な配置を覆っているかは別問題。全数値支持にはroundoff級の尾部も含まれ得るため、支持数そのものを実機で取れる種類数として扱わない。','']
    write(ROOT/'analysis/N71_PAIR_GROUP_COVERAGE_DIAGNOSTIC_JA.md','\n'.join(lines).encode())
    print('N71_DIAGNOSED4_NO_NEW_OUTCOMES',flush=True)

if __name__=='__main__':run()
