"""Audit N64 exact-model support diagnosis without any further sampling."""
from pathlib import Path
from time import process_time,perf_counter
import gzip,json
import numpy as np
from .paths import ROOT
from .support_saturation import LOCAL,half_distribution,member_state,interleaved_to_blocked
from .io import load,save,write,sha256

def run():
    cpu,wall=process_time(),perf_counter()
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h
    assert len(t['rows'])==24 and t['new_model_samples']==t['new_QPU_shots']==0
    assert t['CPU_seconds']<=p['CPU_cap'] and t['wall_seconds']<=p['wall_cap']
    finite=[]
    for f in p['cache_files'][:2]:finite+=load(f)['states']
    audits=[]
    for r in t['rows']:
        assert r['status']=='DISTRIBUTION_COMPLETED'
        assert -1e-8<=r['observed_half_probability_mass']<=1+1e-8
        for e in r['evaluations']:
            root=e['root'];assert root['status']=='COMPLETED'
            assert sha256(Path(root['root_file']).read_bytes())==root['root_SHA256']
            assert max(root['independent_RDM_audit']['errors'].values())<=1e-7
            assert root['result']['residual_norm']<=1e-8
        e=next(e for e in r['evaluations'] if e['R_cap']==128)
        assert e['full_model_half_support_used']
        f=next(f for f in finite if f['draw_id']==r['draw_id'] and f['R_cap']==128 and f.get('context','MAIN')=='MAIN')
        with np.load(e['root']['root_file'],allow_pickle=False) as a:exact=set(map(tuple,a['joint_pairs']))
        with np.load(f['root']['root_file'],allow_pickle=False) as a:observed=set(map(tuple,a['joint_pairs']))
        assert observed<=exact
        x,y=e['root']['result'],f['root']['result']
        assert x['penalized_energy_Ha']<=y['penalized_energy_Ha']+1e-8
        item=next(x for x in p['draws'] if x['draw']['draw_id']==r['draw_id']);d=item['draw'];base=Path(item['base'])
        meta=load(next(c['meta'] for c in p['cases'] if c['case']==r['case']));prob={}
        for m in d['members']:
            packet=json.loads(gzip.decompress((base/'exports'/r['case']/'members'/m['member_id']/'member.json.gz').read_bytes()))
            state,_=member_state(meta,packet)
            for bitstring,amplitude in state.items():
                bitstring=interleaved_to_blocked(int(bitstring),meta['norb']);prob[bitstring]=prob.get(bitstring,0.)+m['shots']/1000*float(abs(amplitude)**2)
        weights,_=half_distribution(prob,meta['norb'],(1<<meta['nalpha'])-1)
        positive=[w for w in weights.values() if w>0]
        tail_profile={str(e):sum(0<w<=e for w in weights.values()) for e in [1e-30,1e-20,1e-12,1e-8,1e-4]}
        audits.append(dict(case=r['case'],arm=r['arm'],seed=r['seed'],half_support=r['half_support'],observed_half_count=r['observed_half_count'],
            positive_half_probability_min=min(positive),half_support_tail_counts=tail_profile,
            observed_half_probability_mass=r['observed_half_probability_mass'],finite_energy_Ha=y['energy_Ha'],full_support_energy_Ha=x['energy_Ha'],
            delta_energy_mHa=1000*(x['energy_Ha']-y['energy_Ha']),delta_penalized_energy_mHa=1000*(x['penalized_energy_Ha']-y['penalized_energy_Ha']),
            finite_R=f['actual_R'],full_R=e['actual_R'],finite_space_contained=True))
    groups=[]
    for case in sorted({r['case'] for r in audits}):
        for arm in ['FULL_N5','FILTERED_SHORT']:
            a=[r for r in audits if r['case']==case and r['arm']==arm];assert len(a)==3
            groups.append(dict(case=case,arm=arm,half_support=[r['half_support'] for r in a],finite_R=[r['finite_R'] for r in a],
                finite_energy_Ha=float(np.mean([r['finite_energy_Ha'] for r in a])),full_support_energy_Ha=float(np.mean([r['full_support_energy_Ha'] for r in a])),
                delta_energy_mHa=float(np.mean([r['delta_energy_mHa'] for r in a])),
                observed_probability_mass_min=min(r['observed_half_probability_mass'] for r in a)))
    s=dict(status='ALL24_SUPPORT_AND48_CI_FIELDS_AUDITED',rows=audits,groups=groups,actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in t['attempts']),
        failed_CI_attempts=sum(a['status']!='COMPLETED' for a in t['attempts']),CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],
        new_model_samples=0,new_QPU_shots=0,all24_full_support_at_R128=True,all24_finite_spaces_contained=True,
        exact_probability_is_privileged=True,support_is_numerical_not_analytic_and_may_include_roundoff=True,
        not_universal_samplability_or_energy_necessity_proof=True,independent_confirmation=False,goal_complete=False)
    save(ROOT/'analysis/N64-support-summary.json',s)
    lines=['# N64：同じ回路を測り続けるべきか','',
        '追補：[N64Bの数値尾部感度検査](N64B_NUMERICAL_TAIL_RESULTS_JA.md)を併読すること。以下の全支持oracle改善の一部は極小の数値残差配置に依存し、実際の追加測定による改善と見なせない。',
        '24保存drawの全支持を数値モデルで再構成し、48評価欄を監査した。新Shotは0。確率が見えるシミュレータ限定の診断で、実機の新手法・総合優越ではない。',
        '全24件でR128以下に数値上の正確率支持が収まり、有限ShotのCI空間がその空間に含まれることと、H+0.1S²目的が悪化しないことを確認した。physical energyの一般的単調性は主張しない。',
        '重要：正確率支持には極小の浮動小数点残差が含まれ得る。解析的に非零と証明した支持ではなく、全配置を実際のShotで回収可能だとも主張しない。oracleによる改善幅を必要Shotの予測に使わない。','',
        '| 条件 | 方策 | 1,000回で見えた実R | 数値モデルの全half支持 | 有限Shot E Ha | 全支持 E Ha | 差 mHa | 観測half配置の確率質量・最小値 |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['case']} | {g['arm']} | {g['finite_R']} | {g['half_support']} | {g['finite_energy_Ha']:.9f} | {g['full_support_energy_Ha']:.9f} | {g['delta_energy_mHa']:.3f} | {100*g['observed_probability_mass_min']:.9f}% |")
    lines+=['','## 読み方と判断','',
        '通常N5は全支持を渡しても、4条件すべてで既存SHORTの有限Shot平均energyより高かった。固定9回路へのShot増加だけを次の救済策にはしない。これはこのensemble・同じ有限penalty目的の数値結果で、任意のN5回路や全てのspin目的に対する不可能性の証明ではない。',
        'SHORTでは99.94%以上のhalf確率質量を観測していても、稀な配置を補うとCI energyが変わる。「ほぼ全確率を観測＝energyが収束」は成り立たない。全支持には極小の数値確率の配置も含む。実際に何Shotで拾えるかは別問題で、oracleの改善幅を実用性能と偽らない。',
        '次は同じ総1,000 Shotで回路ensembleの幅を変える有限比較を検討する。現行9回路の繰返しから、より多い異なる回路に配る方向で、古典構築費用・コンパイル費用・切替数も増える。この方向は元のSqDRIFTにある考えで、新発明とはしない。元論文の多数回路対照を弱い9回路版だけで代用しない。',
        '対照条件を強化した後も利益が残らなければ、短縮方策に固執せず、既存LUCJ/TN/古典HCIとのsource改善へ戻る。過去のpair scale/phase救済、固定U fit救済、VQE期待値だけの調整は、すでに有利でなかったので繰り返さない。','',
        f"実費：新CI根{s['actual_new_CI_roots']}、失敗{s['failed_CI_attempts']}、body CPU {s['CPU_seconds']:.6f}秒、wall {s['wall_seconds']:.6f}秒。cacheは新根ではない。import前・過去の全探索費用は含まない。",'']
    for case in sorted({g['case'] for g in groups}):
        n=next(g for g in groups if g['case']==case and g['arm']=='FULL_N5');b=next(g for g in groups if g['case']==case and g['arm']=='FILTERED_SHORT')
        assert n['full_support_energy_Ha']>b['finite_energy_Ha']
    lines+=['','## 極小確率の注意','',
        '| 条件 | 方策 | seed | 最小正half確率 | 1e-12以下のhalf配置数 |','|---|---|---:|---:|---:|']
    for r in audits:lines.append(f"| {r['case']} | {r['arm']} | {r['seed']} | {r['positive_half_probability_min']:.3e} | {r['half_support_tail_counts']['1e-12']} |")
    s['derived_audit_CPU_seconds']=process_time()-cpu;s['derived_audit_wall_seconds']=perf_counter()-wall
    save(ROOT/'analysis/N64-support-summary.json',s)
    write(ROOT/'analysis/N64_SUPPORT_SATURATION_RESULTS_JA.md','\n'.join(lines).encode())
    print('N64_AUDITED24_48_FIELDS',s['actual_new_CI_roots'],flush=True)

if __name__=='__main__':run()
