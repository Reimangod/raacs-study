"""N69/N70 derived evidence; replay actual model draws, report all controls."""
from pathlib import Path
from io import StringIO
import csv
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .correlated_source_qualification import LOCAL as N69,paired_check,l2,orbit_action
from .pair_group_preflight import LOCAL as N67
from .exact_group_synthesis import LOCAL as N68,FIELDS
from .pair_group_finite import LOCAL,sample,seed
from .paired_lowshot import paired_sector,ranked_half
from .analyze import reference
from paperstudy.lucj import amplitudes_from_saved
from paperstudy.ci import interleaved_to_blocked

def run():
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json');q=load(N69/'protocol.json');qt=load(N69/'terminal.json')
    for f,h in {**q['bindings'],**p['bindings']}.items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    assert len(qt['rows'])==12 and qt['status']=='ALL12_CORRELATED_SOURCE_TWO_ROUTE_CHECKS_PASSED'
    assert len(t['draws'])==22 and len(t['rows'])==66
    samples=0;draws={};resources={};replays=0
    old67=load(N67/'terminal.json');old68=load(N68/'terminal.json')
    for s in p['sources']:
        meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
        half,v,_=paired_sector(n,k,t2);pairids=half|(half<<np.uint64(n))
        pairsource=next(r for r in old67['rows'] if r['case']==s['case'])
        pairres=min(pairsource['pair'],key=lambda x:tuple(x['resources'][field] for field in FIELDS))['resources']
        resources[(s['case'],'PAIR_ONLY')]=dict(mean={key:float(pairres[key]) for key in FIELDS},maximum={key:pairres[key] for key in FIELDS})
        points=[(shots,next(r for r in old68['rows'] if r['case']==s['case'] and r['time']==time)['selected']['resources']) for time,shots in p['time_shots']]
        resources[(s['case'],'PAIR_PLUS_GROUP')]=dict(mean={key:sum(shots*r[key] for shots,r in points)/1000 for key in FIELDS},maximum={key:max(r[key] for _,r in points) for key in FIELDS})
        for d in [d for d in t['draws'] if d['case']==s['case']]:
            f=Path(d['path']);assert sha256(f.read_bytes())==d['SHA256']
            with np.load(f,allow_pickle=False) as a:
                ids=a['blocked_ids'];counts=a['counts'];observed=dict(zip(map(int,ids),map(int,counts)))
            assert sum(observed.values())==1000 and ranked_half(ids,counts,n,(1<<k)-1)==d['rank']
            assert all((int(b)&((1<<n)-1)).bit_count()==k and (int(b)>>n).bit_count()==k for b in ids)
            if d['arm'] in p['arms']:
                combined={}
                for m in d['members']:
                    if d['arm']=='PAIR_ONLY':ii,vv=pairids,v
                    else:
                        r=next(r for r in qt['rows'] if r['case']==s['case'] and r['time']==m['time'])
                        assert sha256(Path(r['state_file']).read_bytes())==r['state_SHA256']==m['source_SHA256']
                        with np.load(r['state_file'],allow_pickle=False) as a:
                            ii=np.array([interleaved_to_blocked(int(b),n) for b in a['source_mode_ids']],np.uint64);vv=a['amplitudes']
                    assert m['seed']==seed(meta['problem_id'],d['arm'],m['time'],m['source_SHA256'])
                    xx,cc=sample(ii,vv,m['shots'],m['seed'])
                    for b,c in zip(xx,cc):combined[int(b)]=combined.get(int(b),0)+int(c)
                    replays+=1
                assert observed==combined;samples+=1000
            draws[(d['case'],d['arm'])]=d
    assert samples==8000 and replays==24
    for r in t['rows']:
        if r['status']=='COMPLETED':
            z=r['root'];assert sha256(Path(z['root_file']).read_bytes())==z['root_SHA256']
            assert z['result']['residual_norm']<=1e-8 and max(z['independent_RDM_audit']['errors'].values())<=1e-7
    n66=load(ROOT/'analysis/N66-full20-breadth-summary.json')['rows'];n67=load(ROOT/'analysis/N67-pair-group-resource-summary.json')['controls']
    flat=[]
    for r in t['rows']:
        d=draws[(r['case'],r['arm'])];hf,ref=reference(r['case']);res=resources.get((r['case'],r['arm']))
        if r['arm']=='REFERENCE_FULL_N20_30':
            z=next(z for z in n66 if z['case']==r['case'] and z['arm']=='FULL_N20' and z['ensemble_size']==30)
            res=dict(mean={key:z[key] for key in FIELDS},maximum={key:z[key+'_max'] for key in FIELDS})
        elif r['arm'].startswith('C5_C_LUCJ'):
            c=next(c for c in n67 if c['case']==r['case'] and c['method']==r['arm'])
            best=min(c['recounted_points'],key=lambda x:tuple(x['resources'][key] for key in FIELDS))['resources']
            res=dict(mean={key:float(best[key]) for key in FIELDS},maximum={key:best[key] for key in FIELDS})
        energy=r['energy_Ha'];flat.append(dict(case=r['case'],arm=r['arm'],context=r['context'],status=r['status'],actual_R=r['actual_R'],actual_D=r['actual_D'],
            energy_Ha=energy,Recovery_pct=100*(hf-energy)/(hf-ref) if energy is not None else None,
            reference_warning='UNVERIFIED_SCI_NOT_FCI' if r['case']=='FE2S2-40-P2' else 'EXISTING_REFERENCE_NOT_NEW_FCI',
            spin_square=r['root'].get('result',{}).get('spin_square'),
            **{key:res['mean'][key] for key in FIELDS},**{key+'_max':res['maximum'][key] for key in FIELDS},
            independent_parameter_rank=None,optimized_variables=0,model_samples_used=1000,actual_new_model_samples=d['actual_new_model_samples'],
            source_role=d['role'],new_QPU_shots=0,basis_id=r['basis_id']))
    contrasts=[]
    for s in p['sources']:
        for context in ['R32_CAP','R128_CAP','COMMON_ACTUAL_R']:
            local=[r for r in flat if r['case']==s['case'] and r['context']==context];hybrid=next(r for r in local if r['arm']=='PAIR_PLUS_GROUP')
            for ref in [r for r in local if r['arm']!='PAIR_PLUS_GROUP']:
                contrasts.append(dict(case=s['case'],context=context,reference=ref['arm'],actual_R=hybrid['actual_R'],reference_R=ref['actual_R'],
                    delta_energy_mHa=1000*(hybrid['energy_Ha']-ref['energy_Ha']) if hybrid['energy_Ha'] is not None and ref['energy_Ha'] is not None else None,
                    **{key+'_reduction_pct':100*(ref[key]-hybrid[key])/ref[key] if ref[key] else None for key in FIELDS}))
    summary=dict(status='ALL66_ROWS_AND8000_NEW_SAMPLES_AUDITED',rows=flat,contrasts=contrasts,
        actual_new_model_samples=samples,new_QPU_shots=0,actual_member_replays=replays,actual_new_CI_roots=t['actual_new_CI_roots'],
        failed_CI_attempts=t['failed_CI_attempts'],source_CPU_seconds=t['source_CPU_seconds'],CI_CPU_seconds=t['CI_CPU_seconds'],
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],matched_LUCJ_compiler=False,matched_cold_work=False,
        independent_confirmation=False,goal_complete=False)
    save(ROOT/'analysis/N70-pair-group-finite-summary.json',summary)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    write(ROOT/'analysis/N70-all66-fields.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda value,digits=3:'欠測' if value is None else f'{value:.{digits}f}'
    lines=['# N70：pairに一groupを加える価値を実際の有限サンプルで検査','',
        '全4条件に新規2arm×1,000、合計8,000 model samplesを実取得。実機Shot0。時間1/2/3に334/333/333を固定配分し、良い時間だけを選んでいない。',
        '既存FullN20とLUCJは参考再評価。LUCJは過去100,000観測からの1,000 subsetでありcold workや独立seedの公平性を満たす確認実験ではない。',
        'Recoveryは既存参照に対する相関energyの回収率。Feの参照は未検証SCIでFCIではない。全根はphysical H+0.1S²/noOOで求め、物理energyを表示。',
        'CNOT/Depthは登録logical規則、各回路のShot加重平均。最大値はCSVに併記。独立Parametersは未検証で0としない。LUCJとの追加compiler機会は未整合。','',
        '| 条件 | 手法 | 評価 | 実R/D | E Ha | Recovery % | CNOT | CX Depth | Total Depth | 状態 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in flat:lines.append(f"| {r['case']} | {r['arm']} | {r['context']} | {r['actual_R']}/{r['actual_D']} | {fmt(r['energy_Ha'],9)} | {fmt(r['Recovery_pct'])} | {fmt(r['CNOT'])} | {fmt(r['CNOT_Depth'])} | {fmt(r['Total_Depth'])} | {r['status']} |")
    lines+=['','## 追加groupの利益：PAIR_PLUS_GROUP − PAIR_ONLY','',
        '| 条件 | R32上限のΔE mHa | 共通実RのΔE mHa | CNOT増分 | CX Depth増分 | Total Depth増分 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in p['sources']:
        a=next(r for r in contrasts if r['case']==s['case'] and r['context']=='R32_CAP' and r['reference']=='PAIR_ONLY')
        b=next(r for r in contrasts if r['case']==s['case'] and r['context']=='COMMON_ACTUAL_R' and r['reference']=='PAIR_ONLY')
        rr={r['arm']:r for r in flat if r['case']==s['case'] and r['context']=='R32_CAP' and r['arm'] in ['PAIR_ONLY','PAIR_PLUS_GROUP']}
        lines.append(f"| {s['case']} | {fmt(a['delta_energy_mHa'],6)} | {fmt(b['delta_energy_mHa'],6)} | "+' | '.join(fmt(rr['PAIR_PLUS_GROUP'][key]-rr['PAIR_ONLY'][key]) for key in FIELDS)+' |')
    lines+=['','## 限界と費用','',
        f"新CI根{t['actual_new_CI_roots']}、失敗{t['failed_CI_attempts']}。source CPU {t['source_CPU_seconds']:.6f}秒、CI処理CPU {t['CI_CPU_seconds']:.6f}秒、実行wall {t['wall_seconds']:.6f}秒。",
        'これらは増分で、初期CCSD/CISD・N67/N68/N69・過去source/CI費用は別。完全cold総費用の優越を主張しない。',
        '小unitary/paired encoding/二経路source検証は通過したが、大40qubit実QPY全体の独立gate-level状態検証は未完。',
        '一seedの開発screenであり、独立確認・実機noise優越・最強論文への総合勝利は示していない。欠測LUCJはナフタレンのL2/L4で、無かったものとして消さない。','']
    write(ROOT/'analysis/N70_PAIR_GROUP_FINITE_RESULTS_JA.md','\n'.join(lines).encode())
    n69lines=['# N69：全12相関sourceの二経路照合','',
        f"全12点成功。CPU {qt['CPU_seconds']:.6f}秒、wall {qt['wall_seconds']:.6f}秒。新Shot/CI/energy/GPUは0。最大複素L2差 {max(r['complex_L2_error'] for r in qt['rows']):.6g}。",'',
        '| 条件 | 時間 | 支持数 | 複素L2差 | 非pair確率 % |','|---|---:|---:|---:|---:|']
    for r in qt['rows']:n69lines.append(f"| {r['case']} | {r['time']} | {r['orbit']['physical_support']} | {r['complex_L2_error']:.3g} | {100*r['nonpaired_probability']:.6f} |")
    n69lines+=['','非pair確率はsourceの性質であり、energy改善や1,000 samplesで実際に発見する確率とは同一ではない。',
        '40qubit全Fock vectorを作らず、最大20qubit paired encodingとrank1 orbitで照合。大QPY全状態検証と同一視しない。','']
    write(ROOT/'analysis/N69_CORRELATED_SOURCE_QUALIFICATION_RESULTS_JA.md','\n'.join(n69lines).encode())
    print('N70_AUDITED8000_66',t['actual_new_CI_roots'],flush=True)

if __name__=='__main__':run()
