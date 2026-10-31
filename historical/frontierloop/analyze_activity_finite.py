"""N72--N74 evidence audit and all-source conditional paired comparisons."""
from pathlib import Path
from io import BytesIO,StringIO
from collections import Counter
import csv,gzip
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .state_activity_preflight import LOCAL as N72,score,choose
from .activity_resource import LOCAL as N73
from .activity_resource_portfolio import LOCAL as N73B,choose_admissible
from .activity_finite import LOCAL,common_seed,inverse_cdf
from .pair_group_preflight import LOCAL as N67,metrics
from .exact_group_synthesis import LOCAL as N68,FIELDS
from .paired_lowshot import paired_sector,ranked_half
from .analyze import reference
from paperstudy.lucj import amplitudes_from_saved
from paperstudy.ci import blocked_to_interleaved,interleaved_to_blocked

def resource_map():
    p=load(LOCAL/'protocol.json');old67=load(N67/'terminal.json');old68=load(N68/'terminal.json');portfolio=load(N73B/'terminal.json');result={}
    for s in p['sources']:
        r=next(r for r in old67['rows'] if r['case']==s['case']);best=min(r['pair'],key=lambda x:tuple(x['resources'][key] for key in FIELDS))['resources']
        result[(s['case'],'PAIR_ONLY')]=dict(mean={k:float(best[k]) for k in FIELDS},maximum={k:best[k] for k in FIELDS})
        envelope=next(c for c in portfolio['cases'] if c['case']==s['case'])['LUCJ_componentwise_resource_envelope']
        for arm,rr in [('RANDOM_GROUP',[r for r in old68['rows'] if r['case']==s['case']]),('ACTIVE_GROUP',[r for r in portfolio['rows'] if r['case']==s['case']])]:
            points=[]
            for r in rr:
                selected=choose_admissible(r['points'],envelope)
                if selected is None:raise ValueError('No admitted resource point')
                if arm=='ACTIVE_GROUP':assert selected==r['selected']
                points.append((r['time'],selected['resources']))
            result[(s['case'],arm)]=dict(mean={k:sum(shots*next(z[k] for t,z in points if t==time) for time,shots in p['time_shots'])/1000 for k in FIELDS},
                maximum={k:max(z[k] for _,z in points) for k in FIELDS})
    return result

def run():
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json');p72=load(N72/'protocol.json');t72=load(N72/'terminal.json');t73b=load(N73B/'terminal.json');raw73=load(N73/'terminal.json')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    assert len(t['draws'])==36 and len(t['rows'])==108 and len(t72['rows'])==120
    resources=resource_map();replays=0;total=0;score_error=0.;qpy_count=0
    for r in raw73['rows']:
        for point in r['points']:
            for role,key in [('preparation','preparation_resources'),('sampling','resources')]:
                f=Path(point['files'][role]['file']);blob=f.read_bytes();assert sha256(blob)==point['files'][role]['SHA256']
                assert metrics(qpy.load(BytesIO(gzip.decompress(blob)))[0])==point[key];qpy_count+=1
    assert qpy_count==96
    for s in p['sources']:
        meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,_=amplitudes_from_saved(s['amplitudes'],n,k);half,v,_=paired_sector(n,k,t2)
        sourceids=np.array([blocked_to_interleaved(int(b)|(int(b)<<n),n) for b in half],np.uint64)
        for c in [c for c in t72['rows'] if c['case']==s['case']]:
            actual=score(sourceids,abs(v)**2,c['group'],k);assert actual['status']==c['score']['status']
            if actual['probability'] is not None:score_error=max(score_error,abs(actual['probability']-c['score']['probability']))
        for r in [r for r in t72['selected'] if r['case']==s['case']]:
            assert r['selection']==choose([c for c in t72['rows'] if c['case']==s['case'] and c['time']==r['time']])
        cache={('PAIR_ONLY',time):(half|(half<<np.uint64(n)),v,sha256(v.tobytes())) for time,_ in p['time_shots']}
        for r in [r for r in p['states'] if r['case']==s['case']]:
            assert sha256(Path(r['state_file']).read_bytes())==r['state_SHA256']
            with np.load(r['state_file'],allow_pickle=False) as a:
                ids=np.array([interleaved_to_blocked(int(b),n) for b in a['source_mode_ids']],np.uint64);amps=a['amplitudes']
            cache[(r['arm'],r['time'])]=(ids,amps,r['state_SHA256'])
        for d in [d for d in t['draws'] if d['case']==s['case']]:
            combined={}
            for m in d['members']:
                ids,amps,h=cache[(d['arm'],m['time'])];assert h==m['source_SHA256']
                rngseed=common_seed(meta['problem_id'],d['replicate'],m['time']);assert rngseed==m['common_readout_seed']
                xx,cc=inverse_cdf(ids,amps,np.random.default_rng(rngseed).random(m['shots']))
                f=Path(m['counts_file']);assert sha256(f.read_bytes())==m['counts_SHA256']
                with np.load(f,allow_pickle=False) as a:assert np.array_equal(xx,a['blocked_ids']) and np.array_equal(cc,a['counts'])
                for b,c in zip(xx,cc):combined[int(b)]=combined.get(int(b),0)+int(c)
                replays+=1
            f=Path(d['counts_file']);assert sha256(f.read_bytes())==d['counts_SHA256']
            with np.load(f,allow_pickle=False) as a:
                assert combined==dict(zip(map(int,a['blocked_ids']),map(int,a['counts'])))
                assert ranked_half(a['blocked_ids'],a['counts'],n,(1<<k)-1)==d['rank']
                assert all((int(b)&((1<<n)-1)).bit_count()==k and (int(b)>>n).bit_count()==k for b in a['blocked_ids'])
            assert sum(combined.values())==1000;total+=1000
    assert replays==108 and total==36000 and score_error<1e-12
    seen=set();flat=[]
    for r in t['rows']:
        c=next(c for c in p['cases'] if c['case']==r['case']);meta=load(c['meta'])
        assert r['basis_id']==digest([meta['problem_id'],r['strings'],p['common_equation'],p['library_SHA256']])
        assert r['new_CI_attempt']==(r['basis_id'] not in seen);seen.add(r['basis_id']);assert r['actual_D']==r['actual_R']**2
        if r['status']=='COMPLETED':
            z=r['root'];assert sha256(Path(z['root_file']).read_bytes())==z['root_SHA256'];assert z['result']['residual_norm']<=1e-8 and max(z['independent_RDM_audit']['errors'].values())<=1e-7
        hf,ref=reference(r['case']);energy=r['energy_Ha'];res=resources[(r['case'],r['arm'])]
        flat.append(dict(case=r['case'],arm=r['arm'],replicate=r['replicate'],context=r['context'],status=r['status'],actual_R=r['actual_R'],actual_D=r['actual_D'],
            energy_Ha=energy,Recovery_pct=100*(hf-energy)/(hf-ref) if energy is not None else None,spin_square=r['root'].get('result',{}).get('spin_square'),
            **res['mean'],**{k+'_max':res['maximum'][k] for k in FIELDS},independent_parameter_rank=None,optimized_variables=0,new_model_samples_per_draw=1000,new_QPU_shots=0,
            reference_warning='UNVERIFIED_SCI_NOT_FCI' if r['case']=='FE2S2-40-P2' else 'EXISTING_REFERENCE',basis_id=r['basis_id']))
    contrasts=[];stats=[]
    for s in p['sources']:
        for context in ['R32_CAP','R128_CAP','COMMON_ACTUAL_R']:
            for ref in ['PAIR_ONLY','RANDOM_GROUP']:
                differences=[]
                for rep in p['replicates']:
                    a=next(r for r in flat if r['case']==s['case'] and r['context']==context and r['arm']=='ACTIVE_GROUP' and r['replicate']==rep)
                    b=next(r for r in flat if r['case']==s['case'] and r['context']==context and r['arm']==ref and r['replicate']==rep)
                    delta=1000*(a['energy_Ha']-b['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None
                    contrasts.append(dict(case=s['case'],context=context,reference=ref,replicate=rep,delta_energy_mHa=delta,actual_R=a['actual_R'],reference_R=b['actual_R']))
                    if delta is not None:differences.append(delta)
                stats.append(dict(case=s['case'],context=context,reference=ref,valid_pairs=len(differences),total_pairs=3,
                    mean_delta_mHa=float(np.mean(differences)) if differences else None,SD_delta_mHa=float(np.std(differences,ddof=1)) if len(differences)>1 else None,
                    min_delta_mHa=min(differences) if differences else None,max_delta_mHa=max(differences) if differences else None,
                    lower_count=sum(x<0 for x in differences),higher_count=sum(x>0 for x in differences),not_independent_source_ensembles=True))
    summary=dict(status='ALL108_STATUSES_AND36000_MODEL_SAMPLES_AUDITED',rows=flat,contrasts=contrasts,statistics=stats,
        new_model_samples=total,new_QPU_shots=0,actual_new_CI_roots=t['actual_new_CI_roots'],failed_CI_attempts=t['failed_CI_attempts'],
        source_CPU_seconds=t['source_CPU_seconds'],CI_CPU_seconds=t['CI_CPU_seconds'],CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],
        audit=dict(actual_member_replays=replays,QPY_recounts=qpy_count,activity_rows_recomputed=120,max_activity_error=score_error),
        matched_LUCJ_compiler_opportunity=False,full_cold_cost_complete=False,independent_source_confirmation=False,goal_complete=False)
    save(ROOT/'analysis/N74-activity-finite-summary.json',summary)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat);write(ROOT/'analysis/N74-all108-fields.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x,d=3:'欠測' if x is None else f'{x:.{d}f}'
    lines=['# N74：状態を動かすgroup選択は、SQD精度にも効いたか','',
        '全4条件×3手法×3読出し反復に各1,000、計36,000 model samples。実QPU0。固定sourceに対する有限読出し反復で、新しいsource ensembleの独立確認ではない。',
        '同じproblem・反復・時間の共通一様乱数を正しい逆CDFに入力。各armの周辺分布を変えず、Shot費用は3手法分として計上した。',
        '全108CI欄を表示。物理H+0.1S²/noOOによる根の物理energy。R上限と実R/Dは区別する。3反復の平均だけで統計的優越を主張しない。','',
        '## 主比較：ACTIVE_GROUP − 対照（負なら改善）','',
        '| 条件 | 評価 | 対照 | 平均ΔE mHa | SD mHa | 範囲 mHa | 低下した反復 | 有効/予定 |',
        '|---|---|---|---:|---:|---|---:|---|']
    for r in stats:lines.append(f"| {r['case']} | {r['context']} | {r['reference']} | {fmt(r['mean_delta_mHa'],6)} | {fmt(r['SD_delta_mHa'],6)} | {fmt(r['min_delta_mHa'],6)}〜{fmt(r['max_delta_mHa'],6)} | {r['lower_count']} | {r['valid_pairs']}/3 |")
    lines+=['','## 回路資源：同じ資源制約付きcompiler選択','',
        '| 条件 | 手法 | 加重CNOT | 加重CX Depth | 加重Total Depth | 最大CNOT/CX Depth/Total |','|---|---|---:|---:|---:|---|']
    for (case,arm),r in resources.items():lines.append(f"| {case} | {arm} | {r['mean']['CNOT']:.3f} | {r['mean']['CNOT_Depth']:.3f} | {r['mean']['Total_Depth']:.3f} | "+'/'.join(str(r['maximum'][k]) for k in FIELDS)+' |')
    lines+=['','## 全108評価欄','',
        '| 条件 | 手法 | 反復 | 評価 | 実R/D | E Ha | Recovery % | 状態 |','|---|---|---:|---|---|---:|---:|---|']
    for r in flat:lines.append(f"| {r['case']} | {r['arm']} | {r['replicate']} | {r['context']} | {r['actual_R']}/{r['actual_D']} | {fmt(r['energy_Ha'],9)} | {fmt(r['Recovery_pct'])} | {r['status']} |")
    lines+=['','## 既存の強い参考対照（N70のR32上限、今回の反復と混ぜない）','',
        '| 条件 | 参考手法 | 実R/D | E Ha | CNOT/CX Depth/Total |','|---|---|---|---:|---|']
    for r in load(ROOT/'analysis/N70-pair-group-finite-summary.json')['rows']:
        if r['context']=='R32_CAP' and r['arm'] not in ['PAIR_ONLY','PAIR_PLUS_GROUP']:
            lines.append(f"| {r['case']} | {r['arm']} | {r['actual_R']}/{r['actual_D']} | {fmt(r['energy_Ha'],9)} | "+'/'.join(fmt(r[k]) for k in FIELDS)+' |')
    lines+=['','## 費用と主張の限界','',
        f"新CI根{t['actual_new_CI_roots']}、失敗{t['failed_CI_attempts']}。source CPU {t['source_CPU_seconds']:.6f}秒、CI処理CPU {t['CI_CPU_seconds']:.6f}秒、計CPU {t['CPU_seconds']:.6f}秒、wall {t['wall_seconds']:.6f}秒。",
        f"状態活性preflight CPU {t72['CPU_seconds']:.6f}秒、resource計算CPU {raw73['CPU_seconds']:.6f}秒、portfolio再計数CPU {t73b['CPU_seconds']:.6f}秒は別に保持した。",
        '初期CCSD/CISD・旧source・旧検証の費用も消していない。完全cold費用の優越は未証明。LUCJ旧1000は100000のsubsetで、追加compiler機会もまだ整合していない。',
        'N73近平衡N₂のCNOT最優先によるdepth不合格を保存。N73Bで上限を緩めず、既存の等価compiler pointを制約内から選んだ。source/group/角度/精度条件は不変。',
        '独立ParametersはUNKNOWN、VQE最適化変数0とは別。Fe Recovery参照は未検証SCIでFCIではない。',
        '小全unitary・paired encoding・二経路source検査は通過済みだが、大40qubit実QPY全状態の独立検証は未完。最強論文への総合優越・実機noise優越・新規性は未証明。','']
    write(ROOT/'analysis/N74_ACTIVITY_FINITE_RESULTS_JA.md','\n'.join(lines).encode())
    pre=['# N72/N73：状態活性と回路資源の事前検査','',
        '| 条件 | 元の期待非pair/1000 | 選択後 | CNOT | CX Depth | Total Depth |','|---|---:|---:|---:|---:|---:|']
    for c in t72['cases']:
        z=next(z for z in t73b['cases'] if z['case']==c['case'])['weighted_mean']
        pre.append(f"| {c['case']} | {c['original_random_expected_observations_per1000']:.6f} | {c['expected_nonpair_observations_per1000']:.6f} | {z['CNOT']:.3f} | {z['CNOT_Depth']:.3f} | {z['Total_Depth']:.3f} |")
    pre+=['','全120候補を再計算し、選択12点と96QPYを監査。非pair化は精度の証明ではなく、有限Shot実験は別のN74に記録した。',
        'N73の元CNOT最優先選択では近平衡N₂のdepth平均67.677が上限67を超えた。N73Bで同じ96QPY内の2pointのみ選び直し、上限を緩めず平均40.663へ下げた。元の不合格を上書きしていない。','']
    write(ROOT/'analysis/N72_N73_ACTIVITY_PREFLIGHT_RESULTS_JA.md','\n'.join(pre).encode())
    print('N74_AUDITED36000_108',t['actual_new_CI_roots'],flush=True)

if __name__=='__main__':run()
