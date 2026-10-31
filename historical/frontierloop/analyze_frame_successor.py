"""N61 full-coverage, cost-aware paired summary and recorded-evidence audit."""
from collections import Counter
from io import BytesIO,StringIO
import csv,gzip,json,sqlite3
import numpy as np
from qiskit import qpy
from .filtered_SqDRIFT_plan import LOCAL,verified
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .strong_projection import choose_safe
from .analyze import reference
from paperstudy.ci import interleaved_to_blocked
from paperstudy.sqdrift_execution import member_state


def run():
    p=verified()
    available=[d for d in p['draws'] if (LOCAL/'draws'/d['draw_id']/'result.json').exists()]
    partial=False
    source=load(LOCAL/'source-frame-successor-v1/combined-source-terminal.json')
    ci=load(LOCAL/'partial-CI-terminal-v3.json')
    prior=load(LOCAL/'partial-CI-terminal-v1.json')
    ci=dict(ci,predecessor_failed_CI_attempts=prior['failed_CI_attempts'],
        predecessor_CI_CPU_seconds=prior['CPU_seconds'],
        predecessor_successful_CI_CPU_seconds=load(LOCAL/'partial-CI-terminal-v2.json')['CPU_seconds'])
    assert len(ci['states'])==48
    assert {x['draw_id'] for x in source['completed']}=={d['draw_id'] for d in available}
    assert len({(r['draw_id'],r['R_cap']) for r in ci['states']})==48
    draw_results={};maxprob=0.;actual_draws=0;recounts=0
    with sqlite3.connect('file:'+str(LOCAL/'shots.sqlite')+'?mode=ro',uri=True) as db:
        assert db.execute('select count(*),sum(actual) from draws where state="COMPLETED"').fetchone()==(9*len(available),1000*len(available))
        assert db.execute('select count(*) from draws where state!="COMPLETED"').fetchone()==(0,)
    for draw in available:
        folder=LOCAL/'draws'/draw['draw_id'];r=load(folder/'result.json');draw_results[draw['draw_id']]=r
        assert load(folder/'request.json')==draw
        assert r['counts_SHA256']==sha256((folder/'counts.npz').read_bytes())
        assert r['budget']['known_actual_shots']==r['new_model_samples']==1000
        assert r['budget']['unknown_or_inflight_draws']==0
        meta=load(next(c['meta'] for c in p['cases'] if c['case']==draw['case']));merged=Counter()
        checks=load(folder/'compiler-check.json')['results'];assert len(checks)==9
        for m,rec in zip(draw['members'],r['members']):
            memberfolder=LOCAL/'exports'/draw['case']/'members'/m['member_id']
            member=json.loads(gzip.decompress((memberfolder/'member.json.gz').read_bytes()))
            assert member['spec']==m['spec'] and digest(m['spec'])==m['member_id']
            assert member['diagonal_terms_retained']==(draw['arm']=='FULL_N20')
            assert rec['request']['source_packet_SHA256']==sha256((memberfolder/'member.json.gz').read_bytes())
            assert rec['policy']==choose_safe(rec['resources'])
            key=draw['case']+':'+str(m['index'])+':'+m['member_id'];check=checks[key]
            assert check['status']=='PASSED' and len(check['check']['circuits'])==3
            for item in check['check']['circuits'].values():
                maxprob=max(maxprob,item['probability_L1_with_leakage_bound'])
            for variant in ['NATIVE','RUSTIQ','PROJECTED']:
                b=(folder/(m['member_id']+'-'+variant+'.qpy.gz')).read_bytes()
                assert sha256(b)==rec['qpy_gz_SHA256'][variant]
                q=qpy.load(BytesIO(gzip.decompress(b)))[0];z=rec['resources'][variant]
                assert [int(q.count_ops().get('cx',0)),int(q.depth(filter_function=lambda x:x.operation.name=='cx')),int(q.depth())]==[z['cnot_count'],z['cnot_depth'],z['total_unitary_depth']]
                recounts+=1
            state,_=member_state(meta,member);ids=np.array(sorted(state),np.uint64)
            probs=np.array([abs(state[int(x)])**2 for x in ids]);probs/=probs.sum()
            expected=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],probs);keep=expected>0
            file=folder/(m['member_id']+'.npz');assert sha256(file.read_bytes())==rec['counts_SHA256']
            with np.load(file,allow_pickle=False) as a:
                assert np.array_equal(a['ids'],ids[keep]) and np.array_equal(a['counts'],expected[keep])
                for x,w in zip(a['ids'],a['counts']):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
            actual_draws+=1
        with np.load(folder/'counts.npz',allow_pickle=False) as a:
            assert dict(zip(map(int,a['blocked_ids']),map(int,a['counts'])))==dict(merged)
    observations={}
    for r in ci['states']:
        if r['status']=='COMPLETED':
            entry=r['root'];file=entry['root_file'];assert sha256(__import__('pathlib').Path(file).read_bytes())==entry['root_SHA256']
            assert max(entry['independent_RDM_audit']['errors'].values())<=1e-7
            assert entry['result']['residual_norm']<=1e-8
            observations[r['basis_id']]=entry
    assert len(observations)==ci['total_unique_roots_in_combined_report']
    groups=[];flat=[];pairs=[]
    for c in p['cases']:
        hf,ref=reference(c['case'])
        for cap in p['R_caps']:
            for arm in ['FULL_N20','FILTERED_SHORT']:
                rows=[r for r in ci['states'] if r['case']==c['case'] and r['arm']==arm and r['R_cap']==cap]
                assert len(rows)==3;ok=[r for r in rows if r['status']=='COMPLETED']
                draws=[draw_results[r['draw_id']] for r in rows if r['draw_id'] in draw_results]
                g=dict(case=c['case'],arm=arm,R_cap=cap,completed=len(ok),planned=3,
                    energy_Ha=float(np.mean([r['energy_Ha'] for r in ok])) if len(ok)==3 else None,
                    Recovery_pct=100*(hf-float(np.mean([r['energy_Ha'] for r in ok])))/(hf-ref) if len(ok)==3 else None,
                    actual_R=[r['actual_R'] for r in rows],actual_D=[r['actual_D'] for r in rows],
                    coefficient_slots_mean=float(np.mean([r['coefficient_slots_mean'] for r in draws])) if draws else None,
                    independent_parameter_rank=None,VQE_optimized_variables=0,
                    resources={name:{f:float(np.mean([d['resources'][name][f] for d in draws])) if draws else None for f in ['cnot_count','cnot_depth','total_unitary_depth']} for name in ['NATIVE','RUSTIQ','PROJECTED','SELECTED']})
                groups.append(g)
                for r in rows:
                    d=draw_results.get(r['draw_id'])
                    flat.append(dict(case=r['case'],arm=r['arm'],seed=r['seed'],R_cap=cap,R=r['actual_R'],D=r['actual_D'],status=r['status'],
                    energy_Ha=r['energy_Ha'],Recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r['energy_Ha'] is not None else None,
                    CNOT=d['resources']['SELECTED']['cnot_count'] if d else None,CNOT_Depth=d['resources']['SELECTED']['cnot_depth'] if d else None,Total_Depth=d['resources']['SELECTED']['total_unitary_depth'] if d else None,
                    coefficient_slots=d['coefficient_slots_mean'] if d else None,basis_id=r['basis_id']))
            for seed in p['seeds']:
                a=next(r for r in flat if r['case']==c['case'] and r['R_cap']==cap and r['seed']==seed and r['arm']=='FULL_N20')
                b=next(r for r in flat if r['case']==c['case'] and r['R_cap']==cap and r['seed']==seed and r['arm']=='FILTERED_SHORT')
                pairs.append(dict(case=c['case'],R_cap=cap,seed=seed,both_numeric=a['energy_Ha'] is not None and b['energy_Ha'] is not None,
                    delta_energy_mHa=1000*(b['energy_Ha']-a['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None,
                    delta_CNOT=b['CNOT']-a['CNOT'] if a['CNOT'] is not None and b['CNOT'] is not None else None,
                    delta_CNOT_Depth=b['CNOT_Depth']-a['CNOT_Depth'] if a['CNOT_Depth'] is not None and b['CNOT_Depth'] is not None else None,
                    delta_Total_Depth=b['Total_Depth']-a['Total_Depth'] if a['Total_Depth'] is not None and b['Total_Depth'] is not None else None))
    same_caps=sum(len({r['basis_id'] for r in ci['states'] if r['draw_id']==d['draw_id'] and r['status']=='COMPLETED'})==1 for d in available)
    out=dict(status='N61_PARTIAL_COHORT_ALL48_STATUSES_AUDITED' if partial else 'N61_ALL_FROZEN_PAIRED_SOURCE_RESULTS_AUDITED',groups=groups,paired_contrasts=pairs,
        source=source,CI=ci,audit=dict(raw_member_draws=actual_draws,actual_QPY_recounts=recounts,max_independent_probability_L1=maxprob,
            replay_is_audit_not_new_model_samples=True),new_model_samples=1000*len(available),planned_model_samples=24000,new_QPU_shots=0,
        identical_R32_R128_basis_draws=same_caps,independent_source_policies_not_optimizations=True,independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(ROOT/'analysis/N61-paired-source-summary.json',out)
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write(ROOT/'analysis/N61-all48-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    lines=['# N61：同Shotでの元SqDRIFTとfiltered短回路の比較','',
        f'予定は全4条件・両方策・3新seed、各draw1000 model samples、合計24000。取得済み{len(available)}/24 draw、{1000*len(available)} samples。実QPUなし。旧18drawはMPS、追加Fe6drawは別契約のgate-level Clifford-frame検査器で全3回路を検査した。旧MPS契約そのものの完了とはしない。',
        'CNOT等は各drawのmember Shot重み付き平均を、3seedで平均した共通logical資源。両armに同じnative/Rustiq/projected機会を与え、独立数値検査器で分布を検査した。',
        '精度は同じ元の全H+0.1S²を解いた根の物理energy。厳密S=0でも、実機noise試験でもない。','',
        '| 条件 | CI R上限 | 方法 | 完了/3 | energy Ha | Recovery % | 実R（3seed） | CNOT | CX Depth | Total Depth | 係数slots |',
        '|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for g in groups:
        z=g['resources']['SELECTED'];e='欠測あり' if g['energy_Ha'] is None else f"{g['energy_Ha']:.9f}";r='欠測あり' if g['Recovery_pct'] is None else f"{g['Recovery_pct']:.3f}"
        fmt=lambda v:'未取得' if v is None else f'{v:.3f}'
        lines.append(f"| {g['case']} | {g['R_cap']} | {g['arm']} | {g['completed']}/3 | {e} | {r} | {g['actual_R']} | {fmt(z['cnot_count'])} | {fmt(z['cnot_depth'])} | {fmt(z['total_unitary_depth'])} | {fmt(g['coefficient_slots_mean'])} |")
    lines+=['','## 何が効いたか・まだ言えないこと','']
    for c in p['cases']:
        for cap in p['R_caps']:
            rr=[r for r in pairs if r['case']==c['case'] and r['R_cap']==cap];ok=[r for r in rr if r['both_numeric']]
            if len(ok)==3:
                de=float(np.mean([r['delta_energy_mHa'] for r in ok]));wins=sum(r['delta_energy_mHa']<0 for r in ok)
                lines.append(f"- {c['case']} / R{cap}：filteredのenergy差は平均{de:+.3f} mHa（負が良い）、3seed中{wins}件で低下。CNOT差平均{np.mean([r['delta_CNOT'] for r in ok]):+.3f}、CX Depth差{np.mean([r['delta_CNOT_Depth'] for r in ok]):+.3f}。")
            else:lines.append(f"- {c['case']} / R{cap}：数値ペア{len(ok)}/3。完了例だけを全体の平均成功として扱わない。")
    lines+=['','対角filterと長さのlambda規則を合わせた方策の試験であり、両成分の因果効果を別々に分離したとはしない。',
        '対角filterは分布を変える既存の公式処置で、新発明やexact compilerではない。',
        'Rは上限で実CI次元はsourceによって違い、同Rでも同古典費用ではない。独立parameter rankは不明、最適化変数は両方0。',
        f'今回{same_caps}/{len(available)} drawはR32/R128で同じbasisになった。2つの独立なR条件での再現性ではなく同じenergyの再利用であり、R拡大への頑健性は未検証。',
        'Recovery参照の精度・種類は既存evaluation-only記録を引き継ぐ。特にFeはSCI参照でFCI保証ではない。','',
        ('source段階の完全なCPU合計は不明。既知body小計 '+str(source.get('known_body_CPU_subtotal'))+'秒。失敗した元controllerの最終reap/close費用が未記録で、0とは扱わない。' if source.get('CPU_accounting_complete') is False else f"source段階CPU（子processを含むbody）{source['CPU_seconds_including_children']:.6f}秒 / wall {source['wall_seconds']:.6f}秒。"),
        f"今回のFe CI段階body CPU {ci['CPU_seconds']:.6f}秒 / wall {ci['wall_seconds']:.6f}秒。新CI根{ci['actual_new_CI_roots']}、旧18根再利用、失敗{ci['failed_CI_attempts']}。同basis cacheの再利用を新計算として数えない。",
        'qualification、import/cold起動、今回の報告監査、旧探索費用は別で、上記を完全cold総費用とは呼ばない。',
        ('旧RDM adapterの失敗'+str(ci.get('predecessor_failed_CI_attempts'))+'件 / '+str(ci.get('predecessor_CI_CPU_seconds'))+' CPU秒も保全した。新しい測定はせず、連続メモリ配置へのコピーだけを修正した。' if 'predecessor_failed_CI_attempts' in ci else ''),
        '元のLUCJ/TN/RAACS全sourceの独立確認や最高論文への総合優越は、このpaired pilotだけでは未証明。','']
    write(ROOT/'analysis/N61_PAIRED_SOURCE_RESULTS_JA.md','\n'.join(lines).encode())
    print('N61_ALL48_AUDITED',ci['completed_endpoints'],ci['actual_new_CI_roots'],flush=True)


if __name__=='__main__':run()
