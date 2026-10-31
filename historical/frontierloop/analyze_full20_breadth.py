"""N66 stronger same-width ordinary FullN20 control, raw and resource audit."""
from collections import Counter
from io import BytesIO,StringIO
from pathlib import Path
from time import process_time,perf_counter
import csv,gzip,json,sqlite3
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .full20_breadth_plan import LOCAL,verified
from .ensemble_breadth_plan import LOCAL as N65,verified as verified65
from .full_N5_plan import LOCAL as N63
from .filtered_SqDRIFT_plan import LOCAL as N57
from .strong_projection import choose_safe
from .analyze import reference
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked

def run():
    cpu,wall=process_time(),perf_counter();p=verified();cost=load(LOCAL/'final-accounting.json');ci=load(LOCAL/'CI-terminal.json')
    assert cost['process_children_reaped'] and len(ci['states'])==32
    sources={};samples=0;qpy_count=0;events=0;max_error=0.;minimum_sampled_probability=1.
    source_status=[]
    for d in p['draws']:
        f=LOCAL/'draws'/d['draw_id'];row=dict(case=d['case'],arm=d['arm'],draw_id=d['draw_id'],status='NOT_COMPLETED')
        old=load(N57/'draws'/d['previous_9_member_draw_id']/'result.json')
        sources[old['draw_id']]=old
        if not (f/'result.json').exists():
            if (f/'failure.json').exists():row.update(status='FAILED_ENGINEERING_PRESERVED',failure=load(f/'failure.json'))
            source_status.append(row);continue
        r=load(f/'result.json');row['status']='COMPLETED';source_status.append(row);sources[d['draw_id']]=r
        assert load(f/'request.json')==d and r['counts_SHA256']==sha256((f/'counts.npz').read_bytes())
        assert r['new_model_samples']==1000 and r['budget']['unknown_or_inflight_draws']==0
        c=next(c for c in p['cases'] if c['case']==d['case']);meta=load(c['meta']);checks=load(f/'compiler-check.json')['results']
        assert len(checks)==30;merged=Counter()
        for m,rec in zip(d['members'],r['members']):
            target=LOCAL/'exports'/d['case']/'members'/m['member_id'];blob=(target/'member.json.gz').read_bytes();member=json.loads(gzip.decompress(blob))
            assert member['spec']==m['spec'] and member['diagonal_terms_retained'] is True
            assert rec['request']['source_packet_SHA256']==sha256(blob) and rec['policy']==choose_safe(rec['resources'])
            proof=checks[d['case']+':'+str(m['index'])+':'+m['member_id']];assert proof['status']=='PASSED'
            assert set(proof['check']['circuits'])=={'NATIVE','RUSTIQ','PROJECTED'}
            for name,ch in proof['check']['circuits'].items():
                err=ch['probability_L1_plus_discard_bound'];assert err<1e-8 and not ch['source_Pauli_sequence_used'];max_error=max(max_error,err)
                b=(f/(m['member_id']+'-'+name+'.qpy.gz')).read_bytes();assert sha256(b)==rec['qpy_gz_SHA256'][name]
                q=qpy.load(BytesIO(gzip.decompress(b)))[0];z=rec['resources'][name]
                assert [int(q.count_ops().get('cx',0)),int(q.depth(filter_function=lambda i:i.operation.name=='cx')),int(q.depth())]==[z['cnot_count'],z['cnot_depth'],z['total_unitary_depth']]
                qpy_count+=1
            state,_=member_state(meta,member);ids=np.asarray(sorted(state),np.uint64);prob=np.array([abs(state[int(x)])**2 for x in ids]);prob/=prob.sum()
            counts=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],prob);keep=counts>0
            minimum_sampled_probability=min(minimum_sampled_probability,float(np.min(prob[keep])))
            file=f/(m['member_id']+'.npz');assert sha256(file.read_bytes())==rec['counts_SHA256']
            with np.load(file,allow_pickle=False) as a:
                assert np.array_equal(a['ids'],ids[keep]) and np.array_equal(a['counts'],counts[keep])
                for x,w in zip(a['ids'],a['counts']):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
            events+=1
        with np.load(f/'counts.npz',allow_pickle=False) as a:assert dict(zip(map(int,a['blocked_ids']),map(int,a['counts'])))==dict(merged)
        assert sum(merged.values())==1000;samples+=1000
        for field in ['cnot_count','cnot_depth','total_unitary_depth']:
            vals=[rec['policy']['selected_resources'][field] for rec in r['members']]
            assert abs(sum(m['shots']*v for m,v in zip(d['members'],vals))/1000-r['resources']['SELECTED'][field])<1e-9
            assert max(vals)==r['resources']['MAX_SELECTED'][field]
    with sqlite3.connect('file:'+str(LOCAL/'shots.sqlite')+'?mode=ro',uri=True) as db:
        known=int(db.execute('select coalesce(sum(actual),0) from draws where state="COMPLETED"').fetchone()[0])
        unknown=int(db.execute('select count(*) from draws where state!="COMPLETED"').fetchone()[0])
    assert known==samples and unknown==0

    # Re-audit the prior same-width arm, not just its derived energy table.
    from .analyze_ensemble_breadth import run as audit65
    audit65()
    for d in verified65()['draws']:
        f=N65/'draws'/d['draw_id'];r=load(f/'result.json')
        assert r['problem_id']==d['problem_id'] and sha256((f/'counts.npz').read_bytes())==r['counts_SHA256']
        sources[d['draw_id']]=r
    oldmain=[]
    for file in [N65/'CI-terminal.json',N57/'partial-CI-terminal-v3.json']:
        oldmain += [r for r in load(file)['states'] if r.get('context','MAIN')=='MAIN' and r['draw_id'] in sources]
    main=[r for r in ci['states'] if r['context']=='MAIN']+oldmain;assert len(main)==32
    for r in ci['states']+oldmain:
        if r['status']=='COMPLETED':
            z=r['root'];assert sha256(Path(z['root_file']).read_bytes())==z['root_SHA256']
            assert z['result']['residual_norm']<=1e-8 and max(z['independent_RDM_audit']['errors'].values())<=1e-7
    flat=[];newids={d['draw_id'] for d in p['draws']};samewidthids={d['draw_id'] for d in verified65()['draws']}
    for r in main:
        source=sources.get(r['draw_id']);hf,ref=reference(r['case']);res=source['resources']['SELECTED'] if source else {}
        flat.append(dict(case=r['case'],arm=r['arm'],ensemble_size=30 if r['draw_id'] in newids|samewidthids else 9,
            R_cap=r['R_cap'],actual_R=r['actual_R'],actual_D=r['actual_D'],status=r['status'],energy_Ha=r['energy_Ha'],
            Recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r['energy_Ha'] is not None else None,
            CNOT=res.get('cnot_count'),CNOT_Depth=res.get('cnot_depth'),Total_Depth=res.get('total_unitary_depth'),
            CNOT_max=max(m['policy']['selected_resources']['cnot_count'] for m in source['members']) if source else None,
            CNOT_Depth_max=max(m['policy']['selected_resources']['cnot_depth'] for m in source['members']) if source else None,
            Total_Depth_max=max(m['policy']['selected_resources']['total_unitary_depth'] for m in source['members']) if source else None,
            coefficient_slots=source.get('coefficient_slots_mean') if source else None,independent_parameter_rank=None,VQE_optimized_variables=0,draw_id=r['draw_id']))
    contrasts=[]
    def difference(a,b,context,pair):
        out=dict(case=a['case'],candidate=a['arm'],reference=b['arm'],candidate_ensemble=a.get('ensemble_size',30),reference_ensemble=b.get('ensemble_size',30),
            context=context,pair=pair,R_cap=a['R_cap'],actual_R=a['actual_R'],reference_actual_R=b['actual_R'],
            delta_energy_mHa=1000*(a['energy_Ha']-b['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None)
        for k in ['CNOT','CNOT_Depth','Total_Depth','coefficient_slots']:
            if k in a and a[k] is not None and b[k] is not None:
                out['delta_'+k]=a[k]-b[k];out[k+'_reduction_pct']=100*(b[k]-a[k])/b[k] if b[k] else None
        return out
    for c in p['cases']:
        for label,candidate,width in [('FULL20_9_TO30','FULL_N20',30),('FULL20_VS_N5_30','FULL_N5',30),('FULL20_VS_SHORT_30','FILTERED_SHORT',30)]:
            for cap in p['R_caps']:
                a=next(r for r in flat if r['case']==c['case'] and r['arm']==candidate and r['ensemble_size']==width and r['R_cap']==cap)
                b=next(r for r in flat if r['case']==c['case'] and r['arm']=='FULL_N20' and r['ensemble_size']==(9 if candidate=='FULL_N20' else 30) and r['R_cap']==cap)
                contrasts.append(difference(a,b,'SAME_R_CAP',label))
            rr=[r for r in ci['states'] if r['case']==c['case'] and r['context']=='COMMON_R' and r['pair']==label]
            assert len(rr)==2
            if candidate=='FULL_N20':a=next(r for r in rr if r['ensemble_size']==30);b=next(r for r in rr if r['ensemble_size']==9)
            else:a=next(r for r in rr if r['arm']==candidate);b=next(r for r in rr if r['arm']=='FULL_N20')
            if a['status']==b['status']=='COMPLETED':assert a['actual_R']==b['actual_R']
            contrasts.append(difference(a,b,'SAME_ACTUAL_R',label))
    summary=dict(status='ALL4_NEW_SOURCE_AND32_CI_FIELDS_AUDITED',source_status=source_status,rows=flat,contrasts=contrasts,
        actual_new_model_samples=samples,new_QPU_shots=0,source_cost=cost,CI_cost=ci,
        audit=dict(actual_QPY_recounts=qpy_count,raw_member_draws=events,max_probability_error=max_error,known_model_samples=known,unknown_model_events=unknown,
            minimum_observed_member_model_probability=minimum_sampled_probability,prior_N65_all720_QPY_reaudited=True,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall),
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(ROOT/'analysis/N66-full20-breadth-summary.json',summary)
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat);write(ROOT/'analysis/N66-all32-main-fields.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x,d=3:'欠測' if x is None else f'{x:.{d}f}'
    lines=['# N66：通常FullN20にも同じ30回路の機会を与えた比較','',
        f'新規{samples}/4000 model samples、実QPU0。旧N65の8000及び旧9回路は再測定せず、全4条件を残した。全てseed570101の開発screenであり、独立確認・原著予算の再現ではない。',
        '同時間1/2/3・総1000samples、同compiler機会、同H+0.1S²/noOO。R上限と実R・Dを併記する。Recoveryは相関energyの回収率で、configuration repair率ではない。','',
        '| 条件 | 方策 | 回路数 | R上限 | 実R / D | E Ha | Recovery % | CNOT | CX Depth | Total Depth | 係数slots |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in flat:lines.append(f"| {r['case']} | {r['arm']} | {r['ensemble_size']} | {r['R_cap']} | {r['actual_R']} / {r['actual_D']} | {fmt(r['energy_Ha'],9)} | {fmt(r['Recovery_pct'])} | {fmt(r['CNOT'])} | {fmt(r['CNOT_Depth'])} | {fmt(r['Total_Depth'])} | {fmt(r['coefficient_slots'])} |")
    lines+=['','## 同じ30回路：SHORT − FullN20','',
        'energy差は負がSHORTで良い。資源の削減率は正がSHORTで小さい。Rによって結論が変わる場合は両方を残す。',
        '| 条件 | R32 ΔE mHa | R128 ΔE mHa | 共通実R ΔE mHa | 共通R | CNOT削減% | CX Depth削減% | Total Depth削減% | 係数slots削減% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in p['cases']:
        rr=[r for r in contrasts if r['case']==c['case'] and r['pair']=='FULL20_VS_SHORT_30']
        a=next(r for r in rr if r['context']=='SAME_R_CAP' and r['R_cap']==32);b=next(r for r in rr if r['context']=='SAME_R_CAP' and r['R_cap']==128);d=next(r for r in rr if r['context']=='SAME_ACTUAL_R')
        lines.append(f"| {c['case']} | {fmt(a['delta_energy_mHa'],6)} | {fmt(b['delta_energy_mHa'],6)} | {fmt(d['delta_energy_mHa'],6)} | {d['actual_R']} | {fmt(a.get('CNOT_reduction_pct'))} | {fmt(a.get('CNOT_Depth_reduction_pct'))} | {fmt(a.get('Total_Depth_reduction_pct'))} | {fmt(a.get('coefficient_slots_reduction_pct'))} |")
    lines+=['','## FullN20の幅拡大・通常N5との比較も残す','',
        '| 条件 | 比較（候補 − 参照） | R32 ΔE mHa | 共通実R ΔE mHa |','|---|---|---:|---:|']
    for c in p['cases']:
        for label in ['FULL20_9_TO30','FULL20_VS_N5_30']:
            rr=[r for r in contrasts if r['case']==c['case'] and r['pair']==label]
            a=next(r for r in rr if r['context']=='SAME_R_CAP' and r['R_cap']==32);b=next(r for r in rr if r['context']=='SAME_ACTUAL_R')
            title='FullN20-30 − FullN20-9' if label=='FULL20_9_TO30' else 'N5-30 − FullN20-30'
            lines.append(f"| {c['case']} | {title} | {fmt(a['delta_energy_mHa'],6)} | {fmt(b['delta_energy_mHa'],6)} |")
    lines+=['','## 費用・限界','',
        f"新source body CPU {cost['CPU_seconds_including_reaped_children']:.6f}秒/wall {cost['wall_seconds']:.6f}秒、新CI根{ci['actual_new_CI_roots']}、失敗{ci['failed_CI_attempts']}、CI CPU {ci['CPU_seconds']:.6f}秒。cacheは新根ではない。source failure={cost['failure']}。",
        '上記には実行した検査とreaped childが含まれるが、親import前・過去全探索・実機切替latencyは含まれず、不明を0にしない。異なる実行回の全cold費用が一致したとは主張しない。',
        '資源はShot重み付きlogical平均。routing/noise/hardware wall timeではない。最大値はCSVへ記録。係数slotsは独立parameter rankではなく、そのrankは不明。VQE最適化変数は全て0。',
        'FeのRecovery参照は精度未保証SCIでFCIではない。小さいsource支持を追跡できるシミュレーションで、一般の40qubit回路の完全シミュレーションではない。',
        '全4条件の良い点だけを抜かず、同実Rの悪化も含めて評価する。同じ幅を与えたFullN20も、最高原著予算・全最強LUCJ/TN/HCIの再現そのものではない。',
        '実用的な総合優越、独立確認、独立parameter削減、完全費用優位は依然として未達。','']
    write(ROOT/'analysis/N66_FULL20_BREADTH_RESULTS_JA.md','\n'.join(lines).encode())
    print('N66_AUDITED',samples,qpy_count,events,flush=True)

if __name__=='__main__':run()

