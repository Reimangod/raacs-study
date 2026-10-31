"""N62 three-arm source ablation: raw audits, complete statuses and costs."""
from collections import Counter
from io import BytesIO,StringIO
from pathlib import Path
import csv,gzip,json,sqlite3
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .filter_only_plan import LOCAL,verified
from .filtered_SqDRIFT_plan import LOCAL as OLD_LOCAL,verified as old_verified
from .io import load,save,write,sha256
from .strong_projection import choose_safe
from .analyze import reference
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked


def run():
    p=verified();old=old_verified();stage=LOCAL/'targeted-successor-v2'
    ci=load(stage/'CI-terminal.json');first=load(LOCAL/'final-accounting.json');previous=load(LOCAL/'targeted-successor-v1/final-accounting.json');extra=load(stage/'final-accounting.json')
    cost=dict(first,CPU_seconds_including_reaped_children=first['CPU_seconds_including_reaped_children']+previous['CPU_seconds_including_children']+extra['CPU_seconds_including_children'],wall_seconds=first['wall_seconds']+previous['wall_seconds']+extra['wall_seconds'],first_attempt=first,numerical_successor_v1=previous,numerical_successor_v2=extra,old_failed_attempt_costs_retained=True)
    old_ci=load(OLD_LOCAL/'partial-CI-terminal-v3.json');assert len(ci['states'])==72 and len(old_ci['states'])==48
    source_results={};QPYs=0;raw_members=0;max_error=0.;samples=0
    for d in p['draws']:
        f=LOCAL/'draws'/d['draw_id']
        if not (f/'result.json').exists():continue
        r=load(f/'result.json');source_results[d['draw_id']]=r
        assert load(f/'request.json')==d and r['counts_SHA256']==sha256((f/'counts.npz').read_bytes())
        assert r['budget']['unknown_or_inflight_draws']==0 and r['new_model_samples']==1000
        meta=load(next(c['meta'] for c in p['cases'] if c['case']==d['case']));checks=load(Path(r.get('compiler_check_file',str(f/'compiler-check.json'))))['results'];assert len(checks)==9
        merged=Counter()
        for m,rec in zip(d['members'],r['members']):
            target=LOCAL/'exports'/d['case']/'members'/m['member_id'];member=json.loads(gzip.decompress((target/'member.json.gz').read_bytes()))
            assert member['spec']==m['spec'] and member['diagonal_terms_retained'] is False and m['spec']['num_terms']==20
            assert rec['request']['source_packet_SHA256']==sha256((target/'member.json.gz').read_bytes())
            assert rec['policy']==choose_safe(rec['resources'])
            check=checks[d['case']+':'+str(m['index'])+':'+m['member_id']]
            assert check['status']=='PASSED' and set(check['check']['circuits'])=={'NATIVE','RUSTIQ','PROJECTED'}
            for ch in check['check']['circuits'].values():
                assert ch['probability_L1_plus_discard_bound']<1e-8 and not ch['source_Pauli_sequence_used']
                max_error=max(max_error,ch['probability_L1_plus_discard_bound'])
            for name in ['NATIVE','RUSTIQ','PROJECTED']:
                b=(f/(m['member_id']+'-'+name+'.qpy.gz')).read_bytes();assert sha256(b)==rec['qpy_gz_SHA256'][name]
                q=qpy.load(BytesIO(gzip.decompress(b)))[0];z=rec['resources'][name]
                assert [int(q.count_ops().get('cx',0)),int(q.depth(filter_function=lambda i:i.operation.name=='cx')),int(q.depth())]==[z['cnot_count'],z['cnot_depth'],z['total_unitary_depth']]
                QPYs+=1
            state,_=member_state(meta,member);ids=np.asarray(sorted(state),np.uint64);probs=np.array([abs(state[int(x)])**2 for x in ids]);probs/=probs.sum()
            counts=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],probs);keep=counts>0
            file=f/(m['member_id']+'.npz');assert sha256(file.read_bytes())==rec['counts_SHA256']
            with np.load(file,allow_pickle=False) as a:
                assert np.array_equal(a['ids'],ids[keep]) and np.array_equal(a['counts'],counts[keep])
                for x,w in zip(a['ids'],a['counts']):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
            raw_members+=1
        with np.load(f/'counts.npz',allow_pickle=False) as a:assert dict(zip(map(int,a['blocked_ids']),map(int,a['counts'])))==dict(merged)
        samples+=1000
    with sqlite3.connect('file:'+str(LOCAL/'shots.sqlite')+'?mode=ro',uri=True) as db:
        ledger=db.execute('select state,count(*),sum(actual) from draws group by state').fetchall()
        known=int(db.execute('select coalesce(sum(actual),0) from draws where state="COMPLETED"').fetchone()[0])
        unknown=int(db.execute('select count(*) from draws where state!="COMPLETED"').fetchone()[0])
    assert known==samples and unknown==0
    for d in old['draws']:source_results[d['draw_id']]=load(OLD_LOCAL/'draws'/d['draw_id']/'result.json')
    unique_roots={}
    for r in ci['states']+old_ci['states']:
        if r['status']!='COMPLETED':continue
        root=r['root'];assert sha256(Path(root['root_file']).read_bytes())==root['root_SHA256']
        assert max(root['independent_RDM_audit']['errors'].values())<=1e-7 and root['result']['residual_norm']<=1e-8
        unique_roots[r['basis_id']]=root
    main=[r for r in ci['states'] if r['context']=='MAIN']+old_ci['states'];assert len(main)==72
    flat=[]
    for r in main:
        s=source_results.get(r['draw_id']);hf,ref=reference(r['case']);z=s['resources']['SELECTED'] if s else {}
        flat.append(dict(case=r['case'],seed=r['seed'],arm=r['arm'],R_cap=r['R_cap'],actual_R=r['actual_R'],actual_D=r['actual_D'],
            status=r['status'],energy_Ha=r['energy_Ha'],Recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r['energy_Ha'] is not None else None,
            CNOT=z.get('cnot_count'),CNOT_Depth=z.get('cnot_depth'),Total_Depth=z.get('total_unitary_depth'),
            coefficient_slots=s['coefficient_slots_mean'] if s else None,independent_parameter_rank=None,VQE_optimized_variables=0,draw_id=r['draw_id']))
    groups=[]
    for c in p['cases']:
        for cap in p['R_caps']:
            for arm in ['FULL_N20','FILTERED_N20','FILTERED_SHORT']:
                rows=[r for r in flat if r['case']==c['case'] and r['R_cap']==cap and r['arm']==arm];assert len(rows)==3
                means={k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in ['energy_Ha','Recovery_pct','CNOT','CNOT_Depth','Total_Depth','coefficient_slots']}
                groups.append(dict(case=c['case'],arm=arm,R_cap=cap,completed=sum(r['status']=='COMPLETED' for r in rows),actual_R=[r['actual_R'] for r in rows],**means))
    contrasts=[]
    for c in p['cases']:
        for seed in p['seeds']:
            for other in ['FULL_N20','FILTERED_SHORT']:
                for cap in p['R_caps']:
                    a=next(r for r in flat if r['case']==c['case'] and r['seed']==seed and r['arm']=='FILTERED_N20' and r['R_cap']==cap)
                    b=next(r for r in flat if r['case']==c['case'] and r['seed']==seed and r['arm']==other and r['R_cap']==cap)
                    diff=lambda k: a[k]-b[k] if a[k] is not None and b[k] is not None else None
                    contrasts.append(dict(case=c['case'],seed=seed,comparator=other,context='SAME_R_CAP',R_cap=cap,
                        delta_energy_mHa=1000*diff('energy_Ha') if diff('energy_Ha') is not None else None,
                        delta_CNOT=diff('CNOT'),delta_CNOT_Depth=diff('CNOT_Depth'),delta_Total_Depth=diff('Total_Depth')))
                rows=[r for r in ci['states'] if r['context']=='COMMON_R' and r['pair']==other and r['case']==c['case'] and r['seed']==seed]
                assert len(rows)==2;a=next(r for r in rows if r['arm']=='FILTERED_N20');b=next(r for r in rows if r['arm']==other)
                assert a['actual_R']==b['actual_R']
                contrasts.append(dict(case=c['case'],seed=seed,comparator=other,context='SAME_ACTUAL_R',actual_R=a['actual_R'],
                    delta_energy_mHa=1000*(a['energy_Ha']-b['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None))
    summary=dict(status='ALL72_THREE_ARM_MAIN_STATUSES_AND48_COMMON_R_ROWS_AUDITED',groups=groups,contrasts=contrasts,
        new_model_samples=samples,planned_new_model_samples=12000,new_QPU_shots=0,new_draws_completed=len(source_results)-24,
        new_main_CI_completed=sum(r['context']=='MAIN' and r['status']=='COMPLETED' for r in ci['states']),
        source_cost=cost,CI_cost=ci,audit=dict(actual_QPY_recounts=QPYs,raw_member_draws=raw_members,max_probability_L1_plus_discard=max_error,
            model_ledger=ledger,known_model_samples=known,unknown_model_events=unknown,counts_replay_not_new_samples=True,distinct_audited_CI_roots=len(unique_roots)),
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(ROOT/'analysis/N62-complete-three-arm-summary.json',summary)
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat);write(ROOT/'analysis/N62-complete-all72-main-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    lines=['# N62：対角filterと短縮長の切り分け','',
        f'追加取得{samples}/12000 model samples、実QPUなし。旧24 drawは再測定せず、3armの全72主評価欄と48同実R評価欄を表示・監査した。',
        'FULL_N20は対角項を残した長さ20、FILTERED_N20は対角項を除いた長さ20、FILTERED_SHORTは同じ除去後にλ比で長さを短縮した既存開発方策。',
        '全armで同じ物理H+0.1S²/noOO。Rは半配置の上限で、実CI次元D=実R²。RecoveryのFe参照は精度未保証SCIで、FCIではない。','',
        '| 条件 | R上限 | 方法 | 数値/3 | energy Ha | Recovery % | 実R | CNOT | CX Depth | Total Depth | 係数slots |',
        '|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|']
    fmt=lambda x,d=3:'欠測' if x is None else f'{x:.{d}f}'
    for g in groups:lines.append(f"| {g['case']} | {g['R_cap']} | {g['arm']} | {g['completed']}/3 | {fmt(g['energy_Ha'],9)} | {fmt(g['Recovery_pct'])} | {g['actual_R']} | {fmt(g['CNOT'])} | {fmt(g['CNOT_Depth'])} | {fmt(g['Total_Depth'])} | {fmt(g['coefficient_slots'])} |")
    lines+=['','## 同じR上限と同じ実Rを両方見る','',
        '差はFILTERED_N20 − 比較相手。負はN20 filteredのenergyが良い。有利なRだけを選ばない。',
        '| 条件 | 比較相手 | R上限32での差 mHa | 最大共通実Rでの差 mHa | 共通R（各seed） |',
        '|---|---|---:|---:|---|']
    for c in p['cases']:
        for other in ['FULL_N20','FILTERED_SHORT']:
            a=[r for r in contrasts if r['case']==c['case'] and r['comparator']==other and r['context']=='SAME_R_CAP' and r['R_cap']==32]
            b=[r for r in contrasts if r['case']==c['case'] and r['comparator']==other and r['context']=='SAME_ACTUAL_R']
            mean=lambda rr:float(np.mean([r['delta_energy_mHa'] for r in rr])) if len(rr)==3 and all(r['delta_energy_mHa'] is not None for r in rr) else None
            lines.append(f"| {c['case']} | {other} | {fmt(mean(a),6)} | {fmt(mean(b),6)} | {[r['actual_R'] for r in b]} |")
    lines+=['','## 費用・限界・次の判断','',
        f"新source body CPU {cost['CPU_seconds_including_reaped_children']:.6f}秒、wall {cost['wall_seconds']:.6f}秒。失敗とchildren回収を含む。stage別は不完全で残りoverheadを0にしない。",
        f"CI追加部分body CPU {ci['CPU_seconds']:.6f}秒、wall {ci['wall_seconds']:.6f}秒、新根{ci['actual_new_CI_roots']}、前段{ci['predecessor_new_CI_roots']}根/{ci['predecessor_CI_CPU_seconds']:.6f} CPU秒も保全。失敗{ci['failed_CI_attempts']}。同じbasisのcacheを新評価に数えない。",
        '旧8drawを再測定せず、元の4未測定drawだけをtargeted Clifford振幅検査器で補完した。物理source・sampling seed・Shotと判定1e-8は同じで、検査器契約だけを加算的に分離した。M4の親process import前と過去全探索のcold費用は上記に含まれない。元sourceの完全cold費用は依然不明で、費用優位を捏造しない。',
        'CNOT/Depthは各drawのShot重み付きlogical資源の3seed平均。全armに同じnative/Rustiq/projected機会を与えた。実機routingやnoise下の性能ではない。',
        '係数slotsは独立parameter rankではない。全armのVQE最適化変数は0。',
        'この3seedは同じ開発対照への追加armであり、未知条件への独立確認ではない。対角filterは既存公式方策で新発明とはしない。',
        '短縮方策の精度がN20 filteredより劣る場合、それを勝利と呼ばず、短い回路とのtradeoffとして残す。長さ/seed/閾値の事後救済はしない。',
        '最高論文への総合優越、強いsource-aligned LUCJ/TN/古典HCI対照、独立確認はなお未達。','']
    write(ROOT/'analysis/N62_COMPLETE_FILTER_ONLY_RESULTS_JA.md','\n'.join(lines).encode())
    print('N62_ALL_AUDITED',samples,summary['new_main_CI_completed'],flush=True)


if __name__=='__main__':run()
