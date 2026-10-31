"""Audit every N65 raw member and actual QPY; report all four cases."""
from collections import Counter
from io import BytesIO,StringIO
from pathlib import Path
from time import process_time,perf_counter
import csv,gzip,json,sqlite3
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .ensemble_breadth_plan import LOCAL,verified
from .full_N5_plan import LOCAL as N63
from .filtered_SqDRIFT_plan import LOCAL as N57
from .strong_projection import choose_safe
from .analyze import reference
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked

def run():
    cpu,wall=process_time(),perf_counter();p=verified();cost=load(LOCAL/'final-accounting.json');ci=load(LOCAL/'CI-terminal.json')
    assert cost['process_children_reaped'] and len(ci['states'])==40
    sources={};samples=0;qpy_count=0;events=0;max_error=0.;minimum_sampled_probability=1.
    source_status=[]
    for d in p['draws']:
        f=LOCAL/'draws'/d['draw_id'];row=dict(case=d['case'],arm=d['arm'],draw_id=d['draw_id'],status='NOT_COMPLETED')
        oldbase=N63 if d['arm']=='FULL_N5' else N57;old=load(oldbase/'draws'/d['previous_9_member_draw_id']/'result.json')
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
            assert member['spec']==m['spec'] and member['diagonal_terms_retained']==(d['arm']=='FULL_N5')
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
    oldmain=[]
    for file in [N63/'CI-terminal.json',N57/'partial-CI-terminal-v3.json']:
        oldmain += [r for r in load(file)['states'] if r.get('context','MAIN')=='MAIN' and r['draw_id'] in sources]
    main=[r for r in ci['states'] if r['context']=='MAIN']+oldmain;assert len(main)==32
    for r in ci['states']+oldmain:
        if r['status']=='COMPLETED':
            z=r['root'];assert sha256(Path(z['root_file']).read_bytes())==z['root_SHA256']
            assert z['result']['residual_norm']<=1e-8 and max(z['independent_RDM_audit']['errors'].values())<=1e-7
    flat=[]
    for r in main:
        source=sources.get(r['draw_id']);hf,ref=reference(r['case']);res=source['resources']['SELECTED'] if source else {}
        flat.append(dict(case=r['case'],arm=r['arm'],ensemble_size=30 if r['draw_id'] in {d['draw_id'] for d in p['draws']} else 9,
            R_cap=r['R_cap'],actual_R=r['actual_R'],actual_D=r['actual_D'],status=r['status'],energy_Ha=r['energy_Ha'],
            Recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r['energy_Ha'] is not None else None,
            CNOT=res.get('cnot_count'),CNOT_Depth=res.get('cnot_depth'),Total_Depth=res.get('total_unitary_depth'),
            CNOT_max=max(m['policy']['selected_resources']['cnot_count'] for m in source['members']) if source else None,
            CNOT_Depth_max=max(m['policy']['selected_resources']['cnot_depth'] for m in source['members']) if source else None,
            Total_Depth_max=max(m['policy']['selected_resources']['total_unitary_depth'] for m in source['members']) if source else None,
            coefficient_slots=source.get('coefficient_slots_mean') if source else None,independent_parameter_rank=None,VQE_optimized_variables=0,draw_id=r['draw_id']))
    contrasts=[]
    for c in p['cases']:
        for arm,label in [('FULL_N5','N5_9_TO30'),('FILTERED_SHORT','SHORT_9_TO30')]:
            for cap in [32,128]:
                rows=[r for r in flat if r['case']==c['case'] and r['arm']==arm and r['R_cap']==cap]
                assert len(rows)==2;a=next(r for r in rows if r['ensemble_size']==30);b=next(r for r in rows if r['ensemble_size']==9)
                contrasts.append(dict(case=c['case'],arm=arm,context='SAME_R_CAP',R_cap=cap,
                    delta_energy_mHa=1000*(a['energy_Ha']-b['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None,
                    delta_CNOT=a['CNOT']-b['CNOT'] if a['CNOT'] is not None else None,
                    delta_CNOT_Depth=a['CNOT_Depth']-b['CNOT_Depth'] if a['CNOT_Depth'] is not None else None,
                    delta_Total_Depth=a['Total_Depth']-b['Total_Depth'] if a['Total_Depth'] is not None else None))
            rows=[r for r in ci['states'] if r['case']==c['case'] and r['context']=='COMMON_R' and r['pair']==label]
            assert len(rows)==2;a=next(r for r in rows if r['ensemble_size']==30);b=next(r for r in rows if r['ensemble_size']==9)
            if a['status']==b['status']=='COMPLETED':assert a['actual_R']==b['actual_R']
            contrasts.append(dict(case=c['case'],arm=arm,context='SAME_ACTUAL_R',actual_R=a['actual_R'],
                delta_energy_mHa=1000*(a['energy_Ha']-b['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None))
    summary=dict(status='ALL8_SOURCE_AND40_CI_STATUSES_AUDITED',source_status=source_status,rows=flat,contrasts=contrasts,
        actual_new_model_samples=samples,new_QPU_shots=0,source_cost=cost,CI_cost=ci,
        audit=dict(actual_QPY_recounts=qpy_count,raw_member_draws=events,max_probability_error=max_error,known_model_samples=known,unknown_model_events=unknown,
            minimum_observed_member_model_probability=minimum_sampled_probability,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall),
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(ROOT/'analysis/N65-ensemble-breadth-summary.json',summary)
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat);write(ROOT/'analysis/N65-all32-main-fields.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x,d=3:'欠測' if x is None else f'{x:.{d}f}'
    lines=['# N65：同じShotで9回路→30回路に広げた結果','',
        f'新規{samples}/8000 model samples。実QPU0。全4条件・両方策・seed570101の一回のscreenであり、統計的優越・独立確認・最強論文再現とはしない。',
        '旧9回路の観測を再利用して比較し、新30回路では1,000回を新たに生成した。時間1/2/3への334/333/333配分、回路内の長さ・filter規則、compilerの機会、H+0.1S²/noOOは不変。','',
        '| 条件 | 方法 | 回路数 | R上限 | 実R / D | E Ha | Recovery % | CNOT平均 | CX Depth平均 | Total Depth平均 | CNOT最大 | 係数slots平均 |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in flat:lines.append(f"| {r['case']} | {r['arm']} | {r['ensemble_size']} | {r['R_cap']} | {r['actual_R']} / {r['actual_D']} | {fmt(r['energy_Ha'],9)} | {fmt(r['Recovery_pct'])} | {fmt(r['CNOT'])} | {fmt(r['CNOT_Depth'])} | {fmt(r['Total_Depth'])} | {fmt(r['CNOT_max'])} | {fmt(r['coefficient_slots'])} |")
    lines+=['','## 30回路 − 9回路の差','',
        'energy差の負が30回路で良い。R上限と実R統一を混同しない。異なる条件の良い方だけを選ばない。',
        '| 条件 | 方法 | R上限32 ΔE mHa | 共通実R ΔE mHa | 共通実R | ΔCNOT | ΔCX Depth | ΔTotal Depth |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in p['cases']:
        for arm in ['FULL_N5','FILTERED_SHORT']:
            a=next(r for r in contrasts if r['case']==c['case'] and r['arm']==arm and r['context']=='SAME_R_CAP' and r['R_cap']==32)
            b=next(r for r in contrasts if r['case']==c['case'] and r['arm']==arm and r['context']=='SAME_ACTUAL_R')
            lines.append(f"| {c['case']} | {arm} | {fmt(a['delta_energy_mHa'],6)} | {fmt(b['delta_energy_mHa'],6)} | {b['actual_R']} | {fmt(a['delta_CNOT'])} | {fmt(a['delta_CNOT_Depth'])} | {fmt(a['delta_Total_Depth'])} |")
    lines+=['','## ensemble内の最も大きい回路','',
        '| 条件 | 方法 | 回路数 | 最大CNOT | 最大CX Depth | 最大Total Depth |','|---|---|---:|---:|---:|---:|']
    for r in flat:
        if r['R_cap']==32:lines.append(f"| {r['case']} | {r['arm']} | {r['ensemble_size']} | {fmt(r['CNOT_max'])} | {fmt(r['CNOT_Depth_max'])} | {fmt(r['Total_Depth_max'])} |")
    lines+=['','## 費用と限界','',
        f"新source CPU {cost['CPU_seconds_including_reaped_children']:.6f}秒、wall {cost['wall_seconds']:.6f}秒。子processの回収を含むbody費用で、親import前・過去の全探索・実機切替latencyは含まれず不明。source failure={cost['failure']}。",
        f"新CI根{ci['actual_new_CI_roots']}、失敗{ci['failed_CI_attempts']}、CI CPU {ci['CPU_seconds']:.6f}秒。cacheは新根ではない。",
        '回路数は3.33倍で、生成・コンパイル・検査・運用切替の負担が増える。旧sourceの完全cold総費用は不明のため、厳密な総費用優越を主張しない。',
        'CNOT/Depthの平均はShot重み付きlogical資源で、最大値は各ensemble内の最大回路。hardware routing/noiseの値ではない。',
        '係数slotsは独立パラメータrankではない。全armでVQE最適化変数は0。FeのRecovery参照は精度未保証SCIでFCIではない。',
        '数値oracleの極小確率配置をCIへ追加するN64診断とは異なり、ここでは実際に有限モデル測定された配置だけを使う。',
        'このscreenの良い点を見て分子別に方策を変えず、全結果を残した上で次の投資を判断する。最高論文との強い同問題LUCJ/TN/HCI対照・独立確認はなお未達。','']
    write(ROOT/'analysis/N65_ENSEMBLE_BREADTH_RESULTS_JA.md','\n'.join(lines).encode())
    print('N65_AUDITED',samples,qpy_count,events,flush=True)

if __name__=='__main__':run()
