"""N30 all-source physical/root/resource/work report, not a paper victory gate."""
from pathlib import Path
from io import StringIO
from collections import Counter
from time import process_time,perf_counter
import csv
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from . import optimized_control_sampling as source
from . import optimized_control_readout as r
from . import finite_source_study as n29
from . import finite_source_readout as parent
from .analyze_finite_source_readout import audit_root,compiler_point


def run():
    cpu,wall=process_time(),perf_counter();p,q,sub=r.verified()
    t=load(r.LOCAL/'terminal.json')
    if t['protocol_SHA256']!=sha256((r.LOCAL/'protocol.json').read_bytes()) or len(t['states'])!=144:
        raise RuntimeError('All144 frozen endpoints required')
    if any(any(row[k]!=task[k] for k in task) for row,task in zip(t['states'],q['tasks'])):raise RuntimeError('Endpoint identity/order mismatch')
    oldp,_=n29.verified();oldsub=load(n29.LUCJ/'protocol.json');sources=r.inventory(p,sub);counts={}
    for row in sources:
        key=row['draw_id'] if row['available'] else row['cell_id']
        counts[key]=parent.audited_counts(oldp,oldsub,row) if row['origin'].startswith('N29_') else r.tuned_counts(p,sub,row)
    if Counter(x['status'] for x in counts.values())!=dict(COMPLETED=63,MISSING_SAME_PROBLEM_SOURCE=9):raise RuntimeError('No pending inputs allowed in final report')
    oldreport=load(ROOT/'analysis/N29-readout-120.json');resources={(x['case'],x['method']):x for x in oldreport['all_resources']}
    for c in p['resources']:
        for point in c['compiler_points']:compiler_point(point)
        main=next(x for x in c['compiler_points'] if x['resources']['optimization_level']==1)
        resources[c['case'],source.METHOD]=dict(main['resources'],**q['parameters'][c['case']],compiler_points=c['compiler_points'],
            declared_parameters=q['parameters'][c['case']]['matched_prefix_suffix_declared_parameters'])
    rows=[];roots={}
    for state in t['states']:
        row={k:state.get(k) for k in ('case','method','replicate','arm','origin','draw_id','status','R','actual_D','basis_id','root_path','root_SHA256','new_CI_attempt','cache_origin')}
        if state['status']=='COMPLETED':
            actual=counts[state['draw_id']]
            if actual!=state['source_evidence']:raise RuntimeError('Actual source binding differs')
            R=actual['R32'] if state['arm']=='R32_CAP' else min(counts[x['draw_id']]['R32'] for x in sources if x['available'] and x['case']==state['case'] and x['replicate']==state['replicate'])
            if R!=state['R'] or sorted(actual['rank'][:R])!=state['half_strings']:raise RuntimeError('Frequency/COMMON_R basis mismatch')
            pairs=[(a,b) for a in state['half_strings'] for b in state['half_strings']];d=p['physical'][state['case']]
            if parent.basis_key(d,pairs,q['library_SHA256'])!=state['basis_id']:raise RuntimeError('Root identity context mismatch')
            if state['basis_id'] not in roots:roots[state['basis_id']]=audit_root(state,d,p['library'])
            row.update(roots[state['basis_id']],HF_half_reference_added=actual['HF_half_reference_added'])
            from .analyze import reference
            hf,ref=reference(state['case']);meta=load(OLD/'data/evaluation-only'/(state['case']+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Reference problem mismatch')
            row.update(recovery_pct=100*(hf-row['energy_Ha'])/(hf-ref),reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED')
            res=resources.get((state['case'],state['method']))
            row.update({k:res.get(k) if res else None for k in ('CNOT','CNOT_Depth','Total_Depth','declared_parameters','optimizer_coordinate_slots','physical_Jacobian_rank')})
        else:row['energy_Ha']=None
        rows.append(row)
    if sum(s.get('new_CI_attempt',False) for s in t['states'])!=len(t['attempts']) or sum(a['status']=='COMPLETED' for a in t['attempts'])!=t['actual_new_CI_roots']:raise RuntimeError('New/reused root work accounting differs')
    groups=[]
    for case,method,arm in sorted({(x['case'],x['method'],x['arm']) for x in rows}):
        local=[x for x in rows if (x['case'],x['method'],x['arm'])==(case,method,arm)];valid=[x for x in local if x['status']=='COMPLETED']
        groups.append(dict(case=case,method=method,arm=arm,planned=len(local),completed=len(valid),
            energy_Ha_mean=float(np.mean([x['energy_Ha'] for x in valid])) if valid else None,
            SD_mHa=float(1000*np.std([x['energy_Ha'] for x in valid],ddof=1)) if len(valid)>1 else None,
            Recovery_mean_pct=float(np.mean([x['recovery_pct'] for x in valid])) if valid else None,
            actual_Ds=[x['actual_D'] for x in valid],statistical_superiority_certified=False))
    comparisons=[]
    for case in p['physical']:
        for arm in parent.ARMS:
            pair=next(g for g in groups if (g['case'],g['method'],g['arm'])==(case,'PAIR_SCALE2_REAL',arm))
            tuned=next(g for g in groups if (g['case'],g['method'],g['arm'])==(case,source.METHOD,arm))
            controls=[g for g in groups if g['case']==case and g['arm']==arm and (g['method'].startswith('C5_') or g['method']==source.METHOD) and g['completed']==3]
            best=min(controls,key=lambda g:g['energy_Ha_mean'])
            comparisons.append(dict(case=case,arm=arm,best_available_LUCJ=best['method'],best_selected_retrospectively=True,
                pair_minus_best_LUCJ_mean_mHa=1000*(pair['energy_Ha_mean']-best['energy_Ha_mean']),
                pair_minus_tuned_LUCJ_mean_mHa=1000*(pair['energy_Ha_mean']-tuned['energy_Ha_mean']) if tuned['completed']==3 else None,
                tuned_missing=tuned['completed']!=3,statistical_superiority_certified=False))
    accounts=[];runtime=[]
    for f in sorted(source.SAMPLING.glob('lane*-collection.json')):
        v=load(f);parts=next(x.split('|') for x in v['accounting'].splitlines() if x.split('|')[0]==v['job']['job_id'])
        if parts[1]!='COMPLETED' or parts[3]!='0:0':raise RuntimeError('Actual GPU lane unsuccessful')
        h,m,s=map(int,parts[2].split(':'));accounts.append(dict(job_id=v['job']['job_id'],GPU_allocation_seconds=3600*h+60*m+s,accounting=v['accounting'],transfer_bytes=v['total_selected_bytes']))
    for c in sub['circuits']:
        f=source.SAMPLING/'collected/circuits'/c['circuit_id']/'output/result.json';v=load(f)
        runtime.append(dict(case=c['case'],CPU_seconds=v['whole_body_CPU_seconds'],wall_seconds=v['whole_body_wall_seconds'],result_SHA256=sha256(f.read_bytes())))
    costfile=OLD/'analysis-v1/TN-optimization-cost-and-independent-quality.csv'
    with costfile.open(newline='') as f:costs=[dict(row) for row in csv.DictReader(f) if row['case_id'] in p['physical']]
    if len(costs)!=3:raise RuntimeError('All3 old tuned optimizer cost histories required')
    output=dict(status='N30_ALL144_ENDPOINTS_AUDITED',rows=rows,groups=groups,comparisons=comparisons,
        endpoint_status_counts=dict(Counter(x['status'] for x in rows)),source_status_counts=dict(Counter(x['status'] for x in counts.values())),
        actual_new_CI_roots=t['actual_new_CI_roots'],failed_CI_attempts=t['failed_CI_attempts'],distinct_saved_CI_roots_audited=len(roots),
        actual_new_model_shots=9000,reused_N29_model_shots=42000,reused_N29_classical_draws=12000,actual_new_QPU_shots=0,
        resources=[dict(case=case,method=method,**{k:v for k,v in res.items() if k not in ('case','method')}) for (case,method),res in sorted(resources.items())],
        GPU_accounting=accounts,source_runtime=runtime,prepare_CPU_seconds=p['prepare_CPU_seconds'],CI_CPU_seconds=t['CPU_seconds'],
        historical_TN_cost_rows=costs,historical_TN_cost_csv_SHA256=sha256(costfile.read_bytes()),
        historical_cost_lower_bounds_not_full_accounting=True,transfer_wall_seconds=None,new_optimizer_calls=0,
        terminal_SHA256=sha256((r.LOCAL/'terminal.json').read_bytes()),protocol_SHA256=sha256((r.LOCAL/'protocol.json').read_bytes()),
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        old_N29_results_preserved=True,native_original_TN_reproduction=False,cold_matched_work=False,goal_complete=False)
    save(ROOT/'analysis/N30-tuned-control-summary.json',output)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(x.keys() for x in rows))));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N30-all144-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x:'未取得/未適用' if x is None else f'{x:.6f}'
    lines=['# N30：調整済みLUCJを加えた有限1,000サンプル比較','',
        f"全144 endpoint、{output['endpoint_status_counts']}。新しいTN draw9件9,000 model-equivalent samples、QPU0。N29の54drawは再利用で新しい確認ではない。",
        f"新CI根{t['actual_new_CI_roots']}、失敗{t['failed_CI_attempts']}。同じ基底のrootは参照再利用し、二重に費用を計上しない。",'',
        '| 条件 | 方法 | 評価 | 数値/予定 | energy平均 Ha | SD mHa | Recovery平均 % | 実CI次元 |','|---|---|---|---:|---:|---:|---:|---|']
    for g in groups:lines.append('| '+' | '.join([g['case'],g['method'],g['arm'],f"{g['completed']}/{g['planned']}",fmt(g['energy_Ha_mean']),fmt(g['SD_mHa']),fmt(g['Recovery_mean_pct']),str(g['actual_Ds'])])+' |')
    lines+=['','## pair側と強化後対照の差（負がpair側低い）','','| 条件 | 評価 | pair−最良LUCJ mHa | pair−TN LUCJ mHa |','|---|---|---:|---:|']
    for c in comparisons:lines.append(f"| {c['case']} | {c['arm']} | {fmt(c['pair_minus_best_LUCJ_mean_mHa'])} | {fmt(c['pair_minus_tuned_LUCJ_mean_mHa'])} |")
    lines+=['','## 回路資源：同じopt1のlogical all-to-all','','| 条件 | 方法 | CNOT | CNOT Depth | Total Depth | 共通chart宣言変数 | optimizer座標slots |','|---|---|---:|---:|---:|---:|---:|']
    for res in output['resources']:lines.append('| '+' | '.join(str(res.get(k,'未適用')) for k in ('case','method','CNOT','CNOT_Depth','Total_Depth','declared_parameters','optimizer_coordinate_slots'))+' |')
    lines+=['','## 費用と解釈','',
        f"新source準備{p['prepare_CPU_seconds']:.6f} CPU秒、小CI{t['CPU_seconds']:.6f} CPU秒、今回audit{output['audit_CPU_seconds']:.6f} CPU秒。A100実割当は{[(x['job_id'],x['GPU_allocation_seconds']) for x in accounts]}。転送時間は未計測で0ではない。",
        'TNの旧調整には原実行TIMEOUTとsuccessorの費用があり、保存された最後のCPU値は下限であって全費用ではない。新1,000 samplesだけを費用としてTNが無料だったとはしない。元の全cost row/記録hashをJSONへ保持した。',
        'N29のpair/Markov/元LUCJは再利用した。同じreplicate labelでも共通乱数ではない。既存sourceを最適化した独立反復ではなく、固定sourceのMonte Carlo draw。3反復で有意な優越を認定しない。',
        'ナフタレンTNは別軌道問題だったため欠測。旧canonical sourceを混ぜない。N₂の2距離は独立分子ではなく、Fe参照はSCI、N₂参照精度は未認証。',
        'TNは我々のcommon-readout/3subproblem variant。原著native Dice/20subproblem・最高予算の完全再現ではない。H+0.1S²有限penaltyとphysical H報告を区別し、厳密singlet最適化とは呼ばない。',
        '宣言変数は同じ初期Slater chartへ合わせる。既知のchart変更を新pruningと呼ばず、optimizer座標数/数値gate slots/実状態Jacobian rankと混同しない。独立rankは未認証。',
        '原著の最高点やnative hardware routing/noiseに勝ったとはまだ主張しない。全方法・不利な条件・欠測・旧費用を保持して次の判断を行う。']
    write(ROOT/'analysis/N30_TUNED_CONTROL_RESULTS_JA.md','\n'.join(lines).encode())
    print(output['status'],'newroots',t['actual_new_CI_roots'],'comparisons',comparisons,flush=True)


if __name__=='__main__':run()
