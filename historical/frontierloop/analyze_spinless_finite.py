"""N35 full192 endpoint audit, paired-encoding resources and paid-cost report."""
from pathlib import Path
from io import BytesIO,StringIO
from collections import Counter
from time import process_time,perf_counter
import csv
import numpy as np
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from . import spinless_finite_study as s
from . import spinless_source_preflight_v2 as preflight
from .analyze_finite_source_readout import audit_root
from .analyze import reference


def run():
    cpu,wall=process_time(),perf_counter();p=s.verified();t=load(s.LOCAL/'readout-terminal.json');draw=load(s.LOCAL/'sample-terminal.json')
    if len(t['states'])!=192 or len(draw['rows'])!=12 or draw['new_model_shots']!=12000:raise RuntimeError('Actual complete sample/CI inventory required')
    if t['protocol_SHA256']!=sha256((s.LOCAL/'protocol.json').read_bytes()) or draw['protocol_SHA256']!=t['protocol_SHA256']:raise RuntimeError('Draw/CI protocol mismatch')
    if any(any(row[k]!=task[k] for k in task) for row,task in zip(t['states'],p['tasks'])):raise RuntimeError('Actual endpoint binding/order mismatch')
    records=s.all_records(p);old=load(ROOT/'analysis/N33-ALL-summary.json');resources={}
    for row in old['rows']:
        if row['status']=='COMPLETED':resources[row['case'],row['method']]={k:row.get(k) for k in ('CNOT','CNOT_Depth','Total_Depth','declared_parameters','optimizer_coordinate_slots','physical_Jacobian_rank')}
    for source in p['sources']:
        for view in source['resources']:
            for point in view['compiler_points']:
                f=point['circuits']['sampling'];blob=Path(f['path']).read_bytes()
                if sha256(blob)!=f['SHA256']:raise RuntimeError('Actual source compiled bytes changed')
                qc=qpy.load(BytesIO(blob))[0];res=point['resources']
                if (res['CNOT'],res['CNOT_Depth'],res['Total_Depth'])!=(qc.count_ops().get('cx',0),qc.depth(filter_function=lambda i:i.operation.name=='cx'),qc.depth()):raise RuntimeError('Actual resource recount mismatch')
        main=next(v for v in source['resources'] if v['view']=='FULL_SPIN_COPY')['compiler_points'][0]['resources']
        resources[source['case'],s.METHOD]=dict(main,declared_parameters=source['known_Slater_chart_declared_parameters'],optimizer_coordinate_slots=source['optimizer_coordinate_slots'],physical_Jacobian_rank=None)
    rows=[];roots={}
    for state in t['states']:
        row={k:state.get(k) for k in ('case','method','replicate','arm','status','R','actual_D','basis_id','root_path','root_SHA256','draw_id','new_CI_attempt','cache_origin','origin')}
        if state['status']=='COMPLETED':
            actual=records[state['draw_id']];d=p['physical'][state['case']]
            if actual!=state['source_evidence']:raise RuntimeError('Actual probability/sample evidence changed')
            R=actual['R32'] if state['arm']=='R32_CAP' else min(records[x['draw_id']]['R32'] for x in p['tasks'] if x['available'] and x['case']==state['case'] and x['replicate']==state['replicate'])
            strings=sorted(actual['rank'][:R]);pairs=[(a,b) for a in strings for b in strings]
            if R!=state['R'] or strings!=state['half_strings'] or s.parent.basis_key(d,pairs,p['library_SHA256'])!=state['basis_id']:raise RuntimeError('Actual frequency/commonR physical basis mismatch')
            if state['basis_id'] not in roots:roots[state['basis_id']]=audit_root(state,d,p['library'])
            row.update(roots[state['basis_id']]);hf,ref=reference(state['case']);meta=load(OLD/'data/evaluation-only'/(state['case']+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Reference problem mismatch')
            row.update(Recovery_pct=100*(hf-row['energy_Ha'])/(hf-ref),reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED',**resources[state['case'],state['method']])
        else:row['energy_Ha']=None
        rows.append(row)
    groups=[]
    for case,method,arm in sorted({(r['case'],r['method'],r['arm']) for r in rows}):
        local=[r for r in rows if (r['case'],r['method'],r['arm'])==(case,method,arm)];valid=[r for r in local if r['status']=='COMPLETED']
        groups.append(dict(case=case,method=method,arm=arm,planned=len(local),completed=len(valid),
            energy_mean_Ha=float(np.mean([r['energy_Ha'] for r in valid])) if valid else None,
            SD_mHa=float(np.std([r['energy_Ha'] for r in valid],ddof=1)*1000) if len(valid)>1 else None,
            Recovery_mean_pct=float(np.mean([r['Recovery_pct'] for r in valid])) if valid else None,
            actual_Ds=[r['actual_D'] for r in valid],statistical_superiority_certified=False))
    comparisons=[]
    for case in p['physical']:
        for arm in s.parent.ARMS:
            new=next(g for g in groups if (g['case'],g['method'],g['arm'])==(case,s.METHOD,arm))
            controls=[g for g in groups if g['case']==case and g['arm']==arm and g['completed']==3 and (g['method'].startswith('C5_') or g['method'] in ('EXISTING_TN_COMMON_L1','ANALYTICAL_OPT_L1_8EVAL'))]
            best=min(controls,key=lambda g:g['energy_mean_Ha']);original=next(g for g in controls if g['method']=='C5_C_LUCJ_COMPRESSED_L1')
            diff=lambda g:1000*(new['energy_mean_Ha']-g['energy_mean_Ha']) if new['completed']==3 else None
            comparisons.append(dict(case=case,arm=arm,best_previous_LUCJ=best['method'],spinless_minus_best_existing_LUCJ_mHa=diff(best),
                spinless_minus_original_L1_mHa=diff(original),retrospective_best_control=True,statistical_superiority_certified=False))
    new_draws=[]
    for req in p['requests']:
        r=next(row for row in draw['rows'] if row['draw_id']==req['draw_id']);d=p['physical'][req['case']];n,k=d['norb'],d['nelec'][0];hf=((1<<k)-1)*(1+(1<<n));mask=(1<<n)-1
        with np.load(r['path'],allow_pickle=False) as z:ids,c=z['blocked_ids'],z['counts']
        new_draws.append(dict(case=req['case'],replicate=req['replicate'],actual_shots=int(c.sum()),unique_full=len(ids),
            unique_half=len({int(x)&mask for x in ids}|{int(x)>>n for x in ids}),HF_frequency=float(c[ids==hf].sum()/c.sum()),payload_SHA256=r['payload_SHA256']))
    n34=load(preflight.LOCAL/'terminal.json');v1=load(preflight.PRIOR/'failed.json')
    out=dict(status='N35_ALL192_ENDPOINTS_PHYSICAL_ROOTS_AND_RESOURCES_AUDITED',rows=rows,groups=groups,comparisons=comparisons,
        endpoint_status_counts=dict(Counter(r['status'] for r in rows)),source_status_counts=dict(Counter(r['status'] for r in records.values())),
        actual_new_model_shots=12000,actual_new_QPU_shots=0,new_GPU_allocations=0,new_optimizer_calls=0,
        actual_new_CI_roots=t['actual_new_CI_roots'],failed_CI_attempts=t['failed_CI_attempts'],distinct_saved_roots_audited=len(roots),
        current_source_support=new_draws,source_resources=[dict(case=x['case'],resources=x['resources'],optimizer_coordinate_slots=x['optimizer_coordinate_slots'],
            known_Slater_chart_declared_parameters=x['known_Slater_chart_declared_parameters'],physical_Jacobian_rank=None,original_C5_L1_resources=x['original_C5_L1_resources']) for x in p['sources']],
        old_initialization_records=[x['old_initialization_record'] for x in p['sources']],
        new_source_preflight_CPU_seconds=n34['CPU_seconds'],old_failed_source_preflight_CPU_seconds=v1['CPU_seconds'],
        sampling_CPU_seconds=draw['CPU_seconds'],CI_CPU_seconds=t['CPU_seconds'],audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        source_protocol_SHA256=sha256((s.LOCAL/'protocol.json').read_bytes()),readout_terminal_SHA256=sha256((s.LOCAL/'readout-terminal.json').read_bytes()),
        old_draws_reused_not_new_confirmation=True,cold_matched_work=False,native_paper_reproduction=False,statistical_superiority_certified=False,goal_complete=False)
    save(ROOT/'analysis/N35-spinless-summary.json',out)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows);write(ROOT/'analysis/N35-all192-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x:'未取得' if x is None else f'{x:.6f}'
    lines=['# N35：spinless LUCJ SDK variantの全4条件・有限Shot比較','',
        f"全192 endpoint、{out['endpoint_status_counts']}。新12,000 model samples、実QPU0、新GPU割当0。新CI根{t['actual_new_CI_roots']}、失敗{t['failed_CI_attempts']}、{len(roots)}保存rootを独立physical-H/penaltyで照合。",'',
        '| 条件 | 方法 | arm | 数値/予定 | SQD energy平均 Ha | SD mHa | Recovery平均 % | 実D |','|---|---|---|---:|---:|---:|---:|---|']
    for g in groups:lines.append('| '+' | '.join([g['case'],g['method'],g['arm'],f"{g['completed']}/{g['planned']}",fmt(g['energy_mean_Ha']),fmt(g['SD_mHa']),fmt(g['Recovery_mean_pct']),str(g['actual_Ds'])])+' |')
    lines+=['','## 精度差：正は新sourceが悪い','','| 条件 | arm | 既存最良LUCJ | 新−最良 mHa | 新−元L1 mHa |','|---|---|---|---:|---:|']
    for c in comparisons:lines.append(f"| {c['case']} | {c['arm']} | {c['best_previous_LUCJ']} | {fmt(c['spinless_minus_best_existing_LUCJ_mHa'])} | {fmt(c['spinless_minus_original_L1_mHa'])} |")
    lines+=['','## logical all-to-all資源：全て同じopt1','','| 条件 | view | qubit | CNOT | CNOT Depth | Total Depth | 共通Slater chart宣言変数 | optimizer coordinate slots |','|---|---|---:|---:|---:|---:|---:|---:|']
    for source in p['sources']:
        case=source['case'];n=source['definition']['norb'];base=source['original_C5_L1_resources'][0]['resources'];oldres=resources[case,'C5_C_LUCJ_COMPRESSED_L1']
        lines.append(f"| {case} | 元C5 full-spin L1 | {2*n} | {base['CNOT']} | {base['CNOT_Depth']} | {base['Total_Depth']} | {oldres['declared_parameters']} | {oldres['optimizer_coordinate_slots']} |")
        for view in source['resources']:
            res=view['compiler_points'][0]['resources']
            lines.append(f"| {case} | {view['view']} | {view['n_qubits']} | {res['CNOT']} | {res['CNOT_Depth']} | {res['Total_Depth']} | {source['known_Slater_chart_declared_parameters']} | {source['optimizer_coordinate_slots']} |")
    lines+=['','## 費用・限界','',f"新sourceの全preflight {n34['CPU_seconds']:.6f} CPU秒（失敗v1 {v1['CPU_seconds']:.6f}秒も別途保持）。sampling {draw['CPU_seconds']:.6f} CPU秒、小CI {t['CPU_seconds']:.6f} CPU秒、今回audit {out['audit_CPU_seconds']:.6f} CPU秒。旧初期化/全探索費用は無料にしない。", 
        'SDK direct restricted-t2 variantであり著者初期化コードの完全再現ではない。CISD fallback2条件を含む。pair encodingとfull-spin-copyは異なる回路/計数点、physical Jacobian rankはunknown。CCSD/回路係数を固定したからparameter0とはしない。',
        '同じ1000 model samplesとR32上限だが、観測support不足なら実Rが小さい。COMMON_Rは全8法の最小で、古い7法だけの図とは異なる。この縮小点だけで勝利としない。',
        '新seed3反復だけで有意な優越を認定しない。旧drawは再利用で独立確認ではない。Fe参照はSCI、N2参照精度未認証、有限S²penaltyは厳密singlet最適化ではない。',
        '同振幅naive full-spin LUCJの安価な構築対照、原著native TN/DiceとSqDRIFTの最高予算・独立確認は残る。これはtriageであり、最強論文への勝利や新規性の証明ではない。']
    write(ROOT/'analysis/N35_SPINLESS_RESULTS_JA.md','\n'.join(lines).encode())
    print(out['status'],comparisons,flush=True)


if __name__=='__main__':run()
