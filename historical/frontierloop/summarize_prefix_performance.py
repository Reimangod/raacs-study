"""Whole N17 descriptive frontier/cost review; never selects new circuits."""
import csv
from collections import Counter,defaultdict
from io import StringIO
import math
from pathlib import Path
import statistics
from .paths import ROOT
from .io import load,save,write,sha256


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in ['case','source','method','arm'])].append(row)
    out=[]
    for key,values in sorted(groups.items()):
        if len(values)!=3 or {r['replicate'] for r in values}!={160101,160102,160103}:
            raise ValueError('All three planned replicates, including missing states, required')
        good=[r for r in values if r['energy_Ha'] is not None]
        item=dict(zip(['case','source','method','arm'],key))
        item.update(planned_replicates=3,complete_replicates=len(good),
            replicate_statuses=dict(Counter(r['status'] for r in values)),
            draw_ids=sorted({r['draw_id'] for r in values if r.get('draw_id')}),
            mean_of_available_only_not_equivalence_certificate=True)
        for metric in ['CNOT','CNOT_Depth','Total_Depth','declared_parameters']:
            present={r[metric] for r in values if r[metric] is not None}
            if len(present)>1:raise ValueError('Replicates unexpectedly changed circuit resources')
            item[metric]=next(iter(present)) if present else None
        for metric in ['energy_Ha','penalized_energy_Ha','recovery_pct','standalone_numeric_CPU_seconds','actual_CI_dimension']:
            available=[r[metric] for r in good]
            if any(not math.isfinite(x) for x in available):raise ValueError('Nonfinite completed metric')
            item[metric+'_mean']=statistics.mean(available) if available else None
        item['energy_SD_mHa']=1000*statistics.stdev(r['energy_Ha'] for r in good) if len(good)>1 else None
        item['energy_min_Ha']=min((r['energy_Ha'] for r in good),default=None)
        item['energy_max_Ha']=max((r['energy_Ha'] for r in good),default=None)
        item['reference_kind']=sorted({r['reference_kind'] for r in good})
        item['reference_precision']=sorted({r['reference_precision'] for r in good})
        points=[r for r in values if r.get('original_opt1_resources')]
        if points:
            item['original_opt1_resources']=points[0]['original_opt1_resources']
            item['original_opt3_resources']=points[0]['original_opt3_resources']
        out.append(item)
    return out


def comparisons(groups):
    bykey={tuple(g[k] for k in ['case','source','method','arm']):g for g in groups};rows=[]
    for g in groups:
        if g['method']=='ORIGINAL':continue
        p=bykey[(g['case'],g['source'],'ORIGINAL',g['arm'])]
        complete=g['complete_replicates']==p['complete_replicates']==3
        row=dict(case=g['case'],source=g['source'],method=g['method'],arm=g['arm'],
            complete_comparison=complete,comparison_status='ALL3_REPLICATES_AVAILABLE_DESCRIPTIVE_ONLY' if complete else 'INCOMPLETE_NOT_A_FAIR_MEAN_COMPARISON',
            energy_mean_delta_mHa=1000*(g['energy_Ha_mean']-p['energy_Ha_mean']) if complete else None,
            penalized_mean_delta_mHa=1000*(g['penalized_energy_Ha_mean']-p['penalized_energy_Ha_mean']) if complete else None,
            energy_equivalence_certified=False)
        for metric in ['CNOT','CNOT_Depth','Total_Depth','declared_parameters']:
            valid=g[metric] is not None and p[metric] is not None
            row[metric+'_delta']=g[metric]-p[metric] if valid else None
            row[metric+'_reduction_pct']=100*(p[metric]-g[metric])/p[metric] if valid and p[metric] else None
        points=[g.get('original_opt1_resources'),g.get('original_opt3_resources')]
        row['selected_resource_nondegradation_vs_every_actual_source_compiler_point']=(
            all(g[k]<=pnt[k] for pnt in points for k in ['CNOT','CNOT_Depth','Total_Depth']) if all(points) else None)
        rows.append(row)
    return rows


def elapsed_seconds(text):
    days,clock=(text.split('-',1) if '-' in text else ('0',text));parts=list(map(int,clock.split(':')))
    if len(parts)!=3:raise ValueError('Explicit Slurm elapsed HH:MM:SS required')
    return int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]


def run():
    paths={batch:ROOT/'analysis'/('N17-'+batch+'-common-readout-summary.json') for batch in ['M4','A100']}
    batches={b:load(p) for b,p in paths.items()};allrows=[];controls=[];protos={}
    for b,summary in batches.items():
        expected='N17_'+b+'_ALL_FROZEN_LOGICAL_READOUTS_AUDITED'
        if summary['status']!=expected:raise RuntimeError('Actual complete batch audit required')
        local=ROOT/('runs/N17-INDEPENDENT-PREFIX-CI-'+b+'-v1');p=load(local/'protocol.json');protos[b]=p
        if summary['protocol_SHA256']!=sha256((local/'protocol.json').read_bytes()) or summary['terminal_SHA256']!=sha256((local/'collected/terminal.json').read_bytes()):
            raise RuntimeError('Actual audited terminal/context changed')
        allrows.extend(summary['rows']);controls.extend(summary['unrestricted_HF_controls'])
    if protos['M4']['worker_SHA256']!=protos['A100']['worker_SHA256'] or protos['M4']['pipeline_SHA256']!=protos['A100']['pipeline_SHA256'] or protos['M4']['full_logical_inventory_SHA256']!=protos['A100']['full_logical_inventory_SHA256']:
        raise RuntimeError('Batches do not share registered policy/math/full inventory')
    if len(allrows)!=216 or len({(r['cell_id'],r['arm']) for r in allrows})!=216 or len(controls)!=4:
        raise RuntimeError('All108x2readouts and4 classical controls required')
    groups=aggregate(allrows);contrast=comparisons(groups)
    if len(groups)!=72 or len(contrast)!=48:raise RuntimeError('All planned group comparisons required')
    sampling=load(ROOT/'analysis/N17-sampling-stage-inventory.json')
    if sampling['logical_status_counts']!={'COMPLETED':90,'MISSING_SAME_PROBLEM_SOURCE':18} or sampling['verified_unique_draws']!=81:
        raise RuntimeError('All actual unique source observations required')
    sample_root=ROOT/'runs/N17-INDEPENDENT-PREFIX-PERFORMANCE-v1';allocations=[]
    for lane in [0,1]:
        collected=load(sample_root/f'lane{lane}-collection.json');job=collected['job']['job_id']
        lines=[s.split('|') for s in collected['accounting'].splitlines() if s.split('|')[0]==job]
        if len(lines)!=1 or lines[0][1]!='COMPLETED' or lines[0][3]!='0:0':raise RuntimeError('Actual GPU accounting incomplete')
        allocations.append(dict(job_id=job,seconds=elapsed_seconds(lines[0][2]),accounting=collected['accounting']))
    exact_alias=[]
    for g in groups:
        if g['method']!='STATE_OVERLAP_ROUND':continue
        angle=next(a for a in groups if (a['case'],a['source'],a['method'],a['arm'])==(g['case'],g['source'],'ANGLE_ONLY_ROUND',g['arm']))
        exact_alias.append(dict(case=g['case'],source=g['source'],arm=g['arm'],
            same_actual_draws=(g['draw_ids']==angle['draw_ids']) if g['draw_ids'] else None,
            state_minus_angle_energy_mean_mHa=1000*(g['energy_Ha_mean']-angle['energy_Ha_mean']) if g['complete_replicates']==angle['complete_replicates']==3 else None))
    result=dict(status='N17_ALL108_CELLS_216_READOUTS_4_HF_CONTROLS_REVIEWED',groups=groups,parent_comparisons=contrast,
        state_angle_comparisons=exact_alias,unrestricted_HF_controls=controls,raw_readout_status_counts=dict(Counter(r['status'] for r in allrows)),
        actual_new_model_shots=sampling['actual_new_model_shots'],actual_new_QPU_shots=0,
        actual_GPU_allocations=allocations,total_GPU_allocation_seconds=sum(a['seconds'] for a in allocations),
        sampling_body_CPU_seconds=sampling['simulation_CPU_seconds_sum'],
        actual_CI_campaign_CPU_seconds=sum(b['CPU_seconds'] for b in batches.values()),
        actual_new_CI_roots=sum(b['actual_new_CI_roots'] for b in batches.values()),
        failed_CI_attempts=sum(b['failed_CI_attempts'] for b in batches.values()),
        original_source_construction_cost_not_zero_and_not_included_here=True,
        policy_compile_qualification_and_analysis_costs_additional=True,
        cold_total_matched_work=False,scientific_novelty_or_best_paper_superiority_established=False,
        measured_chemical_systems=3,measured_physical_conditions=4,three_replicates_not_independent_molecules=True,
        final_CI_state_not_prepared_by_sampler=True,goal_complete=False,
        batch_audit_SHA256={b:sha256(p.read_bytes()) for b,p in paths.items()})
    save(ROOT/'analysis/N17-whole-study-summary.json',result)
    for name,rows in [('groups',groups),('parent-comparisons',contrast)]:
        stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
        write(ROOT/'analysis'/('N17-whole-study-'+name+'.csv'),stream.getvalue().encode('utf-8-sig'))
    def fmt(value,n=3):return '未取得' if value is None else f'{value:.{n}f}'
    lines=['# N17：元LUCJと二つのprefix圧縮の全体結果','',
        f"全108セル×2readoutを終了状態付きで統合。{result['raw_readout_status_counts']}。実観測810万model shots、QPU Shot0。", 
        f"A100割当は計{result['total_GPU_allocation_seconds']}秒。CI campaign {result['actual_CI_campaign_CPU_seconds']:.3f}CPU秒、新CI根{result['actual_new_CI_roots']}。source初期化や過去の探索費用を含むcold総費用ではない。",'',
        '各行は固定した3反復の記述統計。平均の良化だけで精度同等や優越を認定しない。欠測・失敗も同じ予定分母に残す。']
    for case in sorted({g['case'] for g in groups}):
        lines+=['','## '+case,'','| source | 方式 | readout | 完了/3 | energy平均 Ha | 反復SD mHa | Recovery平均 % | CNOT | CX Depth | Total Depth | Parameters* | CI D平均 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g in [g for g in groups if g['case']==case]:
            lines.append('| '+' | '.join([g['source'],g['method'],g['arm'],str(g['complete_replicates']),fmt(g['energy_Ha_mean'],9),fmt(g['energy_SD_mHa']),
                fmt(g['recovery_pct_mean']),str(g['CNOT']),str(g['CNOT_Depth']),str(g['Total_Depth']),str(g['declared_parameters']),fmt(g['actual_CI_dimension_mean'],1)])+' |')
        control=next(c for c in controls if c['case']==case)
        lines+=['',f"同条件の制限なしHF-HCI対照：energy {fmt(control.get('energy_Ha'),9)} Ha、Recovery {fmt(control.get('recovery_pct'))}%、D {control.get('actual_CI_dimension','未取得')}。これは古典CIの結果で、0 CNOTでこの相関状態を準備できる意味ではない。"]
    lines+=['','## 判断に使うときの注意','',
        '- 最も良いseedだけを示さない。3反復全部がある場合だけ親との平均差を比較し、不完全な群を成功扱いしない。親との差と全資源frontierはCSV/JSONへ全件保存。',
        '- 元opt1/opt3双方を残す。CNOT-firstの親に対する改善と、元の最短Total Depthなど全選択肢への改善は別。Parametersは同じprefix/suffix familyの宣言自由実座標で、最小Jacobian rankではない。',
        '- 同じdrawをstate順位とangle順位で共有した場合、二つの独立成功ではない。同じN₂の2距離を独立した2分子に数えない。',
        '- 精度は最終CIのphysical H energy。最適化目的H+0.1S²、実CI次元、native/CI workを区別する。最大Dが同じでも同じ古典仕事量ではない。',
        '- RecoveryのFe参照はSCIでFCIではない。各raw CSVにreference種別・検証状態を保持する。理想noiseless modelの結果を実機のShot優位性へ読み替えない。',
        '- 強い原著TN全workflowや実機routing/noiseまで再現したわけではない。この一試験が終了しても、最強原著への総合的勝利や全goal完了とは呼ばない。']
    write(ROOT/'analysis/N17_WHOLE_STUDY_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],result['raw_readout_status_counts'],'CI CPU',result['actual_CI_campaign_CPU_seconds'],flush=True)


if __name__=='__main__':run()
