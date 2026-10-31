"""Read-only plan coverage. A completed primary grid is not study completion.

This is a report, not an additional authorization gate. Paper-input variants
and resource-limited controls remain explicit limitations even when all their
planned attempts are terminal.
"""
import csv,json,sqlite3
from collections import Counter
from time import time
from .io import ROOT,save_json,atomic_bytes,sha256
from .analysis_products import write_csv

TERMINAL={'COMPLETED','FAILED_ENGINEERING_PRESERVED','RESOURCE_CAPPED',
    'ENGINEERING_OR_RESOURCE_LIMIT_NO_FINAL_STATE','INITIALIZATION_CAPPED'}


def snapshot():
    db=sqlite3.connect(ROOT/'run-queue.sqlite');rows=[]
    for name,table,expected in(('S4_PRIMARY','planned_cells',480),('S5_LENGTH_DEPTH','sensitivity_cells',125),
        ('S_ORBITAL_BASIS','basis_cells',40)):
        exists=db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(table,)).fetchone()
        states=Counter(dict(db.execute('SELECT status,COUNT(*) FROM '+table+' GROUP BY status').fetchall()))if exists else Counter()
        terminal=sum(n for s,n in states.items()if s in TERMINAL)
        rows.append({'requirement':name,'planned':expected,'registered':sum(states.values()),'terminal':terminal,
            'numerically_completed':states.get('COMPLETED',0),'status_counts':dict(states),'coverage_closed':terminal==expected,
            'missing_is_not_zero':True})
    main=db.execute('SELECT case_id,result_path FROM planned_cells WHERE status="COMPLETED"').fetchall();db.close()
    planned_CI=3*len(main);observed_CI=0;complete_extension=0;expected_extension=0
    for case,path in main:
        result=json.loads((ROOT/path).read_text());observed_CI+=len(result['final_CI'])
        if(ROOT/'CI-extension-v1'/case/'trigger.json').exists():
            expected_extension+=1
            out=ROOT/'CI-extension-v1'/case/result['method_id']/(str(result['budget'])+'-'+str(result['seed']))/'CI-r64.json'
            if out.exists():
                v=json.loads(out.read_text())
                if v['source_result_sha256']!=sha256((ROOT/path).read_bytes()):raise RuntimeError('Capacity/source mismatch')
                complete_extension+=1
    rows.append({'requirement':'PRIMARY_CI_FOR_COMPLETED_RUNS','planned_full_grid':1440,'currently_due':planned_CI,
        'numerically_completed':observed_CI,'coverage_closed':observed_CI==1440,'selection_or_shots_added':False})
    triggered=list((ROOT/'CI-extension-v1').glob('*/trigger.json'))
    rows.append({'requirement':'CONDITIONAL_CI4096','triggered_cases':[p.parent.name for p in triggered],
        'planned':120*len(triggered),'currently_due':expected_extension,'numerically_completed':complete_extension,
        'coverage_closed':complete_extension==120*len(triggered),'new_model_shots':0})
    from .control_ci_capacity import sources
    control_sources,_=sources();control_done=0
    for source in control_sources:
        result=json.loads(source.read_text())
        extension=ROOT/'control-CI-extension-v1'/result['lane']/result['case_id']/result['method_id']/(str(result['budget'])+'-'+str(result['seed']))/'CI-r64.json'
        if extension.exists():
            if json.loads(extension.read_text())['source_result_sha256']!=sha256(source.read_bytes()):raise RuntimeError('Control extension/source mismatch')
            control_done+=1
    rows.append({'requirement':'S5_TN_CI4096_FOR_COMPLETED_CONTROLS','currently_due':len(control_sources),
        'numerically_completed':control_done,'coverage_closed':len(control_sources)==control_done,
        'all_future_control_rows_still_required_by_S5_TN_requirements':True,'new_model_shots':0})
    planned_file=ROOT/'analysis-v1/all-planned-cells.csv'
    if planned_file.exists():
        with planned_file.open()as f:planned=list(csv.DictReader(f))
        tn=Counter(r['status']for r in planned if r['lane']=='S_TN_INDEPENDENT_FINAL')
    else:tn=Counter()
    rows.append({'requirement':'TN_INDEPENDENT_FINAL','planned':20,'status_counts':dict(tn),
        'numerically_completed':tn.get('COMPLETED',0),'terminal':sum(v for s,v in tn.items()if s in TERMINAL),
        'coverage_closed':sum(v for s,v in tn.items()if s in TERMINAL)==20,
        'five_readouts_are_not_five_independent_optimizers':True})
    high_spec=ROOT/'TN-high-final-v1/spec.json'
    if high_spec.exists():
        high=json.loads(high_spec.read_text());statuses=Counter()
        for case in high['cases']:
            for seed in high['seeds']:
                path=ROOT/'common-runs'/high['lane']/case/high['method_id']/(str(high['model_shots_per_readout'])+'-'+str(seed))/'result.json'
                statuses['COMPLETED'if path.exists()else('FAILED_ENGINEERING_PRESERVED'if(ROOT/'TN-high-final-v1'/case/'terminal-attention.json').exists()else'NOT_YET_COMPLETED')]+=1
        terminal=sum(v for s,v in statuses.items()if s in TERMINAL)
        rows.append({'requirement':'TN_ORIGINAL_PLAN_HIGH_BUDGET_COMPLETION','planned':20,'model_shots_per_readout':1000000,
            'terminal':terminal,'numerically_completed':statuses.get('COMPLETED',0),'status_counts':dict(statuses),
            'coverage_closed':terminal==20,'old_medium_readouts_retained_as_separate_supplement':True,
            'post_start_not_confirmatory':True,'new_optimizers':0})
    mps=ROOT/'TN-MPS-diagnostic-v1/collected.json'
    rows.append({'requirement':'MPS_FIXED_STATE_DIAGNOSTIC_20Q','planned':1,'terminal':int(mps.exists()),
        'coverage_closed':mps.exists(),'full_32q_40q_MPS_error_not_certified':True,
        'post_start_diagnostic_not_confirmatory':True,'new_model_shots':0})
    p1=ROOT/'paper-anchors/P1-bridge-v1/terminal.json'
    rows.append({'requirement':'P1_SIX_SETTINGS_BRIDGE','planned':6,
        'terminal':json.loads(p1.read_text())['settings_points_completed']if p1.exists()else 0,
        'coverage_closed':p1.exists(),'original_paper_reproduced':False,
        'limitation':'Reconstructed AVAS input, current SDK grouping/compiler and common Cartesian readout'})
    n2=[]
    for case in('N2-32-R1p2-P2','N2-32-R2p4-P2'):
        for kind in('naive','compressed'):
            folder=ROOT/'paper-anchors/P2'/(case+'-'+kind)
            path=folder/'postprocessing-v2/result.json'if(folder/'postprocessing-v2/result.json').exists()else folder/'result.json'
            if path.exists():n2.append({'case_id':case,'kind':kind,'status':json.loads(path.read_text())['status'],
                'source':str(path.relative_to(ROOT)),'sha256':sha256(path.read_bytes())})
    rows.append({'requirement':'P2_N2_SOURCE_ANCHORS','planned':4,'terminal':len(n2),'coverage_closed':len(n2)==4,
        'original_Dice_solver_reproduced':False,'source_results':n2})
    fe=[ROOT/'paper-anchors/P2-Fe-remote-v1'/kind/'collected.json'for kind in('naive','compressed')]
    rows.append({'requirement':'P2_FE_SOURCE_ANCHORS','planned':2,'terminal':sum(p.exists()for p in fe),
        'coverage_closed':all(p.exists()for p in fe),'original_Dice_solver_reproduced':False})
    hardware=list((ROOT/'hardware-postprocessing-v2').glob('*/collected.json'))
    batches=list((ROOT/'hardware-postprocessing-v2').glob('*/output/CI-*-*.json'))
    rows.append({'requirement':'H_PUBLIC_CR_NO_CR_REANALYSIS','planned_cases':3,'terminal_cases':len(hardware),
        'planned_CR_batches':300,'completed_CR_batches':len(batches),'coverage_closed':len(hardware)==3,
        'full_10_iteration_CR_completion':len(batches)==300,'new_QPU_shots':0,
        'resource_capped_cases_are_not_successful_paper_reproduction':True})
    hci=list((ROOT/'classical-controls-v1').glob('*/HCI-epsilon-*/result.json'))
    hf=list((ROOT/'classical-controls-v1').glob('*/HF/*/result.json'))
    rows.append({'requirement':'CLASSICAL_HF_AND_BOUNDED_HCI','planned_HF':120,'terminal_HF':len(hf),
        'planned_HCI':12,'terminal_HCI':len(hci),'coverage_closed':len(hf)==120 and len(hci)==12,
        'optimized_SHCI_or_PT2_reproduced':False})
    compiler=list((ROOT/'rustiq-control-v1/common-readout-resource-joins').glob('*/*.json'))
    rows.append({'requirement':'SQDRIFT_COMPILER_CONTROL','planned':120,'terminal':len(compiler),
        'coverage_closed':len(compiler)==120,'new_model_shots':0,'original_Rustiq0p0p8_reproduced':False})
    result={'observed_at_unix':time(),'coverage':rows,'all_registered_requirements_closed':all(r['coverage_closed']for r in rows),
        'whole_experiment_complete':False,'paper_full_numerical_reproduction_certified':False,
        'report_not_new_gate':True,'missing_or_capped_is_not_scientific_inferiority':True,
        'remaining':[r['requirement']for r in rows if not r['coverage_closed']]}
    save_json(ROOT/'analysis-v1/requirement-coverage.json',result)
    write_csv(ROOT/'analysis-v1/requirement-coverage.csv',[{k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows])
    lines=['# 計画全体の充足状況','',
        '480主セルだけでは全研究の完了ではありません。以下は計画に含まれる各部分の状態です。完了と、原論文の数値再現成功とは分けます。','',
        '| 計画項目 | 予定範囲の処理終了 | 状態の要点 |','|---|---|---|']
    for r in rows:
        note={k:v for k,v in r.items()if k in('planned','terminal','numerically_completed','planned_cases','terminal_cases','planned_CR_batches','completed_CR_batches','currently_due')}
        lines.append('| '+r['requirement']+' | '+('終了'if r['coverage_closed']else'未完了')+' | '+json.dumps(note,ensure_ascii=False)+' |')
    lines+=['','時間・容量上限で終わった対照は、その不足を報告したという意味でterminalです。必要精度に達した・原論文を再現した、という意味にはしません。',
        '最終完了には、この表だけでなく、全予定/欠測表、資源・精度図、共有/単独/失敗を含む費用の解釈、case別の平易な考察が必要です。']
    atomic_bytes(ROOT/'analysis-v1/REQUIREMENT_COVERAGE_JA.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps({'remaining_requirements':result['remaining'],'all_registered_requirements_closed':result['all_registered_requirements_closed']}),flush=True)


if __name__=='__main__':snapshot()
