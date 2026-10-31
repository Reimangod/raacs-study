"""Read-only, evidence-bound synthesis of all15 full-study questions.

Never starts an experiment, overwrites a source report, or calls a partial
cohort complete. CLI outputs content-addressed additive snapshots.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OBJECTIVE=Path('/Users/rei/.codex/attachments/662165c0-336c-4a5f-88b9-6719d6fe76ce/goal-objective.md')
PRIMARY='RAACS:full_cost-qsci_local_grid-last_only-exact_small'
CEO='B1:source_final_ceo_mvp'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'


def question_titles(text):
    body=text.split('# 11. 最終レポートで答える研究上の質問',1)[1].split('# 12.',1)[0]
    found={int(i):title for i,title in re.findall(r'^(\d+)\. (.+)$',body,re.M)}
    if set(found)!=set(range(1,16)):raise ValueError('Authoritative15-question coverage changed')
    return found


def paired_cases(frame,base,cases):
    """All planned cases retained. Seed means already grouped upstream."""
    if frame.duplicated(['case_id','family']).any():raise ValueError('Unaggregated seeds/budgets')
    methods=sorted(set(frame.family));lookup=frame.set_index(['case_id','family']);result=[]
    for case in cases:
        for method in methods:
            a=(case,base);b=(case,method)
            complete=a in lookup.index and b in lookup.index
            if complete and 'all_registered_seeds_available' in frame:
                complete=bool(lookup.loc[a].all_registered_seeds_available and lookup.loc[b].all_registered_seeds_available)
            row=dict(case_id=case,molecule=case.split('_R')[0],baseline=base,family=method,
                status='AVAILABLE_PAIR' if complete else 'MISSING_PAIR_PRESERVED')
            if row['status']=='AVAILABLE_PAIR':
                x=lookup.loc[a];y=lookup.loc[b]
                row.update(delta_error_mHa=y.mean_error_mHa-x.mean_error_mHa,
                    baseline_error_mHa=x.mean_error_mHa,candidate_error_mHa=y.mean_error_mHa,
                    baseline_worst_CNOT=x.worst_registered_component_cnot,candidate_worst_CNOT=y.worst_registered_component_cnot,
                    baseline_worst_depth=x.worst_registered_component_depth,candidate_worst_depth=y.worst_registered_component_depth,
                    candidate_conditional_success=y.mean_conditional_success,
                    candidate_build_plus_final_known=bool(y.full_model_budget_known))
            result.append(row)
    return pd.DataFrame(result)


def equal_molecule_pairs(frame):
    available=frame[frame.status=='AVAILABLE_PAIR']
    bymol=available.groupby(['family','molecule']).agg(paired_conditions=('case_id','size'),
        delta_error_mHa=('delta_error_mHa','mean'),baseline_error_mHa=('baseline_error_mHa','mean'),
        candidate_error_mHa=('candidate_error_mHa','mean')).reset_index()
    result=bymol.groupby('family').agg(molecules=('molecule','size'),paired_conditions=('paired_conditions','sum'),
        equal_molecule_delta_error_mHa=('delta_error_mHa','mean')).reset_index()
    return bymol,result


def trajectory_associations(curves,coverage):
    joined=curves.merge(coverage[['checkpoint_id','nonhf_weighted_coverage']],on='checkpoint_id',validate='one_to_one')
    result=[]
    for (case,method),g in joined.groupby(['case_id','method']):
        valid=g[['mean_error_mHa','nonhf_weighted_coverage']].dropna()
        nonconstant=len(valid)>=3 and valid.nunique().min()>1
        rho=valid.rank().corr().iloc[0,1] if nonconstant else None
        result.append(dict(case_id=case,molecule=case.split('_R')[0],method=method,checkpoints=len(g),
            spearman_coverage_vs_QSCI_error=rho,status='DESCRIPTIVE_ASSOCIATION' if nonconstant else 'CONSTANT_OR_TOO_FEW',
            independent_molecular_samples=False,isolated_causal_effect=False))
    return pd.DataFrame(result)


def render_table(frame):
    def cell(x):
        if pd.isna(x):return 'UNKNOWN / N/A'
        if isinstance(x,float):return f'{x:.6g}'
        return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['|'+'|'.join(frame.columns)+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+
        ['|'+'|'.join(cell(x) for x in r)+'|' for r in frame.itertuples(index=False,name=None)])


def build(output,mode='standard'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<64*2**20:raise RuntimeError('Read-only synthesis requires64MiB free')
    sources={}
    def bind(p):
        p=Path(p);sources[str(p) if not p.is_relative_to(ROOT) else str(p.relative_to(ROOT))]=sha(p);return p
    def read(p):return json.loads(bind(p).read_text())
    def table(p):return pd.read_csv(bind(p))
    titles=question_titles(bind(OBJECTIVE).read_text())
    costs=ROOT/'results/final-cost-frontiers-standard-v2'
    cm=read(costs/'manifest.json');ca=read(costs/'audit-v1/validation.json')
    assert ca['status']=='RAW_COSTS_REPEATS_SEED_COMPLETENESS_FRONTIERS_AND_HASHES_VALIDATED'
    assert ca['input_manifest_sha256']==sha(costs/'manifest.json')
    for rel,d in cm['output_sha256'].items():assert sha(bind(costs/rel))==d
    cases=cm['cases'] if mode=='standard' else ['H2_R0.7','H4_R1.1']
    families=table(costs/'all-seed-family-frontiers.csv');families=families[families.case_id.isin(cases)]
    primary=families[(families.scope=='readout_only')&(families.M==1000)].copy()
    pairs=paired_cases(primary,CEO,cases);molecular,pair_means=equal_molecule_pairs(pairs)
    raw_metrics=table(costs/'all-budget-resource-metrics.csv');raw_metrics=raw_metrics[raw_metrics.case_id.isin(cases)]
    raw_status=table(costs/'all-planned-sampler-statuses.csv');raw_status=raw_status[raw_status.case_id.isin(cases)]
    relations=ROOT/'results/final-relationships-standard-v1'
    read(relations/'presentation-v2/validation-and-style-v2.json')
    curves=table(relations/'growth-curves.csv');coverage=table(relations/'HF-vs-nonHF-coverage.csv')
    curves=curves[curves.case_id.isin(cases)];coverage=coverage[coverage.case_id.isin(cases)]
    association=trajectory_associations(curves,coverage)
    oracle=table(relations/'oracle-random-trained-discovery.csv')
    oracle=oracle[(oracle.M==1000)&oracle.case_id.isin(cases)].copy();oracle['molecule']=oracle.case_id.str.split('_R').str[0]
    discovery=oracle.groupby(['case_id','molecule','method']).nonhf_expected_weighted_coverage.mean().reset_index()
    discovery_mol=discovery.groupby(['molecule','method']).nonhf_expected_weighted_coverage.mean().reset_index()
    discovery_means=discovery_mol.groupby('method').agg(molecules=('molecule','size'),mean_nonHF_expected_coverage=('nonhf_expected_weighted_coverage','mean')).reset_index()
    aroot=ROOT/'results/sampler-study-standard-v1'
    aa=read(aroot/'milestone1-audit.json');assert aa['status']=='PASS_MILESTONE1_CORE_AUDIT'
    stops=table(aroot/'analysis/new-draw-retrospective-stops.csv')
    stops=stops[(stops.shots==1000)&(stops.R_cap==32)&stops.case_id.isin(cases)].copy();stops['molecule']=stops.case_id.str.split('_R').str[0]
    stop_summary=stops.groupby(['method','tolerance_mHa']).agg(conditions=('case_id','size'),
        earlier_than_saved_endpoint=('step_ratio',lambda s:int((s<1).sum())),median_CNOT_reduction_percent=('cnot_reduction_percent','median'),median_depth_reduction_percent=('depth_reduction_percent','median')).reset_index()
    capacity=table(aroot/'analysis/final-capacity-effects.csv');capacity=capacity[(capacity.shots==1000)&capacity.case_id.isin(cases)]
    capacity_summary=capacity.groupby(['molecule','method']).agg(conditions=('case_id','size'),median_error_improvement_R32_to_R512_mHa=('improvement_R32_to_R512_mHa','median')).reset_index()
    croot=ROOT/'results/PhaseC-actual-growth-standard-v1'
    cv=read(croot/'validation.json');read(croot/'evaluation-validation.json')
    cprimary=table(croot/'derived-report-v1/primary-equal-molecule-summary.csv');cprimary=cprimary[cprimary.cap==64]
    online=table(ROOT/'results/online-parameter-standard-v1/report-v1/molecule-means.csv')
    online_summary=online.groupby('mode').agg(molecules=('molecule','size'),delta_QSCI_error_mHa_vs_online_grid=('delta_QSCI_error_mHa_vs_online_grid','mean'),delta_physical_energy_error_mHa_vs_online_grid=('delta_physical_energy_error_mHa_vs_online_grid','mean')).reset_index()
    bank=table(ROOT/'results/fixed-bank-allocation-standard-v1/support-saturation-v1/per-source-support-diagnostic.csv')
    sqd=table(ROOT/'results/full-fermionic-SQD-projected-standard-v1/report-v1/paired-molecule-comparisons.csv')
    sqdv=read(ROOT/'results/full-fermionic-SQD-projected-standard-v1/selected-ground-validation.json')
    sqd_summary=sqd.groupby(['baseline','readout_flip_probability']).agg(molecules=('molecule','size'),mean_RAACS_minus_baseline_SQD_mHa=('mean_RAACS_minus_baseline_SQD_mHa','mean')).reset_index()
    scale=read(ROOT/'results/molecular-MPS-scaling-standard-v1/validation.json')
    e=read(ROOT/'results/molecular-strong-controls-standard-v2/execution-status.json')
    f=read(ROOT/'results/gate-noise-full-standard-v1/execution-status.json')
    fdb=bind(ROOT/'results/gate-noise-full-standard-v1/events.sqlite')
    # SHM contains volatile reader bookkeeping, not experiment records.
    for suffix in ('-wal',):
        p=Path(str(fdb)+suffix)
        if p.exists():bind(p)
    with sqlite3.connect(f'file:{fdb}?mode=ro',uri=True) as db:
        db.execute('BEGIN');fcounts=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
        fchunks,fshots=db.execute('SELECT count(*),sum(shots) FROM chunks').fetchone();db.rollback()
    follow=read(ROOT/'results/integral-followup-sequence-v1/execution-status.json')
    hci_started=(ROOT/'results/integral-threshold-HCI-standard-v1/manifest.json').exists()
    classical=table(ROOT/'results/classical-SCI-standard-v2/summary.csv')
    emol=pd.Series([x['status'] for x in e['rows']]).value_counts().to_dict()
    runtime={'E':{'status':e['status'],'planned':234,'terminal':len(e['rows']),'status_counts':emol},
        'F':{'status':f['status'],'planned':3600,'terminal':sum(fcounts.values()),'status_counts':fcounts,
             'committed_chunks':fchunks,'known_committed_model_shots':fshots,'inflight_cost_if_censored':'UNKNOWN_NOT_ZERO'},
        'followup':follow,'actual_threshold_HCI_standard_started':hci_started,'pending_capacity_is_not_scientific_failure':True}
    latest=bind(ROOT/'results/final-figure-inventory-v1/LATEST.md').read_text()
    inventory_rel=re.search(r'\]\((snapshots/[^)]+/INDEX.md)\)',latest).group(1)
    inv=read(ROOT/'results/final-figure-inventory-v1'/Path(inventory_rel).with_name('inventory.json'))
    figure_status=[dict(figure=r['figure'],title=r['requested_title'],status=r['status'],three_format_candidates=r['candidate_png_svg_pdf_sets'],scientifically_certified_by_synthesis=False) for r in inv['rows']]
    assert len(figure_status)==18
    tables={'all-case-primary-pairs.csv':pairs,'paired-molecule-means.csv':molecular,'equal-molecule-primary-comparisons.csv':pair_means,
        'within-trajectory-coverage-energy-associations.csv':association,'discovery-by-molecule.csv':discovery_mol,
        'discovery-equal-molecule-means.csv':discovery_means,'retrospective-stop-summary.csv':stop_summary,'all-retrospective-stop-cases.csv':stops,
        'capacity-by-molecule.csv':capacity_summary,'actual-FCI-free-primary-stopping.csv':cprimary,'online-angle-controls-molecule-means.csv':online,
        'online-angle-controls-summary.csv':online_summary,'SQD-recovery-paired-molecule-means.csv':sqd,
        'SQD-recovery-equal-molecule-summary.csv':sqd_summary,'planned-sampler-case-status.csv':raw_status,'requested18figures-status.csv':pd.DataFrame(figure_status)}
    metric_fields=['mean_error_mHa','success_fraction',None,'component_cnot_max','component_cnot_depth_max',
                   'native_2Q_count','construction_recorded_seconds','total_classical_wall_seconds','peak_process_RSS_bytes','total_model_shots_per_application']
    tables['comparison-metric-field-availability.csv']=pd.DataFrame([dict(metric=i+1,field=c or 'SEE_SEPARATE_TESTED_GRID_REPORT',
        rows=len(raw_metrics),available_rows=int(raw_metrics[c].notna().sum()) if c else None) for i,c in enumerate(metric_fields)])
    # Remaining report assembly is separate from the numerical tables above.
    return assemble(output,mode,cases,titles,tables,sources,runtime,cv,scale,sqdv,bank,classical,e,fcounts,pair_means,figure_status)


def assemble(output,mode,cases,titles,tables,sources,runtime,cv,scale,sqdv,bank,classical,e,fcounts,pair_means,figure_status):
    statuses=['DESCRIPTIVE_SUPPORTED','ASSOCIATION_NOT_CAUSATION','RETROSPECTIVE_SUPPORTED_NOT_ONLINE',
        'TESTED_NO_PRIMARY_SAVING','DESCRIPTIVE_SUPPORTED_MECHANISM_NOT_IDENTIFIED',
        'CONTROLS_EXECUTED_CAUSE_NOT_ISOLATED','BOUNDED_OPTIMIZER_COMPARISON_ONLY',
        'CONDITIONAL_TRADEOFF_NOT_UNIVERSAL','NO_GENERAL_SUPERIORITY_ESTABLISHED','SCOPED_ABLATIONS_NOT_UNIQUE_CAUSAL_DECOMPOSITION',
        'PARTIAL_COST_SCOPE_TOTAL_WALL_UNKNOWN','PARTIAL_SCALING_PENDING_STRONG_CONTROLS',
        'GEOMETRY_ACTIVE_SPACE_TESTED_NOT_UNSEEN_IDENTITIES','SELECTED_GROUND_TESTED_FULL_NOISE_PENDING',
        'UNRESOLVED_STRONG_THRESHOLD_HCI_PRODUCTION_PENDING']
    answers=[
        'M1000の最終回路を全seed→距離→分子の順で平均した発見率を出力した。FCI配置重みは評価専用。FCI Born分布は実行可能な最適サンプラーでも発見率の上界でもない。',
        '物理条件・poolごとの軌道内でcoverageと独立QSCI平均誤差の順位相関を計算した。異なる分子の絶対energyを混ぜない。相関はoperator追加・全係数再最適化・CI次元の因果分離ではない。',
        '全条件の保存最終点との比をM1000/R32・許容0/.1/.5/1mHaで集計した。将来を見て選ぶretrospective診断であり、真の収束点や実用停止則ではない。',
        f'実成長{cv["actual_cells"]}件・{cv["problems_checked"]}問題を検証済み。事前選択主方式のR64全予算でCNOT/Depth削減率とregretは0。選択済み規則はnativeと同じであり、省計算停止を実証していない。',
        '全予定caseでqubitとCEO-MVPの固定最終点を共通M1000/R64で比較した。実dは一致とは限らない。元のsector診断ではH4_R2.2/qubitのSz逸脱があり、単なる勾配収束を有用性と同一視しない。現象比較は原因の証明ではない。',
        'random構造/角度・trained順序/random角度の全seed対照を保持した。発見率差は算出したが、回路長・演算子列・角度の作用は相互依存する。学習済みtopology継承の費用はUNKNOWNであり無料random baselineではない。',
        '固定LUCJに加えbounded CCSD-VQE/QSCI/MP2/randomを比較した。全予定caseと初期化欠測を残す。bounded optimizer終了は大域最適LUCJを意味せず、CPU/GPUやcacheの違いから速度優位を導かない。',
        '既知の構築+最終model shotsとworst-component論理Depthのfrontierを別途検証済み。同じ総予算でも実d/精度/Depthにtrade-offがある。実機shotsやQPU時間は測っていない。',
        'RAACS主方式を全seed平均し、同じcaseの強baselineと比較した。全手法・全予算での優位は示されていない。M/R固定点の優劣、実d一致対照、構築費用の既知/未知を分離する。',
        f'450件のfresh online angle対照と固定topology対照を区別する。fixed bank{len(bank)}件ではunion追加support0のbankが{int((bank.extra_union_configurations_vs_last==0).sum())}件。これはこのbankの限界であり、全multishallowの否定ではない。独立pool/停止効果の完全な加法因果分解は未達。',
        '構築秒・測定model shots・sampling access・CI lookupを分離した。不明の訓練測定費用や全工程wall timeを0としない。cache lookupを新規取得時間として足せないため、総古典+量子費用の最終frontierは未完成。',
        f'元のmatrix-free/MPS scalingは{scale["planned"]}セル・最大{scale["maximum_actual_molecular_qubits"]}q。強対照は{len(e["rows"])}/234terminalで容量待機。近似不適格を物理手法の敗北と呼ばず、高精度ground referenceがない大型条件の化学精度はUNKNOWN。',
        'LOMO90foldと新geometry/活性空間の実成長を実施したが、主方式の節約は0。未使用の分子identityへの優位を実証したとは言えない。構築seed1と有限個の既知分子という範囲を維持する。',
        f'full fermionic SQD selected-ground successorは{sqdv["completed"]}complete、{sqdv["projected_ground_results"]}selected-root検査。ideal/readoutでRAACSの一般優位はない。gate noise標準は{sum(fcounts.values())}/3600terminalであり、未完了全cohortを最終結論に使わない。相関energy回復率とconfiguration recoveryは別概念。',
        f'旧cardinality-ranked SCIは{len(classical)}R別行を比較済みだが、高性能SHCI/DMRGの代替とは呼ばない。実epsilon-threshold HCIはproduction未完了。強い古典手法に対する量子的追加価値は未確立。']
    evidence=[['discovery-equal-molecule-means.csv'],['within-trajectory-coverage-energy-associations.csv'],
        ['all-retrospective-stop-cases.csv','retrospective-stop-summary.csv'],['actual-FCI-free-primary-stopping.csv'],
        ['all-case-primary-pairs.csv'],['discovery-by-molecule.csv'],['equal-molecule-primary-comparisons.csv','planned-sampler-case-status.csv'],
        ['all-case-primary-pairs.csv'],['paired-molecule-means.csv'],['online-angle-controls-summary.csv'],
        ['source-evidence.json'],['source-evidence.json'],['actual-FCI-free-primary-stopping.csv'],
        ['SQD-recovery-paired-molecule-means.csv'],['source-evidence.json']]
    questions=[dict(number=i,title=titles[i],status=statuses[i-1],answer=answers[i-1],evidence=evidence[i-1]) for i in range(1,16)]
    metrics=[
        (1,'QSCI/SQD Energy Error','AVAILABLE_SCOPED','Fixed M/R and same-problem pairs; large reference errors UNKNOWN.'),
        (2,'Chemical Accuracy Success Probability','AVAILABLE_CONDITIONAL','Conditional readout success, not unconditional construction success.'),
        (3,'Shots to90% Success','AVAILABLE_TESTED_GRID_CENSORING','Observed90% differs from confidence-lower-bound90%; do not interpolate.'),
        (4,'CNOT Count','AVAILABLE_LOGICAL','Z-sampling circuit, not original preparation state.'),
        (5,'CNOT Depth','AVAILABLE_LOGICAL','Multi-circuit maximum and weighted mean differ.'),
        (6,'Native2Q Count/Depth','PARTIAL_SYNTHETIC_TARGET','Source synthetic ECR and F native-CX differ; not calibrated hardware.'),
        (7,'Classical Construction Wall','AVAILABLE_SCOPED_PARTIAL','Scopes and dependencies differ; missing times UNKNOWN.'),
        (8,'Total Classical Wall','MISSING_COMPLETE_TRACE','No complete common-scope from-scratch end-to-end time.'),
        (9,'Peak Memory','AVAILABLE_PARTIAL_PROCESS_HWM','Process high-water, not isolated allocation; unknowns retained.'),
        (10,'End-to-End Quantum Shot Budget','PARTIAL_MODEL_NOT_HARDWARE','Known pilots/readout tracked; VQE/inherited topology UNKNOWN; hardware0.')]
    payload={'schema':'scalable-study-evidence-synthesis-v1','mode':mode,'cases':cases,'questions':questions,
        'primary_metrics':[dict(number=i,name=n,status=s,scope=t) for i,n,s,t in metrics],
        'runtime':runtime,'no_new_experiment_or_solver':True,'hardware_shots':0,'whole_goal_complete':False,
        'figure18_status':'MISSING_COMPLETE_TOTAL_CLASSICAL_WALL_TRACE','source_sha256':dict(sorted(sources.items())),
        'script_sha256':sha(__file__),'scope':'READ_ONLY_PARTIAL_SYNTHESIS_NOT_FINAL_COMPLETION_CERTIFICATE'}
    for p,s in sources.items():assert sha(Path(p) if Path(p).is_absolute() else ROOT/p)==s
    csvtext={name:f.to_csv(index=False) for name,f in tables.items()}
    payload['table_sha256']={name:hashlib.sha256(s.encode()).hexdigest() for name,s in csvtext.items()}
    identity=hashlib.sha256(canonical(payload).encode()).hexdigest()
    dest=output/'snapshots'/identity;dest.mkdir(parents=True,exist_ok=True)
    text=['# QSCI/SQDサンプラー研究：15問・費用・残課題の統合','',
        '**未完了を含む根拠集約であり、研究完了・CEO/RAACS一般優位・実機優位の宣言ではありません。**',
        f'解析mode={mode}、主比較{len(cases)}条件。M1000/R64・最終回路・全登録seed平均。R共通は実CI次元一致ではありません。C/D/E/F引用集計は各独立cohortの全体であり、quick2条件と同じ集団ではありません。',
        '旧CEOはcommon-loop pool実装で、CEO*全subroutineの厳密再現ではありません。',
        '## 実行状態',render_table(pd.DataFrame([dict(phase='E strong molecular',planned=234,terminal=len(e['rows']),status=e['status']),
            dict(phase='F gate noise',planned=3600,terminal=sum(fcounts.values()),status=runtime['F']['status'])])),
        '容量待機・未実行・初期化失敗・近似不適格は、科学的な手法の劣位結果とは区別します。',
        '## 固定最終点の精度比較（負の差は相手の誤差がCEO-MVPより小さい）',render_table(pair_means),
        '分子内geometry平均→分子等重み。距離を独立分子と数えず、欠測は全case表に残します。不明な構築費を免除した費用込み比較ではありません。',
        '## 15の研究質問']
    for q in questions:
        text.extend([f'### {q["number"]}. {q["title"]}',f'状態: `{q["status"]}`',q['answer'],
            '根拠: '+' / '.join(f'[{x}]({x})' for x in q['evidence'])])
    text+=['## 必須10指標',render_table(pd.DataFrame(payload['primary_metrics'])),
        '## 必須18形式',render_table(pd.DataFrame(figure_status)),
        '候補図の存在は科学的完了証明ではありません。Fig18を部分費用図で代用しません。',
        '## 残作業（元のscopeを維持）',
        'E標準234セル・F標準3600セルの未実行分、実threshold HCI quick/standard、native/bond postprocessing、完全費用trace/図18、最終要件別監査を完了する。',
        '完了済み実験を勝てるまで再試行せず、元のqueue/seed/cap/失敗証拠を保持する。LevelC/hardwareは利用可能性と近似資格を満たす場合だけ実施する。',
        '## 根拠ファイル（SHAはsource-evidence.json）']
    text += [f'- [{p}]({os.path.relpath(Path(p) if Path(p).is_absolute() else ROOT/p,dest)})' for p in sources]
    files={**csvtext,'source-evidence.json':canonical(payload),'REPORT_JA.md':'\n\n'.join(text)+'\n'}
    if sum(len(s.encode()) for s in files.values())>8*2**20:raise RuntimeError('Read-only synthesis exceeds8MiB cap')
    for name,body in files.items():
        p=dest/name
        if p.exists():assert p.read_text()==body
        else:p.write_text(body)
    (output/'LATEST.md').write_text(f'# Latest research synthesis\n\n[15-question report](snapshots/{identity}/REPORT_JA.md)\n\nWhole research goal remains incomplete.\n')
    return dict(snapshot=str(dest),mode=mode,questions=15,primary_metrics=10,requested_figure_forms=18,
        paired_case_rows=len(tables['all-case-primary-pairs.csv']),trajectory_associations=len(tables['within-trajectory-coverage-energy-associations.csv']),
        whole_goal_complete=False,new_draws=0,new_solver_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','standard'],default='standard')
    p.add_argument('--output',default=str(ROOT/'results/research-synthesis-v1'));a=p.parse_args()
    print(json.dumps(build(a.output,a.mode),sort_keys=True))


if __name__=='__main__':main()
