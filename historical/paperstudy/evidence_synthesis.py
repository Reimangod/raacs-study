"""Read-only paired comparisons, cost evidence, paper boundaries and plain report.

Never selects a favorable seed or treats an unavailable original implementation
as reproduced. CPU components with overlapping coverage are not added together.
"""
from collections import defaultdict,Counter
import csv,json,sqlite3
from pathlib import Path
from time import time
import numpy as np
from .analysis_products import write_csv
from .io import ROOT,save_json,atomic_bytes,sha256

NAMES={'A_RAACS_CURRENT':'RAACS（測りながら短い回路を構築）','B_LUCJ_NAIVE_L1':'通常初期化LUCJ・1層',
       'C_LUCJ_COMPRESSED_L1':'改善初期化LUCJ・1層','D_SQDRIFT_N20':'SqDRIFT SDK版・20励起'}


def optional(path):return json.loads(path.read_text())if path.exists()else None


def preprocessing_views():
    """Standalone LUCJ preprocessing scenarios are NOT extra executed work.

    Actual shared artifacts are charged once in cost_evidence. Here each row
    states what the registered pipeline would require without a warm cache.
    S5 preparation already contains naive factorization; never add it twice.
    """
    rows=[]
    for pattern in('prepared/*/*/construction.json','sensitivity-prepared/*/*/construction.json'):
        for p in sorted(ROOT.glob(pattern)):
            v=json.loads(p.read_text());case=v['case_id'];method=v['method_id'];S5=pattern.startswith('sensitivity-')
            amp_path=ROOT/'data/initialization-fresh'/case/'result.json';amp=optional(amp_path)
            if not amp:continue
            dependencies=[amp_path];basis_CPU=0.;amp_CPU=amp.get('CPU_process_seconds')
            derived=amp.get('rotation_and_localization_cost_shared_with_physical_preparation',False)
            if derived:
                upstream=ROOT/amp['source_converged_CCSD'];dependencies.append(upstream)
                amp_CPU=json.loads(upstream.read_text()).get('CPU_process_seconds')
                basis=ROOT/'data/physical'/case/'preparation-cost.json';dependencies.append(basis)
                basis_CPU=json.loads(basis.read_text()).get('CPU_seconds')
            compressed='COMPRESSED'in method;init=v['initialization_cost'];compressed_CPU=0.
            if compressed:
                source=ROOT/init['source_result'];dependencies.append(source)
                compressed_CPU=json.loads(source.read_text()).get('CPU_process_seconds')
            build_CPU=v.get('preparation_CPU_seconds')if S5 else v.get('compilation_CPU_seconds')
            # S5 prepare() starts its timer BEFORE naive factorization. Primary
            # prepare_lucj() starts the compilation timer AFTER load_operator().
            naive_extra=0. if S5 or compressed else init.get('naive_factorization_CPU_seconds')
            values=(amp_CPU,basis_CPU,compressed_CPU,build_CPU,naive_extra)
            rows.append({'case_id':case,'method_id':method,'n_reps':v['n_reps'],
                'initialization':amp['selected_method'],'upstream_amplitudes_CPU_seconds':amp_CPU,
                'shared_basis_and_amplitude_transform_CPU_seconds':basis_CPU,
                'basis_work_is_common_not_unique_LUCJ_advantage':derived,
                'compressed_initializer_required':compressed,'compressed_initializer_CPU_seconds':compressed_CPU,
                'configuration_build_CPU_seconds':build_CPU,'naive_factorization_already_in_build':S5 and not compressed,
                'naive_factorization_CPU_seconds_to_add':naive_extra,
                'registered_standalone_preprocessing_CPU_seconds':sum(values)if all(x is not None for x in values)else None,
                'CPU_value_is_measured_component_subtotal_not_complete_end_to_end':True,
                'scenario_given_registered_integrals_and_reference_orbitals':True,
                'excludes_sampling_CI_TN_optimization_and_import_startup':True,
                'counterfactual_view_not_additional_executed_CPU':True,'rows_not_additive':True,
                'not_to_add_to_worker_time_if_worker_already_includes_build':True,
                'dependency_sources_and_SHA256':json.dumps({str(q.relative_to(ROOT)):sha256(q.read_bytes())for q in dependencies},sort_keys=True),
                'construction_source':str(p.relative_to(ROOT))})
    return rows


def paired(rows):
    indexed={(x['case_id'],int(x['budget']),int(x['r_cap']),int(x['seed']),x['method_id']):x
             for x in rows if x['lane']=='C_PRIMARY'}
    individual=[];groups=defaultdict(list)
    for key,a in indexed.items():
        case,B,r,seed,method=key
        if method!='A_RAACS_CURRENT':continue
        for baseline in('B_LUCJ_NAIVE_L1','C_LUCJ_COMPRESSED_L1','D_SQDRIFT_N20'):
            b=indexed.get((case,B,r,seed,baseline))
            if not b:continue
            record={'case_id':case,'budget':B,'r_cap':r,'seed':seed,'candidate':method,'baseline':baseline,
                'energy_delta_mHa':1000*(float(a['energy_Ha'])-float(b['energy_Ha'])),
                'energy_sign_is_descriptive_not_significance':True,'numerical_tolerance_not_equivalence_margin':True}
            for metric in('CNOT','CNOT_depth','total_depth','coefficient_slots_or_member_mean'):
                av,bv=float(a[metric]),float(b[metric]);record[metric+'_delta']=av-bv
                record[metric+'_reduction_pct']=100*(bv-av)/bv if bv>0 else None
            individual.append(record);groups[(case,B,r,baseline)].append(record)
    summary=[]
    for(case,B,r,baseline),entries in sorted(groups.items()):
        x={'case_id':case,'budget':B,'r_cap':r,'baseline':baseline,'observed_pairs':len(entries),'planned_pairs':10}
        for field in('energy_delta_mHa','CNOT_reduction_pct','CNOT_depth_reduction_pct','total_depth_reduction_pct','coefficient_slots_or_member_mean_reduction_pct'):
            v=[e[field]for e in entries if e[field]is not None]
            x['mean_'+field]=float(np.mean(v))if v else None;x['median_'+field]=float(np.median(v))if v else None
            x['SD_'+field]=float(np.std(v,ddof=1))if len(v)>1 else None
        x['pairs_with_lower_RAACS_point_energy']=sum(e['energy_delta_mHa']<0 for e in entries)
        summary.append(x)
    return individual,summary


def cost_evidence():
    rows=[]
    families=[('data/initialization-fresh/*/result.json','shared_amplitudes_once','CPU_process_seconds','total_wall_seconds'),
        ('data/physical/NAPH20-P1/preparation-cost.json','shared_SCF_AVAS_integrals_once','CPU_seconds','wall_seconds'),
        ('data/physical/NAPH20-BLOCK-BOYS/preparation-cost.json','shared_block_Boys_and_covariant_transform_once','CPU_seconds','wall_seconds'),
        ('qualification/*-author-compressed-fresh/result.json','shared_compressed_initialization_once','CPU_process_seconds','total_wall_seconds'),
        ('sensitivity-initialization/*/compressed-L*/result.json','S5_compressed_initialization_once','CPU_process_seconds','total_wall_seconds'),
        ('prepared/*/*/construction.json','shared_compilation_once','compilation_CPU_seconds','compilation_wall_seconds'),
        # S5 starts its timer before naive factorization. Its immutable
        # producer records preparation_*, not primary compilation_* fields.
        ('sensitivity-prepared/*/*/construction.json','S5_shared_preparation_factorization_and_compilation_once','preparation_CPU_seconds','preparation_wall_seconds'),
        ('controller/*-terminal.json','local_primary_worker_including_child_exporters','child_CPU_seconds','wall_seconds'),
        ('sensitivity-controller/*.cost.json','local_S5_subprocess_including_children','child_CPU_seconds','wall_seconds'),
        ('remote-CPU-LUCJ-v1/*/*/remote-result.json','remote_shared_exact_state_and_final_sampling_once','CPU_seconds','wall_seconds'),
        ('remote-S5-LUCJ-v1/*/*/remote-result.json','remote_S5_shared_exact_state_parity_sampling_once','CPU_process_seconds','wall_seconds'),
        ('TN-controls-v1/*/optimization-result.json','TN_classical_surrogate_optimization_once','CPU_seconds','wall_seconds'),
        ('TN-controls-v1/*/terminal-evidence-v1/manifest.json','TN_failed_or_capped_attempt_Slurm_cost_separate','CPU_seconds','wall_seconds'),
        ('TN-controls-v2/*/terminal-evidence/remote/optimization-result.json','TN_checkpointed_successor_optimization_once','CPU_seconds','wall_seconds'),
        ('TN-controls-v2/*/terminal-evidence/remote/progress.json','TN_timeout_progress_CPU_lower_bound_not_additive_to_job_allocation','CPU_seconds','elapsed_wall_seconds'),
        ('TN-controls-v2/*/no-final-state.json','TN_failed_successor_Slurm_cost_separate','CPU_seconds','wall_seconds'),
        ('TN-controls-v1/*/final-readout/remote-result.json','TN_shared_exact_final_state_once','CPU_seconds','wall_seconds'),
        ('TN-high-final-v1/*/remote-result.json','TN_high_budget_shared_exact_final_state_once_not_new_training','CPU_seconds','wall_seconds'),
        ('common-runs/S_TN_INDEPENDENT_FINAL/*/*/*/result.json','TN_medium_final_local_CI_only','this_attempt_CPU_seconds','this_attempt_wall_seconds'),
        ('common-runs/S_TN_HIGH_BUDGET_COMPLETION/*/*/*/result.json','TN_high_final_local_CI_only','this_attempt_CPU_seconds','this_attempt_wall_seconds'),
        ('TN-MPS-diagnostic-v1/output/result.json','fixed20q_MPS_state_diagnostic_no_new_shots','CPU_seconds','wall_seconds'),
        ('basis-sensitivity-v1/controller/*.cost.json','localized_basis_subprocess_including_children','child_CPU_seconds','wall_seconds'),
        ('basis-sensitivity-v1/readout-r64/*/*.json','localized_basis_CI_extension_no_new_shots','CPU_seconds','wall_seconds'),
        ('CI-extension-v1/*/*/*/CI-r64.json','additional_CI_only_no_new_shots','CPU_seconds','wall_seconds'),
        ('control-CI-extension-v1/*/*/*/*/CI-r64.json','S5_TN_additional_CI_only_no_new_shots','CPU_seconds','wall_seconds'),
        ('hardware-postprocessing-v2/*/output/result.json','H_public_data_reanalysis_no_new_QPU_shots','CPU_seconds','wall_seconds'),
        ('paper-anchors/P2-Fe-remote-v1/*/output/result.json','P2_Fe_source_anchor_not_primary','CPU_seconds','wall_seconds'),
        ('paper-anchors/P1-bridge-v1/LUCJ-*/result.json','P1_current_LUCJ_bridge_not_primary','CPU_seconds','wall_seconds'),
        ('common-runs/P1_SETTINGS_BRIDGE_NOT_REPRODUCTION/*/*/*/result.json','P1_SqDRIFT_parent_process_only_exporter_CPU_not_included','this_attempt_CPU_seconds','this_attempt_wall_seconds'),
        ('paper-anchors/P1-bridge-v1/*/full-Cartesian-CI.json','P1_additional_CI_no_new_shots','CPU_seconds','wall_seconds'),
        ('paper-anchors/P2/*/result.json','P2_N2_source_anchor_unmeasured_total_CPU_is_unknown','CPU_seconds','wall_seconds'),
        ('paper-anchors/P2/*/postprocessing-v2/result.json','P2_N2_resumed_postprocessing_CPU_unknown','CPU_seconds','postprocess_wall_seconds'),
        ('paper-anchors/P2-common-readout-crosswalk-v1/*/r*.json','P2_common_readout_post_start_diagnostic_no_new_shots','CPU_seconds','wall_seconds'),
        ('pilot-preparation/*.json','S3_shared_LUCJ_exact_state_preparation_once','simulation_CPU_seconds','simulation_wall_seconds'),
        ('common-runs/S3_ENGINEERING_PILOT/*/*/*/result.json','S3_pilot_parent_process_only_not_complete_end_to_end','this_attempt_CPU_seconds','this_attempt_wall_seconds'),
        ('classical-controls-v1/*/HCI-setup.json','classical_threshold_HCI_setup_once','CPU_seconds','wall_seconds'),
        ('classical-controls-v1/*/HCI-epsilon-*/result.json','classical_threshold_HCI_variational_only','CPU_seconds','total_wall_seconds'),
        ('analysis-v1/Rustiq-exporter-accounting/*.json','resource_compiler_native_export_CPU_separate_from_parent','native_export_CPU_seconds','native_export_wall_seconds_sum'),
        ('rustiq-control-v1/members/*/group-preparations/*.json','resource_compiler_all_recorded_grouping_setup','CPU_seconds','wall_seconds'),
        ('rustiq-control-v1/common-readout-resource-joins/*/*.json','resource_recompile_controller_ONLY_exporter_separate','local_CPU_seconds','wall_seconds')]
    families.append(('filesystem-compression-v1/[0-9a-f]*.json','transparent_filesystem_storage_subprocess_only_not_sampler_work','storage_subprocess_CPU_seconds','wall_seconds'))
    families.extend([
        ('storage-metadata-v1/*.json','empty_archived_directory_metadata_storage_not_sampler_work','CPU_seconds','wall_seconds'),
        ('filesystem-compression-v1/closed-*index.json','offline_index_literal_byte_compression_not_sampler_work','storage_subprocess_CPU_seconds','wall_seconds'),
        ('filesystem-compression-v1/archive-index-layout-completion.json','offline_index_rowid_layout_storage_not_sampler_work','CPU_seconds','wall_seconds'),
        ('filesystem-compression-v1/archive-index-layout-v2-completion.json','offline_index_normalized_layout_storage_not_sampler_work','CPU_seconds','wall_seconds'),
        ('filesystem-compression-v1/archive-index-redundant-secondary-removal.json','offline_index_secondary_removal_storage_not_sampler_work','CPU_seconds','wall_seconds')])
    seen=set()
    for pattern,category,cpu,wall in families:
        for path in sorted(ROOT.glob(pattern)):
            if category=='transparent_filesystem_storage_subprocess_only_not_sampler_work'and not(
                    len(path.stem)==64 and set(path.stem)<=set('0123456789abcdef')):
                continue # Named offline-index evidence has its own nonoverlapping category.
            if category=='TN_timeout_progress_CPU_lower_bound_not_additive_to_job_allocation'and path.with_name('optimization-result.json').exists():
                continue # Normal returned optimizer already includes this work.
            if path in seen:raise RuntimeError('Overlapping canonical cost source')
            seen.add(path);v=json.loads(path.read_text())
            # Compatibility metadata points to the SAME Boys/rotation work,
            # not a second CCSD solve. Retain the pointer but do not add it.
            duplicate_work=v.get('rotation_and_localization_cost_shared_with_physical_preparation',False)
            actual_category=category
            if category=='TN_failed_or_capped_attempt_Slurm_cost_separate'and v.get('final_operator_preserved')and 'COMPLETED'in v.get('actual_Slurm_accounting',''):
                actual_category='TN_successful_terminal_evidence_NOT_additional_optimizer_work'
            rows.append({'category':actual_category,'source':str(path.relative_to(ROOT)),'source_sha256':sha256(path.read_bytes()),
                'CPU_seconds':None if duplicate_work else v.get(cpu),'wall_seconds':None if duplicate_work else v.get(wall),'status':v.get('status',v.get('exit_code')),
                'duplicate_compatibility_record_not_additive':duplicate_work,
                'same_work_canonical_source':'data/physical/NAPH20-BLOCK-BOYS/preparation-cost.json'if duplicate_work else None,
                'missing_is_unknown_not_zero':True,'add_within_category_only':True})
            rows[-1]['CPU_value_is_lower_bound']=category in('TN_timeout_progress_CPU_lower_bound_not_additive_to_job_allocation',
                'transparent_filesystem_storage_subprocess_only_not_sampler_work',
                'P1_SqDRIFT_parent_process_only_exporter_CPU_not_included','S3_pilot_parent_process_only_not_complete_end_to_end')or category.startswith('offline_index_')or category=='empty_archived_directory_metadata_storage_not_sampler_work'
            rows[-1]['missing_CPU_reason']='No whole-attempt CPU timer recorded in this immutable source'if v.get(cpu)is None and not duplicate_work else None
    return rows


def ledger_evidence():
    db=sqlite3.connect(ROOT/'common-budget.sqlite')
    by_state=[{'draw_state':state,'events':n,'known_model_observations':actual,'reserved_upper_bound':upper}
        for state,n,actual,upper in db.execute('SELECT state,COUNT(*),COALESCE(SUM(actual),0),SUM(COALESCE(actual,reserved)) FROM draws GROUP BY state')]
    db.close()
    # Analytic HF counts describe a delta distribution; keep nominal counts
    # out of any claim about stochastic sampling performed on a simulator.
    hf=sum(v['budget_ledger']['known_actual_shots']for p in ROOT.glob('classical-controls-v1/*/HF/*/result.json')
        for v in[json.loads(p.read_text())])
    return {'shared_ledger_including_HF_nominal_counts':by_state,'analytic_HF_nominal_counts':hf,
        'actual_QPU_shots':0,'P2_P3_anchors_not_all_in_this_shared_ledger':True,
        'readout_rows_are_not_additional_quantum_observations':True}


def paper_rows():
    fe_done=all((ROOT/'paper-anchors/P2-Fe-remote-v1'/k/'collected.json').exists()for k in('naive','compressed'))
    hardware=optional(ROOT/'hardware-postprocessing-v2/status.json')or{}
    tn=optional(ROOT/'TN-controls-v1/status.json')or{}
    rows=[{'source':'P1 SqDRIFT v3','requirement':'Original integrals/grouping/author driver/Rustiq0.0.8',
        'status':'NOT_EXACTLY_REPRODUCED','evidence':'Documented geometry/AVAS reconstruction; current SDK grouping and compiler, no original driver identity',
        'claim':'Source-connected reconstruction/variant, not published numerical reproduction'},
        {'source':'P1 Fig6','requirement':'LUCJ1/7/22; SqDRIFT5/20/50; 500 circuits/k,512 samples/member',
        'status':'SETTINGS_BRIDGE_TERMINAL'if(ROOT/'paper-anchors/P1-bridge-v1/terminal.json').exists()else'RUNNING_SETTINGS_BRIDGE',
        'evidence':'paper-anchors/P1-bridge-v1/','claim':'Registered current-runtime points; original grouping and solver not certified'},
        {'source':'P2 improved LUCJ','requirement':'Author amplitudes, naive/compressed,100k,10x4000 batches',
        'status':'N2_CURRENT_RUNTIME_EXECUTED_FE_TERMINAL'if fe_done else'N2_CURRENT_RUNTIME_EXECUTED_FE_IN_PROGRESS','evidence':'paper-anchors/P2/ and P2-Fe-remote-v1/',
        'claim':'Current PySCF instead of original Dice; no curve-match certification'},
        {'source':'P3 original SQD','requirement':'Public distributions, physical integrals, CR/no-CR',
        'status':'CURRENT_REANALYSIS_TERMINAL_CHECK_PER_CASE_STATUS'if hardware.get('terminal_cases')==3 else'PUBLIC_WEIGHTS_AND_PHYSICAL_INPUTS_RESOLVED_REANALYSIS_RUNNING','evidence':'hardware-physical-v1/ and hardware-postprocessing-v2/',
        'claim':'Source values are floating weights, not reconstructed raw shot counts. Original SDK0.1 not reproduced in this runtime; current0.14, not a global unavailability claim'},
        {'source':'P3 hardware TableS1','requirement':'Published gate/depth/shot anchors','status':'SOURCE_QUOTATION_ONLY',
        'evidence':'N2 32/36q:762CNOT,148 totaldepth,100k; Fe40/45q:1100CNOT,173 totaldepth,2457600',
        'claim':'No published CNOT depth in this table; no percentage versus our all-to-all compiler'},
        {'source':'P2-inspired TN','requirement':'Strong optimized LUCJ control','status':('PARTIAL_LIMIT_'+','.join(tn['attention_cases']))if tn.get('attention_cases')else('INDEPENDENT_FINAL_READOUTS_COMPLETE'if tn.get('completed_case_final_readouts')==4 else'BOUNDED_TN_COMMON_READOUT_VARIANT_RUNNING'),
        'evidence':'TN-controls-v1/ and TN_SOURCE_CROSSWALK_JA.md','claim':'CurrentMPS/PSD-MADS commonr32 objective,3subproblems instead of pinned-code20 and no Dice; not exact paper TN'}]
    return rows


def compiler_summary(readouts):
    groups=defaultdict(list)
    for r in readouts:
        if r['lane']=='C_PRIMARY'and r['method_id']=='D_SQDRIFT_N20'and int(r['r_cap'])==32 and r.get('Rustiq_cnot_count'):
            groups[(r['case_id'],int(r['budget']))].append(r)
    result=[]
    for(case,B),rows in sorted(groups.items()):
        record={'case_id':case,'budget':B,'observed_seeds':len(rows),'planned_seeds':10,
            'same_counts_same_energy':True,'new_model_shots':0,'original_P1_Rustiq0p0p8_reproduced':False,
            'mean_energy_Ha':float(np.mean([float(r['energy_Ha'])for r in rows]))}
        for basic,alternative in(('CNOT','Rustiq_cnot_count'),('CNOT_depth','Rustiq_cnot_depth'),('total_depth','Rustiq_total_unitary_depth')):
            record['mean_basic_'+basic]=float(np.mean([float(r[basic])for r in rows]))
            record['mean_Rustiq_'+basic]=float(np.mean([float(r[alternative])for r in rows]))
        result.append(record)
    return result


def run():
    out=ROOT/'analysis-v1';out.mkdir(exist_ok=True)
    if not(out/'all-observed-readouts.csv').exists():return
    with(out/'all-observed-readouts.csv').open()as f:readouts=list(csv.DictReader(f))
    individuals,contrasts=paired(readouts);write_csv(out/'all-paired-run-block-differences.csv',individuals)
    write_csv(out/'all-paired-descriptive-statistics.csv',contrasts)
    costs=cost_evidence();write_csv(out/'canonical-cost-evidence-no-overlap-within-category.csv',costs)
    preprocess=preprocessing_views();write_csv(out/'LUCJ-shared-versus-standalone-preprocessing.csv',preprocess)
    ledger=ledger_evidence();save_json(out/'shared-shot-ledger-summary.json',ledger)
    papers=paper_rows();write_csv(out/'paper-source-claim-boundaries.csv',papers)
    compiler=compiler_summary(readouts);write_csv(out/'same_SqDRIFT_counts_compiler_sensitivity.csv',compiler)
    classical=[]
    for p in ROOT.glob('classical-controls-v1/*/HCI-epsilon-*/result.json'):
        v=json.loads(p.read_text());last=v['history'][-1]if v.get('history')else{}
        classical.append({'case_id':v['case_id'],'setting':p.parent.name,'status':v['status'],
            'energy_Ha':v.get('final_energy_Ha'),'last_dimension':last.get('dimension',last.get('n_determinants')),
            'CPU_seconds':v.get('CPU_seconds'),'is_optimized_SHCI_PT2':False,'source':str(p.relative_to(ROOT))})
    write_csv(out/'bounded-classical-HCI-controls.csv',classical)
    coverage=optional(out/'coverage.json')or{}
    with(out/'all-group-statistics.csv').open()as f:groups=list(csv.DictReader(f))
    stage_note=('登録した数値実行は終了しました。原論文の未再現部分・上限停止は保持し、研究全体の最終判断は上位のFINAL_REPORT_JA.mdを参照してください。'
        if coverage.get('open_scope')==['Final full-scope cost/quality/resource synthesis and audit']else
        '実験は継続中です。未完了セルを失敗や0として埋めず、以下は現在保存されている全runの記述です。')
    lines=['# 研究判断用まとめ：同じ測定予算とCI容量で何が違うか','',
        stage_note,'',
        '## まず結論の読み方','',
        '回路が短くてもenergyが高ければ、精度まで含めた勝利ではありません。逆に、同じShot予算・CI上限で低いenergyなら、その条件では良い配置を集められたと読めます。',
        'RAACSは構築中にも測定予算を使います。LUCJは主に古典CCSD/CISDと初期化、SqDRIFTは多数の回路生成に費用がかかります。Shotが同じだから計算全体の費用も同じ、とは言いません。','',
        '## 主比較：同じ条件の全seed平均','',
        '| 物理条件 | 総model Shot | 方法 | seed数 | 相関energy回収率 % | CNOT平均 | CNOT Depth平均 | Total Depth平均 | 係数スロット平均 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['lane']!='C_PRIMARY'or int(g['r_cap'])!=32:continue
        def number(key):return f"{float(g[key]):.2f}"if g.get(key)else'不明'
        lines.append('| '+ ' | '.join([g['case_id'],g['budget'],NAMES.get(g['method_id'],g['method_id']),g['observed_runs'],
            number('mean_correlation_recovery_pct'),number('mean_CNOT'),number('mean_CNOT_depth'),
            number('mean_total_depth'),number('mean_coefficient_slots_or_member_mean')])+' |')
    lines +=['','この表はr=32、CI実次元d≤1024の比較です。異なるrの全値、各seedの値、SD・中央値はCSVにあります。係数スロットは数学的に独立なparameter数の証明ではありません。',
        '回収率の参照はケースごとに明示しています。著者報告の近似SCI/DMRG値を独立に厳密解と認証したものではありません。',
        'CI容量を増やすと平均順位が逆転する条件があります。[分子別解説のr32/r64比較](CASE_INTERPRETATION_JA.md#ciを大きくすると何が変わるか)も必ず参照してください。小さいCIでの見かけの優位を、同精度での一般的な資源優位とはしません。','',
        '## SqDRIFTを不利なcompilerだけで評価しない','',
        '同じ回路順・同じ測定結果を保ったRustiq合成でも資源を再計数しました。energyは変わりません。以下は同じseed群についての比較です。元の論文版Rustiq0.0.8そのものではありません。','',
        '| 条件 | Shot | seed数 | 基本合成CNOT | Rustiq CNOT | 基本CNOT Depth | Rustiq CNOT Depth |','|---|---:|---:|---:|---:|---:|---:|']
    lines +=[f"| {c['case_id']} | {c['budget']} | {c['observed_seeds']} | {c['mean_basic_CNOT']:.1f} | {c['mean_Rustiq_CNOT']:.1f} | {c['mean_basic_CNOT_depth']:.1f} | {c['mean_Rustiq_CNOT_depth']:.1f} |"for c in compiler]
    lines +=['','RAACSの資源上の利点を判断するときは、この改善後SqDRIFTと、通常/改善初期化・多層・TNのLUCJ対照も見る必要があります。基本合成の弱い点だけを相手にしません。','',
        '## 論文との接続：再現できた部分と残る差','',
        '| 出典 | 状態 | 大切な制約 |','|---|---|---|']
    lines +=['| '+r['source']+' | '+r['status']+' | '+r['claim']+' |'for r in papers]
    lines +=['','公開P3のN₂ファイルは、repository READMEでは平衡距離1.0975135Åの倍数です。積分からのHF計算でも確認しました。主比較の1.2/2.4Åと同じ点だと見なしていません。FeもP2/P3のHamiltonianを区別しています。','',
        '## 古典対照の限界','',
        'HF対照と、同じ積分に対するthreshold-HCIの12点も保存しています。ただしHCIは512次元・120秒のvariational段階だけで、optimized SHCIやPT2を含む強い古典手法ではありません。',
        '全threshold候補を加える前に容量上限へ達し、最後に保存できた値がHFのままという点もあります。これを「古典計算に勝った」根拠にはしません。各点のstatusとenergyはbounded-classical-HCI-controls.csvにあります。','',
        '## 費用は何を数えたか','',
        '| 種類 | 扱い |','|---|---|',
        '| model Shot | 共有ledgerの実消費と未確定予約を区別。R違いのCIを測定数に足さない |',
        '| 実QPU Shot | 今回の新規計算は0。理想simulationと公開hardware再解析を分離 |',
        '| HF対照 | 既知の1配置からの解析的分布。名目Shotを実stochastic samplingと呼ばない |',
        '| CCSD/CISD・改善初期化 | 同じ計算を共有した場合1回。独立に運用する場合の必要費用とは分ける |',
        '| CPU時間 | categoryごとの正規出典を保存。worker CPUと同じworker内のCI時間を二重加算しない |',
        '| GPU時間 | 実Slurm区間と割当数を記録。kernel時間と割当時間を別々に扱う |',
        '| TN対照 | classical surrogate最適化1回と独立final readout5回。5個の独立optimizerではない |',
        '| 過去sourceの作成費用 | 不明ならUNKNOWN。ダウンロード時間を元CCSD計算費用にしない |','',
        '費用の正規出典はcanonical-cost-evidence-no-overlap-within-category.csv、Shotの全体像はshared-shot-ledger-summary.jsonです。category間でcoverageが重なる時間を無条件に足した総CPU秒は出していません。','',
        'P1のSqDRIFT検算とS3 pilotのprocess計時も別categoryへ保存します。これは親processの既知CPUで、別processのexporter・起動費用すべてを含むend-to-end値ではありません。P2のN₂検算には元の全attempt CPU計時がないため、未知を0とせず、記録された再開postprocessingのwall時間も別欄に残します。これらの補助作業を主480セルの費用へ付け替えません。','',
        '### 共有した初期化を無料に見せない','',
        'LUCJ-shared-versus-standalone-preprocessing.csvでは、通常/改善初期化、各層数、局在軌道の各設定について、cacheなしでこの登録pipelineを使う場合の初期化費用を出しています。各rowは反実仮想の単独運用で、実消費に足す追加計算ではありません。',
        '例えば共通CCSD/CISDは実験では1回だけ作り、複数設定が共有しています。単独運用の表では必要な上流費用を各設定に表示しますが、それらを全部足して実験総額にはしません。Boysの共変変換も、同じ作業をHamiltonian準備と振幅準備へ二重計上しません。',
        'この表はsampling、CI、TN調整、process起動時間を含む総CPU時間ではありません。RAACS/SqDRIFTがこのCCSD初期化を必要としないことと、候補生成や回路合成が無料であることも別です。','',
        '## まだ主張しないこと','',
        '全手法・全条件での優越、原論文曲線の完全再現、実機での高速化、未知分子への一般化は主張しません。主比較の完了と、原runtimeの再現不可・追加対照のresource capは別の状態です。',
        '全セルと欠測理由、容量感度、強い対照を揃えてから、48qubit以上へ進む価値を判断します。','',
        '詳しい元データ：all-observed-readouts.csv / all-circuit-resource-views.csv / all-planned-cells.csv。全paired差分も残してあり、勝ったseedだけを選んでいません。']
    atomic_bytes(out/'DECISION_REPORT_JA.md',('\n'.join(lines)+'\n').encode())
    save_json(out/'synthesis-status.json',{'observed_at_unix':time(),'paired_individual_rows':len(individuals),
        'paired_group_rows':len(contrasts),'canonical_cost_sources':len(costs),'coverage':coverage,'whole_experiment_complete':False})
    print(json.dumps({'paired_rows':len(individuals),'cost_sources':len(costs),'whole_experiment_complete':False}),flush=True)


if __name__=='__main__':run()
