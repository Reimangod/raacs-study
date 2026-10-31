"""Source-specific anchors remain separate from common-budget results."""
import json
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from .io import ROOT,save_json,atomic_bytes,sha256,digest
from .analysis_products import resource_rows,write_csv
from .research_report import reference
from .ci import solve


def P2_common_readout_crosswalk():
    """Post-start diagnostic of existing P2 counts, not a new paper anchor.

    The native large-batch result remains immutable, including its timeout.
    Small common-CI readouts disentangle source initialization from readout
    capacity, without pretending to be a cost-matched C-lane comparison.
    """
    rows=[];sidecar=ROOT/'paper-anchors/P2-common-readout-crosswalk-v1'
    for case in('N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'):
        physical=ROOT/'data/physical'/case
        meta=json.loads((physical/'problem.json').read_text())
        integral_file=physical/'integrals.npz'
        with np.load(integral_file,allow_pickle=False)as z:h1,eri=z['h1'].copy(),z['eri'].copy()
        for kind in('naive','compressed'):
            folder=(ROOT/'paper-anchors/P2-Fe-remote-v1'/kind/'output'if case.startswith('FE')else
                    ROOT/'paper-anchors/P2'/(case+'-'+kind))
            counts_file=folder/'raw-counts.npz'
            original=(folder/'postprocessing-v2/result.json'if(folder/'postprocessing-v2/result.json').exists()else folder/'result.json')
            if not counts_file.exists()or not original.exists():continue
            source=json.loads(original.read_text())
            if source.get('problem_id')is not None:
                if source['problem_id']!=meta['problem_id']:raise RuntimeError('P2 diagnostic problem mismatch')
                binding='EXPLICIT_SOURCE_PROBLEM_ID'
            else:
                # Early postprocessing successors saved the case and source
                # counts digest but not a second copy of ProblemID. Preserve
                # this weaker provenance instead of inventing a source field.
                if source.get('case_id')!=case or source.get('source_counts_sha256')!=sha256(counts_file.read_bytes()):
                    raise RuntimeError('P2 resumed source case/count binding missing')
                binding='REGISTERED_CASE_PHYSICAL_INPUT_AND_RESUMED_COUNTS_HASH_NOT_SOURCE_PROBLEM_ID'
            with np.load(counts_file,allow_pickle=False)as z:ids,counts=z['blocked_ids'].copy(),z['counts'].copy()
            if sum(map(int,counts))!=100000:raise RuntimeError('P2 source counts must remain100k')
            for r in(8,16,32,64):
                request={'case_id':case,'kind':kind,'problem_id':meta['problem_id'],'r_cap':r,
                    'source_counts_sha256':sha256(counts_file.read_bytes()),'integrals_sha256':sha256(integral_file.read_bytes()),
                    'CI_code_sha256':sha256(Path(__file__).with_name('ci.py').read_bytes()),
                    'classical_seconds_cap':60,'post_start_diagnostic_not_primary':True}
                path=sidecar/(case+'-'+kind)/('r'+str(r)+'.json');request_id=digest(request)
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['request_id']!=request_id:raise RuntimeError('P2 common-readout identity changed')
                else:
                    wall,cpu=perf_counter(),process_time()
                    try:
                        ci=solve(h1,eri,meta['core_energy'],(meta['nalpha'],meta['nbeta']),ids,counts,r,seconds=60)
                        record={'status':'COMMON_READOUT_COMPLETED','CI':ci}
                    except (TimeoutError,MemoryError)as e:
                        record={'status':'COMMON_READOUT_RESOURCE_CAPPED','CI':None,'failure':repr(e)}
                    record.update(request=request,request_id=request_id,CPU_seconds=process_time()-cpu,
                        wall_seconds=perf_counter()-wall,new_model_quantum_shots=0,source_model_quantum_shots=100000,
                        actual_QPU_shots=0,whole_run_replicates=1,original_native_result_unchanged=True)
                    save_json(path,record)
                ci=record.get('CI')or{}
                rows.append({'case_id':case,'kind':kind,'r_cap':r,'status':record['status'],
                    'source_native_status':source['status'],'energy_Ha':ci.get('energy_Ha'),'actual_d':ci.get('actual_d'),
                    'CI_converged':ci.get('converged'),'residual_norm':ci.get('residual_norm'),
                    'whole_run_replicates':1,'new_model_quantum_shots':0,'source_amplitude_creation_CPU_seconds':None,
                    'post_start_diagnostic_not_primary':True,'original_native_result_unchanged':True,
                    'physical_binding_evidence':binding,
                    'source_counts':str(counts_file.relative_to(ROOT)),'source_counts_sha256':request['source_counts_sha256'],
                    'source_result':str(original.relative_to(ROOT)),'source_result_sha256':sha256(original.read_bytes()),
                    'diagnostic_result':str(path.relative_to(ROOT))})
    write_csv(ROOT/'analysis-v1/P2-source-counts-common-CI-crosswalk.csv',rows)
    return rows


def run():
    out=ROOT/'analysis-v1';rows=[];Eref,quality=reference('NAPH20-P1')
    for source in sorted((ROOT/'paper-anchors/P1-bridge-v1').glob('*/full-Cartesian-CI.json')):
        folder=source.parent;full=json.loads(source.read_text());saved=folder/'result.json'
        if saved.exists():
            result=json.loads(saved.read_text());label=folder.name;shots=result['model_shots']
            q=result['resource_views']['Z_sampling_equivalent'];slots=result['coefficient_slots'];CI=result['final_CI']
        else:
            link=json.loads((folder/'source-result.json').read_text());result=json.loads((ROOT/link['result']).read_text())
            label=folder.name;shots=result['budget'];r=resource_rows(result)
            q={x['metric']:x['shot_weighted_mean']for x in r if x['view']=='Z_sampling_equivalent'}
            slots=next(x['shot_weighted_mean']for x in r if x['metric']=='coefficient_slots');CI=result['final_CI']
        ci=full['CI'];small=next(v for v in CI if v['r_cap']==32)
        rows.append({'lane':'P1_CURRENT_RUNTIME_SETTINGS_BRIDGE_NOT_ORIGINAL_REPRODUCTION','setting':label,'model_shots':shots,
            'whole_run_replicates':1,'r32_actual_d':small['actual_d'],'r32_energy_Ha':small['energy_Ha'],
            'full_observed_Cartesian_d':ci.get('actual_d'),'full_observed_energy_Ha':ci.get('energy_Ha'),
            'energy_lowering_from_readout_only_mHa':1000*(small['energy_Ha']-ci['energy_Ha'])if'energy_Ha'in ci else None,
            'approx_reference_error_mHa':1000*(ci['energy_Ha']-Eref)if Eref is not None and'energy_Ha'in ci else None,
            'reference_quality':quality,'CNOT_sampling_mean':q['cnot_count'],'CNOT_depth_sampling_mean':q['cnot_depth'],
            'total_depth_sampling_mean':q['total_unitary_depth'],'coefficient_slots_or_member_mean':slots,
            'full_CI_converged':ci.get('converged'),'full_CI_residual_norm':ci.get('residual_norm'),
            'no_new_shots_for_expanded_CI':True,'original_author_grouping_compiler_solver_not_certified':True,
            'source':str(source.relative_to(ROOT))})
    write_csv(out/'P1-settings-vs-readout-capacity.csv',rows)
    p2=P2_common_readout_crosswalk()
    lines=['# 論文設定に接続する検算：主比較とは分けて読む','',
        '元の著者driver・旧runtimeを完全再現した結果ではありません。公開geometryから再構成した同じナフタレンに、論文の回路長・層数・測定配分を適用した現行runtimeの結果です。','',
        '## ナフタレン：回路とCI容量を分ける','',
        '| 設定 | model Shot | CNOT平均 | CNOT Depth平均 | Total Depth平均 | 係数スロット平均 | 小CI d | 全観測からのCI d | CI拡大だけによるenergy改善 mHa | 参照との差 mHa |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def n(k):return f"{r[k]:.3f}"if isinstance(r[k],(int,float))else'不明'
        lines.append('| '+' | '.join([r['setting'],str(r['model_shots']),n('CNOT_sampling_mean'),n('CNOT_depth_sampling_mean'),
            n('total_depth_sampling_mean'),n('coefficient_slots_or_member_mean'),str(r['r32_actual_d']),str(r['full_observed_Cartesian_d']),
            n('energy_lowering_from_readout_only_mHa'),n('approx_reference_error_mHa')])+' |')
    lines+=['','大きなポイントは、同じcountsを大きなCIで読み直しただけでもenergyが改善することです。新しい回路・追加の量子測定による改善ではありません。',
        'したがって、小さい共通CIでの勝敗だけから、sampler本来の到達精度を断定できません。同じ小CIでの安い運用比較と、より大きいCIで引き出せる精度を分けて判断します。',
        'SqDRIFTはk=1/2/3それぞれ500回路、各512 samples、合計768,000の配分です。LUCJもここでは同じ総数を読みました。各settingはwhole-run1回で、1500memberを1500個の独立研究試行とは数えません。',
        '全観測Cartesian CIは元論文のsolver/後処理そのものではなく、独立したsource-connected検算です。残差・参照の数値精度もCSVに保持し、µHa単位の差を過大評価しません。',
        'CNOT等は今回の共通all-to-all論理compilerの値です。論文のhardware-native値や原著Rustiq0.0.8の値と直接引き算しません。','',
        '## P2：同じ著者入力・同じ保存countsを、違うCI容量で読む','',
        '| 条件 | 初期化 | r32のenergy Ha | r64のenergy Ha | 元の大batch検算の平均energy Ha | 元の検算状態 |',
        '|---|---|---:|---:|---:|---|']
    for case in('N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'):
        for kind in('naive','compressed'):
            values=[x for x in p2 if x['case_id']==case and x['kind']==kind]
            if not values:continue
            by_r={x['r_cap']:x for x in values};source=json.loads((ROOT/values[0]['source_result']).read_text())
            def E(v):return '未取得'if v is None else f'{v:.8f}'
            lines.append('| '+' | '.join([case,kind,E(by_r[32]['energy_Ha']),E(by_r[64]['energy_Ha']),
                E(source.get('mean_energy_Ha')),source['status']])+' |')
    lines+=['','全設定を同じrで読めば、元の大batch検算との差にCI容量がどれだけ関係するかを確認できます。小CIで良くないことと、保存countsに有用な配置がないことは同義ではありません。',
        'これは著者保存振幅を使った1回のsource-connected診断です。新規CCSD/CISDから作った主比較とは初期化も反復数も異なり、source振幅の作成費用も不明です。この表を主比較の勝敗や全費用整合の証拠にはしません。',
        '元の検算が時間切れだった場合、r32/64が完了しても元の大batch再現が成功したとは扱いません。','',
        '## 他のsource-connected検算','',
        '- P2：6設定の保存countsを同じr=8/16/32/64で読み直した診断をP2-source-counts-common-CI-crosswalk.csvへ分離。追加Shotは0。主比較開始後の診断であり、新しい確認的試験や元solverの再現ではない。',
        '- 保存source振幅の作成費用は不明。主比較の新規CCSD/CISD費用へ置換せず、この診断から総費用をそろえた優越性を主張しない。大batchの元結果が時間切れでも、その状態は変更しない。',
        '- P3：公開hardware分布は浮動weightsのまま保持し、source固有のHamiltonianでCRを実行。理想RAACSの勝敗に混ぜない。',
        '- すべての検算結果を480主セルの成功数に足さない。']
    atomic_bytes(out/'PAPER_BRIDGES_JA.md',('\n'.join(lines)+'\n').encode())
    save_json(out/'paper-bridge-report-status.json',{'P1_observed_settings':len(rows),'P2_diagnostic_common_readouts':len(p2),'original_paper_reproduction_claimed':False,
        'whole_experiment_complete':False})
    print(json.dumps({'P1_settings':len(rows),'P2_diagnostic_common_readouts':len(p2),'additional_quantum_shots_for_CI_capacity':0}),flush=True)


if __name__=='__main__':run()
