"""P2/P3 tables: source-specific readouts, never additional C-lane wins."""
import json
from .io import ROOT,save_json,atomic_bytes,sha256
from .analysis_products import write_csv


def optional(p):return json.loads(p.read_text())if p.exists()else None


def run():
    out=ROOT/'analysis-v1';p2=[];hardware=[]
    for case in('N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'):
        for kind in('naive','compressed'):
            if case.startswith('FE'):
                root=ROOT/'paper-anchors/P2-Fe-remote-v1'/kind;source=root/'output/result.json'
            else:
                root=ROOT/'paper-anchors/P2'/(case+'-'+kind)
                source=root/'postprocessing-v2/result.json'if(root/'postprocessing-v2/result.json').exists()else root/'result.json'
            v=optional(source)
            if not v:
                p2.append({'case_id':case,'kind':kind,'status':'NOT_YET_COLLECTED','source':str(source.relative_to(ROOT))});continue
            history=v.get('history',v.get('iterations',[]));last=history[-1]if history else{}
            dimensions=last.get('actual_batch_dimensions',last.get('batch_dimensions'))
            p2.append({'case_id':case,'kind':kind,'status':v['status'],'model_shots':100000,'new_QPU_shots':0,
                'problem_id':v.get('problem_id'),'mean_energy_Ha':v.get('mean_energy_Ha'),'minimum_batch_energy_Ha':v.get('minimum_energy_Ha'),
                'maximum_batch_energy_Ha':v.get('maximum_energy_Ha'),'batch_count':len(last.get('batch_energies_Ha',[])),
                'actual_dimensions':dimensions,'mean_batch_equals_independent_run_replicates':False,
                'original_Dice_solver_reproduced':False,'published_curve_match_verified':False,
                'source':str(source.relative_to(ROOT)),'source_sha256':sha256(source.read_bytes())})
    for case in('P3-N2-ratio1p00','P3-N2-ratio2p00','P3-FE2S2-40'):
        root=ROOT/'hardware-postprocessing-v2'/case/'output';meta=optional(ROOT/'hardware-physical-v1'/case/'physical.json')
        ref=optional(ROOT/'hardware-physical-v1'/case/'evaluation-reference.json')
        terminal=optional(root.parent/'collected.json');v=optional(root/'result.json')
        no=optional(root/'no-CR.json');progress=optional(root/'progress.json')or{}
        iterations=(v or progress).get('iterations',[])
        start=optional(root/'started.json')or{}
        Eref=ref['reference_energy_Ha'];den=meta['hf_energy_Ha']-Eref
        def row(arm,energy,dimension,status,iteration=None):
            return {'case_id':case,'problem_id':meta['problem_id'],'arm':arm,'iteration':iteration,'energy_Ha':energy,
                'actual_dimension_or_batch_dimensions':dimension,'status':status,
                'correlation_recovery_vs_reported_reference_pct':100*(meta['hf_energy_Ha']-energy)/den if energy is not None and den>0 else None,
                'reference_kind':ref['kind'],'absolute_reference_precision_certified':ref['absolute_precision_bound_certified'],
                'reference_sector_independently_verified':ref['reference_sector_independently_verified'],
                'initial_valid_sector_weight':start.get('valid_sector_weight'),'configuration_repair_fraction':None,
                'repair_fraction_reason':'Not separately instrumented; not interchangeable with correlation-energy recovery',
                'completed_full_CR_iterations':len(iterations),'planned_full_CR_iterations':10,
                'completed_CI_batches':len(list(root.glob('CI-*-*.json'))),'planned_CI_batches':100,
                'new_QPU_shots':0,'public_weights_not_raw_integer_counts':True,
                'same_problem_as_primary_C_assumed':False,'source_directory':str(root.relative_to(ROOT))}
        if no:
            ci=no.get('CI',{});hardware.append(row('CR_OFF_ALL_VALID',ci.get('energy_Ha'),ci.get('actual_d'),no['status']))
        else:hardware.append(row('CR_OFF_ALL_VALID',None,None,'NOT_YET_COLLECTED'if not terminal else'TERMINAL_WITHOUT_NO_CR_RESULT'))
        for item in iterations:
            r=row('CR_ON_BATCH_MEAN',item['mean_energy_Ha'],item['actual_d'],'FULL_ITERATION_COMPLETED',item['iteration'])
            r.update(minimum_batch_energy_Ha=item['minimum_energy_Ha'],maximum_batch_energy_Ha=item['maximum_energy_Ha'],
                CI_all_converged=item['all_CI_converged'],CI_max_residual=item['max_residual_norm'],batch_range_is_not_confidence_interval=True)
            hardware.append(r)
        if not iterations:hardware.append(row('CR_ON_BATCH_MEAN',None,None,(v or terminal or{}).get('status','NOT_YET_COLLECTED')))
    write_csv(out/'P2-all-six-source-anchor-statuses.csv',p2)
    write_csv(out/'H-public-data-CR-and-recovery.csv',hardware)
    lines=['# 原論文入力の検算と公開実機分布の再解析','',
        '主比較とは別のデータです。同じ分子名でもHamiltonianや距離が違う場合があるため、ここからRAACSとの削減率を作りません。','',
        '## P2：著者の初期振幅を使ったLUCJ','',
        '| 条件 | 初期化 | 状態 | 平均energy Ha | batch数 | 実CI次元 |','|---|---|---|---:|---:|---|']
    def number(v):return '未取得'if v is None else f'{v:.8f}'
    for r in p2:
        dims=r.get('actual_dimensions');label='未取得'if not dims else str(min(dims))+'–'+str(max(dims))
        lines.append(f"| {r['case_id']} | {r['kind']} | {r['status']} | {number(r.get('mean_energy_Ha'))} | {r.get('batch_count','未取得')} | {label} |")
    lines+=['','ここでは100k samplesを元に10batchを読む設定へ接続しています。batchがすべて同じ配置空間になる場合、同じenergyの繰り返しは再現性の10回実証ではありません。現行PySCFは元のDiceではなく、数値曲線の完全一致を認証していません。','',
        '## H：Configuration Recoveryの前後','',
        '| 元の物理条件 | CR | iteration | 状態 | energy Ha | 相関energy回収率 % | このcase全体のCR batch進捗 / 100 |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in hardware:
        pct=r['correlation_recovery_vs_reported_reference_pct'];percent='未取得'if pct is None else f'{pct:.3f}'
        lines.append(f"| {r['case_id']} | {r['arm']} | {r['iteration'] if r['iteration'] is not None else '—'} | {r['status']} | {number(r['energy_Ha'])} | {percent} | {r['completed_CI_batches']} / 100 |")
    lines+=['','右端はcase全体の全CR iterationを合計した進捗で、同じcaseの各行に繰り返し表示しています。各energy行で75個等のbatchを平均した意味ではなく、完全なCR iterationのenergyは10batch平均です。CR_OFFは別の単一CI空間の値で、このCR進捗の分母・分子へ足していません。未完了iterationの個別batchは別CSVと図へ全件残しています。',
        '相関energy回収率は、公開参照値との差をどれだけ取り戻したかです。測定bitstringを何%修復したか、とは別です。参照は丸められたFCI表や近似DMRGで、絶対精度とspin sectorを独立に認証したものではありません。',
        'CRでenergyが下がれば、この公開分布と後処理では改善があったと読めます。ただしCI次元も変わり得るため、CR効果だけを計算量不変の改善と解釈しません。CRなしの空間が上限を超えた場合は、簡単な一部分に切って成功値を作らず欠測として残します。',
        '各点は公開浮動weightsをそのまま使った現行SDK0.14の再解析です。新しいQPU実行は0で、生の整数countsや元SDK0.1の完全再現とは呼びません。10iterationの途中で時間切れなら、終了iterationと未完了batchを明示します。']
    atomic_bytes(out/'SOURCE_REANALYSIS_JA.md',('\n'.join(lines)+'\n').encode())
    save_json(out/'source-reanalysis-status.json',{'P2_planned_anchors':6,'P2_collected_results':sum(r['status']!='NOT_YET_COLLECTED'for r in p2),
        'H_rows':len(hardware),'original_paper_reproduction_certified':False,'whole_experiment_complete':False})
    print(json.dumps({'P2_rows':len(p2),'H_rows':len(hardware)}),flush=True)


if __name__=='__main__':run()
