"""All16 D2 cells, all spin definitions, costs and unfinishedness are visible."""
import csv
from io import StringIO
import sqlite3
from .paths import ROOT
from .io import load,save,write,sha256
from .compression_trial import DIRECTORY,guard,validate_row
from .analyze import reference


def main():
    spec=guard();rows=[];statuses=[];draws=0;reads=0
    for case in spec['cases']:
        hf,ref=reference(case)
        for point in spec['points']:
            request=validate_row(case,point);d=DIRECTORY/case/point;q=request['resources']['Z_sampling_equivalent']
            draw=load(d/'draw-result.json') if (d/'draw-result.json').exists() else None
            if draw:
                assert sha256((d/'final-counts.npz').read_bytes())==draw['counts_SHA256']
                draws+=1
            for mode in spec['readouts']:
                path=d/mode/'result.json';r=load(path) if path.exists() else None
                status='COMPLETED' if r else 'FAILED_ENGINEERING_PRESERVED' if (d/mode/'failure.json').exists() else 'NOT_COMPLETED'
                statuses.append({'case':case,'point':point,'readout':mode,'draw_completed':bool(draw),'status':status})
                if r:
                    assert r['source_counts_SHA256']==draw['counts_SHA256'];reads+=1
                source_compile=None
                if point!='EXACT':
                    # The manifest's frozen source-result hash identifies the
                    # measured compiler cost; do not charge a future ideal fit.
                    matches=list((ROOT/'runs/D1-MERGED-RESOURCE'/case).glob('*/result.json'))
                    bound=[p for p in matches if sha256(p.read_bytes())==request['source_D1_result_SHA256']]
                    if len(bound)!=1:raise RuntimeError('D1 compiler-cost provenance missing')
                    source_compile=load(bound[0])['CPU_seconds']
                rows.append({'case':case,'point':point,'readout':mode,'status':status,
                    'energy_Ha':r['final_energy_Ha'] if r else None,
                    'recovery_pct':100*(hf-r['final_energy_Ha'])/(hf-ref) if r else None,
                    'spin_square':r['history'][-1]['spin_square'] if r else None,
                    'CNOT':q['cnot_count'],'CNOT_depth':q['cnot_depth'],'total_depth':q['total_unitary_depth'],
                    'numeric_primitive_slots_not_independent_parameters':request['numeric_primitive_coefficient_slots_not_rank'],
                    'independent_parameter_rank':None,'model_shots':draw['actual_model_shots'] if draw else None,
                    'actual_QPU_shots':0,'r_cap':spec['r_cap'],
                    'actual_CI_dimension':r['actual_CI_dimension'] if r else None,
                    'singlet_dimension':(r.get('spin_basis') or {}).get('singlet_dimension') if r else None,
                    'observed_D1_fit_and_compilation_CPU_seconds':source_compile,
                    'D2_input_preparation_CPU_seconds':request['input_preparation_CPU_seconds'],
                    'simulation_CPU_seconds':draw['whole_body_CPU_seconds'] if draw else None,
                    'simulation_wall_seconds':draw['whole_body_wall_seconds'] if draw else None,
                    'GPU_used':draw['GPU_used'] if draw else None,
                    'OO_CPU_seconds':r['CPU_seconds'] if r else None,
                    'source_initialization_record':request['source_initialization_provenance'],
                    'readout_result':str(path) if r else None,'readout_SHA256':sha256(path.read_bytes()) if r else None})
    def csvout(name,data):
        out=StringIO();w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        write(ROOT/'analysis'/name,out.getvalue().encode('utf-8-sig'))
    csvout('D2-all-data.csv',rows);csvout('D2-all-statuses.csv',statuses)
    with sqlite3.connect(ROOT/'production-shots.sqlite') as db:
        known,upper,unknown=db.execute('SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE actual END),0),SUM(actual IS NULL) FROM draws').fetchone()
    snapshot={'completed_draws':draws,'planned_draws':16,'completed_readouts':reads,'planned_readouts':48,
        'known_global_production_model_shots':known,'charged_global_upper_bound_model_shots':upper,
        'unknown_or_inflight_global_draws':unknown or 0,'D2_known_model_shots':draws*spec['shots_per_cell'],
        'D2_all_draws_and_readouts_complete':draws==16 and reads==48,'goal_complete':False}
    save(ROOT/'analysis/D2-summary.json',snapshot)
    lines=['# LUCJ圧縮D2：同じ測定予算で精度も保てるか','',
        f"固定した16 samplerセル中{draws}件が測定済み、48 readout中{reads}件が完了。全4条件が終わる前の最良点選択や総合優越の主張はしない。",'',
        '各samplerは総100,000 model Shot、r上限64。3種類の層上限を全て報告し、grid全体は最大160万Shot。最良層を選ぶための探索が無料という意味ではない。現在の実QPU Shotは0。', '',
        'CNOT/depthはsamplerの共通論理資源。Recoveryは参照依存の相関energy回収率で、energyと独立の成功指標ではない。Parametersの独立rankは未認証で、数値ゲート係数の個数を代わりに「独立parameter削減」と呼ばない。','']
    for case in spec['cases']:
        lines+=['## '+case,'',
            '| 設定 | CNOT | CNOT Depth | Total Depth | C2 energy Ha | C3 energy Ha | C4 energy Ha |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for point in spec['points']:
            rs=[r for r in rows if r['case']==case and r['point']==point];q=rs[0]
            es=[next(r for r in rs if r['readout']==mode)['energy_Ha'] for mode in spec['readouts']]
            lines.append(f"| {point} | {q['CNOT']} | {q['CNOT_depth']} | {q['total_depth']} | "+' | '.join('未完了' if e is None else f'{e:.8f}' for e in es)+' |')
        baseline=next(r for r in rows if r['case']==case and r['point']=='EXACT' and r['readout']=='C3-SPIN-OO')
        if baseline['energy_Ha'] is not None:
            lines+=['','C3の同じ出発回路・精度評価に対する差（正なら圧縮でenergy悪化）：']
            for r in rows:
                if r['case']==case and r['point']!='EXACT' and r['readout']=='C3-SPIN-OO' and r['energy_Ha'] is not None:
                    lines.append(f"- {r['point']}: ΔE = {1000*(r['energy_Ha']-baseline['energy_Ha']):+.6f} mHa、CNOT {100*(1-r['CNOT']/baseline['CNOT']):.2f}%削減。")
        lines.append('')
    lines+=['## いま分かったことと判断の限界','',
        'ナフタレンでは弱い圧縮F75の精度悪化が小さいが、無圧縮より低いenergyではなく、統計的な同等も未証明。F25/F50は明確な精度悪化を伴う。回路だけ小さい点を成功にしない。層を増やせばenergyが単調に良くなるとも限らない。','',
        'C2は無罰則、C3は有限spin罰則、C4は選択空間に含まれる厳密S=0。勝つ定義だけを採用しない。最終CI/OO状態はsamplerが準備した状態ではない。古典CISD配置選択と強い量子対照の結果はB4_C6_DECISION_JA.mdとC5_CONTROLS_JA.mdに残す。','',
        'native compression自体、SQD向け圧縮、半qubit SQDは既存研究。これらとの差や最強原著条件への優越は未証明。D2は固定sourceの開発screenで独立な確認試験ではない。','',
        '## 費用と待機','',
        f"今回の開発namespaceのproduction ledgerにある既知model Shot {known:,}、予約込み上限 {upper:,}、未確定/処理待ちdraw {unknown or 0}。これは過去の全project・外部sourceのShotを含む総計ではない。C5等で再利用した旧sourceの費用はsource provenanceへ別に残す。予約を実測Shotとして足さず、pending GPU仕事を失敗や0-shot完了にも変更しない。",'',
        'CSVには実際のD1 fit/合成、D2準備、simulation、OO費用を別に記載。初期化の元記録も保持し、未知を0にしない。scopeが重なる値を機械的に足さない。GPU計算のCPU process timeだけをGPU費用と呼ばず、Slurm allocationとwall timeも保存する。','',
        'AICの他チャットjobを停止しない。QOSMaxGRESPerUserでpendingなら同じjobを観測し、同じdrawをCPUへ重複投入しない。研究rootは小さいcountsと回路のみ保存し、大きいstatevectorは保存しない。','',
        '[D1全資源点](D1_MERGED_RESOURCE_JA.md)、[既知論文との境界](LITERATURE_BOUNDARY_UPDATE_JA.md)。']
    write(ROOT/'analysis/D2_COMPRESSION_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(snapshot,flush=True)


if __name__=='__main__':main()
