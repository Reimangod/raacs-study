"""All paired OO results, including spin, cost and non-quantum controls."""
from io import StringIO
import csv
from collections import defaultdict
import sqlite3
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256


def main():
    spec=load(ROOT/'protocol-C2-OO.json');rows=[];group=defaultdict(dict)
    names={'RAACS_B3':'残差beam RAACS','LUCJ_COMPRESSED_L1':'Compressed LUCJ L1','CLASSICAL_UNIFORM_SECTOR':'古典uniform配置（弱い診断対照）'}
    for case in spec['cases']:
        ref=load(OLD/'data/evaluation-only'/(case+'.json'))['energies_Ha']
        hf=ref['hf_energy'];reference=ref['fci_energy']if ref.get('fci_energy')is not None else ref['sci_energy']
        for method in spec['methods']:
            path=ROOT/'runs/C2-OO'/case/method/'result.json'
            r=load(path);source=r['source'];res=source.get('sampler_resources',{})
            if any(b['energy_Ha']>a['energy_Ha']+1e-8 for a,b in zip(r['history'],r['history'][1:])):
                raise RuntimeError('Nonmonotone OO history')
            row={'case_id':case,'method':method,'n_readout_sources':1,'r_cap':64,
                'actual_CI_dimension':r['original_counts_context']['actual_d'],
                'energy_before_Ha':r['initial_energy_Ha'],'energy_after_Ha':r['final_energy_Ha'],
                'correlation_recovery_after_pct':100*(hf-r['final_energy_Ha'])/(hf-reference),
                'spin_square_after':r['history'][-1]['spin_square'],'CI_residual_norm_after':r['history'][-1]['residual_norm'],
                'sampler_CNOT':res.get('cnot_count'),'sampler_CNOT_Depth':res.get('cnot_depth'),
                'sampler_Total_Depth':res.get('total_depth',res.get('total_unitary_depth')),
                'sampler_coefficient_slots':source.get('sampler_coefficient_slots'),
                'classical_orbital_coordinates':r['classical_orbital_coordinates'],'extra_quantum_model_shots':0,
                'source_total_quantum_model_shots':source.get('source_total_budget',0),
                'source_final_quantum_model_shots':source.get('source_final_model_shots',0),
                'OO_CPU_seconds':r['CPU_seconds'],'optimizer_nfev':sum(x.get('optimizer_nfev',0)for x in r['history']),
                'optimizer_steps':sum(x.get('optimizer_nit',0)for x in r['history']),
                'all_optimizer_stages_converged':all(x.get('optimizer_success',True)for x in r['history']),
                'source_result':str(path),'source_SHA256':sha256(path.read_bytes()),'optimized_CI_state_prepared_by_sampler':False}
            rows.append(row);group[case][method]=row
    out=StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/paired-OO-all-data.csv',out.getvalue().encode('utf-8-sig'))
    contrasts=[];lines=['# 別方向の比較：samplerと軌道最適化を組み合わせる','',
        '**Fe₂S₂の1試行では、RAACS＋軌道最適化がLUCJ＋同じ軌道最適化より低いCI energyになりました。他の3条件ではまだLUCJが上です。全体の目標は未達で、研究は継続します。**','',
        '全条件でr=64、同じ3外側反復・各L-BFGS-B最大20反復/40function evaluationsです。既存countsを再利用したため追加model Shotは0。これは新しい量子測定ではなく、明示的な古典変分処理です。初期のsampler作成時には探索pilot込み100,000 Shotを使っています。','',
        '**重要：表のCNOT/Depthはサンプリング回路の値です。軌道最適化した最終CI波動関数を、この小回路が準備したという意味ではありません。** 古典軌道変数とCI係数は別に数えます。','']
    for case in spec['cases']:
        data=group[case];a=data['RAACS_B3'];b=data['LUCJ_COMPRESSED_L1'];delta=1000*(a['energy_after_Ha']-b['energy_after_Ha'])
        contrasts.append({'case_id':case,'RAACS_minus_LUCJ_after_OO_mHa':delta,'independent_replicates':1,
            'singlet_certified':False,'formal_superiority_proved':False})
        lines+=['## '+case,'','| 手法 | OO前 energy Ha | OO後 energy Ha | 回収率 % | S² | 実CI次元 |',
            '|---|---:|---:|---:|---:|---:|']
        for method in spec['methods']:
            r=data[method];lines.append(f"| {names[method]} | {r['energy_before_Ha']:.8f} | {r['energy_after_Ha']:.8f} | {r['correlation_recovery_after_pct']:.2f} | {r['spin_square_after']:.5f} | {r['actual_CI_dimension']} |")
        lines+=['',f'RAACS−LUCJの差は **{delta:+.3f} mHa**（負ならRAACSが低いenergy）。'+
            ('この条件では有望な開発上の比較点が得られました。ただし1試行であり、原著最強対照や統計的優越の認定ではありません。'if delta<0 else'この条件ではまだLUCJ側が低いenergyです。'),'',
            '| sampler | CNOT | CNOT Depth | Total Depth | sampler係数スロット平均 | 追加の古典軌道変数 |','|---|---:|---:|---:|---:|---:|']
        for method in('RAACS_B3','LUCJ_COMPRESSED_L1'):
            r=data[method];lines.append('| '+' | '.join([names[method],*[f"{r[k]:.2f}"for k in('sampler_CNOT','sampler_CNOT_Depth','sampler_Total_Depth','sampler_coefficient_slots')],str(r['classical_orbital_coordinates'])])+' |')
        lines+=['','古典uniformは量子回路を使いません。その0 CNOTを、同じground stateを量子的に準備できたという成果にはしません。これは弱い診断対照で、強い古典SCIを代用しません。','',
            '| 手法 | 追加OO CPU秒 | optimizer評価数 | optimizer反復数 | 全stageのoptimizer収束 |','|---|---:|---:|---:|---|']
        for method in spec['methods']:
            r=data[method];lines.append(f"| {names[method]} | {r['OO_CPU_seconds']:.3f} | {r['optimizer_nfev']} | {r['optimizer_steps']} | {'はい'if r['all_optimizer_stages_converged']else'未達stageあり'} |")
        lines+=['','未収束stageがあっても、保存された変分状態と再計算energyが有効なら結果として保持します。収束未達を成功へ書き換えず、同時に有効な中間結果も捨てません。元sampler構築、LUCJの初期化、sampling simulation等は別費用です。この表のOO CPUだけをend-to-end費用とはしません。','']
    lines+=['## 発見した科学的な問題と次の検証','',
        'N₂ 2.4 ÅのRAACS＋OOはS²≈2.72で、singlet純度を満たしていません。Feでも両法のS²は厳密な0ではありません。Nα=Nβであることだけでsingletと扱ってはいけません。今回の数値比較は同じM_S=0空間のものですが、原著のsinglet-targeted比較として確定するには、全手法共通のspin-targeted readoutを追加し、物理Hamiltonianのenergyとpenalty energyを分けて再確認する必要があります。','',
        '次はspinの扱いをそろえた再解析、Feの追加独立sampler/readout、強いTN対照と古典SCIの比較です。Feだけを後から選んで全4条件成功に変更しません。軌道最適化という既存技術自体を新規性にせず、小回路samplerと組み合わせたときの精度・費用の関係を調べます。','',
        '別案の単純な古典readout tail除去は、小系検算では正しかったものの、LUCJ L1の4条件で削減0でした。この方向へ同じ試験を増やさず、限界を記録して優先度を下げました。','',
        '[公式OO-SQD手順](https://qiskit.github.io/qiskit-addon-sqd/guides/optimize_orbitals.html)に基づき、PySCF RDMとffsim energyの一致、直交軌道変換、変分単調性を確認しています。小系の独立FCI検算は合成fixtureのみで、分子のFCI係数を探索に使っていません。']
    write(ROOT/'analysis/ORBITAL_DIRECTION_JA.md',('\n'.join(lines)+'\n').encode())
    save(ROOT/'analysis/OO-contrasts.json',contrasts)
    # No uncertain or indirect evidence is promoted to completion.
    with sqlite3.connect('file:'+str(ROOT/'production-shots.sqlite')+'?mode=ro',uri=True)as db:
        known,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL)FROM draws').fetchone()
    save(ROOT/'CURRENT_GOAL_AUDIT.json',{'goal_achieved':False,'status':'ACTIVE_RESEARCH',
        'known_new_molecular_model_shots':known,'unknown_or_inflight_draws':unknown,
        'completed_new_sampler_runs':28,'completed_new_Fe_control_readouts':3,'completed_OO_cells':12,
        'new_quantum_shots_for_OO':0,'four_case_target':'NOT_MET',
        'strongest_source_aligned_TN_and_SqDRIFT_comparison':'INCOMPLETE',
        'fixed_policy_independent_confirmation':'NOT_DONE','strong_classical_SCI_control':'INCOMPLETE',
        'spin_target_certification':'INCOMPLETE','alternative_directions_experimentally_checked':2,
        'next_action':'Common spin-targeted readout with physical energy/penalty separated; then independent repeats and broader direction choice',
        'old_scientific_outcomes_changed':False})
    print({'OO_cells':12,'contrasts':contrasts,'new_model_shots_this_goal':known,'goal_achieved':False},flush=True)


if __name__=='__main__':main()
