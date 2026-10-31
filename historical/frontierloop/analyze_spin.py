"""Rebuild all paired spin diagnostics from immutable result records."""
import csv
from io import StringIO
from .paths import ROOT
from .io import load,save,write,sha256
from .analyze import reference


def main():
    spec=load(ROOT/'protocol-C3-SPIN-OO.json')
    rows=[]
    for cycle in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO'):
        for case in spec['cases']:
            hf,ref=reference(case)
            for method in spec['methods']:
                path=ROOT/'runs'/cycle/case/method/'result.json'
                if not path.exists():
                    continue
                r=load(path)
                s=r['source'];rr=s.get('sampler_resources',{})
                last=r['history'][-1]
                previous=load(ROOT/'runs/C2-OO'/case/method/'request.json')
                assert previous['CI_strings']==r['CI_strings'] and previous['source']==s
                assert r['new_quantum_model_shots']==0
                if cycle=='C4-SINGLET-OO':
                    assert all(abs(x['spin_square'])<1e-8 for x in r['history'])
                rows.append({
                    'cycle':cycle,'case':case,'method':method,'n_sources':1,
                    'energy_initial_Ha':r['initial_energy_Ha'],'energy_final_Ha':r['final_energy_Ha'],
                    'correlation_recovery_pct':100*(hf-r['final_energy_Ha'])/(hf-ref),
                    'spin_square':last['spin_square'],'residual_norm':last['residual_norm'],
                    'exact_singlet_projection':last.get('exact_singlet_projection',False),
                    'CI_determinant_dimension':len(r['CI_strings'])**2,
                    'CI_singlet_dimension':r.get('spin_basis',{}).get('singlet_dimension'),
                    'r_cap':64,'total_model_shots_in_source':s.get('source_total_budget',0),
                    'final_model_shots_in_source':s.get('source_final_model_shots',0),
                    'additional_model_shots':0,'sampler_CNOT':rr.get('cnot_count'),
                    'sampler_CNOT_depth':rr.get('cnot_depth'),
                    'sampler_total_depth':rr.get('total_depth',rr.get('total_unitary_depth')),
                    'sampler_coefficient_slots_mean':s.get('sampler_coefficient_slots'),
                    'classical_orbital_coordinates':r['classical_orbital_coordinates'],
                    'additional_classical_CPU_seconds':r['CPU_seconds'],
                    'optimizer_nfev':sum(x.get('optimizer_nfev',0) for x in r['history']),
                    'all_optimizer_stages_converged':all(x.get('optimizer_success',True) for x in r['history']),
                    'source_result':str(path),'source_SHA256':sha256(path.read_bytes()),
                    'world_best_claim_authorized':False})
    out=StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/paired-spin-all-data.csv',out.getvalue().encode('utf-8-sig'))
    contrasts=[]
    for cycle in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO'):
        for case in spec['cases']:
            rs={r['method']:r for r in rows if r['cycle']==cycle and r['case']==case}
            if not all(m in rs for m in spec['methods']):
                continue
            a,b=rs['RAACS_B3'],rs['LUCJ_COMPRESSED_L1']
            contrasts.append({'cycle':cycle,'case':case,
                'RAACS_minus_LUCJ_mHa':1000*(a['energy_final_Ha']-b['energy_final_Ha']),
                'matched_source_counts_and_subspace_with_C2':True,
                'independent_new_full_optimization_replicates':0,
                'number_of_sources_per_method':1,'formal_superiority_proved':False})
    save(ROOT/'analysis/spin-contrasts.json',contrasts)
    lines=['# スピン条件をそろえた再解析：C2 / C3 / C4','',
        '目的は既存の良い結果を守ることではなく、正しい状態の比較で差が残るかを確かめることです。C2の無罰則、C3の有限spin罰則0.1 Ha、C4の選択空間内の厳密S=0部分空間を別々に保存しています。','',
        'いずれも各手法1つの保存済みsourceを使う探索的解析です。総予算100,000 model Shot（RAACSはpilot込み）、r上限64、OO外側3反復・各最大20 optimizer反復です。追加Shotは0、実QPU Shotも0。CI実次元は手法ごとに違うため別に示します。','',
        'CNOT/Depthはサンプリング回路です。最終CI状態や軌道最適化状態をこの小回路が準備したという意味ではありません。係数スロット平均は独立parameter rankの証明ではなく、ensemble全体の総数とも違います。Recoveryは参照依存の相関energy回収率です。公開N₂/Fe参照の独立再認証は未完了で、energyとの二重の独立勝利とはしません。','']
    for cycle in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO'):
        lines+=['## '+cycle,'',
            '| 条件 | 手法 | energy Ha | Recovery % | S² | CI配置数 / singlet次元 | OO CPU秒 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['cycle']==cycle:
                lines.append(f"| {r['case']} | {r['method']} | {r['energy_final_Ha']:.8f} | {r['correlation_recovery_pct']:.2f} | {r['spin_square']:.6g} | {r['CI_determinant_dimension']} / {r['CI_singlet_dimension'] if r['CI_singlet_dimension'] is not None else '未投影'} | {r['additional_classical_CPU_seconds']:.3f} |")
        lines.append('')
    lines+=['## 同じsourceでの変化の意味','',
        '| 条件 | C2 RAACS−LUCJ mHa | C3 RAACS−LUCJ mHa | C4 RAACS−LUCJ mHa |',
        '|---|---:|---:|---:|']
    for case in spec['cases']:
        values=[]
        for cycle in ('C2-OO','C3-SPIN-OO','C4-SINGLET-OO'):
            x=next((c for c in contrasts if c['case']==case and c['cycle']==cycle),None)
            values.append(f"{x['RAACS_minus_LUCJ_mHa']:+.3f}" if x else '未完了')
        lines.append('| '+case+' | '+' | '.join(values)+' |')
    lines+=['','負ならRAACS側のenergyが低い。ただし有限罰則の異なるS²を同じsingletと呼ばない。C4では新しい配置を増やさず、P S² Pの零空間にCIを制限し、全状態のS²期待値も確認します。S²が正半定値なので、選択空間に含まれる零固有ベクトルは全空間でもsingletです。','',
        '## 資源と費用','',
        '| 条件 | sampler | CNOT平均 | CNOT Depth平均 | Total Depth平均 | 係数スロット平均 | 追加古典軌道座標 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['cycle']=='C3-SPIN-OO' and r['method']!='CLASSICAL_UNIFORM_SECTOR':
            lines.append(f"| {r['case']} | {r['method']} | {r['sampler_CNOT']:.2f} | {r['sampler_CNOT_depth']:.2f} | {r['sampler_total_depth']:.2f} | {r['sampler_coefficient_slots_mean']:.2f} | {r['classical_orbital_coordinates']} |")
    lines+=['','OO CPUは追加の古典処理のみ。sampling構築・CCSD/CISD・分解・TN・simulation・CI・失敗attemptを含む総費用とはしません。全method同じOO反復上限でも実時間は違います。各optimizerの未収束stageと全評価数はCSVに残します。','',
        '## 結論の境界','',
        'C3/C4は診断であって、固定policyの新しい独立試験ではありません。弱い古典uniformだけに勝っても量子samplingの有用性は証明できません。原著の最強TN LUCJ、改善合成済みSqDRIFT-5/20/50、強い古典SCIとの同じ後処理・同予算比較と、別seed確認が必要です。4条件を後からFeだけに変更せず、最終goalはactiveのままです。','',
        '[公式SQD orbital optimization guide](https://qiskit.github.io/qiskit-addon-sqd/guides/optimize_orbitals.html)を参照。有限罰則版のphysical energyとpenalty energyを分離し、C4の厳密選択空間投影を公式手順そのものの再現と呼ばない。']
    write(ROOT/'analysis/SPIN_DIRECTION_JA.md',('\n'.join(lines)+'\n').encode())
    print({'rows':len(rows),'contrasts':len(contrasts),'new_shots':0,'goal_complete':False})


if __name__=='__main__':
    main()
