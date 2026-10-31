"""Joint interpretation of the policy screen and stronger classical control."""
import csv
from io import StringIO
from .paths import ROOT
from .io import load,save,write,sha256
from .analyze import reference


def output_csv(path,rows):
    if not rows:return
    out=StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(path,out.getvalue().encode('utf-8-sig'))


def main():
    cases=load(ROOT/'protocol-B4.json')['cases'];rows=[]
    for case in cases:
        hf,ref=reference(case);p=ROOT/'runs/B4'/case/'74001/result.json';r=load(p)
        old=load(ROOT/'runs/C4-SINGLET-OO'/case/'RAACS_B3/result.json')
        q=r['sampling_resources'];oo=r['final_singlet_OO']
        rows.append({'case':case,'seed':74001,'energy_Ha':oo['final_energy_Ha'],
            'old_B3_seed':73001,'old_B3_energy_Ha':old['final_energy_Ha'],
            'B4_minus_B3_mHa':1000*(oo['final_energy_Ha']-old['final_energy_Ha']),
            'recovery_pct':100*(hf-oo['final_energy_Ha'])/(hf-ref),'total_model_shots':r['budget'],
            'final_model_shots':r['final_model_shots'],'r_cap':64,
            'actual_CI_dimension':r['final_CI'][-1]['actual_d'],'singlet_dimension':oo['spin_basis']['singlet_dimension'],
            'CNOT_mean':q['cnot_count'],'CNOT_depth_mean':q['cnot_depth'],'total_depth_mean':q['total_depth'],
            'coefficient_slots_mean':r['coefficient_slots_shot_weighted_mean'],
            'construction_CPU_seconds':r['construction_CPU_seconds'],'OO_CPU_seconds':oo['CPU_seconds'],
            'basis_cache_hits':r['spin_score_same_basis_cache_hits'],'completed_depth':r['completed_depth'],
            'stop':r['construction_stop_reason'],'different_seeds_not_a_causal_ablation':True,
            'source_result':str(p),'source_SHA256':sha256(p.read_bytes())})
    output_csv(ROOT/'analysis/B4-all-results.csv',rows)
    spec=load(ROOT/'protocol-C6-CLASSICAL-CISD.json');sources=[];classical=[];contrasts=[]
    for case in cases:
        hf,ref=reference(case);p=ROOT/'runs/C6-CLASSICAL-CISD'/case/'source/result.json'
        if not p.exists():continue
        s=load(p);sources.append({'case':case,'CISD_source_energy_Ha':s['energy_Ha'],
            'fresh_CISD_CPU_seconds':s['CPU_seconds'],'fresh_CISD_whole_body_CPU_seconds':s['whole_body_CPU_seconds'],
            'CISD_nonzero_configurations':s['sparse_nonzero_configurations'],
            'CISD_coefficient_slots':s['CISD_coefficient_slots'],'residual_norm':s['residual_norm'],
            'classical_draws_generated_once':s['classical_draws'],'quantum_model_shots':0,
            'not_the_matched_R64_endpoint':True})
        for variant in spec['variants']:
            for mode in spec['readouts']:
                p=ROOT/'runs/C6-CLASSICAL-CISD'/case/variant/mode/'result.json'
                if not p.exists():continue
                r=load(p);classical.append({'case':case,'variant':variant,'readout':mode,
                    'energy_Ha':r['final_energy_Ha'],'recovery_pct':100*(hf-r['final_energy_Ha'])/(hf-ref),
                    'spin_square':r['history'][-1]['spin_square'],'r_cap':64,
                    'actual_CI_dimension':len(r['CI_strings'])**2,
                    'singlet_dimension':(r.get('spin_basis') or {}).get('singlet_dimension'),
                    'classical_draws_used_for_space_selection':r['classical_draws'],
                    'quantum_model_shots':0,'quantum_sampler_required':False,
                    'OO_CPU_seconds':r['CPU_seconds'],'classical_orbital_coordinates':r['classical_orbital_coordinates'],
                    'backend':'QUALIFIED_PARTICLE_HOLE' if mode=='C4-SINGLET-OO' else 'ORIGINAL_ELECTRON',
                    'source_result':str(p),'source_SHA256':sha256(p.read_bytes())})
                if variant=='DETERMINISTIC_CISD_MARGINAL':
                    a=load(ROOT/'runs'/mode/case/'RAACS_B3/result.json')
                    contrasts.append({'case':case,'readout':mode,'RAACS_B3_energy_Ha':a['final_energy_Ha'],
                        'classical_CISD_marginal_energy_Ha':r['final_energy_Ha'],
                        'RAACS_minus_classical_mHa':1000*(a['final_energy_Ha']-r['final_energy_Ha']),
                        'R64_cap_matched':True,'actual_dimensions_differ':len(r['CI_strings'])**2!=len(a['CI_strings'])**2,
                        'classical_no_quantum_sampler_not_a_zero_gate_correlated_state':True,
                        'formal_superiority':False})
    output_csv(ROOT/'analysis/C6-classical-source-costs.csv',sources)
    output_csv(ROOT/'analysis/C6-all-readouts.csv',classical)
    save(ROOT/'analysis/C6-contrasts.json',contrasts)
    lines=['# 新方策と古典CISD対照から分かったこと','',
        'B4は全4条件・各100k model Shotで実行した。C6は物理積分から直接CISDを計算し、その係数から配置を選んだ古典対照で、量子回路を使っていない。どちらも開発データで、独立な確認試験ではない。','',
        '## S=0評価へ変更した方策は改善したか','',
        '| 条件 | 旧B3＋同じS=0/OO energy Ha | 新B4 energy Ha | 新−旧 mHa | CNOT | CNOT Depth | Total Depth |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['old_B3_energy_Ha']:.8f} | {r['energy_Ha']:.8f} | {r['B4_minus_B3_mHa']:+.3f} | {r['CNOT_mean']:.2f} | {r['CNOT_depth_mean']:.2f} | {r['total_depth_mean']:.2f} |")
    lines+=['','今回のscreenでは、評価をS=0へそろえただけでは全体改善を得られなかった。伸長N₂はわずかに改善したが、他の3条件は悪化した。seedも異なるため、これだけでspin目的の因果効果や一般的な無効性を断定しない。ただし、この結果のままB4を大規模展開したり、勝つseedが出るまで追加したりしない。','',
        '## 直接CISDの費用と元の波動関数','',
        '| 条件 | CISD source energy Ha | CISD CPU秒 | 非零配置数 | 量子Shot |',
        '|---|---:|---:|---:|---:|']
    for s in sources:
        lines.append(f"| {s['case']} | {s['CISD_source_energy_Ha']:.8f} | {s['fresh_CISD_CPU_seconds']:.4f} | {s['CISD_nonzero_configurations']} | 0 |")
    lines+=['','CPU欄は共通の物理積分を受け取った後のCISD計算自体。プロセス起動、積分生成、後のOOまで含む総時間ではない。全process時間はcontrollerの記録に分けて残す。上のenergyは全CISD空間の値であり、次のr64の比較と混ぜない。一方、実用的な総費用を議論するとき、この安価な古典計算を無視もしない。','',
        '## 同じr64上限とOOで比較した結果','',
        '| 条件 | 読出し | RAACS B3 energy Ha | 古典CISD配置選択 energy Ha | RAACS−古典 mHa |',
        '|---|---|---:|---:|---:|']
    for r in contrasts:
        lines.append(f"| {r['case']} | {r['readout']} | {r['RAACS_B3_energy_Ha']:.8f} | {r['classical_CISD_marginal_energy_Ha']:.8f} | {r['RAACS_minus_classical_mHa']:+.3f} |")
    lines+=['','負ならRAACSのenergyが低い。ここでは事前指定したdeterministic marginalを主たる古典対照として使い、最も良かった古典drawを後から選ばない。100kのclassical drawを使う別variantも全てCSVへ保存している。各caseはCISD source一つで、2variant×3読出しを6個の独立分子や独立optimizer成功とは数えない。','',
        '同じr64でも実CI次元は異なる。C6はCISD係数の計算費用も払い、その情報から選択した対照である。CISD sourceの空間自体はr64に固定していないので、同じ情報量や同じ全中間メモリーを使った対照とは呼ばない。','',
        '## 研究の判断','',
        '量子回路が短いことだけで全体が実用的とは言えない。古典対照が同程度または低いenergyを出す条件では、RAACSの量子samplingに付加価値を示せていない。厳密S=0だけで勝った場合も、無罰則・有限罰則の結果と全CISD sourceの費用を隠さない。','',
        '次はHFからのbeamを微調整し続けるのではなく、強いLUCJを出発点にsamplingの品質を保った回路圧縮を検討する。これも同じ古典対照と全4条件で比較し、near-equilibriumだけの勝利や古典計算を量子0 CNOTの成果へ読み替えない。','',
        '現在のr64結果だけでは原著の大きなCI空間・最強TN・実機資源まで超えたとは言えない。goalは未達。新規性・量子優位・投稿可能性も、この数値だけから認定しない。']
    write(ROOT/'analysis/B4_C6_DECISION_JA.md',('\n'.join(lines)+'\n').encode())
    print({'B4_runs':len(rows),'classical_sources':len(sources),'classical_readouts':len(classical),'goal_complete':False})


if __name__=='__main__':main()
