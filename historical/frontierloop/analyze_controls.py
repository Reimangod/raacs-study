"""All C5 outcomes, missing cells and descriptive comparator envelope."""
import csv
from io import StringIO
from pathlib import Path
from collections import Counter
from .paths import ROOT
from .io import load,save,write,sha256
from .analyze import reference


def main():
    spec=load(ROOT/'protocol-C5-CONTROLS.json')
    rows=[];status_rows=[]
    for case in spec['cases']:
        hf,ref=reference(case)
        for method in spec['methods']:
            for readout in spec['readouts']:
                d=ROOT/'runs/C5-CONTROLS'/case/method/readout
                p=d/'result.json';failure=d/'failure.json'
                r=load(p) if p.exists() else load(failure) if failure.exists() else None
                status=r['status'] if r else 'NOT_TERMINAL'
                status_rows.append({'case':case,'method':method,'readout':readout,'status':status,
                    'source':str(p if p.exists() else failure) if r else None,
                    'error':r.get('error') if r else None})
                if status!='COMPLETED':
                    continue
                s=r['source'];source=load(s['source_result']);rr=s['sampler_resources']
                construction=source.get('construction',{});init=construction.get('initialization_cost',{})
                row={'case':case,'method':method,'readout':readout,'n_sources':1,
                    'source_seed':s['source_seed'],'total_model_shots_in_source':s['source_total_budget'],
                    'final_model_shots_in_source':s['source_final_model_shots'],'additional_model_shots':0,
                    'r_cap':64,'CI_determinant_dimension':len(r['CI_strings'])**2,
                    'CI_singlet_dimension':r.get('spin_basis',{}).get('singlet_dimension'),
                    'energy_initial_Ha':r['initial_energy_Ha'],'energy_final_Ha':r['final_energy_Ha'],
                    'correlation_recovery_pct':100*(hf-r['final_energy_Ha'])/(hf-ref),
                    'spin_square':r['history'][-1]['spin_square'],
                    'CNOT_mean':rr['cnot_count'],'CNOT_depth_mean':rr['cnot_depth'],
                    'total_depth_mean':rr['total_unitary_depth'],
                    'coefficient_slots_mean':s['sampler_coefficient_slots_mean'],'member_count':s['member_count'],
                    'classical_orbital_coordinates':r['classical_orbital_coordinates'],
                    'new_OO_CPU_seconds':r['CPU_seconds'],'new_OO_wall_seconds':r['wall_seconds'],
                    'historical_initialization_CPU_seconds':init.get('CPU_seconds'),
                    'historical_preparation_CPU_seconds':construction.get('preparation_CPU_seconds'),
                    'historical_this_attempt_CPU_seconds':source.get('this_attempt_CPU_seconds'),
                    'historical_this_attempt_is_CI_only':source.get('local_attempt_time_is_CI_only'),
                    'historical_cost_categories_not_blindly_additive':True,
                    'all_optimizer_stages_converged':all(x.get('optimizer_success',True) for x in r['history']),
                    'source_result':str(p),'source_SHA256':sha256(p.read_bytes()),
                    **s['Rustiq_resources']}
                rows.append(row)
    root=ROOT/'analysis'
    def csv_file(name,data):
        if not data:
            return
        out=StringIO();w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        write(root/name,out.getvalue().encode('utf-8-sig'))
    csv_file('C5-all-control-readouts.csv',rows)
    csv_file('C5-all-planned-statuses.csv',status_rows)
    contrasts=[]
    for case in spec['cases']:
        for readout in spec['readouts']:
            controls=[r for r in rows if r['case']==case and r['readout']==readout]
            if not controls:
                continue
            best=min(controls,key=lambda r:(r['energy_final_Ha'],r['method']))
            a=load(ROOT/'runs'/readout/case/'RAACS_B3/result.json')
            contrasts.append({'case':case,'readout':readout,
                'RAACS_energy_Ha':a['final_energy_Ha'],'best_observed_control':best['method'],
                'best_observed_control_energy_Ha':best['energy_final_Ha'],
                'RAACS_minus_best_control_mHa':1000*(a['final_energy_Ha']-best['energy_final_Ha']),
                'controls_completed':len(controls),'controls_planned':len(spec['methods']),
                'descriptive_retrospective_envelope_not_free_online_selection':True,
                'independent_confirmation':False})
    save(root/'C5-contrasts.json',contrasts)
    statuses=dict(Counter(r['status'] for r in status_rows))
    save(root/'C5-summary.json',{'planned_readouts':len(status_rows),'statuses':statuses,
        'new_model_shots':0,'new_OO_CPU_seconds':sum(r['new_OO_CPU_seconds'] for r in rows),
        'goal_complete':False})
    lines=['# 強い対照にも同じ後処理を認めた比較','',
        '同じHamiltonianと軌道基底の100k model Shotの保存済みcountsを使用する。各methodの最小seedを結果と無関係に固定した開発screenで、各1source。総Shotには元の探索pilotを含むが、追加model Shotと実QPU Shotは0。全methodともr64、同じ3段OOを適用する。','',
        '無罰則、有限spin罰則、厳密S=0を分けて示す。後から勝つ読出しだけを選ばない。最終CI状態はsamplerが準備した状態ではなく、古典CIと軌道最適化の出力である。','',
        '## 全予定セルの状態','',f'全120セルの現在の内訳：{statuses}。',
        'ナフタレンのcanonical軌道countsをBoys軌道のcountsに置き換えない。欠測も表から消さない。','',
        '## 各条件で観測した最も低い対照energyとの比較','',
        '| 条件 | 読出し | RAACS energy Ha | 最小energyの対照 | 対照energy Ha | RAACS−対照 mHa | 比較済み対照数 |',
        '|---|---|---:|---|---:|---:|---:|']
    for c in contrasts:
        lines.append(f"| {c['case']} | {c['readout']} | {c['RAACS_energy_Ha']:.8f} | {c['best_observed_control']} | {c['best_observed_control_energy_Ha']:.8f} | {c['RAACS_minus_best_control_mHa']:+.3f} | {c['controls_completed']} / {c['controls_planned']} |")
    lines+=['','負ならRAACSのenergyが低い。これは対照集合のretrospective envelopeで、一つの運用可能な手法が無料で最良設定を選んだ結果ではない。独立確認や原著最強の再現を証明しない。','']
    for readout in spec['readouts']:
        lines+=['## '+readout+' の全対照','',
            '| 条件 | 手法 | energy Ha | Recovery % | S² | CNOT | CNOT Depth | Total Depth | 係数スロット | 実配置 / S=0次元 | 追加OO CPU秒 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['readout']!=readout:
                continue
            lines.append(f"| {r['case']} | {r['method']} | {r['energy_final_Ha']:.8f} | {r['correlation_recovery_pct']:.2f} | {r['spin_square']:.5g} | {r['CNOT_mean']:.2f} | {r['CNOT_depth_mean']:.2f} | {r['total_depth_mean']:.2f} | {r['coefficient_slots_mean']:.2f} | {r['CI_determinant_dimension']} / {r['CI_singlet_dimension'] if r['CI_singlet_dimension'] is not None else '未投影'} | {r['new_OO_CPU_seconds']:.3f} |")
        lines.append('')
    lines+=['## 読み方と次の判断','',
        'CNOT/Depthは共通論理Z-sampling資源、ensembleはShot重み付き平均。SqDRIFTの改善合成が保存されている場合のRustiq資源もCSVに残す。係数スロットは非冗長な独立parameter rankではない。Recoveryは参照に依存するenergyの換算で、独立の勝利数に足さない。','',
        'OO CPUだけをend-to-end費用としない。初期化・回路準備・旧attemptの記録値を別列へ保持し、未知を0としない。CPU時間の範囲が重なる旧項目は足し算しない。TN surrogate費用は元sourceの作成費用として残り、再解析したため無料になったわけではない。','',
        'Compressed L1への勝利が強い対照で消えた場合、その事実で次の方策を変える。次の候補はspinに整合したscoreのbeamと、強いLUCJのsampling-aware圧縮。対照を弱くする、spin条件を勝つ方へ選ぶ、分子を除外することで目標達成にしない。','',
        '[改善初期化とTN LUCJ](https://arxiv.org/abs/2511.22476v1)、[SqDRIFT v3](https://arxiv.org/abs/2508.02578v3)とのsource/runtime/CI/solverの相違は保持する。今回のTNは原著設定を全部満たした最強対照ではない。']
    write(root/'C5_CONTROLS_JA.md',('\n'.join(lines)+'\n').encode())
    print({'statuses':statuses,'new_OO_CPU_seconds':sum(r['new_OO_CPU_seconds'] for r in rows),'goal_complete':False})


if __name__=='__main__':
    main()
