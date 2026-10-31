"""Derived same-basis N20/N21 cost review; no numerical solver is called."""
from .paths import ROOT
from .io import load,save,write,sha256


def run():
    a=load(ROOT/'analysis/N20-nested-CI-summary.json')
    b=load(ROOT/'analysis/N21-nested-CI-summary.json')
    for label,s in [('N20',a),('N21',b)]:
        if s['status']!=label+'_ALL16_POINTS_AND_ALL_ACTUAL_SAVED_ROOTS_AUDITED':
            raise RuntimeError('All sixteen planned points must be audited')
    by={r['task_id']:r for r in a['rows']};rows=[]
    if len(a['rows'])!=16 or len(b['rows'])!=16:raise RuntimeError('All planned points required')
    for child in b['rows']:
        key=(child['case'],child['requested_R'])
        parents=[r for r in a['rows'] if (r['case'],r['requested_R'])==key]
        if len(parents)!=1:raise RuntimeError('Ambiguous physical point')
        parent=parents[0]
        if parent['task_id']!=child['task_id'] or parent['actual_CI_dimension']!=child['actual_CI_dimension']:
            raise RuntimeError('Changed source basis, not solver-only comparison')
        both=parent['energy_Ha'] is not None and child['energy_Ha'] is not None
        diff=child['energy_Ha']-parent['energy_Ha'] if both else None
        if both and abs(diff)>1e-8:raise RuntimeError('Same projected operator energy parity failed')
        rows.append(dict(case=key[0],requested_R=key[1],status_N20=parent['status'],status_N21=child['status'],
            energy_delta_Ha=diff,N20_body_CPU_seconds=parent['root_CPU_seconds'],N21_body_CPU_seconds=child['root_CPU_seconds'],
            alias=child['alias_of_same_task']))
    report=dict(status='N20_N21_SAME16_BASIS_SOLVER_REVIEW_COMPLETE',rows=rows,
        N20_child_CPU_seconds=a['CPU_seconds_children'],N21_child_CPU_seconds=b['CPU_seconds_children'],
        child_CPU_reduction_pct=100*(a['CPU_seconds_children']-b['CPU_seconds_children'])/a['CPU_seconds_children'],
        max_absolute_energy_difference_Ha=max(abs(r['energy_delta_Ha']) for r in rows if r['energy_delta_Ha'] is not None),
        new_CI_solves=0,new_model_QPU_shots=0,no_more_automatic_Fe_solver_or_cap_rescues=True,
        repeated_timing_statistical_speedup_established=False,source_or_circuit_resources_changed=False,
        best_paper_superiority=False,goal_complete=False,
        input_SHA256={label:sha256((ROOT/f'analysis/{label}-nested-CI-summary.json').read_bytes()) for label in ('N20','N21')})
    save(ROOT/'analysis/N20-N21-solver-cost-review.json',report)
    lines=['# N20/N21：CI上限と数値解法を分けて考える','',
        f"同じ16点・同じ基底・同じ式。得られたenergyの最大差は{report['max_absolute_energy_difference_Ha']:.3g} Ha。",
        f"全子CPUは{a['CPU_seconds_children']:.3f}→{b['CPU_seconds_children']:.3f}秒（今回の実測で{report['child_CPU_reduction_pct']:.1f}%減）。単回の数値比較で普遍的speedupの証明ではない。",'',
        '両試験とも新root8、同じrootの再表示5、Fe R256の時間上限1、後続未実行2。Feを成功させるためだけに時間上限・solver救済を再追加しない。失敗も費用も保持する。',
        '大きいCIの利益は回路圧縮の利益ではない。同sourceの強いHCI対照より低いenergyはNaph1.521、近N₂0.635、遠N₂0.055 mHa。弱いfrequency対照だけへの27.9mHa改善を主要成果として強調しない。',
        'CI拡大・数値実装の単独延長より、次はsourceそのものを変える安い対照を試す。N22は全4条件のGaussian prefix-only ablation。同じsource初期化費用を保持し、既知のclassical DPPと明記する。']
    write(ROOT/'analysis/N20_N21_COST_AND_DECISION_JA.md',('\n'.join(lines)+'\n').encode())
    print(report['status'],report['child_CPU_reduction_pct'],flush=True)


if __name__=='__main__':run()
