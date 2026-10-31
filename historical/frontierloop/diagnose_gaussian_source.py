"""N22 retrospective support diagnostic; no new energy, draws, or policy tuning."""
import statistics
from pathlib import Path
import ffsim
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .gaussian_source import LOCAL as SAMPLES,probability
from paperstudy.lucj import make_circuit


def statistics_for_counts(path,n,nelec):
    with np.load(path,allow_pickle=False) as z:ids,counts=z['blocked_ids'],z['counts']
    hf=(1<<nelec)-1;mask=(1<<n)-1
    rank=np.array([2*nelec-(int(x)&mask&hf).bit_count()-((int(x)>>n)&hf).bit_count() for x in ids])
    if int(counts.sum())!=100000:raise RuntimeError('Same100k reference draws required')
    return dict(HF_observed=int(counts[rank==0].sum()),rank_at_most2_observed=int(counts[rank<=2].sum()),
        rank_at_most4_observed=int(counts[rank<=4].sum()),payload_SHA256=sha256(Path(path).read_bytes()))


def run():
    summary=load(ROOT/'analysis/N22-gaussian-CI-summary.json');sample=load(SAMPLES/'terminal.json')
    if summary['status']!='N22_ALL16_TASKS28_READOUTS_AND_ALL12_SOURCES_AUDITED':raise RuntimeError('Audit every readout first')
    parent_root=ROOT/'runs/N17-INDEPENDENT-PREFIX-PERFORMANCE-v1';parent=load(parent_root/'protocol.json')
    rows=[];structure=[]
    for source in sample['sources']:
        case,n,nelec=source['case'],source['norb'],source['nelec'][0]
        registered=source['problem_source']
        with np.load(registered['operator'],allow_pickle=False) as z:
            op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        q=ffsim.qiskit.PRE_INIT.run(make_circuit(op,n,tuple(source['nelec'])))
        first=q.data[0].operation
        a,b=first.orbital_rotation_a.copy(),first.orbital_rotation_b.copy()
        names=[]
        for instruction in q.data[1:]:
            g=instruction.operation;names.append(g.name)
            if g.name=='diag_coulomb_jw':continue
            if g.name!='orb_rot_jw':raise RuntimeError('Unrecognized source order')
            a=g.orbital_rotation_a@a;b=g.orbital_rotation_b@b
        error=float(max(np.max(abs(a-op.final_orbital_rotation)),np.max(abs(b-op.final_orbital_rotation))))
        if error>1e-10:raise RuntimeError('Actual Gaussian backbone does not telescope as expected')
        hf=(1<<nelec)-1;x=hf|(hf<<n);occupied=tuple(first.occupied_orbitals)
        structure.append(dict(case=case,actual_suffix_gate_order=names,
            initial_Gaussian_prefix_HF_probability=probability(x,n,first.orbital_rotation_a,first.orbital_rotation_b,occupied),
            all_J_removed_Gaussian_backbone_HF_probability=probability(x,n,a,b,occupied),
            Gaussian_product_vs_actual_final_rotation_max_error=error,
            backbone_not_numerically_evaluated_as_CI_or_sampling_source=True))
        groups=[('GAUSSIAN_PREFIX',[Path(d['payload']) for d in sample['draws'] if d['case']==case]),
            ('FULL_LUCJ_L1',[parent_root/'decoded'/(c['draw_id']+'.npz') for c in parent['cells'] if c['case']==case and c['source']=='C5_C_LUCJ_COMPRESSED_L1' and c['method']=='ORIGINAL'])]
        for label,files in groups:
            if len(files)!=3:raise RuntimeError('All3draws required')
            ss=[statistics_for_counts(path,n,nelec) for path in files]
            rows.append(dict(case=case,source=label,draws=ss,
                HF_frequency_pct=statistics.mean(x['HF_observed'] for x in ss)/1000,
                excitation_rank_at_most2_frequency_pct=statistics.mean(x['rank_at_most2_observed'] for x in ss)/1000,
                excitation_rank_at_most4_frequency_pct=statistics.mean(x['rank_at_most4_observed'] for x in ss)/1000))
    parity=[]
    prior=load(ROOT/'analysis/N18-half-source-native-summary.json')
    for r in summary['rows']:
        if r['method']!='UNRESTRICTED_HF_HCI':continue
        before=next(x for x in prior['rows'] if x['case']==r['case'] and x['method']=='UNRESTRICTED_HF_HCI')
        error=abs(r['energy_Ha']-before['energy_Ha'])
        if error>1e-8:raise RuntimeError('Fresh strong classical control parity failed')
        parity.append(dict(case=r['case'],absolute_energy_difference_Ha=error))
    out=dict(status='N22_ALL4_SUPPORT_AND_ACTUAL_UNITARY_ORDER_DIAGNOSED',rows=rows,source_structure=structure,
        fresh_unrestricted_HF_control_parities=parity,new_CI_solves=0,new_model_QPU_shots=0,
        decision='STOP_PREFIX_ONLY_NO_SHOT_OR_SEED_EXPANSION',
        Gaussian_family_falsified=False,causal_unique_mechanism_proven=False,goal_complete=False,
        summary_SHA256=sha256((ROOT/'analysis/N22-gaussian-CI-summary.json').read_bytes()))
    save(ROOT/'analysis/N22-support-and-structure-diagnostic.json',out)
    lines=['# N22の判断：大きなgate削減だけでは採用しない','',
        '全4条件で最初のGaussian prefixだけでは精度が悪化した。これは技術的失敗・solver上限ではなく、全28readoutが通過したsource品質の不成功。追加seed/Shotでこのprefix-onlyを救済しない。','',
        '| 条件 | source | HF観測 % | HFから2重励起以内 % | 4重励起以内 % |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['source']} | {r['HF_frequency_pct']:.6f} | {r['excitation_rank_at_most2_frequency_pct']:.6f} | {r['excitation_rank_at_most4_frequency_pct']:.6f} |")
    lines+=['','これらは有限観測の分布診断。HFに近いほど常に良いという一般則でも、全失敗原因を一意に証明する因果分解でもない。特に強相関系ではHF以外の配置も不可欠である。','',
        '## このablation自体の限界も認める','',
        '実際のLUCJ sourceは、最初のSlater準備→対角相互作用→軌道回転という順序である。prefix-onlyは相互作用だけでなく、後ろの基底を戻す回転も捨てた非常に強い切断。従って今回だけで「Gaussian源では不可能」「量子相関層が不可欠」とは言えない。',
        '全J相互作用を0にしつつ全軌道回転を残す別のGaussian-backbone controlでは、回転積が実際のfinal_orbital_rotationへ全4条件で1e-10以内に一致した。これは既存unitaryの代数検査であり、新sourceのsampling/CI/energy成功ではない。','',
        '| 条件 | prefix-onlyの解析的HF確率 | 全Gaussian回転を残した場合のHF確率 | 回転積の最大差 |',
        '|---|---:|---:|---:|']
    for s in structure:
        lines.append(f"| {s['case']} | {s['initial_Gaussian_prefix_HF_probability']:.4g} | {s['all_J_removed_Gaussian_backbone_HF_probability']:.6f} | {s['Gaussian_product_vs_actual_final_rotation_max_error']:.3g} |")
    lines+=['','## 次の投資方針','',
        '1. prefix-onlyは終了。69〜74%のCNOT削減だけを成果として採用しない。N22全失敗・費用とN17/N18/N20/N21の結果を保全。',
        '2. 次の小対照は「相互作用を外してもGaussianの基底変換を保持する」構造的に整合したablationと、単純なclassical support。新しいoptimizer、分子、閾値、巨大CIを先に足さない。原著LUCJ全層と強い古典対照も残す。',
        '3. ただしこれはGaussianの新規アルゴリズムではない。これも悪ければGaussianだけのsourceに更なる計算を投入せず、相関層を保ったsampling-aware objectiveに研究資源を戻す。原著TNは同objectiveの既知方式である。',
        '4. 新sourceに利益が出ても、同じ小さい実D/費用の比較とgeneric classical controlsを先に行い、basis増加や古典HCIの仕事を量子sourceの利益へすり替えない。',
        '5. 最強論文への総合優越は未達。次のproduction protocol/jobはまだない。N22の完了を全goal達成とは呼ばない。',
        '',f"今回のsampling/compile {summary['sampling_and_compile_CPU_seconds']:.3f} CPU秒、CI campaign {summary['CI_campaign_CPU_seconds']:.3f} CPU秒。新規draw120万、QPU/GPU0。これは初期化・過去の探索・監査/転送を含むcold総費用ではない。"]
    write(ROOT/'analysis/N22_DECISION_AND_NEXT_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['decision'],flush=True)


if __name__=='__main__':run()
