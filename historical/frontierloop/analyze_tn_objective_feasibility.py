"""Audit all N19 actual MPS counts and original batch dimensions, no energy."""
from collections import Counter
import numpy as np
from .paths import ROOT
from .io import load, save, write, sha256
from .tn_objective_feasibility import LOCAL
from .tn_objective_feasibility_worker import upstream_subsample, batch_spaces


def run():
    spec=load(LOCAL/'protocol.json'); collected=LOCAL/'collected'
    started=load(collected/'started.json'); terminal=load(collected/'terminal.json')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Actual N19 protocol or threading changed')
    for name,h in spec['input_SHA256'].items():
        if sha256((LOCAL/name).read_bytes())!=h: raise RuntimeError('Frozen N19 executable/input changed')
    qualification=load(collected/'qualification/result.json')
    if qualification['status']!='TINY_MPS_FULL_FOCK_AND_AUTHOR_BATCH_PARITY_PASSED' or qualification['maximum_state_error']>1e-10:
        raise RuntimeError('Actual independent qualification missing')
    if qualification['reference_SHA256']!=spec['reference_SHA256'] or sha256((LOCAL/'sqd-subsampling.py').read_bytes())!=spec['reference_SHA256']:
        raise RuntimeError('Unmodified author subsample identity changed')
    states={s['case']:s for s in terminal['states']}
    if len(states)!=4 or set(states)!={s['case'] for s in spec['cases']}:
        raise RuntimeError('All four actual terminal states required')
    sample=upstream_subsample(LOCAL/'sqd-subsampling.py'); rows=[]
    for entry in spec['cases']:
        case=entry['case']; row=states[case]; folder=collected/'cells'/case
        if row['status']!='COMPLETED':
            if load(folder/'failure.json')!=row: raise RuntimeError('Failure record missing')
            rows.append(row);continue
        if load(folder/'result.json')!=row or row['problem_id']!=entry['problem_id'] or row['operator_SHA256']!=entry['operator_SHA256']:
            raise RuntimeError('Actual source/problem binding differs')
        if row['MPS_max_bond']!=50 or row['MPS_cutoff']!=1e-10 or row['CI_energy_evaluations']!=0 or row['actual_classical_surrogate_samples']!=10000:
            raise RuntimeError('Unregistered molecular evaluation or MPS settings')
        if sha256((folder/'counts.npz').read_bytes())!=row['counts_SHA256'] or sha256((folder/'batch-spaces.npz').read_bytes())!=row['spaces_SHA256']:
            raise RuntimeError('Actual counts or batch arrays changed')
        with np.load(folder/'counts.npz',allow_pickle=False) as a:
            ids=a['blocked_ids']; counts=a['counts']
            if int(counts.sum())!=10000 or len(ids)!=row['unique_full_configurations']:
                raise RuntimeError('Actual surrogate sample count mismatch')
            spaces, arrays, rejected=batch_spaces(ids,counts,entry['norb'],tuple(entry['nelec']),sample,spec['batch_seed'])
        if spaces!=row['batch_spaces'] or rejected!=row['wrong_sector_samples']:
            raise RuntimeError('Original subspace selection was not reproduced')
        with np.load(folder/'batch-spaces.npz',allow_pickle=False) as a:
            if set(a.files)!=set(arrays) or any(not np.array_equal(a[k],v) for k,v in arrays.items()):
                raise RuntimeError('Saved actual CI spaces differ')
        progress=load(folder/'samples-prefix.json')
        if progress!={'generated':10000,'complete':True,'actual_QPU_shots':0}:
            raise RuntimeError('Actual terminal sampling prefix does not match')
        result=dict(row)
        result['joint_dimension_min']=min(s['joint_dimension'] for s in spaces)
        result['joint_dimension_max']=max(s['joint_dimension'] for s in spaces)
        result['max_dimension_over_current_D16384']=result['joint_dimension_max']/16384
        result['maximum_single_complex_vector_MiB']=max(s['coefficient_vector_complex128_bytes'] for s in spaces)/2**20
        rows.append(result)
    out=dict(status='N19_ALL4_ACTUAL_ATTEMPTS_MPS_COST_AND_CI_SIZE_AUDITED',rows=rows,
        state_counts=dict(Counter(r['status'] for r in rows)),
        known_completed_surrogate_samples=sum(r.get('actual_classical_surrogate_samples',0) for r in rows),
        failed_or_partial_sample_counts_not_imputed_zero=True,
        CPU_seconds_children=terminal['CPU_seconds_children'],qualification_CPU_seconds=qualification['CPU_seconds'],
        wall_seconds_including_qualification=terminal['wall_seconds_including_qualification'],
        new_CI_energy_evaluations=0,new_exact_model_shots=0,actual_QPU_shots=0,GPU_allocations=0,
        full_paper_reproduction=False,independent_quantum_confirmation=False,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((collected/'terminal.json').read_bytes()))
    save(ROOT/'analysis/N19-tn-objective-feasibility-summary.json',out)
    lines=['# N19：原著型MPS samplingの実費とCI空間サイズ','',
           '元L1 operatorを変更せず、全4条件でMPS χ50/cutoff1e-10の近似samplingを各10,000回試した。',
           'これは最適化・精度比較ではない。新しいCI energy・QPU Shot・exact model Shot・GPU割当は0。', '',
           f"全状態：{out['state_counts']}。完了したsurrogate samplesは{out['known_completed_surrogate_samples']}。失敗・途中数は0へ補完しない。",
           f"子process合計CPU {out['CPU_seconds_children']:.3f}秒、別途tiny qualification {out['qualification_CPU_seconds']:.3f}CPU秒。全wall {out['wall_seconds_including_qualification']:.3f}秒。",'',
           '| 条件 | 状態 | MPS CPU秒 | build wall秒 | sample wall秒 | wrong-sector/10k | unique full配置 | 原著型joint D範囲 | 現D上限との最大比 | peak RSS MiB |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def f(x,n=3): return '未取得' if x is None else f'{x:.{n}f}'
    for r in rows:
        span=f"{r['joint_dimension_min']}〜{r['joint_dimension_max']}" if r['status']=='COMPLETED' else '未取得'
        lines.append('| '+' | '.join([r['case'],r['status'],f(r.get('CPU_seconds')),f(r.get('MPS_build_wall_seconds')),f(r.get('sampling_wall_seconds')),
            f(r.get('wrong_sector_samples'),0),f(r.get('unique_full_configurations'),0),span,f(r.get('max_dimension_over_current_D16384')),
            f(r['peak_RSS_KiB']/1024) if r.get('peak_RSS_KiB') is not None else '未取得'])+' |')
    lines+=['','## 解釈上の注意','',
            '- 4000は片spinの上限であり、joint4000ではない。10batchesの空間は重複し得る。sampleを量子Shotとして数えたり、10倍の独立物理配置と呼ばない。',
            '- 今回はCI diagonalizationを行っていない。大きい空間ほどenergyが低くなる保証は、異なるsample集合にはそのまま適用できない。次は同じsource countsで空間を入れ子にした費用付き評価が必要。',
            '- 4qubit exact parityはコード経路の検査であり、分子MPSのexact fidelityやQSCI精度保証ではない。normが1に近くても正しい分布とは限らない。sector逸脱は除外前に明示した。',
            '- 原著のsource初期値・全optimizer・native Dice等を再現したものではなく、その一部分の同じ物理問題へのfeasibility試験。原著最高点や最強論文への優越は未達。',
            '- 原著batch sampling関数はSQD 0.12.0の原文を使用した。現在のruntimeにSQD/Dice addon全体をインストールした意味ではない。',
            '- 原著source構築、過去のsampling、テスト・転送・回収・解析は上記子process CPUに含めていない。CPU上の小検証であり、A100を割り当てなかったことはGPU最適化失敗ではない。']
    write(ROOT/'analysis/N19_TN_FEASIBILITY_RESULTS_JA.md', ('\n'.join(lines)+'\n').encode())
    print(out['status'],out['state_counts'],flush=True)


if __name__=='__main__':run()
