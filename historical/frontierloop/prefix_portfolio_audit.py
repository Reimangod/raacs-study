"""Expose all exact compiler controls, not only the CNOT-first chosen point.

This is an additive resource-only diagnostic before any candidate energy or
sampling. It preserves the frozen N15 selection and distinguishes that point
from the full opt1/opt3 resource frontier.
"""
from io import BytesIO
from time import process_time,perf_counter
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .slater_prefix_preflight import LOCAL


def counters(path):
    q=qpy.load(BytesIO(path.read_bytes()))[0]
    return dict(CNOT=int(q.count_ops().get('cx',0)),CNOT_Depth=int(q.depth(filter_function=lambda i:i.operation.name=='cx')),Total_Depth=int(q.depth()),
        numeric_primitive_coefficient_slots_not_independent_rank=sum(len(i.operation.params) for i in q))


def run():
    cpu,tick=process_time(),perf_counter();p=load(LOCAL/'protocol.json');r=load(LOCAL/'result.json');controls={};records=[]
    for source in p['sources']:
        if source['status']!='AVAILABLE':continue
        key=(source['case'],source['method']);folder=LOCAL/source['case']/source['method']/'EXACT';cs=[]
        for level in p['compiler_levels']:
            file=folder/f'opt{level}-sampling.qpy';preparation=folder/f'opt{level}-preparation.qpy'
            cs.append(dict(resources={**counters(file), 'optimization_level':level},sampling_SHA256=sha256(file.read_bytes()),preparation_SHA256=sha256(preparation.read_bytes()),
                sampling=str(file.relative_to(LOCAL)),preparation=str(preparation.relative_to(LOCAL)),preparation_resources=counters(preparation)))
        controls[key]=cs
    metrics=['CNOT','CNOT_Depth','Total_Depth'];keyfn=lambda o:tuple(o['resources'][k] for k in metrics)
    for row in r['rows']:
        if row['status']=='MISSING_SAME_PROBLEM_SOURCE':continue
        cs=controls[(row['case'],row['method'])];best=min(cs,key=keyfn)
        if row['exact_control']['resources']!=best['resources'] or row['exact_control']['sampling']['SHA256']!=best['sampling_SHA256']:raise RuntimeError('Chosen source is not actual CNOT-first portfolio point')
        minimum={k:min(c['resources'][k] for c in cs) for k in metrics}
        strict=[o for o in row['candidate_compiler_outputs'] if all(o['resources'][k]<=minimum[k] for k in metrics) and any(o['resources'][k]<minimum[k] for k in metrics)]
        selected=min(strict,key=keyfn) if strict else None
        records.append(dict(case=row['case'],method=row['method'],policy=row['policy'],tier=row['tier'],
            frozen_N15_adopted=row['adopted'],all_exact_controls=cs,componentwise_minimum_not_a_physical_circuit=minimum,
            strict_all_controls_candidate_exists=bool(selected),strict_candidate_resources=selected['resources'] if selected else None,
            strict_candidate_sampling=selected['sampling'] if selected else None,
            note='DIAGNOSTIC_ONLY_ORIGINAL_SELECTION_UNCHANGED_NO_ENERGY_OUTCOME_OBSERVED'))
    result=dict(status='N15_ALL10_EXACT_COMPILER_PORTFOLIOS_AND60_CANDIDATES_AUDITED',rows=records,
        exact_source_compiler_points=20,exact_source_QPYs_recounted=40,
        strict_all_controls_candidate_rows=sum(x['strict_all_controls_candidate_exists'] for x in records),
        frozen_N15_adopted_rows=sum(x['frozen_N15_adopted'] for x in records),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-tick,
        original_result_SHA256=sha256((LOCAL/'result.json').read_bytes()),new_CI_solves=0,new_model_shots=0,goal_complete=False)
    save(ROOT/'analysis/N15-control-portfolio-audit.json',result)
    lines=['# N15補足：CNOT-firstの対照と全compiler選択肢を分ける','',
        f"全10sourceのopt1/opt3計20点・40 QPYを再計数。元N15採用{result['frozen_N15_adopted_rows']}行のうち、全source compiler選択肢の各指標にも非増加で、少なくとも一指標がstrictに下がる候補が存在する行は{result['strict_all_controls_candidate_rows']}。新Shot/CI0。",'',
        '原N15は事前固定したCNOT-first対照との3指標guardであり、元回路の全Pareto選択肢へ同時に勝つ保証ではない。特にopt3でCNOTが少し下がる代わりにTotal Depthがopt1より長い場合がある。この差を隠して総合的に改善したとは呼ばない。','',
        '今回のcomponentwise minimumは複数対照の最良値を束ねた保守的な判定用で、一つの実在回路として表へ出さない。元N15の採否・raw artifactは変更していない。次の独立性能実験でどの規則を使うかは、energy取得前の別protocolで明示する。', '',
        '| 条件 | source | tier | policy | N15採用 | 全対照への非増加候補 |','|---|---|---:|---|---|---|']
    for x in records:
        if x['tier']==.01:lines.append(f"| {x['case']} | {x['method']} | {x['tier']} | {x['policy']} | {x['frozen_N15_adopted']} | {x['strict_all_controls_candidate_exists']} |")
    write(ROOT/'analysis/N15_CONTROL_PORTFOLIO_JA.md',('\n'.join(lines)+'\n').encode());print(result['status'],result['strict_all_controls_candidate_rows'],flush=True)


if __name__=='__main__':run()
