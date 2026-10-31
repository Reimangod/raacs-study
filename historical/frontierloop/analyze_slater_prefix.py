"""N15 all-row audit and typed selected-output decoder/fidelity view."""
import csv
from collections import Counter
from io import BytesIO,StringIO
from pathlib import Path
from time import process_time,perf_counter
import ffsim
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .slater_prefix_preflight import LOCAL
from .slater_prefix_compression import parse_prefix
from .slater_prefix_rounding import trajectory,choose
from paperstudy.lucj import make_circuit


def selected_decoder(row,nqubits):
    # Raw candidate map is not a decoder for an EXACT fallback.
    return row['decoder_logical_to_physical'] if row['adopted'] else list(range(nqubits))


def run():
    start,cpu=perf_counter(),process_time();spec=load(LOCAL/'protocol.json');result=load(LOCAL/'result.json');started=load(LOCAL/'started.json')
    protocol_sha=sha256((LOCAL/'protocol.json').read_bytes())
    if result['protocol_SHA256']!=protocol_sha or started['protocol_SHA256']!=protocol_sha:raise RuntimeError('Executed protocol mismatch')
    for name,value in spec['code_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Frozen code changed')
    if len(result['rows'])!=72 or len(result['sources'])!=10:raise RuntimeError('All72 rows/10 source executions required')
    seen=set();recount={};traces={};rows=[]
    def check_output(output):
        for role in ['preparation','sampling']:
            record=output[role];path=LOCAL/record['path']
            if LOCAL not in path.resolve().parents or sha256(path.read_bytes())!=record['SHA256']:raise RuntimeError('Actual saved QPY identity mismatch')
            if record['path'] not in recount:
                q=qpy.load(BytesIO(path.read_bytes()))[0]
                if any(i.operation.name not in spec['basis']+['global_phase'] for i in q):raise RuntimeError('Unregistered resource primitive')
                recount[record['path']]=dict(CNOT=int(q.count_ops().get('cx',0)),CNOT_Depth=int(q.depth(filter_function=lambda i:i.operation.name=='cx')),
                    Total_Depth=int(q.depth()),numeric_primitive_coefficient_slots_not_independent_rank=sum(len(i.operation.params) for i in q))
            if role=='sampling' and any(recount[record['path']][k]!=v for k,v in output['resources'].items() if k!='optimization_level'):raise RuntimeError('Direct QPY counters differ')
    for source in spec['sources']:
        case=source['case'];method=source['method'];matches=[r for r in result['rows'] if r['case']==case and r['method']==method]
        if len(matches)!=6 or {(r['policy'],r['tier']) for r in matches}!={(p,t) for p in spec['policies'] for t in spec['infidelity_tiers']}:raise RuntimeError('All source policy/tiers required')
        if source['status']=='AVAILABLE':
            if sha256(Path(source['operator']).read_bytes())!=source['operator_SHA256']:raise RuntimeError('Source operator changed')
            with np.load(source['operator'],allow_pickle=False) as a:op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
            q=ffsim.qiskit.PRE_INIT.run(make_circuit(op,source['norb'],tuple(source['nelec'])))
            prefix=parse_prefix(q.data[0].operation.definition,source['norb'])
            for policy in spec['policies']:traces[(case,method,policy)]=trajectory(prefix,policy,spec['max_polynomial_infidelity'])
        for raw in matches:
            key=(case,method,raw['policy'],raw['tier'])
            if key in seen:raise RuntimeError('Duplicate planned row')
            seen.add(key)
            if source['status']!='AVAILABLE':
                if raw['status']!='MISSING_SAME_PROBLEM_SOURCE':raise RuntimeError('Missing source fabricated')
                rows.append(dict(case=case,method=method,policy=raw['policy'],tier=raw['tier'],status=raw['status']));continue
            expected=choose(traces[(case,method,raw['policy'])],raw['tier'])
            if expected['actions']!=raw['actions'] or abs(expected['infidelity']-raw['certified_original_input_many_body_infidelity'])>1e-12:raise RuntimeError('State policy not reproducible')
            if expected['infidelity']>raw['tier']+1e-12:raise RuntimeError('Registered fidelity tier exceeded')
            for output in [raw['exact_control'],raw['selected'],*raw['candidate_compiler_outputs']]:check_output(output)
            if raw['adopted']:
                if raw['selected'] not in raw['candidate_compiler_outputs'] or not raw['actions']:raise RuntimeError('Unbound candidate selection')
                if any(raw['selected']['resources'][k]>raw['exact_control']['resources'][k] for k in ['CNOT','CNOT_Depth','Total_Depth']):raise RuntimeError('Resource safeguard failed')
            elif raw['selected']!=raw['exact_control']:raise RuntimeError('Fallback is not exact source')
            decoder=selected_decoder(raw,2*source['norb'])
            if sorted(decoder)!=list(range(2*source['norb'])):raise RuntimeError('Selected decoder malformed')
            resources=raw['selected']['resources'];baseline=raw['exact_control']['resources']
            rows.append(dict(case=case,method=method,policy=raw['policy'],tier=raw['tier'],status=raw['status'],adopted=raw['adopted'],
                selected_decoder_logical_to_physical=decoder,selected_preparation=raw['selected']['preparation'],selected_sampling=raw['selected']['sampling'],
                selected_preparation_infidelity=expected['infidelity'] if raw['adopted'] else 0.,
                selected_Z_measurement_TV_upper_bound=float(np.sqrt(expected['infidelity'])) if raw['adopted'] else 0.,
                CNOT_before=baseline['CNOT'],CNOT=resources['CNOT'],CNOT_reduction_pct=100*(baseline['CNOT']-resources['CNOT'])/baseline['CNOT'],
                CNOT_Depth_before=baseline['CNOT_Depth'],CNOT_Depth=resources['CNOT_Depth'],Total_Depth_before=baseline['Total_Depth'],Total_Depth=resources['Total_Depth'],
                numeric_primitive_coefficient_slots_before=baseline['numeric_primitive_coefficient_slots_not_independent_rank'],
                numeric_primitive_coefficient_slots=resources['numeric_primitive_coefficient_slots_not_independent_rank'],independent_parameter_rank=None))
    contrasts=[]
    for source in spec['sources']:
        if source['status']!='AVAILABLE':continue
        for tier in spec['infidelity_tiers']:
            a,b=[next(r for r in rows if r['case']==source['case'] and r['method']==source['method'] and r['tier']==tier and r['policy']==p) for p in spec['policies']]
            delta=a['CNOT']-b['CNOT'];contrasts.append(dict(case=source['case'],method=source['method'],tier=tier,state_minus_angle_CNOT=delta,
                status='STATE_LOWER' if delta<0 else 'STATE_HIGHER' if delta>0 else 'EQUAL_CNOT',not_independent_molecular_samples=True))
    out=dict(status='N15_ALL72_ROWS_30_POLICY_CONTRASTS_SAVED_QPY_RECOUNTS_AUDITED',rows=rows,contrasts=contrasts,
        contrast_counts=dict(Counter(c['status'] for c in contrasts)),adopted_points=sum(r.get('adopted',False) for r in rows),
        unique_saved_compiled_QPY_recounts=len(recount),processing_CPU_seconds=process_time()-cpu,processing_wall_seconds=perf_counter()-start,
        original_result_SHA256=sha256((LOCAL/'result.json').read_bytes()),analyzer_SHA256=sha256(Path(__file__).read_bytes()),
        semantic_clarifications=['Raw decoder_logical_to_physical is a CANDIDATE map; EXACT fallback uses identity.',
            'Raw adopted_output_many_body_infidelity refers to PREPARATION, not terminal-phase-stripped sampling state.',
            'All raw rows remain unchanged; typed selected decoder and sampling TV bound are in this additive view.'],
        new_model_shots=0,new_CI_solves=0,molecular_energy_preservation_verified=False,goal_complete=False)
    save(ROOT/'analysis/N15-prefix-rounding-summary.json',out)
    f=StringIO();writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N15-all72-prefix-rounding.csv',f.getvalue().encode('utf-8-sig'))
    lines=['# N15：prefix丸めの状態・資源preflight','',
        f"72予定行を監査。10 source/60数値行、2 source/12欠測行。採用{out['adopted_points']}点、保存compiled QPY {len(recount)}件を直接再計数。policy対比{out['contrast_counts']}。",'',
        f"元preflight {result['CPU_seconds']:.3f}CPU秒/{result['wall_seconds']:.3f}wall秒。派生監査のCPUはJSONへ別計上。新energy/CI/Shot/GPU0。",'',
        '| 条件 | source | fidelity階層 | CNOT：元→state順 | Depth：元→state順 | Total：元→state順 | 準備状態infidelity | state−angle CNOT |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status']=='MISSING_SAME_PROBLEM_SOURCE' or r['policy']!='STATE_OVERLAP_ROUND':continue
        c=next(c for c in contrasts if (c['case'],c['method'],c['tier'])==(r['case'],r['method'],r['tier']))
        lines.append(f"| {r['case']} | {r['method']} | {r['tier']} | {r['CNOT_before']}→{r['CNOT']} | {r['CNOT_Depth_before']}→{r['CNOT_Depth']} | {r['Total_Depth_before']}→{r['Total_Depth']} | {r['selected_preparation_infidelity']:.8g} | {c['state_minus_angle_CNOT']:+d} |")
    lines+=['','## 正確な意味','',
        '状態重なりの式は指定HF入力と共通unitary suffixに対するPREPARATIONの証明。末尾対角位相を除いたsampling回路の状態fidelityとは区別する。そのsampling分布についてはTV≤sqrt(infidelity)という上限を使えるが、CI energy保持は別に測る。',
        'raw候補のdecoderを、元回路へfallbackした出力に流用してはいけない。新typed viewはselected decoderを明示し、fallbackはidentityとした。元artifactを変更していない。',
        '全回路はall-to-all logical。同じopt1/3 portfolioからCNOT-firstで元回路を選んだ。実機routing/noise、原著TN最高性能、独立Parameters rankは未確認。数値primitive係数slotを独立parameter数へ読み替えない。',
        'generic Slater圧縮、Clifford丸め、virtual SWAPそのものは既知。今回の処理が新規理論とは主張しない。全比較はpreflightであり、次に新しい独立drawと同じCI機会で精度・費用を確認する必要がある。']
    write(ROOT/'analysis/N15_PREFIX_ROUNDING_RESULTS_JA.md',('\n'.join(lines)+'\n').encode());print(out['status'],out['contrast_counts'],flush=True)


if __name__=='__main__':run()
