"""N29 saved-root/source audit and complete-or-explicitly-partial reporting."""
from pathlib import Path
from io import BytesIO, StringIO
from time import process_time, perf_counter
from collections import Counter
import csv
import numpy as np
from qiskit import qpy
from .io import load, save, write, sha256
from .paths import ROOT, OLD
from . import finite_source_readout as readout
from . import finite_source_study as source
from .joint_ci_sparse import matrices


def audit_root(row, d, library):
    file = Path(row['root_path'])
    if sha256(file.read_bytes()) != row['root_SHA256']:
        raise RuntimeError('Saved root hash differs')
    with np.load(file, allow_pickle=False) as z:
        pairs, v = z['joint_pairs'], z['coefficients']
    expected = np.array([(a,b) for a in row['half_strings'] for b in row['half_strings']], np.uint64)
    if not np.array_equal(pairs, expected) or len(pairs) != row['actual_D'] or row['actual_D'] != row['R']**2:
        raise RuntimeError('Actual basis/dimension mismatch')
    file = Path(d['local_integrals'])
    if sha256(file.read_bytes()) != d['integrals_SHA256']:
        raise RuntimeError('Actual integral digest differs')
    with np.load(file, allow_pickle=False) as z:
        H, S, _ = matrices(z['h1'], z['eri'], tuple(d['nelec']), pairs, library, seconds=20)
    energy, s2 = float(v@(H@v)+d['core_energy']), float(v@(S@v))
    objective = energy + .1*s2
    residual = float(np.linalg.norm((H+.1*S)@v-(objective-d['core_energy'])*v))
    if abs(v@v-1)>1e-10 or residual>1e-8 or any(abs(a-row['result'][k])>1e-8 for k,a in [('energy_Ha',energy),('spin_square',s2),('penalized_energy_Ha',objective)]):
        raise RuntimeError('Independent physical/sector-root/residual check failed')
    return dict(energy_Ha=energy, spin_square=s2, penalized_energy_Ha=objective, residual=residual)


def compiler_point(point):
    f = point['circuits']['sampling']
    blob = Path(f['path']).read_bytes()
    if sha256(blob) != f['SHA256']:
        raise RuntimeError('Bound compiler QPY differs')
    q = qpy.load(BytesIO(blob))[0]
    measured = dict(CNOT=q.count_ops().get('cx',0), CNOT_Depth=q.depth(filter_function=lambda i:i.operation.name=='cx'), Total_Depth=q.depth())
    if any(measured[k] != point['resources'][k] for k in measured):
        raise RuntimeError('Actual resource recount failed')
    return measured


def run():
    cpu, wall = process_time(), perf_counter()
    p, q, sub = readout.verified()
    terminals = []
    for batch in ('M4','A100_INPUTS'):
        file = readout.LOCAL / batch / 'terminal.json'
        if file.exists():
            t = load(file)
            if t['batch'] != batch or t['protocol_SHA256'] != sha256((readout.LOCAL/'protocol.json').read_bytes()):
                raise RuntimeError('Readout terminal context differs')
            expected = [x for x in q['tasks'] if x['owner']==batch]
            if len(expected) != len(t['states']) or any(any(row[k]!=task[k] for k in task) for row,task in zip(t['states'],expected)):
                raise RuntimeError('Frozen endpoint inventory differs')
            terminals.append(t)
    if not terminals:
        raise RuntimeError('No terminal readout; do not fabricate comparison')
    original_inventory = load(ROOT/'analysis/N17-sampling-stage-inventory.json')
    resources = {}
    for s in p['cpu_sources']:
        for point in s['all_compiler_points']:
            compiler_point(point)
        main = next(x for x in s['all_compiler_points'] if x['encoding']=='FULL_SPIN_COPY' and x['resources']['optimization_level']==1)
        resources[s['case'],'PAIR_SCALE2_REAL'] = dict(main['resources'], declared_parameters=s['declared_real_pair_variables'], all_points=s['all_compiler_points'])
    for c in sub['circuits']:
        old = next(x for x in original_inventory['rows'] if x['case']==c['case'] and x['source']==c['source'] and x['method']=='ORIGINAL')
        resources[c['case'],c['source']] = dict(old['original_opt1_resources'], declared_parameters=old['declared_parameters'], opt3_alternative=old['original_opt3_resources'])
    expected_rows = readout.inventory(p,sub)
    actual_sources = {}
    for r in expected_rows:
        actual_sources[r['draw_id'] if r['available'] else r['cell_id']] = readout.audited_counts(p,sub,r)
    roots, rows, attempts = {}, [], []
    for terminal in terminals:
        attempts.extend(terminal['attempts'])
        for r in terminal['states']:
            row = {k:r.get(k) for k in ('case','method','replicate','arm','draw_id','status','R','actual_D','basis_id','root_path','root_SHA256','new_CI_attempt')}
            if r['status']=='COMPLETED':
                actual = actual_sources[r['draw_id']]
                if actual != r['source_evidence']:
                    raise RuntimeError('Source audit binding differs')
                R = actual['R32'] if r['arm']=='R32_CAP' else min(actual_sources[x['draw_id']]['R32'] for x in expected_rows if x['available'] and x['case']==r['case'] and x['replicate']==r['replicate'])
                if R!=r['R'] or sorted(actual['rank'][:R])!=r['half_strings']:
                    raise RuntimeError('Frozen frequency selection differs')
                d = p['physical'][r['case']]
                pairs = [(a,b) for a in r['half_strings'] for b in r['half_strings']]
                if readout.basis_key(d,pairs,q['library_SHA256']) != r['basis_id']:
                    raise RuntimeError('Basis context identity differs')
                if r['basis_id'] not in roots:
                    roots[r['basis_id']] = audit_root(r,d,p['library'])
                row.update(roots[r['basis_id']], HF_half_reference_added=actual['HF_half_reference_added'],
                    HF_full_configuration_observed=actual['HF_full_configuration_observed'])
                from .analyze import reference
                hf, ref = reference(r['case'])
                meta = load(OLD/'data/evaluation-only'/(r['case']+'.json'))
                if meta['problem_id']!=d['problem_id']:
                    raise RuntimeError('Evaluation reference problem differs')
                row.update(recovery_pct=100*(hf-row['energy_Ha'])/(hf-ref),
                    reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
                    reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED')
                resource = resources.get((r['case'],r['method']))
                row.update({k:resource[k] if resource else None for k in ('CNOT','CNOT_Depth','Total_Depth','declared_parameters')})
            else:
                row['energy_Ha']=None
            rows.append(row)
    if len(roots)!=sum(a['status']=='COMPLETED' for a in attempts) or sum(r['new_CI_attempt'] is True for r in rows)!=len(attempts):
        raise RuntimeError('Actual CI attempts/aliases differ')
    groups=[]
    for case,method,arm in sorted({(r['case'],r['method'],r['arm']) for r in rows}):
        allrows=[r for r in rows if (r['case'],r['method'],r['arm'])==(case,method,arm)]
        valid=[r for r in allrows if r['status']=='COMPLETED']
        groups.append(dict(case=case,method=method,arm=arm,planned=len(allrows),completed=len(valid),
            energy_Ha_mean=float(np.mean([r['energy_Ha'] for r in valid])) if valid else None,
            energy_SD_mHa=float(1000*np.std([r['energy_Ha'] for r in valid],ddof=1)) if len(valid)>1 else None,
            recovery_pct_mean=float(np.mean([r['recovery_pct'] for r in valid])) if valid else None,
            actual_D_values=[r['actual_D'] for r in valid], independent_MC_draws=len(valid),
            distinct_CI_basis_count=len({r['basis_id'] for r in valid}), statistical_superiority_certified=False))
    comparisons=[]
    for case in p['physical']:
        for arm in readout.ARMS:
            pair=next((g for g in groups if (g['case'],g['method'],g['arm'])==(case,'PAIR_SCALE2_REAL',arm)),None)
            lucj=[g for g in groups if g['case']==case and g['arm']==arm and g['method'].startswith('C5_') and g['completed']==3]
            if pair and pair['completed']==3 and lucj:
                best=min(lucj,key=lambda g:g['energy_Ha_mean'])
                comparisons.append(dict(case=case,arm=arm,best_available_LUCJ=best['method'],
                    pair_minus_best_LUCJ_mean_mHa=1000*(pair['energy_Ha_mean']-best['energy_Ha_mean']),
                    best_LUCJ_selected_retrospectively=True, n_MC_draws_each=3, statistical_superiority_certified=False))
    sampling_runtime=[]
    for c in sub['circuits']:
        folder=source.LUCJ/('collected/circuits' if c['backend']=='A100' else 'circuits')/c['circuit_id']/'output'
        if (folder/'result.json').exists():
            result=load(folder/'result.json')
            sampling_runtime.append(dict(case=c['case'],method=c['source'],backend=c['backend'],
                CPU_seconds=result['whole_body_CPU_seconds'],wall_seconds=result['whole_body_wall_seconds'],
                result_SHA256=sha256((folder/'result.json').read_bytes())))
    gpu_accounting=[]
    for f in sorted(source.LUCJ.glob('lane*-collection.json')):
        collected=load(f);job=collected['job'];parts=next(line.split('|') for line in collected['accounting'].splitlines() if line.split('|')[0]==job['job_id'])
        if parts[1]!='COMPLETED' or parts[3]!='0:0':raise RuntimeError('GPU accounting not successful')
        h,m,s=map(int,parts[2].split(':'))
        gpu_accounting.append(dict(job_id=job['job_id'],elapsed_seconds=3600*h+60*m+s,
            one_A100_allocated=True,accounting=collected['accounting'],selected_transfer_bytes=collected['total_selected_bytes']))
    init=load(ROOT/'runs/N25-PAIRED-LOWSHOT-v1/protocol.json')['sources']
    initialization=[]
    for s in init:
        if sha256(Path(s['initialization_record_path']).read_bytes())!=s['initialization_record_SHA256']:raise RuntimeError('Old initializer evidence changed')
        initialization.append(dict(case=s['case'],old_CPU_seconds=s['initialization_record']['CPU_process_seconds'],
            selected_method=s['initialization_record']['selected_method'],attempts=s['initialization_record'].get('attempts'),
            complete_old_record=s['initialization_record'],
            extra_upstream_CCSD_CPU_seconds=s['initialization_record'].get('upstream_CCSD_CPU_seconds'),
            old_record_SHA256=s['initialization_record_SHA256']))
    count=len(rows);base=ROOT/'analysis'/f'N29-readout-{count}'
    output=dict(status='N29_ALL120_READOUTS_AUDITED' if count==120 else 'N29_PARTIAL48_READOUTS_AUDITED_GPU_INPUTS_PENDING',
        rows=rows,groups=groups,comparisons=comparisons,all_resources=resources_to_rows(resources),
        sources_registered=60,source_status_counts=dict(Counter(x['status'] for x in actual_sources.values())),
        endpoint_status_counts=dict(Counter(r['status'] for r in rows)),actual_new_CI_roots=len(roots),failed_CI_attempts=sum(a['status']!='COMPLETED' for a in attempts),
        CI_CPU_seconds=sum(t['CPU_seconds'] for t in terminals),CI_wall_seconds_sum=sum(t['wall_seconds'] for t in terminals),
        source_CPU_terminal=load(source.LOCAL/'cpu-terminal.json'),
        LUCJ_runtime=sampling_runtime,GPU_accounting=gpu_accounting,
        actual_new_model_shots=sum(1000 for r in expected_rows if r['available'] and r['method']!='MARKOV_SCALE2' and actual_sources[r['draw_id']]['status']=='COMPLETED'),
        actual_new_classical_Markov_draws=sum(1000 for r in expected_rows if r['available'] and r['method']=='MARKOV_SCALE2' and actual_sources[r['draw_id']]['status']=='COMPLETED'),
        actual_new_QPU_shots=0,old_initialization=initialization,
        N28_search_CPU_seconds=load(source.N28/'terminal.json')['CPU_seconds'],
        transfer_wall_seconds=None,transfer_time_missing_reason='NOT_SEPARATELY_INSTRUMENTED_DO_NOT_INVENT_ZERO',
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        readout_protocol_SHA256=sha256((readout.LOCAL/'protocol.json').read_bytes()),
        terminal_SHA256={t['batch']:sha256((readout.LOCAL/t['batch']/'terminal.json').read_bytes()) for t in terminals},
        old_initialization_and_search_costs_retained=True,cold_matched_work=False,original_paper_reproduction=False,
        exact_source_not_QPU=True,goal_complete=False)
    save(base.with_suffix('.json'),output)
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
    write(base.with_suffix('.csv'),stream.getvalue().encode('utf-8-sig'))
    lines=['# N29：新しい1,000サンプルでの比較（途中集計）' if count<120 else '# N29：新しい1,000サンプルでの全比較','',
        f"予定120 endpoint中{count}を監査。状態{output['endpoint_status_counts']}。実際の新CI根{len(roots)}、失敗{output['failed_CI_attempts']}。",
        'sourceは54利用可能・6既存欠測。GPU待機は欠測ではなく未取得。energyはsampler自身ではなく共通の古典CI後。','',
        '| 条件 | 方法 | 評価 | 数値/予定 | energy平均 Ha | 反復SD mHa | 実CI次元 | Recovery平均 % |',
        '|---|---|---|---:|---:|---:|---|---:|']
    fmt=lambda x:'未取得/未適用' if x is None else f'{x:.6f}'
    for g in groups:
        lines.append('| '+' | '.join([g['case'],g['method'],g['arm'],f"{g['completed']}/{g['planned']}",fmt(g['energy_Ha_mean']),fmt(g['energy_SD_mHa']),str(g['actual_D_values']),fmt(g['recovery_pct_mean'])])+' |')
    lines+=['','## 同じ1,000サンプルでの差','',
        '| 条件 | 評価 | pair − 最良利用可能LUCJ mHa（負がpair側低い） |','|---|---|---:|']
    for c in comparisons:lines.append(f"| {c['case']} | {c['arm']} | {c['pair_minus_best_LUCJ_mean_mHa']:.6f} |")
    lines+=['','## 論理回路の資源（全法opt1）','','| 条件 | 方法 | CNOT | CNOT Depth | Total Depth | 宣言変数 |','|---|---|---:|---:|---:|---:|']
    for resource in output['all_resources']:
        lines.append('| '+' | '.join(str(resource[k]) for k in ('case','method','CNOT','CNOT_Depth','Total_Depth','declared_parameters'))+' |')
    lines+=['','## 今回の費用','',
        f"新model-equivalent samples {output['actual_new_model_shots']}、新Markov古典draws {output['actual_new_classical_Markov_draws']}、QPU0。旧100k drawの再利用ではない。",
        f"Mac source stage {output['source_CPU_terminal']['CPU_seconds']:.6f} CPU秒（ナフタレンLUCJを含む）、新CI {output['CI_CPU_seconds']:.6f} CPU秒、今回の監査 {output['audit_CPU_seconds']:.6f} CPU秒。互いを二重加算しない。",
        f"A100 jobは{[(x['job_id'],x['elapsed_seconds']) for x in gpu_accounting]}、合計{sum(x['elapsed_seconds'] for x in gpu_accounting)} GPU割当秒。queue待機は含まない。A100上のCPU時間とMac CPUは同じ性能単位ではない。",
        f"旧初期化adapter {sum(x['old_CPU_seconds'] for x in initialization):.6f} CPU秒、別途記録されたupstream CCSD {sum(x['extra_upstream_CCSD_CPU_seconds'] or 0 for x in initialization):.6f} CPU秒、N28探索{output['N28_search_CPU_seconds']:.6f} CPU秒を保持。元LUCJ生成/compile・その他過去探索費用まで含めたcold matched-workではない。転送wall時間は個別未計測で、0とはしない。"]
    lines+=['','## 読み方と限界','',
        'R32_CAPは全法に同じ最大Rを与えるが、観測poolが足りない法は実Dが小さい。COMMON_Rは各条件・反復で同じ実Dに揃える。両方を見る。',
        '3反復は同じ固定sourceの新しいMonte Carlo drawsで、独立分子や独立source最適化ではない。CI基底が一致してrootを再利用してもraw drawの独立性は保持される。ここでは有意差や一般化を主張しない。',
        'Markovの量子CNOT/Depth/Parametersは未適用で、ゼロコストの量子回路という意味ではない。pairとLUCJの資源はlogical all-to-allのopt1を主表とし、別compiler点もJSONへ保持する。',
        'N₂の参照精度は未確認、Feの参照はSCIでFCIではない。Recoveryはこれらの参照に条件付けた値。H+0.1S²による有限penaltyで、厳密singlet証明ではない。',
        'Mac source CPUと新CI費用に加え、旧初期化・CCSD失敗/CISD fallback・N28探索・A100割当/転送費用がある。新1,000 Shotだけでcold構築費用を消さない。',
        '原著最高予算・native TN/QSCI/SqDRIFT・参照精度の未検証が残り、最強論文への勝利ではない。未取得のGPU結果を見ずに設定を変更しない。']
    write(base.with_suffix('.md'),'\n'.join(lines).encode())
    print(output['status'],count,'roots',len(roots),'comparisons',comparisons,flush=True)


def resources_to_rows(resources):
    return [dict(case=case,method=method,**resource) for (case,method),resource in sorted(resources.items())]


if __name__=='__main__':run()
