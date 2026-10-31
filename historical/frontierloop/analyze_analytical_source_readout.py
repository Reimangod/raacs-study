"""Audit N33 terminal partitions, physical CI and all available controls."""
from pathlib import Path
from io import BytesIO
from collections import Counter
from time import process_time,perf_counter
import numpy as np
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from . import analytical_source_readout as r
from . import analytical_source_sampling as source
from . import analytical_source_optimization as n32
from .analyze_finite_source_readout import audit_root
from .analyze import reference


def run(partition):
    if partition not in ('M4','A100_INPUTS','ALL'):raise ValueError('Explicit scope')
    cpu,wall=process_time(),perf_counter();p,q,sub=r.verified()
    parts=['M4','A100_INPUTS'] if partition=='ALL' else [partition]
    terminals=[load(r.LOCAL/(part+'-terminal.json')) for part in parts]
    states=[x for t in terminals for x in t['states']]
    cases={x['case'] for x in states};tasks=[t for t in q['tasks'] if t['case'] in cases]
    identity=lambda x:(x['case'],x['method'],x['replicate'],x['arm'])
    byid={identity(x):x for x in states}
    if len(states)!=len(tasks) or len(byid)!=len(tasks):raise RuntimeError('Actual endpoint coverage differs')
    for task in tasks:
        actual=byid[identity(task)]
        if any(actual[k]!=v for k,v in task.items()):raise RuntimeError('Endpoint/task identity differs')
    for t in terminals:
        if t['protocol_SHA256']!=sha256((r.LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Partition protocol drift')
    inventory=[x for x in r.inventory(p,sub) if x['case'] in cases];counts=r.records(p,sub,inventory)
    if any(x['status']=='WAITING_FOR_FROZEN_INPUTS' for x in counts.values()):raise RuntimeError('Terminal without actual completed draws')
    old=load(ROOT/'analysis/N30-tuned-control-summary.json');resources={(x['case'],x['method']):x for x in old['resources']}
    for s in p['sources']:
        for point in s['resources']:
            f=point['files']['sampling'];blob=Path(f['file']).read_bytes()
            if sha256(blob)!=f['SHA256']:raise RuntimeError('Actual QPY resource hash changed')
            circuit=qpy.load(BytesIO(blob))[0];res=point['resources']
            if (res['CNOT'],res['CNOT_Depth'],res['Total_Depth'])!=(circuit.count_ops().get('cx',0),circuit.depth(filter_function=lambda x:x.operation.name=='cx'),circuit.depth()):raise RuntimeError('Actual logical resource recount differs')
        main=s['resources'][0]['resources']
        circuit=next(c for c in sub['circuits'] if c['case']==s['case'])
        parameter_source=dict(operator=s['operator'],norb=circuit['norb'],old_surrogate_coordinate_slots=s['optimizer_coordinate_slots'])
        parameters=r.oldread.parameter_count(parameter_source,circuit)
        resources[s['case'],source.METHOD]=dict(main,**parameters,declared_parameters=parameters['matched_prefix_suffix_declared_parameters'])
    roots={};rows=[]
    for task in tasks:
        state=byid[identity(task)];out={k:state.get(k) for k in ('case','method','replicate','arm','status','R','actual_D','basis_id','root_path','root_SHA256','origin','draw_id','new_CI_attempt','cache_origin')}
        if state['status']=='COMPLETED':
            actual=counts[state['draw_id']];d=p['physical'][state['case']]
            if actual!=state['source_evidence']:raise RuntimeError('Actual count binding differs')
            R=actual['R32'] if state['arm']=='R32_CAP' else min(counts[x['draw_id']]['R32'] for x in inventory if x['available'] and x['case']==state['case'] and x['replicate']==state['replicate'])
            strings=sorted(actual['rank'][:R]);pairs=[(a,b) for a in strings for b in strings]
            if R!=state['R'] or strings!=state['half_strings'] or r.parent.basis_key(d,pairs,q['library_SHA256'])!=state['basis_id']:raise RuntimeError('Actual common-R root context differs')
            if state['basis_id'] not in roots:roots[state['basis_id']]=audit_root(state,d,p['library'])
            out.update(roots[state['basis_id']]);hf,ref=reference(state['case']);meta=load(OLD/'data/evaluation-only'/(state['case']+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Evaluation reference identity differs')
            out.update(Recovery_pct=100*(hf-out['energy_Ha'])/(hf-ref),reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED')
            res=resources.get((state['case'],state['method']),{})
            out.update({k:res.get(k) for k in ('CNOT','CNOT_Depth','Total_Depth','declared_parameters','optimizer_coordinate_slots','physical_Jacobian_rank')})
        else:out['energy_Ha']=None
        rows.append(out)
    groups=[]
    for case,method,arm in sorted({(x['case'],x['method'],x['arm']) for x in rows}):
        local=[x for x in rows if (x['case'],x['method'],x['arm'])==(case,method,arm)];valid=[x for x in local if x['status']=='COMPLETED']
        groups.append(dict(case=case,method=method,arm=arm,completed=len(valid),planned=len(local),
            energy_mean_Ha=float(np.mean([x['energy_Ha'] for x in valid])) if valid else None,
            SD_mHa=float(np.std([x['energy_Ha'] for x in valid],ddof=1)*1000) if len(valid)>1 else None,
            Recovery_mean_pct=float(np.mean([x['Recovery_pct'] for x in valid])) if valid else None,
            actual_Ds=[x['actual_D'] for x in valid],statistical_superiority_certified=False))
    comparisons=[]
    for case in sorted(cases):
        for arm in r.parent.ARMS:
            new=next(g for g in groups if (g['case'],g['method'],g['arm'])==(case,source.METHOD,arm))
            controls=[g for g in groups if g['case']==case and g['arm']==arm and g['completed']==3 and (g['method'].startswith('C5_') or g['method']=='EXISTING_TN_COMMON_L1')]
            best=min(controls,key=lambda g:g['energy_mean_Ha']) if controls else None
            original=next((g for g in controls if g['method']=='C5_C_LUCJ_COMPRESSED_L1'),None)
            difference=lambda control:1000*(new['energy_mean_Ha']-control['energy_mean_Ha']) if control and new['completed']==3 else None
            comparisons.append(dict(case=case,arm=arm,best_previous_LUCJ=best['method'] if best else None,
                optimized_minus_best_previous_LUCJ_mHa=difference(best),
                optimized_minus_initial_L1_mHa=difference(original),
                retrospective_best_control=True,statistical_superiority_certified=False))
    accounts=[];runtime=[]
    for f in sorted(source.SAMPLING.glob('lane*-collection.json')):
        v=load(f);account_parts=next(line.split('|') for line in v['accounting'].splitlines() if line.split('|')[0]==v['job']['job_id'])
        if account_parts[1]!='COMPLETED' or account_parts[3]!='0:0':raise RuntimeError('Actual Slurm result unsuccessful')
        h,m,s=map(int,account_parts[2].split(':'));accounts.append(dict(job_id=v['job']['job_id'],GPU_allocation_seconds=3600*h+60*m+s,accounting=v['accounting'],transfer_bytes=v['total_selected_bytes']))
    for c in sub['circuits']:
        if c['case'] not in cases:continue
        folder=source.SAMPLING/('collected/circuits' if c['backend']=='A100' else 'circuits')/c['circuit_id']/'output'
        v=load(folder/'result.json')
        runtime.append(dict(case=c['case'],backend=c['backend'],CPU_seconds=v['whole_body_CPU_seconds'],wall_seconds=v['whole_body_wall_seconds'],result_SHA256=sha256((folder/'result.json').read_bytes())))
    out=dict(status='N33_PARTITION_PHYSICAL_READOUTS_AUDITED',scope=partition,rows=rows,groups=groups,comparisons=comparisons,
        endpoint_status_counts=dict(Counter(x['status'] for x in rows)),distinct_saved_roots_audited=len(roots),
        actual_new_CI_roots=sum(a['status']=='COMPLETED' for t in terminals for a in t['attempts']),
        failed_CI_attempts=sum(a['status']!='COMPLETED' for t in terminals for a in t['attempts']),
        new_model_shots=3000*len(cases),new_QPU_shots=0,old_draws_reused_not_new_confirmation=True,
        N32_additional_all4_child_CPU_seconds=load(n32.LOCAL/'terminal.json')['child_CPU_seconds'],
        N32_old_initializer_records_retained=[x['old_initialization_cost'] for x in p['sources']],
        GPU_accounting=accounts,source_runtime=runtime,prepare_CPU_seconds=p['prepare_CPU_seconds'],transfer_wall_seconds=None,
        actual_partition_CI_CPU_seconds=sum(t['CPU_seconds'] for t in terminals),
        source_protocol_SHA256=sha256((source.LOCAL/'protocol.json').read_bytes()),
        terminal_SHA256={part:sha256((r.LOCAL/(part+'-terminal.json')).read_bytes()) for part in parts},
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        new_CI_physical_energy_not_VQE_expectation=True,native_paper_reproduction=False,cold_matched_work=False,goal_complete=False)
    save(ROOT/'analysis'/f'N33-{partition}-summary.json',out)
    fmt=lambda x:'未取得' if x is None else f'{x:.6f}'
    lines=[f'# N33 {partition}：期待値調整後の有限Shot SQD結果','',
        f"全{len(rows)} endpoint、{out['endpoint_status_counts']}。新model samples {out['new_model_shots']}、実QPU0。新CI根{out['actual_new_CI_roots']}、失敗{out['failed_CI_attempts']}。",'',
        '| 条件 | 方法 | arm | 数値/予定 | SQD energy平均 Ha | SD mHa | Recovery平均 % | 実D |','|---|---|---|---:|---:|---:|---:|---|']
    for g in groups:lines.append('| '+' | '.join([g['case'],g['method'],g['arm'],f"{g['completed']}/{g['planned']}",fmt(g['energy_mean_Ha']),fmt(g['SD_mHa']),fmt(g['Recovery_mean_pct']),str(g['actual_Ds'])])+' |')
    lines+=['','| 条件 | arm | 既存最良LUCJ | 新−最良既存 mHa | 新−元L1 mHa |','|---|---|---|---:|---:|']
    for c in comparisons:lines.append(f"| {c['case']} | {c['arm']} | {c['best_previous_LUCJ']} | {fmt(c['optimized_minus_best_previous_LUCJ_mHa'])} | {fmt(c['optimized_minus_initial_L1_mHa'])} |")
    lines+=['','正の差は新sourceが悪い。3反復のばらつきも残し、有意な優越としない。期待値低下はSQD精度向上を保証しない。R32-capと全法同じ実Rを分離する。失敗・欠測を除いた都合のよい平均は作らない。',
        '',f"新CI {out['actual_partition_CI_CPU_seconds']:.6f} CPU秒、今回audit {out['audit_CPU_seconds']:.6f} CPU秒。N32全4追加構築82.477387 child CPU秒、N33準備{p['prepare_CPU_seconds']:.6f} CPU秒。回収済GPU割当{[(x['job_id'],x['GPU_allocation_seconds']) for x in accounts]}。N31両試行・旧初期化・sampling/転送費用は別途必要。転送wallは未計測で0ではない。実QPU Shotは0。model countsとsimulation費用を実量子測定の実費に置き換えない。",
        'CNOT/DepthはN32のopt1/3 QPYを再計数済み。optimizer coordinate slotsは独立状態パラメータrankではない。Fe参照はSCI、N2参照精度未認証。原著最高予算/実hardwareの完全再現ではなく、最強論文への勝利は未証明。']
    write(ROOT/'analysis'/f'N33_{partition}_RESULTS_JA.md','\n'.join(lines).encode())
    print('N33_AUDITED',partition,len(rows),'endpoints',comparisons,flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
