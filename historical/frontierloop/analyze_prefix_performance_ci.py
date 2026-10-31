"""N17 all-cell energy/resource/work report with actual-root verification."""
import argparse
import csv
import gzip
import hashlib
import math
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from . import prefix_performance_ci as study
from .prefix_performance_sampling import LOCAL as SAMPLES,verified_protocol
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row
from .analyze import reference
from .audit_prefix_sampling import parameter_row,all_control_outputs


def basis_digest(pairs):
    return hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest()


def check_observation(item,case,pairs,byid,seen):
    key=(case,basis_digest(pairs));o=byid[item['observation_id']]
    if o['case']!=case or o['basis_digest']!=key[1] or item['row']!=o['row'] or item['evaluation_origin']!=o['origin']:
        raise RuntimeError('CI reuse does not match the actual same-problem/basis observation')
    if item['new_CI_solve']!=(key not in seen) or item['row']['actual_CI_dimension']!=len(pairs):
        raise RuntimeError('Actual solve/reuse/dimension accounting differs')
    seen.add(key);return item['row']


def run(batch):
    local=study.LOCAL;spec=load(local/'protocol.json');collected=local/'collected';terminal=load(collected/'terminal.json')
    started=load(collected/'started.json');sampling=verified_protocol();protocol_sha=sha256((local/'protocol.json').read_bytes())
    if started['protocol_SHA256']!=protocol_sha or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Actual protocol/thread binding differs')
    if sha256((local/'worker.py').read_bytes())!=spec['worker_SHA256'] or sha256(study.generate().encode())!=spec['worker_SHA256'] or sha256(Path(study.__file__).read_bytes())!=spec['preparer_SHA256']:
        raise RuntimeError('Frozen common numerical pipeline/preparer changed')
    if spec['sampling_protocol_SHA256']!=sha256((SAMPLES/'protocol.json').read_bytes()):raise RuntimeError('Sampling protocol changed')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((local/'pipeline'/name).read_bytes())!=value:raise RuntimeError('Numerical helper changed')
    if load(collected/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':
        raise RuntimeError('Actual tiny native export qualifier did not pass')
    plans=spec['plans'];states={s['task_id']:s for s in terminal['states']}
    if len(states)!=len(plans) or len(terminal['states'])!=len(plans) or set(states)!={p['task_id'] for p in plans}:
        raise RuntimeError('Incomplete planned terminal-state inventory')
    observations=load(collected/'observations.json');attempts=load(collected/'ci-attempts.json');byid={o['id']:o for o in observations}
    completed=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(observations) or len(completed)!=len(observations) or len(observations)!=terminal['actual_new_CI_roots'] or len(attempts)!=terminal['ci_attempts'] or {a['observation_id'] for a in completed}!=set(byid):
        raise RuntimeError('Actual CI evaluation accounting mismatch')
    for o in observations:
        verify_row(o['row']);path=collected/o['vector_file']
        if sha256(path.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved CI vector changed')
        with np.load(path,allow_pickle=False) as a:pairs=a['joint_pairs'];vector=a['coefficients']
        if len(pairs)!=len(vector) or not np.isfinite(vector).all() or abs(vector@vector-1)>1e-10:
            raise RuntimeError('Saved CI vector invalid')
        h=basis_digest(pairs)
        if o['basis_digest']!=h or o['id']!=hashlib.sha256((o['case']+h+protocol_sha).encode()).hexdigest() or o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:
            raise RuntimeError('CI problem/basis/kernel identity mismatch')
    results={};seen=set();native_CPU=0.;bundle=load(local/'bundle-collection-v1.json')
    for plan in plans:
        state=states[plan['task_id']];case=plan['case'];definition=spec['physical'][case];folder=collected/'cells'/plan['task_id']
        if load(folder/'result.json')!=state or state['case']!=case or state['source']!=plan['source'] or state['selection_mode']!=plan['selection_mode']:
            raise RuntimeError('Actual case/source/input state mismatch')
        source=plan['source'];spaces={};native=None
        if source['status']!='AVAILABLE':
            if state['status']!='MISSING_SAME_PROBLEM_SOURCE' or state['arms']:raise RuntimeError('Missing source fabricated')
        elif source['kind']=='QUANTUM_FULL_COUNTS':
            rebuilt,strings,pool=study.draw_source(source['draw_id'],case,definition)
            if rebuilt!=source or strings!=plan['frequency_strings'] or pool!=plan['allowed_strings']:
                raise RuntimeError('Readout input not reconstructed from actual new observations')
            spaces['SOURCE_FREQUENCY_R128']=[(a,b) for a in strings for b in strings]
        if state.get('native_attempt'):
            native_CPU+=state['native_attempt']['CPU_seconds']
            if load(folder/'native-attempt.json')!=state['native_attempt']:raise RuntimeError('Native work accounting mismatch')
            for spin in ['Alpha','Beta']:
                if (folder/(spin+'Dets.bin')).read_bytes()!=plan['HF_string'].to_bytes(16,'big'):
                    raise RuntimeError('Actual native start not HF')
        if 'native' in state:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());h=sha256(raw)
            if h!=state['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+plan['task_id']+'/dets.bin')!=h:
                raise RuntimeError('Native raw/lossless evidence mismatch')
            n,native=decode(raw);joint,selection=joint_pairs(native,n,tuple(definition['nelec']),plan['HF_string'])
            if n!=definition['norb'] or selection!=state['selection']:raise RuntimeError('Native coefficient/sector/selection mismatch')
            pool=plan['allowed_strings']
            if pool is not None:
                if (folder/'AllowedHalfDets.bin').read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in pool):
                    raise RuntimeError('Observed pool bytes differ')
                if [int(x) for x in (folder/'CEO-sample-pool-v1.txt').read_text().split()]!=pool or any(a not in pool or b not in pool for a,b,_ in native):
                    raise RuntimeError('Native HCI escaped the observed pool')
                if 'CEO_SAMPLE_POOL_V1 ' not in (folder/'native.log').read_text():raise RuntimeError('Actual constrained kernel evidence missing')
            elif (folder/'CEO-sample-pool-v1.txt').exists():raise RuntimeError('Observed restriction leaked into unrestricted control')
            spaces['JOINT_TOP_D16384']=joint
        for arm,item in state['arms'].items():
            if arm not in plan['requested_readouts'] or arm not in spaces:raise RuntimeError('Unplanned or unbound successful readout')
            row=check_observation(item,case,spaces[arm],byid,seen)
            hf,ref=reference(case);reference_file=OLD/'data/evaluation-only'/(case+'.json');meta=load(reference_file)
            if meta['problem_id']!=definition['problem_id']:raise RuntimeError('Evaluation reference problem mismatch')
            final_set=set(spaces[arm])
            results[(plan['task_id'],arm)]=dict(energy_Ha=row['energy_Ha'],penalized_energy_Ha=row['penalized_energy_Ha'],spin_square=row['spin_square'],
                recovery_pct=100*(hf-row['energy_Ha'])/(hf-ref),reference_gap_mHa=1000*(row['energy_Ha']-ref),
                reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') is True else 'UNVERIFIED',
                actual_CI_dimension=row['actual_CI_dimension'],CI_CPU_seconds=row['CPU_seconds'],
                standalone_numeric_CPU_seconds=row['CPU_seconds']+(state.get('native_attempt',{}).get('CPU_seconds',0.) if arm=='JOINT_TOP_D16384' else 0.),
                new_CI_solve=item['new_CI_solve'],observation_id=item['observation_id'],
                final_basis_spin_flip_closed=all((b,a) in final_set for a,b in spaces[arm]))
    if len(seen)!=len(observations) or native_CPU+sum(a['CPU_seconds'] for a in attempts)>terminal['CPU_seconds']+1e-5:
        raise RuntimeError('Unattributed or double-counted numerical work')
    samples={c['cell_id']:c for c in sampling['cells']};bindings=load(ROOT/'analysis/N15-declared-coordinate-bindings.json')['rows']
    raw=load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/result.json')['rows'];control_cache={};rows=[]
    for logical in spec['logical_cells']:
        cell=samples[logical['cell_id']];state=states[logical['task_id']]
        if any(cell[k]!=logical[k] for k in ['case','source','method','replicate']):raise RuntimeError('Logical method mapping differs')
        for arm in ['SOURCE_FREQUENCY_R128','JOINT_TOP_D16384']:
            numeric=results.get((logical['task_id'],arm));record=dict(logical,arm=arm,
                status='COMPLETED' if numeric else 'MISSING_SAME_PROBLEM_SOURCE' if cell['status']=='MISSING_SAME_PROBLEM_SOURCE' else 'READOUT_FAILED_PRESERVED',
                task_status=state['status'],energy_Ha=None,recovery_pct=None,model_shots=100000 if cell.get('draw_id') else None,
                new_QPU_shots=0,CNOT=None,CNOT_Depth=None,Total_Depth=None,declared_parameters=None,
                error=None if numeric else state.get('errors',{}).get(arm,state.get('error')))
            if cell.get('draw_id'):
                resource=cell['selected_resources'];key=(cell['case'],cell['source'])
                if key not in control_cache:control_cache[key]=all_control_outputs(*key,raw)
                record.update(CNOT=resource['CNOT'],CNOT_Depth=resource['CNOT_Depth'],Total_Depth=resource['Total_Depth'],
                    declared_parameters=parameter_row(cell,bindings),preparation_infidelity=cell['preparation_infidelity'],
                    original_opt1_resources=control_cache[key][0]['resources'],original_opt3_resources=control_cache[key][1]['resources'])
            if numeric:record.update(numeric)
            rows.append(record)
    controls=[]
    for plan in plans:
        if plan['method']=='UNRESTRICTED_HF_HCI':
            record=dict(case=plan['case'],method=plan['method'],status=states[plan['task_id']]['status'],
                arm='JOINT_TOP_D16384',CNOT=None,CNOT_Depth=None,Total_Depth=None,declared_parameters=None)
            record.update(results.get((plan['task_id'],'JOINT_TOP_D16384'),{}));controls.append(record)
    contrasts=[]
    for r in rows:
        if r['method']=='ORIGINAL':continue
        original=next(x for x in rows if (x['case'],x['source'],x['replicate'],x['arm'],x['method'])==
            (r['case'],r['source'],r['replicate'],r['arm'],'ORIGINAL'))
        numeric=r['energy_Ha'] is not None and original['energy_Ha'] is not None
        contrast=dict(case=r['case'],source=r['source'],method=r['method'],replicate=r['replicate'],arm=r['arm'],
            status='NUMERIC_PAIR' if numeric else 'MISSING_PAIR',
            energy_delta_mHa=1000*(r['energy_Ha']-original['energy_Ha']) if numeric else None,
            penalized_delta_mHa=1000*(r['penalized_energy_Ha']-original['penalized_energy_Ha']) if numeric else None,
            shared_actual_draw=(r['draw_id']==original['draw_id']) if r['draw_id'] is not None else None)
        for metric in ['CNOT','CNOT_Depth','Total_Depth','declared_parameters']:
            contrast[metric+'_delta']=r[metric]-original[metric] if r[metric] is not None and original[metric] is not None else None
        contrasts.append(contrast)
    result=dict(status='N17_'+batch+'_ALL_FROZEN_LOGICAL_READOUTS_AUDITED',execution_batch=batch,rows=rows,unrestricted_HF_controls=controls,
        contrasts=contrasts,status_counts=dict(Counter(r['status'] for r in rows)),logical_cell_count=len(spec['logical_cells']),
        full_logical_inventory_count=108,other_execution_batch_incomplete_or_separately_reported=spec['other_batch_cells_waiting_or_separately_accounted'],
        actual_new_CI_roots=len(observations),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        actual_native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),
        CPU_seconds=terminal['CPU_seconds'],wall_seconds=terminal['wall_seconds'],
        native_CPU_seconds=native_CPU,CI_actual_attempt_CPU_seconds=sum(a['CPU_seconds'] for a in attempts),
        same_dimension_cap_not_equal_actual_work=True,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        all_source_construction_and_sampling_costs_additional=True,final_CI_state_not_prepared_by_sampler=True,
        full_paper_superiority=False,goal_complete=False,protocol_SHA256=protocol_sha,terminal_SHA256=sha256((collected/'terminal.json').read_bytes()))
    out=ROOT/'analysis'/('N17-'+batch+'-common-readout-summary.json');save(out,result)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    write(out.with_suffix('.csv'),stream.getvalue().encode('utf-8-sig'))
    def fmt(x,n=5):return '未取得' if x is None else f'{x:.{n}f}'
    lines=['# N17 '+batch+'：同じShot条件の圧縮前後比較','',
        f"{len(spec['logical_cells'])}論理セル、{len(rows)}予定readout：{result['status_counts']}。新CI根{len(observations)}、失敗CI試行{result['failed_CI_attempts']}。", 
        f"今回のCIは{terminal['CPU_seconds']:.3f}CPU秒/{terminal['wall_seconds']:.3f}wall秒。新model/QPU Shot0（先行sampling費用は別計上）。",'',
        '| 条件 | source | method | replicate | readout | energy Ha | Recovery % | CNOT | CNOT Depth | Total Depth | Parameters* | CI D | 数値CPU proxy秒 |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([r['case'],r['source'],r['method'],str(r['replicate']),r['arm'],fmt(r['energy_Ha'],10),fmt(r['recovery_pct'],3),
            str(r['CNOT']),str(r['CNOT_Depth']),str(r['Total_Depth']),str(r['declared_parameters']),str(r.get('actual_CI_dimension')),fmt(r.get('standalone_numeric_CPU_seconds'),3)])+' |')
    lines+=['','* Parametersは同じprefix/suffix familyの宣言自由実座標。元ffsim chartからの既知の表現削減・数値slot・optimizer変数0・最小Jacobian rankとは区別する。',
        '', '## 元回路との差（負が低いenergy）','',
        '| source | method | replicate | readout | Δenergy mHa | ΔCNOT | ΔCNOT Depth | ΔTotal Depth | ΔParameters |','|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in contrasts:
        lines.append('| '+' | '.join([r['source'],r['method'],str(r['replicate']),r['arm'],fmt(r['energy_delta_mHa'],6),
            str(r['CNOT_delta']),str(r['CNOT_Depth_delta']),str(r['Total_Depth_delta']),str(r['declared_parameters_delta'])])+' |')
    lines+=['','## 解釈の範囲','',
        '- 全108セルのうち、この実行batchだけの暫定結果。残りbatchの条件をこの結果で変更しない。全研究や最強論文への勝利とは呼ばない。',
        '- 両readoutを残す。frequencyには不要なnative HCI費用を加えない。表のCPUは単独実行の数値proxyで、campaign実費・source構築・圧縮/compile・sampling費用とは別。cache再利用を新計算に水増ししない。',
        '- 同じ回路のpolicy名違いで再利用したcountsは独立試験ではない。3replicateは小さな開発反復で、未知分子一般化や統計的非劣性の証明ではない。',
        '- 保存QPYはall-to-all logical。元opt1/opt3の全資源点はJSON/CSVに保存。元CNOT-first点より下がっても、元の最小Total Depthより長い場合はtrade-off。',
        '- 最終CI状態はsampler回路が直接準備した状態ではない。H+0.1S²は厳密singlet制約ではなく、SCI archiveはFCIではない。Recoveryの参照種別・未検証状態はJSON/CSVに付記する。']
    write(ROOT/'analysis'/('N17_'+batch+'_COMMON_READOUT_RESULTS_JA.md'),('\n'.join(lines)+'\n').encode())
    print(result['status'],result['status_counts'],'CPU',result['CPU_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',choices=['M4','A100','ALL'],default='M4');a=p.parse_args();study.configure(a.batch);run(a.batch)
