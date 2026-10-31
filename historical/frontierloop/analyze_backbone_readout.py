"""N23 all-source descriptive review; all adverse/missing endpoints retained."""
from io import BytesIO,StringIO
import csv,math,statistics
from pathlib import Path
from collections import Counter
from time import process_time,perf_counter
import ffsim
import numpy as np
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest
from .backbone_controls import LOCAL as SAMPLES,backbone,one_spin_distribution,draw
from .backbone_readout import LOCAL,source_input
from .common_readout_audit import audit
from .gaussian_source import checked_counts
from .analyze import reference


def audit_sampling():
    cpu,wall=process_time(),perf_counter();p=load(SAMPLES/'protocol.json');t=load(SAMPLES/'terminal.json')
    if t['protocol_SHA256']!=sha256((SAMPLES/'protocol.json').read_bytes()) or sha256((ROOT/'frontierloop/backbone_controls.py').read_bytes())!=p['code_SHA256']:raise RuntimeError('Sampling implementation/protocol changed')
    if len(t['draws'])!=24 or {d['status'] for d in t['draws']}!={'COMPLETED'} or len({d['draw_id'] for d in t['draws']})!=24:raise RuntimeError('All24actualdraws required')
    if t['exact_Gaussian_model_equivalent_samples']!=1200000 or t['classical_uniform_samples']!=1200000 or t['known_generated_classical_samples_including_failures']!=2400000 or t['attempts_with_unknown_generated_sample_count']!=0:raise RuntimeError('Actual draw accounting differs')
    attempts=load(SAMPLES/'draw-attempts.json')
    if len(attempts)!=24 or {a['status'] for a in attempts}!={'COMPLETED'} or {a['draw_id'] for a in attempts}!={d['draw_id'] for d in t['draws']}:raise RuntimeError('Actual draw attempts not bound')
    sources={s['case']:s for s in t['sources']};proofs=[]
    for case,s in sources.items():
        folder=SAMPLES/case
        if load(folder/'source.json')!=s:raise RuntimeError('Actual source/terminal binding differs')
        original=s['original_source']
        for field,hfield in [('operator','operator_SHA256'),('construction','construction_SHA256'),('source_result','source_result_SHA256')]:
            if sha256(Path(original[field]).read_bytes())!=original[hfield]:raise RuntimeError('Original source bytes changed')
        with np.load(original['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        qc,a,b,occupied,skipped=backbone(op,s['norb'],tuple(s['nelec']))
        payload=Path(s['distribution'])
        if sha256(payload.read_bytes())!=s['distribution_SHA256']:raise RuntimeError('Frozen Gaussian probability input changed')
        with np.load(payload,allow_pickle=False) as z:
            ids,pa,pb=z['half_strings'],z['probabilities_a'],z['probabilities_b']
            if not np.array_equal(z['rotation_a'],a) or not np.array_equal(z['rotation_b'],b):raise RuntimeError('Actual source frame differs')
        ids2,p2,norm=one_spin_distribution(s['norb'],occupied[0],a)
        if not np.array_equal(ids,ids2) or np.max(abs(pa-p2))>1e-14 or not np.array_equal(pa,pb) or norm!=s['Cauchy_Binet_norm']:raise RuntimeError('Actual determinant probability table differs')
        for output in s['all_compiler_points']:
            for role in ('preparation','sampling'):
                qpath=Path(output[role]['path'])
                if sha256(qpath.read_bytes())!=output[role]['SHA256']:raise RuntimeError('Actual archived source circuit changed')
                circuit=qpy.load(BytesIO(qpath.read_bytes()))[0]
                if circuit.num_qubits!=2*s['norb']:raise RuntimeError('Wrong physical register size')
                if role=='sampling':
                    counters=dict(CNOT=int(circuit.count_ops().get('cx',0)),CNOT_Depth=int(circuit.depth(filter_function=lambda x:x.operation.name=='cx')),Total_Depth=int(circuit.depth()))
                    if any(output['resources'][k]!=v for k,v in counters.items()):raise RuntimeError('Actual QPY resource recount differs')
        uniform=np.full(len(ids),1/len(ids))
        for d in [d for d in t['draws'] if d['case']==case]:
            path=Path(d['payload'])
            if sha256(path.read_bytes())!=d['payload_SHA256'] or d['source_record_SHA256']!=sha256((folder/'source.json').read_bytes()):raise RuntimeError('Actual draw/source linkage differs')
            seed=int(digest(['N23',case,d['method'],d['replicate'],original['operator_SHA256']])[:16],16)%(1<<63)
            expected_id=digest([case,d['method'],seed,100000,s['problem_id']])
            if seed!=d['seed'] or expected_id!=d['draw_id']:raise RuntimeError('Frozen RNG/input identity differs')
            # Deterministic audit reconstruction, NOT new experimental draws.
            raw=draw(ids,pa if d['method']=='GAUSSIAN_BACKBONE' else uniform,pb if d['method']=='GAUSSIAN_BACKBONE' else uniform,s['norb'],100000,seed)
            values,counts=checked_counts(raw,s['norb'],tuple(s['nelec']),100000)
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['blocked_ids'],values) or not np.array_equal(z['counts'],counts):raise RuntimeError('Actual sample counts not reproducible from frozen inputs')
        proofs.append(dict(case=case,Cauchy_Binet_norm=norm,one_spin_dimension=len(ids),removed_J_blocks=skipped,all6_draws_reconstructed=True,all4_QPY_hashes_and_resource_counts_checked=True))
    out=dict(status='N23_ALL24_DRAWS_ALL4_SOURCE_TABLES_QPYS_AND_INPUTS_VERIFIED',proofs=proofs,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        deterministic_reconstruction_samples=2400000,new_independent_experimental_samples=0,new_QPU_shots=0,
        numerical_qualification_scope='SMALL_COMPLEX_ELECTRON_HOLE_QISKIT_PLUS_MOLECULAR_DETERMINANT_TABLE_AND_ACTUAL_QPY_COUNTERS_NOT_ALL40Q_FULL_STATEVECTOR')
    save(ROOT/'analysis/N23-sampling-stage-audit.json',out)
    return t,sources,out


def run():
    cpu,wall=process_time(),perf_counter();sample,sources,sampleaudit=audit_sampling();spec=load(LOCAL/'protocol.json')
    if sha256((ROOT/'frontierloop/backbone_readout.py').read_bytes())!=spec['preparer_SHA256'] or sha256((SAMPLES/'terminal.json').read_bytes())!=spec['sampling_terminal_SHA256'] or sha256((SAMPLES/'protocol.json').read_bytes())!=spec['sampling_protocol_SHA256']:raise RuntimeError('Frozen sampling/readout context changed')
    draws={d['draw_id']:d for d in sample['draws']}
    result=audit(LOCAL,lambda p,d:source_input(draws[p['source']['draw_id']],d,spec['sampling_terminal_SHA256']))
    rows=[]
    for r in result['rows']:
        case=r['case'];data=r['result'];source=r['source'];method=source.get('source_method','UNRESTRICTED_HF_HCI')
        hf,ref=reference(case);meta=load(OLD/'data/evaluation-only'/(case+'.json'))
        if meta['problem_id']!=spec['physical'][case]['problem_id']:raise RuntimeError('Reporting reference problem differs')
        gaussian=method=='GAUSSIAN_BACKBONE';state=sources[case];res=state['all_compiler_points'][0]['resources'] if gaussian else {}
        item={k:v for k,v in r.items() if k not in ('result','source','method')}
        item.update(method=method,replicate=source.get('replicate'),energy_Ha=data['energy_Ha'] if data else None,
            penalized_energy_Ha=data['penalized_energy_Ha'] if data else None,spin_square=data['spin_square'] if data else None,
            recovery_pct=100*(hf-data['energy_Ha'])/(hf-ref) if data else None,reference_gap_mHa=1000*(data['energy_Ha']-ref) if data else None,
            reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
            reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') is True else 'UNVERIFIED',
            CNOT=res.get('CNOT'),CNOT_Depth=res.get('CNOT_Depth'),Total_Depth=res.get('Total_Depth'),
            declared_parameters=state['declared_parameters'] if gaussian else None,
            sample_role=source.get('sample_role','NO_SOURCE_DRAW_HF_INITIALIZED_CLASSICAL_HCI'),
            classical_samples=source.get('actual_classical_samples'),new_QPU_shots=0,
            observed_half_pool_fraction=(r['observed_half_pool_size']/math.comb(state['norb'],state['nelec'][0])) if r['observed_half_pool_size'] is not None else None,
            final_CI_state_not_prepared_by_sampler=True)
        rows.append(item)
    old=load(ROOT/'analysis/N17-whole-study-summary.json');groups=[]
    for case in sorted(sources):
        for method in ('GAUSSIAN_BACKBONE','UNIFORM_SECTOR'):
            for arm in ('SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'):
                rs=[r for r in rows if (r['case'],r['method'],r['arm'])==(case,method,arm)];good=[r for r in rs if r['energy_Ha'] is not None]
                if len(rs)!=3 or {r['replicate'] for r in rs}!={230101,230102,230103}:raise RuntimeError('All3planned replicates inclmissing required')
                parent=next(g for g in old['groups'] if (g['case'],g['source'],g['method'],g['arm'])==(case,'C5_C_LUCJ_COMPRESSED_L1','ORIGINAL',arm))
                energy=statistics.mean(r['energy_Ha'] for r in good) if good else None;complete=len(good)==parent['complete_replicates']==3
                hfcontrol=next(r for r in rows if r['case']==case and r['method']=='UNRESTRICTED_HF_HCI')
                point=dict(case=case,method=method,arm=arm,planned_replicates=3,completed_replicates=len(good),
                    distinct_actual_CI_observations=len({r['observation_id'] for r in good}),
                    energy_Ha_mean=energy,energy_SD_mHa=1000*statistics.stdev(r['energy_Ha'] for r in good) if len(good)>1 else None,
                    recovery_pct_mean=statistics.mean(r['recovery_pct'] for r in good) if good else None,
                    actual_CI_dimension_mean=statistics.mean(r['actual_CI_dimension'] for r in good) if good else None,
                    native_plus_CI_CPU_mean=statistics.mean(r['standalone_native_plus_CI_CPU_seconds'] for r in good) if good else None,
                    parent_full_LUCJ_mean_Ha=parent['energy_Ha_mean'],energy_delta_vs_full_LUCJ_mean_mHa=1000*(energy-parent['energy_Ha_mean']) if complete else None,
                    energy_delta_vs_fresh_HF_HCI_mHa=1000*(energy-hfcontrol['energy_Ha']) if energy is not None and hfcontrol['energy_Ha'] is not None else None,
                    CNOT=rs[0]['CNOT'],CNOT_Depth=rs[0]['CNOT_Depth'],Total_Depth=rs[0]['Total_Depth'],declared_parameters=rs[0]['declared_parameters'],
                    parent_declared_parameters=parent['declared_parameters'],statistical_equivalence_or_superiority_certified=False,
                    same_dimension_cap_not_equal_actual_D_or_CPU=True)
                if method=='GAUSSIAN_BACKBONE':
                    point['matched_compiler_portfolio']=[dict(level=o['resources']['optimization_level'],candidate=o['resources'],
                        full_LUCJ_parent=parent['original_opt1_resources' if o['resources']['optimization_level']==1 else 'original_opt3_resources']) for o in sources[case]['all_compiler_points']]
                groups.append(point)
    strong_LUCJ=[g for g in old['groups'] if g['method']=='ORIGINAL']
    prior=load(ROOT/'analysis/N22-gaussian-CI-summary.json');controlparity=[]
    for r in rows:
        if r['method']!='UNRESTRICTED_HF_HCI' or r['energy_Ha'] is None:continue
        oldhf=next(a for a in prior['rows'] if a['case']==r['case'] and a['method']=='UNRESTRICTED_HF_HCI')
        error=abs(r['energy_Ha']-oldhf['energy_Ha'])
        if error>1e-8:raise RuntimeError('Fresh classical control physical parity differs')
        controlparity.append(dict(case=r['case'],absolute_energy_difference_Ha=error))
    t=result['terminal'];out=dict(status='N23_ALL28_TASKS52_READOUTS24_SOURCES_AUDITED',rows=rows,groups=groups,
        stronger_existing_full_LUCJ_L1_L2_L4_groups=strong_LUCJ,fresh_HF_control_parities=controlparity,
        actual_new_CI_roots=result['actual_new_CI_roots'],failed_CI_attempts=result['failed_CI_attempts'],
        CI_campaign_CPU_seconds=t['CPU_seconds'],CI_campaign_wall_seconds=t['wall_seconds'],
        sampling_compile_distribution_CPU_seconds=sample['CPU_seconds'],sampling_compile_distribution_wall_seconds=sample['wall_seconds'],
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        new_exact_Gaussian_model_equivalent_samples=1200000,new_classical_uniform_samples=1200000,new_QPU_shots=0,GPU_allocations=0,
        repeated_identical_CI_roots_not_independent_method_confirmations=True,original_initialization_costs_additional=True,
        cold_total_matched_work=False,original_highest_TN_and_SqDRIFT_paper_reproduction=False,
        best_paper_superiority=False,goal_complete=False,protocol_SHA256=result['protocol_SHA256'],terminal_SHA256=result['terminal_SHA256'],
        source_audit_SHA256=sha256((ROOT/'analysis/N23-sampling-stage-audit.json').read_bytes()))
    save(ROOT/'analysis/N23-backbone-uniform-summary.json',out)
    for name,values in [('all52-readouts',rows),('all16-group-comparisons',groups)]:
        stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in values))));writer.writeheader();writer.writerows(values)
        write(ROOT/f'analysis/N23-{name}.csv',stream.getvalue().encode('utf-8-sig'))
    def fmt(v,n=3):return '未取得' if v is None else f'{v:.{n}f}'
    lines=['# N23：相関を外すなら基底回転を残す。ただしそれでも良いsourceとは限らない','',
        f"全52readout：{dict(Counter(r['status'] for r in rows))}。実際の新CI根{out['actual_new_CI_roots']}、失敗CI{out['failed_CI_attempts']}。",
        f"Gaussian120万model-equivalent drawsとuniform120万classical drawsを別計上。QPU/GPU0。sampling/compile/分布表 {sample['CPU_seconds']:.3f} CPU秒、CI campaign {t['CPU_seconds']:.3f} CPU秒。",'',
        '| 条件 | 方法 | readout | 完了/3 | 異なるCI観測数 | energy平均 Ha | SD mHa | full LUCJとの差 mHa | Recovery平均 % | CNOT | CX Depth | Total Depth | Parameters* | 実D平均 | native+CI CPU平均秒 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append('| '+' | '.join([g['case'],g['method'],g['arm'],str(g['completed_replicates']),str(g['distinct_actual_CI_observations']),fmt(g['energy_Ha_mean'],9),fmt(g['energy_SD_mHa']),fmt(g['energy_delta_vs_full_LUCJ_mean_mHa']),fmt(g['recovery_pct_mean']),fmt(g['CNOT'],0),fmt(g['CNOT_Depth'],0),fmt(g['Total_Depth'],0),fmt(g['declared_parameters'],0),fmt(g['actual_CI_dimension_mean'],1),fmt(g['native_plus_CI_CPU_mean'])])+' |')
    lines+=['','## 強いLUCJ対照を省略しない','',
        '| 条件 | full LUCJ | readout | 完了/3 | energy平均 Ha | CNOT | CX Depth | Total Depth | Parameters | 実D平均 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in strong_LUCJ:
        lines.append('| '+' | '.join([g['case'],g['source'],g['arm'],str(g['complete_replicates']),fmt(g['energy_Ha_mean'],9),fmt(g['CNOT'],0),fmt(g['CNOT_Depth'],0),fmt(g['Total_Depth'],0),fmt(g['declared_parameters'],0),fmt(g['actual_CI_dimension_mean'],1)])+' |')
    lines+=['','## 古典HF-HCI対照','',
        '| 条件 | energy Ha | Recovery % | 実CI D | native+CI CPU秒 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['method']=='UNRESTRICTED_HF_HCI':lines.append('| '+' | '.join([r['case'],fmt(r['energy_Ha'],9),fmt(r['recovery_pct']),fmt(r['actual_CI_dimension'],0),fmt(r['standalone_native_plus_CI_CPU_seconds'])])+' |')
    lines+=['','## 読み方・限界','',
        '負のenergy差は元full LUCJ L1の全3反復平均より低い。平均比較は独立確認・精度同等の統計的証明ではない。同じ最終基底になった反復は同じrootを再利用し、異なるCI観測数を明示。3回の同じ結果を3つの独立成功としない。',
        'R128/最大D16384の規則は同じでも、実DとCPUは異なる。小さいsource poolを観測外の架空配置で埋めない。大きいCI空間・古典HCI選抜の利益を量子samplingの利益と読み替えない。厳密なsource優越には同じ実Dでの対照がなお必要。',
        'uniformは古典サンプリングで、対応する量子回路を準備・実行していない。資源を0で捏造せず未適用とした。最終の古典CI状態をGaussian samplerが準備した意味でもない。',
        '* Parametersは同じshared-spin complex Slater chartの宣言座標。最小Jacobian rankや数値gateスロットではない。今回optimizer変数0でも、ansatz Parameters0ではない。JSONに同level opt1/opt3の全compiler比較を残した。',
        'N23の枚挙は各spinで最大15,504要素を使うサイズ限定backend。一般には指数的で、汎用40q statevector simulator・量子優位・新しいSlater理論ではない。N22の遅いDPP費用も過去実費として消さない。',
        '表のCPUはcold総費用ではない。旧CCSD/CISD/圧縮初期化、原著の最適化、queue/転送、検証費用は別。各source分布生成は反復/両readout間で共有し二重計上しない。',
        'H+0.1S²で選んだCI根のphysical Hをenergyとして報告。純粋なS=0を保証しない。Recoveryはenergy換算で独立指標ではない。Fe SCI/N₂ archive FCIの参照精度未認証を全CSVに保存。',
        '原著最高TN最適化・SqDRIFT原著予算・NOCI-SQD最新版の再現ではない。最強論文への総合優越と全goal達成は未証明。']
    write(ROOT/'analysis/N23_BACKBONE_UNIFORM_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['actual_new_CI_roots'],out['CI_campaign_CPU_seconds'],flush=True)


if __name__=='__main__':run()
