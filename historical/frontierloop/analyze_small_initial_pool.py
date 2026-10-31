"""All28 N7 outcomes and the registered within-source R8/R128 trade-off."""
import csv,gzip,hashlib
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .small_initial_pool import LOCAL
from .small_initial_pool_plan import initial_strings
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row,direction
from .analyze import reference


def run():
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';terminal=load(c/'terminal.json');started=load(c/'started.json')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Exact executed protocol/thread binding mismatch')
    if sha256((ROOT/'frontierloop/small_initial_pool_worker.py').read_bytes())!=spec['worker_SHA256'] or sha256((ROOT/'frontierloop/small_initial_pool_plan.py').read_bytes())!=spec['initial_rule_SHA256']:
        raise RuntimeError('Frozen N7 initialization/executor changed')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified numerical/decoder source drift')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Export qualifier missing')
    previous_file=ROOT/'analysis/N6-combined-joint-comparison.json';previous=load(previous_file)
    if previous['N5_G5_source_SHA256']!=sha256((ROOT/'analysis/N5-G5-bound-resource-comparison.json').read_bytes()) or previous['N6_source_SHA256']!=sha256((ROOT/'analysis/N6-source-controls-summary.json').read_bytes()):
        raise RuntimeError('Prior all-source comparison changed')
    old={(r['case'],r['method']):r for r in previous['rows']};states={s['task_id']:s for s in terminal['states']};plans=spec['plans']
    if len(plans)!=28 or len(terminal['states'])!=28 or set(states)!={p['task_id'] for p in plans} or len(old)!=28:
        raise RuntimeError('All4x7 current and prior source cells required')
    obs=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in obs}
    complete=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(obs) or len(complete)!=len(obs) or len(obs)!=terminal['actual_new_CI_roots'] or {a['observation_id'] for a in complete}!=set(byid):
        raise RuntimeError('Actual CI accounting mismatch')
    for o in obs:
        verify_row(o['row']);file=c/o['vector_file']
        if sha256(file.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved actual CI vector changed')
        with np.load(file,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        if len(v)!=len(pairs) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:raise RuntimeError('Saved CI vector normalization')
        digest=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest()
        if digest!=o['basis_digest'] or o['id']!=hashlib.sha256((o['case']+digest+started['protocol_SHA256']).encode()).hexdigest():raise RuntimeError('CI state/problem identity mismatch')
        if o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Wrong actual compiled CI kernel')
    rows=[];seen=set();native_cpu=0.;initial_groups={};transport_anomalies=[]
    bundle=load(LOCAL/'bundle-collection-v1.json') if (LOCAL/'bundle-collection-v1.json').exists() else None
    for p in plans:
        s=states[p['task_id']];source=p['source'];case=p['case'];definition=spec['physical'][case];folder=c/'cells'/p['task_id']
        if load(folder/'result.json')!=s or s['source']!=source or s['method']!=p['method'] or s['case']!=case:raise RuntimeError('Cell/source identity mismatch')
        if source['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if name in source and sha256(Path(source[name]).read_bytes())!=source[key]:raise RuntimeError('Original source evidence drift')
            expected_initial=initial_strings(source,definition['norb'],tuple(definition['nelec']),r=8)
            if expected_initial!=p['initial_strings']:raise RuntimeError('Pool was not outcome-blind source top8')
            initial_groups.setdefault((case,tuple(expected_initial)),[]).append(p['method'])
            for spin in ['Alpha','Beta']:
                f=folder/(spin+'Dets.bin')
                if f.exists() and f.read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in expected_initial):
                    raise RuntimeError('Actual native input differs from prepared space')
        elif s['status']!='MISSING_SAME_PROBLEM_SOURCE' or s['arms']:raise RuntimeError('Missing source fabricated')
        if s.get('native_attempt'):native_cpu+=s['native_attempt']['CPU_seconds']
        r=None;item=s['arms'].get('JOINT_TOP_D16384')
        if item:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes())
            if sha256(raw)!=s['native']['raw_determinants_SHA256']:raise RuntimeError('Native archive mismatch')
            if bundle:
                relative='cells/'+p['task_id']+'/dets.bin'
                if bundle['native_raw_SHA256'].get(relative)!=sha256(raw):raise RuntimeError('Remote raw/lossless archive digest mismatch')
            elif not (folder/'dets.bin').exists() or sha256((folder/'dets.bin').read_bytes())!=sha256(raw):
                raise RuntimeError('Neither complete raw copy nor authenticated lossless bundle available')
            if (folder/'dets.bin').exists() and sha256((folder/'dets.bin').read_bytes())!=sha256(raw):
                transport_anomalies.append(dict(file=str((folder/'dets.bin').relative_to(LOCAL)),
                    status='INCOMPLETE_OLD_SFTP_COPY_PRESERVED_NOT_AUTHORITATIVE_VERIFIED_GZIP_USED',
                    expected_raw_SHA256=sha256(raw),partial_SHA256=sha256((folder/'dets.bin').read_bytes())))
            n,native=decode(raw);pairs,selection=joint_pairs(native,n,tuple(definition['nelec']),p['HF_string'])
            if n!=definition['norb'] or selection!=s['selection']:raise RuntimeError('Joint policy reconstruction mismatch')
            digest=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest();o=byid[item['observation_id']];key=(case,digest)
            if o['case']!=case or digest!=o['basis_digest'] or o['row']!=item['row'] or o['origin']!=item['evaluation_origin']:
                raise RuntimeError('Final CI not bound to exact proposal and case')
            if item['new_CI_solve']!=(key not in seen) or item['row']['actual_CI_dimension']!=len(pairs):raise RuntimeError('Solve/reuse count mismatch')
            seen.add(key);r=item['row']
        baseline=old[(case,p['method'])];hf,ref=reference(case);ncpu=s.get('native_attempt',{}).get('CPU_seconds')
        cpu=ncpu+r['CPU_seconds'] if r else None;delta=r['energy_Ha']-baseline['energy_Ha'] if r and baseline['energy_Ha'] is not None else None
        cost_delta=cpu-baseline['native_plus_CI_standalone_numeric_CPU'] if cpu is not None and baseline['native_plus_CI_standalone_numeric_CPU'] is not None else None
        rows.append(dict(case=case,method=p['method'],status='COMPLETED' if r else s['status'],
            energy_R8_Ha=r['energy_Ha'] if r else None,energy_R128_Ha=baseline['energy_Ha'],
            energy_R8_minus_R128_mHa=1000*delta if delta is not None else None,
            energy_direction=direction(delta) if delta is not None else 'MISSING_PAIR',
            penalized_R8_Ha=r['penalized_energy_Ha'] if r else None,spin_square_R8=r['spin_square'] if r else None,
            recovery_R8_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,recovery_R128_pct=baseline['recovery_pct'],
            actual_dimension_R8=r['actual_CI_dimension'] if r else None,actual_dimension_R128=baseline['actual_dimension'],
            initial_strings_R8=len(p['initial_strings']) if p['initial_strings'] is not None else None,initial_strings_R128=p['prior_initial_string_count'],
            native_CPU_R8_seconds=ncpu,CI_CPU_R8_seconds=r['CPU_seconds'] if r else None,
            native_plus_CI_CPU_R8=cpu,native_plus_CI_CPU_R128=baseline['native_plus_CI_standalone_numeric_CPU'],CPU_change=cost_delta,
            CNOT=baseline['CNOT'],CNOT_Depth=baseline['CNOT_Depth'],Total_Depth=baseline['Total_Depth'],
            coefficient_slots=baseline['coefficient_slots'],independent_parameter_count=None,
            source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),
            source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),
            evaluation_origin=item['evaluation_origin'] if item else None,actual_new_solve=item['new_CI_solve'] if item else None,
            task_id=p['task_id'],new_model_shots=0,new_QPU_shots=0,error=s.get('error') or s.get('errors',{}).get('JOINT_TOP_D16384',{}).get('error')))
    if len(seen)!=len(obs):raise RuntimeError('Unattributed saved root')
    if native_cpu+sum(a['CPU_seconds'] for a in attempts)>terminal['CPU_seconds']+1e-5:raise RuntimeError('Stage CPU exceeds campaign')
    classical={'CLASSICAL_HF_ONLY','CLASSICAL_CISD_100K_DRAWS','DETERMINISTIC_CISD_MARGINAL'};cases=[]
    for case in spec['physical']:
        good=[r for r in rows if r['case']==case and r['energy_R8_Ha'] is not None]
        classic=min((r for r in good if r['method'] in classical),key=lambda r:r['energy_R8_Ha'])
        quantum=min((r for r in good if r['method'] not in classical),key=lambda r:r['energy_R8_Ha'])
        cases.append(dict(case=case,best_classical_R8_method=classic['method'],best_quantum_R8_method=quantum['method'],
            quantum_minus_classical_R8_mHa=1000*(quantum['energy_R8_Ha']-classic['energy_R8_Ha']),
            classical_numeric_CPU=classic['native_plus_CI_CPU_R8'],quantum_numeric_CPU=quantum['native_plus_CI_CPU_R8'],
            not_total_cost_or_quantum_advantage=True))
    summary=dict(status='ALL28_SMALL_INITIAL_POOL_CELLS_AUDITED',rows=rows,cases=cases,
        completed=sum(r['status']=='COMPLETED' for r in rows),direction_counts=dict(Counter(r['energy_direction'] for r in rows)),
        actual_new_CI_roots=len(obs),native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),
        actual_unique_initial_spaces=len(initial_groups),
        shared_initial_spaces=[dict(case=case,strings=list(strings),methods=methods) for (case,strings),methods in initial_groups.items() if len(methods)>1],
        CPU_seconds=terminal['CPU_seconds'],wall_seconds=terminal['wall_seconds'],new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        original_shot_budget_not_reduced=True,independent_quantum_confirmation=False,goal_complete=False,
        preserved_transport_anomalies=transport_anomalies,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((c/'terminal.json').read_bytes()),
        R128_comparison_SHA256=sha256(previous_file.read_bytes()))
    save(ROOT/'analysis/N7-small-initial-pool-summary.json',summary)
    f=StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/N7-small-initial-pool-all28.csv',f.getvalue().encode('utf-8-sig'))
    names={'CLASSICAL_HF_ONLY':'古典HF-HCI','C5_C_LUCJ_COMPRESSED_L1':'圧縮LUCJ','C5_D_SQDRIFT_N20':'SqDRIFT20','RAACS_B4':'RAACS',
        'CLASSICAL_CISD_100K_DRAWS':'古典CISD draws','DETERMINISTIC_CISD_MARGINAL':'決定論的CISD','C5_E_LUCJ_TN_COMMON_L1':'限定TN/LUCJ'}
    def fmt(x,n=3):return '未取得/該当なし' if x is None else f'{x:.{n}f}'
    lines=['# N7：初期pool8で費用とsourceの差を調べた結果','',
        f"28予定セル中{summary['completed']}数値結果。比較{summary['direction_counts']}。native探索{summary['native_attempts']}、新CI根{len(obs)}。",
        f"CPU{summary['CPU_seconds']:.2f}秒、wall{summary['wall_seconds']:.2f}秒。新model/QPU Shot0、GPU割当0。",'',
        '変更したのはsourceから渡す初期pool上限128→8だけ（HF-onlyは1のまま）。HCIとjoint readoutの式・設定は同じ。元の量子Shot予算や回路CNOTが減った意味ではない。','']
    for case in spec['physical']:
        lines += ['## '+case,'','| source | 128 energy Ha | 8 energy Ha | 8−128 mHa（負が良い） | native＋CI CPU秒：128→8 | joint実次元：128→8 |','|---|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if r['case']==case]:
            lines.append('| '+' | '.join([names[r['method']],fmt(r['energy_R128_Ha'],10),fmt(r['energy_R8_Ha'],10),fmt(r['energy_R8_minus_R128_mHa'],6),
                fmt(r['native_plus_CI_CPU_R128'])+' → '+fmt(r['native_plus_CI_CPU_R8']),str(r['actual_dimension_R128'])+' → '+str(r['actual_dimension_R8'])])+' |')
        s=next(s for s in cases if s['case']==case)
        lines += ['',f"小pool内で最も低いenergyの古典sourceは{names[s['best_classical_R8_method']]}、量子sourceは{names[s['best_quantum_R8_method']]}。量子−古典の差は{s['quantum_minus_classical_R8_mHa']:+.6f}mHa。これはenergyの観測差で、全費用や独立確認を含む優越認定ではない。",'']
    lines += ['## 解釈の境界','',
        '- CPUはnative＋CIの単独数値proxy。source構築・モデルShot・cold初期化・全研究探索は別。速くなっても、精度が悪化すればtrade-offとして残す。',
        '- 元R128はN5/N6の既存対照。元の結果を書き換えていない。sourceごとのnative初期化は同一cutoff/上限、上流nativeRNGは明示seed固定ではない。',
        '- 初期poolが同じsource同士や同じ最終basisのcache再利用を、独立な発見/成功例と数えない。',
        '- 回路資源はN6の同一source比較から継承。SqDRIFTのcompiler資源は全bank認証由来の旧観測再利用で、独立サンプリングではない。',
        '- 最終CI状態をsamplerが準備した意味ではなく、古典行は0CNOTの相関状態準備ではない。独立Parameter rankも未確定。',
        '- 小poolで利益が無ければ、このmarginal pool-size調整の自動拡張はしない。source情報が全般に無価値という証明ではなく、marginal化でjoint情報を落とす別要因などは切り分けが必要。',
        '- 元論文最高の全設定、独立量子確認、実機routing/noise、cold総費用、publication上の新規性、研究goalの達成は未認定。','']
    write(ROOT/'analysis/N7_SMALL_INITIAL_POOL_RESULTS_JA.md','\n'.join(lines).encode())
    print(summary['status'],summary['completed'],summary['direction_counts'],flush=True)


if __name__=='__main__':run()
