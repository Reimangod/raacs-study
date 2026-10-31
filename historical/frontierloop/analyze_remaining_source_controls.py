"""Audit all new N8 outcomes, then join the complete existing source inventory."""
import csv,gzip,hashlib,math
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .remaining_source_controls import LOCAL,WORKER,choose_sources
from .native_source_control_plan import initial_strings
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row
from .analyze import reference


def run():
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';t=load(c/'terminal.json');started=load(c/'started.json')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Executed protocol/thread mismatch')
    if sha256(WORKER.read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Reused executor drift')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified helper drift')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Native tiny parity missing')
    prevfile=ROOT/'analysis/N6-combined-joint-comparison.json'
    if sha256(prevfile.read_bytes())!=spec['prior_28_comparison_SHA256']:raise RuntimeError('Prior comparison changed')
    previous=load(prevfile)['rows'];r2file=ROOT/'runs/R2-COMMON-CI-v1/freeze.json'
    if sha256(r2file.read_bytes())!=spec['R2_source_freeze_SHA256']:raise RuntimeError('Original source manifest changed')
    choose_sources(load(r2file),{(r['case'],r['method']) for r in previous},spec['physical'])
    plans=spec['plans'];states={s['task_id']:s for s in t['states']}
    if len(plans)!=32 or len(t['states'])!=32 or set(states)!={p['task_id'] for p in plans}:raise RuntimeError('All32 outcomes including missing required')
    obs=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in obs}
    complete=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(obs) or len(complete)!=len(obs) or len(obs)!=t['actual_new_CI_roots'] or {a['observation_id'] for a in complete}!=set(byid) or len(attempts)!=t['ci_attempts']:
        raise RuntimeError('Actual attempt/solve accounting mismatch')
    for o in obs:
        verify_row(o['row']);f=c/o['vector_file']
        if sha256(f.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved CI vector changed')
        with np.load(f,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        if len(v)!=len(pairs) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:raise RuntimeError('CI normalization/shape mismatch')
        d=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest()
        if d!=o['basis_digest'] or o['id']!=hashlib.sha256((o['case']+d+started['protocol_SHA256']).encode()).hexdigest():raise RuntimeError('CI identity mismatch')
        if o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Wrong sparse kernel')
    bundle=load(LOCAL/'bundle-collection-v1.json');rows=[];seen=set();native_cpu=0
    for p in plans:
        s=states[p['task_id']];source=p['source'];case=p['case'];definition=spec['physical'][case];folder=c/'cells'/p['task_id']
        if load(folder/'result.json')!=s or s['source']!=source or s['method']!=p['method'] or s['case']!=case:raise RuntimeError('Cell/source mismatch')
        if source['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if sha256(Path(source[name]).read_bytes())!=source[key]:raise RuntimeError('Source drift')
            strings=initial_strings(source,definition['norb'],tuple(definition['nelec']))
            if strings!=p['initial_strings']:raise RuntimeError('Not registered R128 ranking')
            for spin in ['Alpha','Beta']:
                if (folder/(spin+'Dets.bin')).read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in strings):raise RuntimeError('Actual native input mismatch')
        elif s['status']!='MISSING_SAME_PROBLEM_SOURCE' or s['arms']:raise RuntimeError('Fabricated missing source')
        if s.get('native_attempt'):native_cpu+=s['native_attempt']['CPU_seconds']
        item=s['arms'].get('JOINT_TOP_D16384');r=None
        if item:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());d=sha256(raw)
            if d!=s['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+p['task_id']+'/dets.bin')!=d:raise RuntimeError('Native raw evidence mismatch')
            n,native=decode(raw);pairs,selection=joint_pairs(native,n,tuple(definition['nelec']),p['HF_string'])
            if n!=definition['norb'] or selection!=s['selection']:raise RuntimeError('Joint selection drift')
            d=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest();o=byid[item['observation_id']];key=(case,d)
            if o['case']!=case or o['basis_digest']!=d or item['row']!=o['row'] or item['evaluation_origin']!=o['origin']:raise RuntimeError('Result not bound to actual selected space')
            if item['new_CI_solve']!=(key not in seen) or o['row']['actual_CI_dimension']!=len(pairs):raise RuntimeError('Solve/reuse mismatch')
            seen.add(key);r=item['row']
        hf,ref=reference(case);res=source.get('sampler_resources') or {};ncpu=s.get('native_attempt',{}).get('CPU_seconds')
        rows.append(dict(case=case,method=p['method'],stage='N8',status='COMPLETED' if r else s['status'],
            energy_Ha=r['energy_Ha'] if r else None,penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,
            recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,spin_square=r['spin_square'] if r else None,
            actual_dimension=r['actual_CI_dimension'] if r else None,dimension_cap=16384,initial_marginal_R_cap=128,
            native_CPU_seconds=ncpu,CI_CPU_seconds=r['CPU_seconds'] if r else None,
            native_plus_CI_standalone_numeric_CPU=ncpu+r['CPU_seconds'] if r else None,
            CNOT=res.get('cnot_count'),CNOT_Depth=res.get('cnot_depth'),Total_Depth=res.get('total_unitary_depth',res.get('total_depth')),
            coefficient_slots=source.get('sampler_coefficient_slots_mean'),independent_parameter_count=None,
            source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),
            source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),
            new_model_shots=0,new_QPU_shots=0,task_id=p['task_id'],
            actual_new_solve=item['new_CI_solve'] if item else None,
            error=s.get('error') or s.get('errors',{}).get('JOINT_TOP_D16384',{}).get('error')))
    if len(seen)!=len(obs) or native_cpu+sum(a['CPU_seconds'] for a in attempts)>t['CPU_seconds']+1e-5:raise RuntimeError('Unattributed/double-counted work')
    combined=previous+rows
    if len(combined)!=60 or len({(r['case'],r['method']) for r in combined})!=60:raise RuntimeError('All60 source inventory required')
    # The prior exact-source N20 join includes G5's qualified compiler. No such
    # whole-bank compiler pass was executed here for N50 or new LUCJ sources.
    # Make that asymmetry explicit rather than imply compiler-matched resources.
    for r in combined:
        r['resource_treatment']=('G5_WHOLE_BANK_PROBABILITY_CERTIFIED_LOGICAL_OPT1'
            if r['method']=='C5_D_SQDRIFT_N20' else
            'ORIGINAL_SOURCE_REGISTERED_LOGICAL_RESOURCES_NO_NEW_COMPILER_PASS')
        r['initial_marginal_R_cap']=128
        r['final_joint_dimension_cap']=16384
    classical={'CLASSICAL_HF_ONLY','CLASSICAL_CISD_100K_DRAWS','DETERMINISTIC_CISD_MARGINAL'};contrasts=[]
    for case in spec['physical']:
        good=[r for r in combined if r['case']==case and r['energy_Ha'] is not None]
        cb=min((r for r in good if r['method'] in classical),key=lambda r:r['energy_Ha'])
        qb=min((r for r in good if r['method'] not in classical),key=lambda r:r['energy_Ha'])
        contrasts.append(dict(case=case,best_classical=cb['method'],best_quantum=qb['method'],
            classical_energy_Ha=cb['energy_Ha'],quantum_energy_Ha=qb['energy_Ha'],
            quantum_minus_classical_mHa=1000*(qb['energy_Ha']-cb['energy_Ha']),not_total_cost_or_independent_confirmation=True))
    summary=dict(status='ALL32_NEW_AND60_COMBINED_SOURCE_CELLS_AUDITED',rows=rows,combined=combined,cases=contrasts,
        completed_new=sum(r['energy_Ha'] is not None for r in rows),combined_status_counts=dict(Counter(r['status'] for r in combined)),
        actual_new_CI_roots=len(obs),actual_CI_attempts=len(attempts),native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        independent_quantum_confirmation=False,goal_complete=False,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        terminal_SHA256=sha256((c/'terminal.json').read_bytes()),prior_comparison_SHA256=sha256(prevfile.read_bytes()))
    save(ROOT/'analysis/N8-remaining-source-controls-summary.json',summary)
    fields=list(dict.fromkeys(k for r in combined for k in r));stream=StringIO();w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(combined)
    write(ROOT/'analysis/N8-all60-source-comparison.csv',stream.getvalue().encode('utf-8-sig'))
    lines=['# N8：全ての既存source対照を同じ後処理まで補完','',
        f"新規32予定のうち数値{summary['completed_new']}件。統合60予定：{summary['combined_status_counts']}。",
        f"新規native {summary['native_attempts']}件、CI根{len(obs)}、CPU {t['CPU_seconds']:.2f}秒、wall {t['wall_seconds']:.2f}秒。追加Shot/GPU割当0。",'',
        '同じ物理問題・初期R128・最終joint上限16384・H＋0.1S²で比較した。実次元・古典探索量は同一とは限らない。既存sourceのShot/構築費用は別途必要。N7のR8を都合よく混ぜていない。',
        '回路資源は全方式の新compiler-matched比較ではない。既存N20だけはG5の全bank確率証明済みopt1資源、他は元sourceの登録値。CSVにresource_treatmentを明記した。原著最高コンパイラやrouting条件の勝利へ読み替えない。','',
        '| 条件 | 最低energyの量子source | 量子energy Ha | 最良古典source | 古典energy Ha | 量子−古典 mHa |',
        '|---|---|---:|---|---:|---:|']
    for x in contrasts:lines.append(f"| {x['case']} | {x['best_quantum']} | {x['quantum_energy_Ha']:.10f} | {x['best_classical']} | {x['classical_energy_Ha']:.10f} | {x['quantum_minus_classical_mHa']:.6f} |")
    lines+=['','負ならこの開発比較では量子source由来CIが低い。正なら古典sourceが低い。CI最終状態をsamplerが直接準備したとは言わない。回路・費用とのtrade-offを含む総合優越や原著最高設定の再現は未証明。',
        '独立Parameter数はunknown。係数slot数を独立自由度に読み替えない。入力欠測と計算不成功は残し、全結果はCSVに保存した。']
    write(ROOT/'analysis/N8_REMAINING_SOURCE_CONTROLS_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(summary['status'],summary['completed_new'],summary['combined_status_counts'],flush=True)


if __name__=='__main__':run()
