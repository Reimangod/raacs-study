"""All N10 endpoints, equal-size paired contrasts, saved-state/cost audits."""
import csv,gzip,hashlib
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from .paired_pool_study import LOCAL,PARENT
from .paired_pool_plan import source_pairs
from .paired_pool_executor import generate
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row,direction
from .analyze import reference


def run():
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';t=load(c/'terminal.json');started=load(c/'started.json')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Actual protocol/thread binding mismatch')
    if sha256(PARENT.read_bytes())!=spec['mechanical_parent_SHA256'] or sha256(generate(PARENT.read_text()).encode())!=spec['worker_SHA256'] or sha256((LOCAL/'worker.py').read_bytes())!=spec['worker_SHA256']:
        raise RuntimeError('Frozen generated executor mismatch')
    for name,key in [('paired_pool_plan.py','input_rule_SHA256'),('paired_pool_executor.py','executor_generator_SHA256')]:
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=spec[key]:raise RuntimeError('Frozen input generation changed')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified CI helper changed')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Default tiny startup parity failed')
    plans=spec['plans'];states={s['task_id']:s for s in t['states']}
    if len(plans)!=120 or len(t['states'])!=120 or set(states)!={p['task_id'] for p in plans}:raise RuntimeError('All120 planned arm states required')
    obs=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in obs}
    complete=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(obs) or len(complete)!=len(obs) or len(obs)!=t['actual_new_CI_roots'] or {a['observation_id'] for a in complete}!=set(byid) or len(attempts)!=t['ci_attempts']:
        raise RuntimeError('Actual solve accounting mismatch')
    for o in obs:
        verify_row(o['row']);f=c/o['vector_file']
        if sha256(f.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved CI vector changed')
        with np.load(f,allow_pickle=False) as a:ps=a['joint_pairs'];v=a['coefficients']
        if len(ps)!=len(v) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:raise RuntimeError('Saved CI vector norm/shape mismatch')
        d=hashlib.sha256(np.ascontiguousarray(ps,dtype=np.uint64).tobytes()).hexdigest()
        if d!=o['basis_digest'] or o['id']!=hashlib.sha256((o['case']+d+started['protocol_SHA256']).encode()).hexdigest():raise RuntimeError('CI state/problem identity mismatch')
        if o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Wrong sparse kernel')
    bundle=load(LOCAL/'bundle-collection-v1.json');seen=set();rows=[];native_cpu=0;grouped={}
    for p in plans:
        s=states[p['task_id']];source=p['source'];case=p['case'];d=spec['physical'][case];folder=c/'cells'/p['task_id']
        if load(folder/'result.json')!=s or s['source']!=source or s['method']!=p['method'] or s['initialization_arm']!=p['initialization_arm']:
            raise RuntimeError('Actual cell/source/arm mismatch')
        if source['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if name in source and sha256(Path(source[name]).read_bytes())!=source[key]:raise RuntimeError('Original source drift')
            strings,product,paired,info=source_pairs(source,d['norb'],tuple(d['nelec']))
            expected=product if p['initialization_arm']=='PRODUCT_R8' else paired
            if strings!=p['initial_strings'] or info!=p['selection_diagnostics'] or [list(x) for x in expected]!=p['expected_initial_pairs']:
                raise RuntimeError('Outcome-blind input selection not reconstructed')
            if len(product)!=len(paired) or len(expected)!=p['initial_joint_dimension']:raise RuntimeError('Initial dimension not equal')
            if s.get('native_attempt'):
                for spin,index in [('Alpha',0),('Beta',1)]:
                    values=strings if p['initialization_arm']=='PRODUCT_R8' else [q[index] for q in paired]
                    if (folder/(spin+'Dets.bin')).read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in values):raise RuntimeError('Actual native input bytes differ')
                audit=folder/'CEO-initial-pairs-v1.txt'
                if p['initialization_arm']=='OBSERVED_JOINT_EQUAL_D' and s.get('native'):
                    if [list(map(int,l.split())) for l in audit.read_text().splitlines()]!=p['expected_initial_pairs']:raise RuntimeError('Actual native paired domain differs')
                if p['initialization_arm']=='PRODUCT_R8' and audit.exists():raise RuntimeError('Paired mode leaked into control')
        elif s['status']!='MISSING_SAME_PROBLEM_SOURCE' or s['arms']:raise RuntimeError('Missing source fabricated')
        if s.get('native_attempt'):native_cpu+=s['native_attempt']['CPU_seconds']
        item=s['arms'].get('JOINT_TOP_D16384');r=None
        if item:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());h=sha256(raw)
            if h!=s['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+p['task_id']+'/dets.bin')!=h:raise RuntimeError('Native export digest mismatch')
            n,pp=decode(raw);final,selection=joint_pairs(pp,n,tuple(d['nelec']),p['HF_string'])
            if n!=d['norb'] or selection!=s['selection']:raise RuntimeError('Final-space derivation differs')
            h=hashlib.sha256(np.ascontiguousarray(final,dtype=np.uint64).tobytes()).hexdigest();o=byid[item['observation_id']];key=(case,h)
            if o['case']!=case or o['basis_digest']!=h or item['row']!=o['row'] or item['evaluation_origin']!=o['origin']:raise RuntimeError('CI result not bound to actual selected space')
            if item['new_CI_solve']!=(key not in seen) or item['row']['actual_CI_dimension']!=len(final):raise RuntimeError('CI reuse/dimension mismatch')
            seen.add(key);r=item['row']
        hf,ref=reference(case);ncpu=s.get('native_attempt',{}).get('CPU_seconds');res=source.get('sampler_resources') or {}
        rf=OLD/'data/evaluation-only'/(case+'.json');rm=load(rf)
        if rm['problem_id']!=d['problem_id']:raise RuntimeError('Recovery reference has a different physical problem')
        row=dict(case=case,method=p['method'],arm=p['initialization_arm'],status='COMPLETED' if r else s['status'],
            energy_Ha=r['energy_Ha'] if r else None,penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,
            recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,spin_square=r['spin_square'] if r else None,
            recovery_reference_kind='FCI_ARCHIVE' if rm['energies_Ha'].get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
            recovery_reference_precision_status='VERIFIED' if rm.get('sector_and_reference_precision_verified') is True else 'NOT_VERIFIED',
            recovery_reference_SHA256=sha256(rf.read_bytes()),reference_relative_gap_mHa=1000*(r['energy_Ha']-ref) if r else None,
            initial_dimension=p['initial_joint_dimension'],final_dimension=r['actual_CI_dimension'] if r else None,final_D_cap=16384,
            native_CPU_seconds=ncpu,CI_CPU_seconds=r['CPU_seconds'] if r else None,
            native_plus_CI_standalone_numeric_CPU=ncpu+r['CPU_seconds'] if r else None,
            CNOT=res.get('cnot_count'),CNOT_Depth=res.get('cnot_depth'),Total_Depth=res.get('total_unitary_depth',res.get('total_depth')),
            coefficient_slots=source.get('sampler_coefficient_slots_mean'),independent_parameter_count=None,
            source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),
            source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),
            same_initial_sets=(p['selection_diagnostics'] or {}).get('equal_product_and_joint_spaces'),
            paired_initial_spin_flip_closed=(p['selection_diagnostics'] or {}).get('joint_spin_flip_closed'),
            actual_new_solve=item['new_CI_solve'] if item else None,task_id=p['task_id'],new_model_shots=0,new_QPU_shots=0,
            error=s.get('error') or s.get('errors',{}).get('JOINT_TOP_D16384',{}).get('error'))
        rows.append(row);grouped.setdefault((case,p['method']),{})[p['initialization_arm']]=row
    if len(seen)!=len(obs) or native_cpu+sum(a['CPU_seconds'] for a in attempts)>t['CPU_seconds']+1e-5:raise RuntimeError('Unattributed/double-counted work')
    contrasts=[]
    for (case,method),arms in grouped.items():
        a,b=arms['PRODUCT_R8'],arms['OBSERVED_JOINT_EQUAL_D'];good=a['energy_Ha'] is not None and b['energy_Ha'] is not None
        delta=b['energy_Ha']-a['energy_Ha'] if good else None
        contrasts.append(dict(case=case,method=method,status=direction(delta) if good else 'MISSING_PAIR',
            energy_paired_minus_product_mHa=1000*delta if good else None,
            penalized_paired_minus_product_mHa=1000*(b['penalized_energy_Ha']-a['penalized_energy_Ha']) if good else None,
            numeric_CPU_paired_minus_product=b['native_plus_CI_standalone_numeric_CPU']-a['native_plus_CI_standalone_numeric_CPU'] if good else None,
            same_initial_sets=a['same_initial_sets'],not_independent_quantum_confirmation=True))
    if len(contrasts)!=60:raise RuntimeError('All60 within-source contrasts required')
    classical={'CLASSICAL_HF_ONLY','DETERMINISTIC_CISD_MARGINAL','CLASSICAL_CISD_100K_DRAWS'};best=[]
    for case in spec['physical']:
        for arm in ['PRODUCT_R8','OBSERVED_JOINT_EQUAL_D']:
            good=[r for r in rows if r['case']==case and r['arm']==arm and r['energy_Ha'] is not None]
            q=[r for r in good if r['method'] not in classical];cl=[r for r in good if r['method'] in classical]
            if not q or not cl:
                best.append(dict(case=case,arm=arm,status='MISSING_GROUP'));continue
            qb=min(q,key=lambda r:r['energy_Ha']);cb=min(cl,key=lambda r:r['energy_Ha'])
            best.append(dict(case=case,arm=arm,status='DEVELOPMENT_PHYSICAL_ENERGY_COMPARISON_NOT_SPIN_CONDITIONED',
                best_quantum_method=qb['method'],best_classical_method=cb['method'],
                quantum_minus_classical_mHa=1000*(qb['energy_Ha']-cb['energy_Ha']),
                quantum_spin_square=qb['spin_square'],classical_spin_square=cb['spin_square'],
                quantum_numeric_CPU=qb['native_plus_CI_standalone_numeric_CPU'],classical_numeric_CPU=cb['native_plus_CI_standalone_numeric_CPU']))
    summary=dict(status='ALL120_ARM_STATES_AND60_SOURCE_CONTRASTS_AUDITED',rows=rows,contrasts=contrasts,
        completed=sum(r['energy_Ha'] is not None for r in rows),status_counts=dict(Counter(r['status'] for r in rows)),
        contrast_counts=dict(Counter(r['status'] for r in contrasts)),best_source_comparisons=best,
        actual_new_CI_roots=len(obs),actual_CI_attempts=len(attempts),native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        circuit_resource_change=0,independent_quantum_confirmation=False,independent_RDM_for_each_joint_root=False,
        final_CI_state_not_prepared_by_sampler=True,full_cold_cost_reconstructed=False,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((c/'terminal.json').read_bytes()),
        analyzer_SHA256=sha256(Path(__file__).read_bytes()))
    save(ROOT/'analysis/N10-paired-initial-pool-summary.json',summary)
    for name,rs in [('N10-all120-arm-results.csv',rows),('N10-all60-source-contrasts.csv',contrasts)]:
        f=StringIO();w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs);write(ROOT/'analysis'/name,f.getvalue().encode('utf-8-sig'))
    lines=['# N10：初期ペア情報の全source比較','',f"120予定arm：{summary['status_counts']}。60source対比：{summary['contrast_counts']}。",
        f"native {summary['native_attempts']}件、新CI根{len(obs)}、CPU{t['CPU_seconds']:.3f}秒/wall{t['wall_seconds']:.3f}秒。新Shot/GPU割当0。",'',
        '同じsource・初期実次元・native cutoff/iteration・最終joint上限をそろえた。実際の最終次元とCPUは変わる。元sourceの構築/測定費用を除いたnumeric CPUを完全総費用やMeasurement Costに読み替えない。',
        'sampler回路は変えていないためCNOT/Depth/Parametersの新削減ではない。最終CI状態をsamplerが直接準備したわけでもない。有限S²罰則でありexact singlet保証はなく、energyと罰則energyの両方をCSVに保持した。',
        'N9のtoy RDM資格を各分子CI根の独立RDM検査へ拡張解釈しない。分子根は資格済みH/S²疎行列と保存vector/residualで確認した。','',
        '| 条件 | pairedで低下 | 上昇 | 同等 | 欠測対 |','|---|---:|---:|---:|---:|']
    for case in spec['physical']:
        counts=Counter(r['status'] for r in contrasts if r['case']==case)
        if sum(counts.values())!=15 or sum(counts[k] for k in ['LOWER','HIGHER','EQUAL_WITHIN_TOLERANCE','MISSING_PAIR'])!=15:
            raise RuntimeError('Per-case table must account for all15 sources, including ties and missingness')
        lines.append(f"| {case} | {counts['LOWER']} | {counts['HIGHER']} | {counts['EQUAL_WITHIN_TOLERANCE']} | {counts['MISSING_PAIR']} |")
    lines+=['','これは同じ保存観測の開発診断で、独立量子確認でも原著最高設定の再現でもない。古典CISD/HFにも同じpaired機会を与えた。良い点だけを選んで総合優位と呼ばず、全120armと60対比を参照する。']
    lines+=['','因果帰属の限界：同じ初期次元でも、paired集合は8種類を超えるα/β stringsを含み得る。従って、改善があっても純粋なco-occurrence情報だけの効果とは断定できず、marginal coverageやspin-flip閉包の変化を含む初期化パッケージの比較である。N₂の2距離も同じ分子であり、source variantの53対を53個の独立分子標本に数えない。']
    lines+=['','Recoveryは同じ物理問題の保存referenceに対する相関energy回収率。Fe₂S₂はSCI referenceでFCIではなく、archive metadataではsector/precision未検証。数値の比は計算できても、厳密なground-state-quality証明に使わない。CSVへreference種別・検証状態・hashを付けた。']
    write(ROOT/'analysis/N10_PAIRED_POOL_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(summary['status'],summary['status_counts'],summary['contrast_counts'],flush=True)


if __name__=='__main__':run()
