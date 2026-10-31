"""Audit all N13 native inputs, membership, saved CI roots and actual costs."""
import csv,gzip,hashlib,math
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from .sample_constrained_study import LOCAL,PARENT,generate,plans_from_existing
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row
from .analyze import reference


def family(method):
    if method=='UNRESTRICTED_HF_HCI':return 'Unrestricted HF-HCI'
    if method=='CLASSICAL_HF_ONLY':return 'Sample-constrained HF'
    if method in ['DETERMINISTIC_CISD_MARGINAL','CLASSICAL_CISD_100K_DRAWS']:return 'CISD source'
    if method.startswith('RAACS'):return 'RAACS'
    if 'LUCJ_NAIVE' in method:return 'Naive LUCJ'
    if 'LUCJ_COMPRESSED' in method:return 'Compressed LUCJ'
    if 'SQDRIFT' in method:return 'SqDRIFT'
    if 'LUCJ_TN' in method:return 'Limited TN/LUCJ'
    raise ValueError('Unknown family')


def run():
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';t=load(c/'terminal.json');started=load(c/'started.json')
    protocol_sha=sha256((LOCAL/'protocol.json').read_bytes())
    if started['protocol_SHA256']!=protocol_sha or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:raise RuntimeError('Executed protocol/thread mismatch')
    if sha256(PARENT.read_bytes())!=spec['mechanical_parent_SHA256'] or sha256(generate(PARENT.read_text()).encode())!=spec['worker_SHA256'] or sha256((LOCAL/'worker.py').read_bytes())!=spec['worker_SHA256']:raise RuntimeError('Frozen executor changed')
    if sha256((ROOT/'frontierloop/sample_constrained_study.py').read_bytes())!=spec['preparer_SHA256'] or sha256((ROOT/'frontierloop/sample_universe_audit.py').read_bytes())!=spec['pool_builder_SHA256']:raise RuntimeError('Frozen input builder changed')
    priorfile=ROOT/'runs/N10-PAIRED-INITIAL-POOL-v1/protocol.json';qfile=ROOT/'analysis/N12-sample-constraint-qualification-summary.json'
    if sha256(priorfile.read_bytes())!=spec['source_protocol_SHA256'] or sha256(qfile.read_bytes())!=spec['qualification_SHA256']:raise RuntimeError('Prior evidence changed')
    q=load(qfile);prior=load(priorfile)
    if plans_from_existing(prior,spec['policy'],q['binary_SHA256'])!=spec['plans'] or spec['physical']!=prior['physical']:raise RuntimeError('Input plan cannot be reproduced')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((LOCAL/'pipeline'/name).read_bytes())!=value or sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Qualified CI helper changed')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Actual export startup qualifier failed')
    plans=spec['plans'];states={s['task_id']:s for s in t['states']}
    if len(plans)!=64 or len(t['states'])!=64 or set(states)!={p['task_id'] for p in plans}:raise RuntimeError('All64 terminal states required')
    obs=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in obs};completed=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(obs) or len(completed)!=len(obs) or len(obs)!=t['actual_new_CI_roots'] or len(attempts)!=t['ci_attempts'] or {a['observation_id'] for a in completed}!=set(byid):raise RuntimeError('CI actual work accounting mismatch')
    for o in obs:
        verify_row(o['row']);file=c/o['vector_file']
        if sha256(file.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved CI root changed')
        with np.load(file,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        if len(pairs)!=len(v) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:raise RuntimeError('Saved vector norm/shape mismatch')
        h=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest()
        if o['basis_digest']!=h or o['id']!=hashlib.sha256((o['case']+h+protocol_sha).encode()).hexdigest() or o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('CI kernel/problem/basis binding mismatch')
    bundle=load(LOCAL/'bundle-collection-v1.json');rows=[];seen=set();native_CPU=0.
    for p in plans:
        s=states[p['task_id']];case=p['case'];d=spec['physical'][case];source=p['source'];folder=c/'cells'/p['task_id'];pool=p['allowed_strings']
        if load(folder/'result.json')!=s or s['case']!=case or s['method']!=p['method'] or s['source']!=source or s['selection_mode']!=p['selection_mode']:raise RuntimeError('Actual state/plan mismatch')
        if source['status']!='AVAILABLE' and (s['status']!='MISSING_SAME_PROBLEM_SOURCE' or s['arms']):raise RuntimeError('Missing source fabricated')
        if s.get('native_attempt'):
            native_CPU+=s['native_attempt']['CPU_seconds']
            if load(folder/'native-attempt.json')!=s['native_attempt']:raise RuntimeError('Native attempt mismatch')
            for spin in ['Alpha','Beta']:
                if (folder/(spin+'Dets.bin')).read_bytes()!=int(p['HF_string']).to_bytes(16,'big'):raise RuntimeError('Actual start is not HF')
            if pool is not None and (folder/'AllowedHalfDets.bin').read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in pool):raise RuntimeError('Actual pool bytes differ')
        item=s['arms'].get('JOINT_TOP_D16384');r=None;native=None;selection=None;final=None
        if item:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());h=sha256(raw)
            if h!=s['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+p['task_id']+'/dets.bin')!=h:raise RuntimeError('Native raw proof mismatch')
            n,native=decode(raw);final,selection=joint_pairs(native,n,tuple(d['nelec']),p['HF_string'])
            if n!=d['norb'] or selection!=s['selection']:raise RuntimeError('Native sector/norm/final policy mismatch')
            energy=(folder/'shci.e').read_bytes()
            import struct
            if len(energy)!=8 or struct.unpack('<d',energy)[0]!=s['native']['native_diagnostic_spinfree_energy_Ha'] or not math.isfinite(s['native']['native_diagnostic_spinfree_energy_Ha']):raise RuntimeError('Native diagnostic energy mismatch')
            if pool is not None:
                domain=set(pool)
                if [int(x) for x in (folder/'CEO-sample-pool-v1.txt').read_text().split()]!=pool or any(a not in domain or b not in domain for a,b,_ in native) or any(a not in domain or b not in domain for a,b in final):raise RuntimeError('Candidate/final space outside observed pool')
                if s['native']['allowed_pool_SHA256']!=sha256((folder/'AllowedHalfDets.bin').read_bytes()) or s['native']['allowed_pool_size']!=len(pool) or 'CEO_SAMPLE_POOL_V1 ' not in (folder/'native.log').read_text():raise RuntimeError('Actual candidate filter evidence missing')
            elif (folder/'CEO-sample-pool-v1.txt').exists():raise RuntimeError('Restriction leaked into unrestricted control')
            h=hashlib.sha256(np.ascontiguousarray(final,dtype=np.uint64).tobytes()).hexdigest();o=byid[item['observation_id']];key=(case,h)
            if o['case']!=case or o['basis_digest']!=h or item['row']!=o['row'] or item['evaluation_origin']!=o['origin'] or item['new_CI_solve']!=(key not in seen) or item['row']['actual_CI_dimension']!=len(final):raise RuntimeError('Final observation reuse/basis mismatch')
            seen.add(key);r=item['row']
        hf,ref=reference(case);rf=OLD/'data/evaluation-only'/(case+'.json');rm=load(rf)
        if rm['problem_id']!=d['problem_id']:raise RuntimeError('Reference physical mismatch')
        resource=source.get('sampler_resources') or {};ncpu=s.get('native_attempt',{}).get('CPU_seconds')
        final_set=set(final) if final is not None else None
        rows.append(dict(case=case,method=p['method'],family=family(p['method']),mode=p['selection_mode'],status='COMPLETED' if r else s['status'],
            energy_Ha=r['energy_Ha'] if r else None,penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,spin_square=r['spin_square'] if r else None,
            native_spinfree_energy_Ha=s.get('native',{}).get('native_diagnostic_spinfree_energy_Ha'),
            reference_gap_mHa=1000*(r['energy_Ha']-ref) if r else None,recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,
            recovery_reference_kind='FCI_ARCHIVE' if rm['energies_Ha'].get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
            recovery_reference_precision_status='VERIFIED' if rm.get('sector_and_reference_precision_verified') is True else 'NOT_VERIFIED',reference_SHA256=sha256(rf.read_bytes()),
            observed_half_pool_size=len(pool) if pool is not None else None,potential_sample_universe_dimension=len(pool)**2 if pool is not None else None,
            native_dimension=len(native) if native is not None else None,final_dimension=r['actual_CI_dimension'] if r else None,final_D_cap=16384,
            final_basis_spin_flip_closed=all((b,a) in final_set for a,b in final) if final is not None else None,
            native_CPU_seconds=ncpu,CI_CPU_seconds=r['CPU_seconds'] if r else None,native_plus_CI_standalone_numeric_CPU=ncpu+r['CPU_seconds'] if r else None,
            actual_new_CI_solve=item['new_CI_solve'] if item else None,
            CNOT=resource.get('cnot_count'),CNOT_Depth=resource.get('cnot_depth'),Total_Depth=resource.get('total_unitary_depth',resource.get('total_depth')),
            coefficient_slots=source.get('sampler_coefficient_slots_mean'),independent_parameter_count=None,
            source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),source_classical_draws=source.get('classical_draws'),
            source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),task_id=p['task_id'],new_model_shots=0,new_QPU_shots=0,
            error=s.get('error') or s.get('errors',{}).get('JOINT_TOP_D16384',{}).get('error')))
    if len(seen)!=len(obs) or native_CPU+sum(a['CPU_seconds'] for a in attempts)>t['CPU_seconds']+1e-5:raise RuntimeError('Work unattributed/double counted')
    cases=[];best_by_family=[]
    for case in spec['physical']:
        allcase=[r for r in rows if r['case']==case]
        if len(allcase)!=16:raise RuntimeError('All16 source/control states per case required')
        control=next(r for r in allcase if r['method']=='UNRESTRICTED_HF_HCI')
        for r in allcase:r['energy_minus_unrestricted_HF_mHa']=1000*(r['energy_Ha']-control['energy_Ha']) if r['energy_Ha'] is not None and control['energy_Ha'] is not None else None
        for f in sorted({r['family'] for r in allcase}):
            good=[r for r in allcase if r['family']==f and r['energy_Ha'] is not None]
            if good:best_by_family.append(dict(min(good,key=lambda r:r['energy_Ha']),selection_scope='RETROSPECTIVE_BEST_OBSERVED_VARIANT_NOT_FIXED_POLICY_CONFIRMATION'))
        quantum=[r for r in allcase if r['energy_Ha'] is not None and r['family'] in ['RAACS','Naive LUCJ','Compressed LUCJ','SqDRIFT','Limited TN/LUCJ']]
        classical=[r for r in allcase if r['energy_Ha'] is not None and r['family'] in ['CISD source','Sample-constrained HF','Unrestricted HF-HCI']]
        if quantum and classical:
            qb=min(quantum,key=lambda r:r['energy_Ha']);cb=min(classical,key=lambda r:r['energy_Ha'])
            cases.append(dict(case=case,best_quantum_method=qb['method'],best_classical_method=cb['method'],
                quantum_minus_classical_mHa=1000*(qb['energy_Ha']-cb['energy_Ha']),quantum_gap_mHa=qb['reference_gap_mHa'],classical_gap_mHa=cb['reference_gap_mHa'],
                quantum_numeric_CPU=qb['native_plus_CI_standalone_numeric_CPU'],classical_numeric_CPU=cb['native_plus_CI_standalone_numeric_CPU']))
    result=dict(status='ALL64_SAMPLE_CONSTRAINED_SOURCE_AND_UNRESTRICTED_HF_STATES_AUDITED',rows=rows,best_by_family=best_by_family,cases=cases,
        status_counts=dict(Counter(r['status'] for r in rows)),completed=sum(r['energy_Ha'] is not None for r in rows),actual_new_CI_roots=len(obs),
        native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],actual_stage_CPU_seconds=native_CPU+sum(a['CPU_seconds'] for a in attempts),
        same_max_D_not_equal_actual_D=True,actual_work_not_matched=True,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        circuit_resource_change=0,independent_quantum_confirmation=False,independent_RDM_for_each_molecular_root=False,full_paper_reproduction=False,
        final_CI_state_not_prepared_by_sampler=True,goal_complete=False,protocol_SHA256=protocol_sha,terminal_SHA256=sha256((c/'terminal.json').read_bytes()),
        analyzer_SHA256=sha256(Path(__file__).read_bytes()))
    save(ROOT/'analysis/N13-sample-constrained-summary.json',result)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N13-all64-sample-constrained.csv',stream.getvalue().encode('utf-8-sig'))
    def fmt(x,n=3):return '未取得/該当なし' if x is None else f'{x:.{n}f}'
    lines=['# N13：観測配置内の選抜を全sourceへ公平に適用した結果','',
        f"64予定：{result['status_counts']}。native {result['native_attempts']}件、新CI根{len(obs)}、CPU{t['CPU_seconds']:.3f}秒/wall{t['wall_seconds']:.3f}秒。新Shot/GPU0。",'',
        '全てHF開始。同じ閾値5e-4/10iteration/最終D上限16384で、実次元と実CPUは異なる。sample制限は原著の既知成分であり新規発明ではない。元sampler回路は変更していない。']
    for case in spec['physical']:
        lines+=['','## '+case,'','| 手法群内で最も低いenergyの実測例 | error mHa | Recovery % | CNOT | CNOT Depth | Total Depth | 係数slot | CI実次元 | native＋CI CPU秒 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in best_by_family if r['case']==case]:
            lines.append('| '+' | '.join([r['family']+' / '+r['method'],fmt(r['reference_gap_mHa']),fmt(r['recovery_pct']),fmt(r['CNOT']),fmt(r['CNOT_Depth']),fmt(r['Total_Depth']),fmt(r['coefficient_slots']),str(r['final_dimension']),fmt(r['native_plus_CI_standalone_numeric_CPU'])])+' |')
        s=next((x for x in cases if x['case']==case),None)
        if s:lines+=['',f"量子source群で最も低いenergyは{s['best_quantum_method']}、古典群は{s['best_classical_method']}。量子−古典は{s['quantum_minus_classical_mHa']:+.6f} mHa（負なら量子source側が低い）。これは同一保存観測を使う開発比較で、独立確認でも全費用をそろえた優越性でもない。"]
        lines+=['','### 全16予定行（不利な結果・欠測を含む）','','| method | 状態 | energy Ha | S² | pool数 | native D | final D | HF制限なしとの差 mHa | CPU秒 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if r['case']==case]:lines.append('| '+' | '.join([r['method'],r['status'],fmt(r['energy_Ha'],10),fmt(r['spin_square'],6),str(r['observed_half_pool_size']),str(r['native_dimension']),str(r['final_dimension']),fmt(r['energy_minus_unrestricted_HF_mHa'],6),fmt(r['native_plus_CI_standalone_numeric_CPU'])])+' |')
    lines+=['','## 読み方・限界','',
        '- 群内bestは探索後の記述で、固定法の独立性能認定ではない。N₂の2距離や同じsourceの多数variantを独立分子数へ水増ししない。',
        '- Recovery/errorは保存reference相対。FeはSCI archiveでFCIではなく、sector/precision未検証。原著値と絶対値を直接競わせない。',
        '- HCIのnative spinfree energyと最終H+0.1S²で求めたphysical energyは異なる。spin交換閉包も各basisへ記録し、exact S=0は主張しない。',
        '- 今回の各分子rootは資格済みH/S²行列・保存vector・residualで確認した。全rootの独立RDM再検証とは呼ばない。',
        '- sampler CNOT/Depth/係数slotは元sourceから継承。全sourceのcompiler処理が同一ではない。独立Parameter rankはunknown、古典行の回路資源は該当なし。最終CI状態を0CNOTで準備できる意味ではない。',
        '- CPUはnative＋CIの単独数値proxy。元初期化/Shot/探索費用は別途必要。cache再利用を新評価へ重複計上せず、campaign費用は一度だけ報告。全cold費用や完全matched-workではない。',
        '- 元の量子観測を再利用したため、新しい独立Shotや実機検証は0。最強論文の完全再現、hardware優位、総合goal達成は未認定。','']
    write(ROOT/'analysis/N13_SAMPLE_CONSTRAINED_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],result['status_counts'],'CPU',t['CPU_seconds'],flush=True)


if __name__=='__main__':run()
