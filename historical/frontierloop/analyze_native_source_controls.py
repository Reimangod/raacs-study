"""All N6 cells and combined source comparison; no reference-driven selection."""
import csv,gzip,hashlib,math
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .native_source_controls import LOCAL,N5
from .native_source_control_plan import initial_strings
from .native_dice_binary import decode
from .native_hci_plan import marginal_strings
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row,direction
from .analyze import reference


def run():
    spec=load(LOCAL/'protocol.json');collected=LOCAL/'collected';terminal=load(collected/'terminal.json')
    started=load(collected/'started.json');observations=load(collected/'observations.json');attempts=load(collected/'ci-attempts.json')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise RuntimeError('N6 frozen protocol changed')
    if set(started['thread_env'].values())!={'1'} or started['pyscf_threads']!=1:raise RuntimeError('Executed thread contract mismatch')
    if sha256((ROOT/'frontierloop/native_source_control_worker.py').read_bytes())!=spec['worker_SHA256']:
        raise RuntimeError('Executed worker changed')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Executed evaluator/decoder drift')
    if sha256((ROOT/'frontierloop/native_source_control_plan.py').read_bytes())!=spec['initial_rule_SHA256']:
        raise RuntimeError('Registered source initialization changed')
    if sha256((ROOT/'runs/R2-COMMON-CI-v1/freeze.json').read_bytes())!=spec['R2_source_freeze_SHA256']:
        raise RuntimeError('Original source manifest drift')
    if load(collected/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':
        raise RuntimeError('Native export path not qualified')
    plans=spec['plans'];states={s['task_id']:s for s in terminal['states']}
    if len(plans)!=8 or len(terminal['states'])!=8 or set(states)!={p['task_id'] for p in plans}:
        raise RuntimeError('All8 including missingness required')
    obsbyid={o['id']:o for o in observations}
    successful=[a for a in attempts if a['status']=='COMPLETED']
    if len(obsbyid)!=len(observations) or len(observations)!=terminal['actual_new_CI_roots'] or len(successful)!=len(observations):
        raise RuntimeError('Actual solve/observation count mismatch')
    if {a['observation_id'] for a in successful}!=set(obsbyid) or len(attempts)!=terminal['ci_attempts']:
        raise RuntimeError('Actual attempts unbound')
    for obs in observations:
        verify_row(obs['row']);file=collected/obs['vector_file']
        if sha256(file.read_bytes())!=obs['vector_SHA256']:raise RuntimeError('Saved CI state drift')
        with np.load(file,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        if len(pairs)!=len(v) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:raise RuntimeError('Final CI state not normalized')
        d=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest()
        if d!=obs['basis_digest'] or obs['id']!=hashlib.sha256((obs['case']+d+started['protocol_SHA256']).encode()).hexdigest():
            raise RuntimeError('CI state/problem identity mismatch')
        if obs['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Wrong actual sparse library')
    seen=set();rows=[];native_cpu=0.;contrasts=[]
    for plan in plans:
        s=states[plan['task_id']];case=plan['case'];source=plan['source'];definition=spec['physical'][case]
        folder=collected/'cells'/plan['task_id']
        if load(folder/'result.json')!=s or s['source']!=source or s['method']!=plan['method'] or s['case']!=case:
            raise RuntimeError('Cell/terminal/source identity mismatch')
        expected={}
        if source['status']=='AVAILABLE':
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if sha256(Path(source[name]).read_bytes())!=source[key]:raise RuntimeError('Source evidence changed')
            if initial_strings(source,definition['norb'],tuple(definition['nelec']))!=plan['initial_strings']:
                raise RuntimeError('Initial source ranking not reconstructed')
        elif s['status']!='MISSING_SAME_PROBLEM_SOURCE' or s['arms']:raise RuntimeError('Missing source incorrectly executed')
        if s.get('native'):
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes())
            if sha256(raw)!=s['native']['raw_determinants_SHA256'] or sha256((folder/'dets.bin').read_bytes())!=sha256(raw):
                raise RuntimeError('Lossless native export mismatch')
            n,pairs=decode(raw);strings,norm=marginal_strings(pairs,n,tuple(definition['nelec']),plan['HF_string'])
            joint,selection=joint_pairs(pairs,n,tuple(definition['nelec']),plan['HF_string'])
            if n!=definition['norb'] or s['selection']!=selection or s['product_strings']!=strings:
                raise RuntimeError('Native derived spaces differ from fixed policies')
            expected={'PRODUCT_R128':[(a,b) for a in strings for b in strings],'JOINT_TOP_D16384':joint}
            if not math.isfinite(s['native']['native_diagnostic_spinfree_energy_Ha']):raise RuntimeError('Native diagnostic nonfinite')
        if s.get('native_attempt'):native_cpu+=s['native_attempt']['CPU_seconds']
        hf,ref=reference(case);resource=source.get('sampler_resources') or {};arms={}
        for arm in ['PRODUCT_R128','JOINT_TOP_D16384']:
            item=s['arms'].get(arm);r=None
            if item:
                pairs=np.ascontiguousarray(expected[arm],dtype=np.uint64)
                d=hashlib.sha256(pairs.tobytes()).hexdigest();o=obsbyid[item['observation_id']];key=(case,d)
                if o['case']!=case or o['basis_digest']!=d or item['row']!=o['row'] or item['evaluation_origin']!=o['origin']:
                    raise RuntimeError('CI result not bound to actual selected space')
                if item['new_CI_solve']!=(key not in seen) or item['row']['actual_CI_dimension']!=len(pairs):
                    raise RuntimeError('Solve/reuse or dimension miscount')
                seen.add(key);r=item['row']
            row=dict(case=case,method=plan['method'],arm=arm,status='COMPLETED' if r else s['status'],
                energy_Ha=r['energy_Ha'] if r else None,penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,
                recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,spin_square=r['spin_square'] if r else None,
                actual_dimension=r['actual_CI_dimension'] if r else None,dimension_cap=16384,
                CI_CPU_seconds=r['CPU_seconds'] if r else None,CI_wall_seconds=r['wall_seconds'] if r else None,
                native_CPU_seconds=s.get('native_attempt',{}).get('CPU_seconds'),
                native_plus_CI_standalone_numeric_CPU=s['native_attempt']['CPU_seconds']+r['CPU_seconds'] if r else None,
                source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),
                source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),
                source_classical_draws=source.get('classical_draws'),
                CNOT=resource.get('cnot_count'),CNOT_Depth=resource.get('cnot_depth'),Total_Depth=resource.get('total_unitary_depth',resource.get('total_depth')),
                coefficient_slots=source.get('sampler_coefficient_slots_mean'),independent_parameter_count=None,
                actual_new_solve=item['new_CI_solve'] if item else None,evaluation_origin=item['evaluation_origin'] if item else None,
                new_model_shots=0,new_QPU_shots=0,task_id=plan['task_id'],error=s.get('error') or s.get('errors',{}).get(arm,{}).get('error'))
            rows.append(row);arms[arm]=row
        p,j=arms['PRODUCT_R128'],arms['JOINT_TOP_D16384'];contrast=dict(case=case,method=plan['method'],status='MISSING_PAIR',physical_change_mHa=None,penalized_change_mHa=None)
        if p['energy_Ha'] is not None and j['energy_Ha'] is not None:
            delta=j['energy_Ha']-p['energy_Ha'];obj=j['penalized_energy_Ha']-p['penalized_energy_Ha']
            contrast.update(status=direction(delta),physical_change_mHa=1000*delta,penalized_change_mHa=1000*obj,penalized_direction=direction(obj))
        contrasts.append(contrast)
    if len(seen)!=len(observations):raise RuntimeError('Unattributed actual CI observation')
    actual=sum(a['CPU_seconds'] for a in attempts)+native_cpu
    if actual>terminal['CPU_seconds']+1e-5:raise RuntimeError('Actual stage CPU double-counted')
    summary=dict(status='ALL8_SOURCE_CONTROLS_AND16_ARMS_AUDITED',rows=rows,contrasts=contrasts,
        contrast_counts=dict(Counter(c['status'] for c in contrasts)),completed_readouts=sum(r['energy_Ha'] is not None for r in rows),
        actual_new_CI_roots=len(observations),native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),
        failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        CPU_seconds=terminal['CPU_seconds'],wall_seconds=terminal['wall_seconds'],actual_cell_native_CPU_seconds=native_cpu,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,independent_quantum_confirmation=False,
        scope='EXISTING_DETERMINISTIC_CISD_AND_LIMITED_TN_NOT_FULL_HIGHEST_PAPER',
        independent_RDM_for_each_joint_root=False,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((collected/'terminal.json').read_bytes()))
    save(ROOT/'analysis/N6-source-controls-summary.json',summary)
    def csv_bytes(rs):
        f=StringIO();w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs);return f.getvalue().encode('utf-8-sig')
    write(ROOT/'analysis/N6-source-controls-all16.csv',csv_bytes(rows))
    old=load(ROOT/'analysis/N5-G5-bound-resource-comparison.json')
    n5=load(ROOT/'analysis/N5-joint-CI-summary.json')
    if old['N5_summary_SHA256']!=sha256((ROOT/'analysis/N5-joint-CI-summary.json').read_bytes()) or n5['completed_readouts']!=40:
        raise RuntimeError('N5 exact-source compiler comparison changed')
    fields=['case','method','status','energy_Ha','penalized_energy_Ha','recovery_pct','spin_square','actual_dimension',
        'CI_CPU_seconds','native_plus_CI_standalone_numeric_CPU','source_total_model_budget','source_final_model_shots',
        'source_construction_CPU_seconds','CNOT','CNOT_Depth','Total_Depth','coefficient_slots','independent_parameter_count']
    combined=[{**{k:r.get(k) for k in fields},'stage':stage} for stage,rs in [('N5',old['rows']),('N6',[r for r in rows if r['arm']=='JOINT_TOP_D16384'])] for r in rs]
    save(ROOT/'analysis/N6-combined-joint-comparison.json',dict(rows=combined,new_shots_for_join=0,new_CI_for_join=0,goal_complete=False,
        N5_G5_source_SHA256=sha256((ROOT/'analysis/N5-G5-bound-resource-comparison.json').read_bytes()),
        N6_source_SHA256=sha256((ROOT/'analysis/N6-source-controls-summary.json').read_bytes())))
    write(ROOT/'analysis/N6-combined-joint-comparison.csv',csv_bytes(combined))
    names={'CLASSICAL_HF_ONLY':'古典HF-HCI','C5_C_LUCJ_COMPRESSED_L1':'圧縮LUCJ','C5_D_SQDRIFT_N20':'SqDRIFT20＋認証compiler',
        'RAACS_B4':'RAACS','CLASSICAL_CISD_100K_DRAWS':'古典CISD・100k draws','DETERMINISTIC_CISD_MARGINAL':'決定論的CISD','C5_E_LUCJ_TN_COMMON_L1':'限定TN/LUCJ'}
    def fmt(v,d=3):return '未取得/該当なし' if v is None else f'{v:.{d}f}'
    lines=['# N6：追加対照と全7sourceの現在地','',
        f"N6は8予定セル/16arm。数値結果{summary['completed_readouts']}、新CI根{len(observations)}。比較{summary['contrast_counts']}。",
        f"新native探索{summary['native_attempts']}件、campaign CPU{summary['CPU_seconds']:.2f}秒、wall{summary['wall_seconds']:.2f}秒。追加model/QPU Shot0、GPU割当0。",'',
        '以下は同じ物理問題・同じH＋0.1Ha S²・同じjoint最大16384配置での開発比較。実次元やnative探索量は異なり、exact matched CPUや独立量子確認ではない。新native探索費用を含むproxyと、source構築/全研究費用を区別する。','']
    for case in spec['physical']:
        lines += ['## '+case,'','| source | Energy Ha | Recovery % | CNOT | CNOT Depth | Total Depth | 元係数slot平均 | native＋CI CPU秒 | 実CI次元 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        group=[r for r in combined if r['case']==case]
        for r in group:
            lines.append('| '+' | '.join([names[r['method']],fmt(r['energy_Ha'],10),fmt(r['recovery_pct']),*[fmt(r[k],2) for k in ['CNOT','CNOT_Depth','Total_Depth','coefficient_slots','native_plus_CI_standalone_numeric_CPU']],str(r['actual_dimension'])])+' |')
        valid=[r for r in group if r['energy_Ha'] is not None]
        best=min(valid,key=lambda r:r['energy_Ha']);raacs=next(r for r in group if r['method']=='RAACS_B4')
        lines += ['',f"今回の最小energyは{names[best['method']]}。RAACSとの差は{1000*(raacs['energy_Ha']-best['energy_Ha']):.6f}mHa。これは精度だけの順位で、全費用・全資源の勝者認定ではない。",'']
    lines += ['## N6だけの積空間からjointへの変化','',
        '| 条件 | source | joint−product mHa（負が良い） | 状態 |','|---|---|---:|---|']
    for c in contrasts:lines.append('| '+c['case']+' | '+names[c['method']]+' | '+fmt(c['physical_change_mHa'],6)+' | '+c['status']+' |')
    lines += ['','## 限界と費用の扱い','',
        '- ナフタレンblock-BoysのTNは同一problemのsourceが無く欠測のまま。別orbitalを代用していない。利用可能なTNも元論文の最高の完全TN ansatzではない。',
        '- 元samplerのCNOT/Depthと、古典CI後の最終energyを組にしたSQD型評価。最終CI状態をその回路が直接準備した意味ではない。古典行は0CNOTの相関状態準備ではない。',
        '- 係数slotは独立Parameter数ではない。全rowの独立rankは未確定で、都合よく0へ置換しない。',
        '- TN/LUCJの旧100k model Shotsと古典CISDの係数全体利用は異なる入力。古典対照を量子Shotと同数だったように偽装しない。RAACSは総予算100kの一部をpilotへ使う。',
        '- native＋CI CPUは単独で必要な数値費用proxy。source構築費用の未知は未知のまま。旧研究・compiler・資格・解析・cold初期化を含む全費用ではない。',
        '- G5のSqDRIFT compiler資源は同一sourceへ結合した全bankの過去証拠を再利用。新しい独立サンプリングではなく、実機noise/routingでもない。',
        '- H/S²行列と残差の資格に基づく。各joint rootの独立RDM検証、FCI・QPU実証、最強論文への総合優越は未認定。','']
    write(ROOT/'analysis/N6_SOURCE_CONTROLS_RESULTS_JA.md','\n'.join(lines).encode())
    print(summary['status'],summary['completed_readouts'],'COMBINED_ROWS',len(combined),flush=True)


if __name__=='__main__':run()
