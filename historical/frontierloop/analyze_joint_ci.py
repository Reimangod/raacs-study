"""All40 frozen N5 readouts, actual CI state evidence and cost accounting."""
import csv,gzip,hashlib,math
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .joint_ci_dispatch import LOCAL
from .native_hci_bootstrap import LOCAL as PARENT
from .joint_ci_plan import joint_pairs
from .native_dice_binary import decode
from .analyze import reference


def direction(delta,tol=1e-8):
    return 'LOWER' if delta< -tol else 'HIGHER' if delta>tol else 'EQUAL_WITHIN_TOLERANCE'


def verify_row(row):
    required=['energy_Ha','penalized_energy_Ha','spin_square','residual_norm','CPU_seconds','wall_seconds']
    if not all(math.isfinite(row[k]) for k in required):raise RuntimeError('Nonfinite numerical result')
    if row['common_equation']!='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO':raise RuntimeError('Wrong spin equation')
    if abs(row['penalized_energy_Ha']-row['energy_Ha']-.1*row['spin_square'])>1e-8 or row['spin_square']< -1e-8:
        raise RuntimeError('Physical/spin expectation mismatch')
    if row['residual_norm']>1e-8 or max(row['random_bilinear_hermiticity_errors'])>1e-10:
        raise RuntimeError('Numerical residual/Hermiticity check failed')
    if row['matrix_element_threshold']!=0.:raise RuntimeError('Unregistered matrix-element truncation')


def run():
    collected=LOCAL/'collected';spec=load(LOCAL/'protocol.json');terminal=load(collected/'terminal.json')
    started=load(collected/'started.json');qualification=load(collected/'qualification.json')
    observations=load(collected/'observations.json');attempts=load(collected/'attempts.json')
    legacy=load(collected/'legacy-comparisons.json') if (collected/'legacy-comparisons.json').exists() else []
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Frozen protocol drift')
    if started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:raise RuntimeError('Thread contract not executed')
    for name,value in spec['task_implementation_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Frozen implementation drift')
    if sha256((PARENT/'terminal.json').read_bytes())!=spec['parent_terminal_SHA256']:raise RuntimeError('Parent scientific results changed')
    plans=spec['plans'];states={s['task_id']:s for s in terminal['states']}
    if len(plans)!=20 or len(terminal['states'])!=20 or set(states)!={p['task_id'] for p in plans}:
        raise RuntimeError('All20 source cells with missingness required')
    byid={o['id']:o for o in observations}
    if len(byid)!=len(observations) or len(observations)!=terminal['actual_unique_CI_roots']:raise RuntimeError('Root count mismatch')
    successful=[a for a in attempts if a['status']=='COMPLETED']
    if len(successful)!=len(observations) or len(attempts)!=terminal['actual_attempts']:
        raise RuntimeError('Actual attempt/observation accounting mismatch')
    if {a['observation_id'] for a in successful}!=set(byid):raise RuntimeError('Actual completed evaluations unbound')
    if any(a['status'] not in ['COMPLETED','FAILED_PRESERVED'] for a in attempts):raise RuntimeError('Nonterminal attempt')
    build=load(collected/'library/build.json');binary=collected/('library/libjoint_ci.dylib' if (collected/'library/libjoint_ci.dylib').exists() else 'library/libjoint_ci.so')
    if sha256(binary.read_bytes())!=build['binary_SHA256'] or build['source_SHA256']!=spec['pipeline_SHA256']['joint_ci_sparse.c']:
        raise RuntimeError('Actual compiled-kernel identity mismatch')
    for obs in observations:
        path=collected/obs['vector_file']
        if sha256(path.read_bytes())!=obs['vector_SHA256']:raise RuntimeError('Saved final CI state changed')
        with np.load(path,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        if len(v)!=len(pairs) or not np.isfinite(v).all() or abs(v@v-1)>1e-10:raise RuntimeError('Saved CI state normalization mismatch')
        digest=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest()
        if digest!=obs['basis_digest'] or obs['id']!=hashlib.sha256((obs['case']+digest+started['protocol_SHA256']).encode()).hexdigest():
            raise RuntimeError('Saved state/physical problem/evaluation ID mismatch')
        if obs['row']['binary_SHA256']!=build['binary_SHA256']:raise RuntimeError('Root executed by different numerical kernel')
        verify_row(obs['row'])
    qualified=qualification['status']=='ALL4_REAL_FULL_PRODUCT_C7_PARITY_PASSED'
    if qualified and (len(qualification['rows'])!=4 or max(max(q['errors'].values()) for q in qualification['rows'])>1e-8):
        raise RuntimeError('Four-case product qualifier incomplete')
    lookup={(o['case'],o['basis_digest']):o for o in observations};seen=set()
    def call(item,case,pairs):
        digest=hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.uint64).tobytes()).hexdigest();key=(case,digest)
        o=lookup.get(key)
        if o is None or item['observation_id']!=o['id'] or item['row']!=o['row'] or item['evaluation_origin']!=o['origin']:
            raise RuntimeError('Call did not reuse exact physical observation')
        if item['new_CI_solve']!=(key not in seen):raise RuntimeError('Actual new solve/reuse event misclassified')
        if item['row']['actual_CI_dimension']!=len(pairs):raise RuntimeError('Actual determinant dimension mismatch')
        seen.add(key);return item['row']
    for q in qualification['rows']:
        qp=next(p for p in spec['qualification'] if p['case']==q['case'])
        with np.load(LOCAL/qp['basis_file'],allow_pickle=False) as a:call(q['result'],q['case'],a['product'])
    rows=[];contrasts=[]
    for plan in plans:
        record=states[plan['task_id']];case=plan['case'];definition=spec['physical'][case]
        if load(collected/'cells'/(plan['task_id']+'.json'))!=record:raise RuntimeError('Cell/terminal mismatch')
        basis=LOCAL/plan['basis_file']
        if sha256(basis.read_bytes())!=plan['basis_SHA256']:raise RuntimeError('Frozen basis changed')
        raw=gzip.decompress((PARENT/'collected'/plan['parent_task_id']/'dets.bin.gz').read_bytes())
        if sha256(raw)!=plan['native_raw_SHA256']:raise RuntimeError('Paid native coefficients changed')
        n,native=decode(raw)
        oldplan=next(p for p in load(PARENT/'protocol.json')['plans'] if p['task_id']==plan['parent_task_id'])
        selected,meta=joint_pairs(native,n,definition['nelec'],oldplan['HF_string'])
        if meta!=plan['selection']:raise RuntimeError('Outcome-blind joint selection not reproducible')
        hf,ref=reference(case);source=plan['source'];resource=source.get('sampler_resources') or {};armrows={}
        with np.load(basis,allow_pickle=False) as data:
            if not np.array_equal(data['joint'],selected):raise RuntimeError('Native joint ranking differs from registered rule')
            for arm,key in [('PRODUCT_R128','product'),('JOINT_TOP_D16384','joint')]:
                item=record['arms'].get(arm);r=call(item,case,data[key]) if item else None
                row=dict(case=case,method=plan['method'],arm=arm,status='COMPLETED' if r else record['status'],
                    energy_Ha=r['energy_Ha'] if r else None,penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,
                    recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,spin_square=r['spin_square'] if r else None,
                    actual_dimension=r['actual_CI_dimension'] if r else None,dimension_cap=16384,
                    CI_CPU_seconds=r['CPU_seconds'] if r else None,CI_wall_seconds=r['wall_seconds'] if r else None,
                    matrix_build_CPU_seconds=r['matrix_build_CPU_seconds'] if r else None,H_nonzeros=r['H_nonzeros'] if r else None,
                    CSR_bytes=r['CSR_bytes'] if r else None,
                    inherited_native_CPU_seconds=plan['inherited_native_CPU_seconds'],
                    native_plus_CI_standalone_numeric_CPU=plan['inherited_native_CPU_seconds']+r['CPU_seconds'] if r else None,
                    actual_new_solve=item['new_CI_solve'] if item else None,evaluation_origin=item['evaluation_origin'] if item else None,
                    source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),
                    source_classical_draws=source.get('classical_draws'),source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),
                    CNOT=resource.get('cnot_count'),CNOT_Depth=resource.get('cnot_depth'),Total_Depth=resource.get('total_unitary_depth',resource.get('total_depth')),
                    coefficient_slots=source.get('sampler_coefficient_slots_mean'),independent_parameter_count=None,
                    new_model_shots=0,new_QPU_shots=0,task_id=plan['task_id'],
                    error=record.get('errors',{}).get(arm,{}).get('error'))
                rows.append(row);armrows[arm]=row
        p,j=armrows['PRODUCT_R128'],armrows['JOINT_TOP_D16384']
        contrast=dict(case=case,method=plan['method'],physical_change_mHa=None,penalized_change_mHa=None,status='MISSING_PAIR')
        if p['energy_Ha'] is not None and j['energy_Ha'] is not None:
            delta=j['energy_Ha']-p['energy_Ha'];obj=j['penalized_energy_Ha']-p['penalized_energy_Ha']
            contrast.update(physical_change_mHa=1000*delta,penalized_change_mHa=1000*obj,
                status=direction(delta),penalized_direction=direction(obj),
                product_actual_dimension=p['actual_dimension'],joint_actual_dimension=j['actual_dimension'],
                product_CI_CPU_seconds=p['CI_CPU_seconds'],joint_CI_CPU_seconds=j['CI_CPU_seconds'])
        contrasts.append(contrast)
    if seen!=set(lookup):raise RuntimeError('Unattributed solved observations')
    actual_numeric=sum(a['CPU_seconds'] for a in attempts)+sum(a['actual_CPU_seconds'] for a in legacy)
    if actual_numeric>terminal['CPU_seconds']+1e-5:raise RuntimeError('Actual CPU double counted or campaign undercounted')
    summary=dict(status='ALL40_PLANNED_READOUTS_AUDITED' if qualified else 'QUALIFICATION_FAILED_WITHOUT_NEW_JOINT_CANDIDATES',
        rows=rows,contrasts=contrasts,contrast_counts=dict(Counter(c['status'] for c in contrasts)),
        all40_planned_rows=len(rows),completed_readouts=sum(r['energy_Ha'] is not None for r in rows),
        actual_new_explicit_CI_roots=len(observations),failed_explicit_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        legacy_timing_comparisons=legacy,legacy_timing_roots=sum(a['status']=='COMPLETED' for a in legacy),
        CPU_seconds=terminal['CPU_seconds'],wall_seconds=terminal['wall_seconds'],actual_numeric_CPU_seconds=actual_numeric,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,final_CI_state_not_prepared_by_sampler=True,
        qualification_scope_note='The frozen legacy-named independent_RDM_qualification field refers to H/S2 matrix parity tests, NOT a new independent RDM test of every arbitrary-joint root. Historical bytes preserved; do not promote that field to a RDM certificate.',
        independent_RDM_validation_for_each_new_joint_root=False,
        independent_quantum_confirmation=False,same_max_dimension_not_equal_actual_dimension=True,
        legacy_validation_work_is_different=True,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((collected/'terminal.json').read_bytes()))
    save(ROOT/'analysis/N5-joint-CI-summary.json',summary)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N5-joint-CI-all40.csv',stream.getvalue().encode('utf-8-sig'))
    names={'CLASSICAL_HF_ONLY':'古典HF','C5_C_LUCJ_COMPRESSED_L1':'圧縮LUCJ','C5_D_SQDRIFT_N20':'SqDRIFT20','RAACS_B4':'RAACS','CLASSICAL_CISD_100K_DRAWS':'古典CISD draws'}
    def fmt(x,dec=3):return '未取得' if x is None else f'{x:.{dec}f}'
    lines=['# N5：joint配置を直接使った結果','',f"40予定arm中{summary['completed_readouts']}件の数値結果。比較：{summary['contrast_counts']}。",
        f"実測CPU{summary['CPU_seconds']:.2f}秒、wall{summary['wall_seconds']:.2f}秒。新model/QPU Shot0、GPU割当0。",'',
        '両方式の最終CIは共通のH＋0.1Ha S²。表のenergyは物理期待値。積空間もjoint空間も同じ新疎行列kernelで再計算し、同じ最大16,384配置を与えた。実次元は必ずしも等しくない。','']
    for case in spec['physical']:
        lines += ['## '+case,'','| 開始方法 | 積空間 energy Ha | joint energy Ha | 差 mHa（負が良い） | joint Recovery % | 実次元：積 / joint | CI CPU秒：積 / joint |','|---|---:|---:|---:|---:|---:|---:|']
        for plan in [x for x in plans if x['case']==case]:
            rs={r['arm']:r for r in rows if r['task_id']==plan['task_id']};p,j=rs['PRODUCT_R128'],rs['JOINT_TOP_D16384'];c=next(c for c in contrasts if c['case']==case and c['method']==plan['method'])
            lines.append('| '+' | '.join([names[plan['method']],fmt(p['energy_Ha'],10),fmt(j['energy_Ha'],10),fmt(c['physical_change_mHa'],6),fmt(j['recovery_pct']),
                str(p['actual_dimension'])+' / '+str(j['actual_dimension']),fmt(p['CI_CPU_seconds'])+' / '+fmt(j['CI_CPU_seconds'])])+' |')
        cs=[c for c in contrasts if c['case']==case]
        lines += ['',f"この条件のjoint変更は、改善{sum(c['status']=='LOWER' for c in cs)}・悪化{sum(c['status']=='HIGHER' for c in cs)}・同等{sum(c['status']=='EQUAL_WITHIN_TOLERANCE' for c in cs)}・比較不能{sum(c['status']=='MISSING_PAIR' for c in cs)}。低いenergyだけを選んで元の積方式を失敗へ変更していない。",'']
    lines += ['## 同じ機械・1threadの旧CIとの数値費用診断','',
        '| 条件 | 旧C7 CPU秒 | 新product CPU秒 | 備考 |','|---|---:|---:|---|']
    for old in legacy:
        q=next(x for x in qualification['rows'] if x['case']==old['case']);new=q['result']['row']
        lines.append('| '+old['case']+' | '+fmt(old.get('row',{}).get('CPU_seconds'))+' | '+fmt(new['CPU_seconds'])+' | '+old['status']+' |')
    lines += ['','旧C7は各rootのRDM検査を含み、新実装は小系/部分空間で独立に検証した行列式を使用する。検証作業量も異なるため、CPU差の全てが新たな物理アルゴリズムの利益ではない。matrix build費用は新CI CPUに含めた。資格実行・失敗・全armを含むcampaign費用を各rowへ重ねて足していない。','',
        '## 限界と資源の読み方','',
        '- 回路のCNOT/Depth/係数slotはN4と同じsourceのまま。今回の改善はreadout側で、最終CI状態が元samplerにより準備されたという意味ではない。classical行は0 CNOTの相関状態ではない。',
        '- source構築・旧native HCI探索は別費用。CSVでは各arm単独の比較に必要なnative＋CI CPUも示したが、coldな全研究費用ではない。',
        '- 任意joint選択は既知のHCI型処理で、最大次元が同じでも実次元/結合数/CPUは異なる。物理energy・罰則energy・S²を全てCSVに保存した。spinが異なる効果を黙って精度利益と呼ばない。',
        '- 旧観測sourceを使った開発で、独立量子確認ではない。新しいFCI、QPU実証、Parameter独立rank、最高論文の完全再現や総合優越は主張しない。','',
        '保存metadataの`independent_RDM_qualification`という名前は不正確である。今回の独立資格はH/S²行列のPySCF照合で、各joint rootの独立RDM検証ではない。元の実行bytesは保存し、この説明で検証範囲を限定する。物理式と結果を変更していない。','',
        '## 欠測一覧','']
    missing=[r for r in rows if r['energy_Ha'] is None]
    lines.extend(['- '+r['case']+' / '+r['method']+' / '+r['arm']+': '+str(r['error']) for r in missing] or ['40予定armの欠測なし。'])
    write(ROOT/'analysis/N5_JOINT_CI_RESULTS_JA.md','\n'.join(lines).encode())
    print(summary['status'],summary['contrast_counts'],'CPU',summary['CPU_seconds'],flush=True)


if __name__=='__main__':run()
