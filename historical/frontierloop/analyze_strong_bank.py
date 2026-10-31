"""G5 whole-bank audit and context-bound reuse of existing CI readouts."""
from collections import Counter
import csv
import gzip
from io import BytesIO,StringIO
import json
from pathlib import Path
import sqlite3
from time import process_time,perf_counter
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .strong_projection_bank import FREEZE,RUN,binding
from .strong_projection import choose_safe,FIELDS
from .pooled_selection_screen import FREEZE as T2_FREEZE,binding as t2_binding
from .analyze import reference
from paperstudy.lucj import circuit_views


def rows_csv(rows):
    out=StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def analyze():
    cpu,wall=process_time(),perf_counter()
    frozen=load(FREEZE);terminal=load(RUN/'terminal.json')
    if binding()!=frozen or terminal['freeze_SHA256']!=sha256(FREEZE.read_bytes()):
        raise RuntimeError('G5 actual frozen inputs changed')
    if terminal['statuses']!={'CERTIFIED_RESOURCE_POINT':3132}: raise RuntimeError('No partial-bank energy join')
    path=RUN/'compiled-members.sqlite'
    if sha256(path.read_bytes())!=terminal['database_SHA256']: raise RuntimeError('G5 DB changed')
    t2=load(T2_FREEZE)
    if t2_binding()!=t2: raise RuntimeError('Original T2 inputs/physical problems changed')
    r2_path=ROOT/'analysis/R2-common-CI-all.csv'
    if sha256(r2_path.read_bytes())!=load(ROOT/'freeze-G3-FULLBANK-PROJECTION.json')['R2_comparison_csv_SHA256']:
        raise RuntimeError('Historical all-control comparison changed')
    r2=list(csv.DictReader(r2_path.open(encoding='utf-8-sig')))
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]: raise RuntimeError('DB integrity')
    members=[];recounts=0;flat=[]
    for case,idx,mid,record,cert,native,rustiq,projected in db.execute('SELECT * FROM members ORDER BY case_id,idx'):
        r=json.loads(record);source=next(s for s in frozen['sources'] if s['case']==case);plan=source['members'][idx]
        if r['status']!='CERTIFIED_RESOURCE_POINT' or r['member_id']!=mid or mid!=plan['member_id'] or r['spec']!=plan['spec']:
            raise RuntimeError('Member/context mismatch')
        if r['problem_id']!=source['problem_id'] or r['source_artifacts']!=plan['artifacts']:
            raise RuntimeError('Physical/raw-source mismatch')
        if r['original_member_shot_weight']!=plan['historical_member_shots']:
            raise RuntimeError('Original ensemble weights changed')
        if sha256(cert)!=r['certificate_SHA256'] or json.loads(gzip.decompress(cert))['original_member_id']!=mid:
            raise RuntimeError('Certificate binding')
        if choose_safe(r['resources'])!=r['policy']: raise RuntimeError('Outcome-free guard changed')
        if not r['policy']['no_increase_vs_native']: raise RuntimeError('Native metric increased')
        for name,blob in [('NATIVE',native),('RUSTIQ',rustiq),('PROJECTED',projected)]:
            if sha256(blob)!=r['qpy_gz_SHA256'][name]: raise RuntimeError('Actual QPY hash mismatch')
            q=qpy.load(BytesIO(gzip.decompress(blob)))[0]
            _,_,views=circuit_views(q)
            if views['Z_sampling_equivalent']!=r['resources'][name]: raise RuntimeError('Saved QPY resource recount mismatch')
            c=r['independent_check']['circuits'][name]
            if c['status']!='INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED' or c['probability_L1_with_leakage_bound']>1e-8:
                raise RuntimeError('Actual circuit not independently checked')
            recounts+=1
        members.append(r)
        row=dict(case=case,index=idx,member_id=mid,shot_weight=r['original_member_shot_weight'],
            original_qubits=r['original_qubits'],active_qubits=r['active_qubits'],
            selected_qubits=r['active_qubits'] if r['policy']['selected']=='PROJECTED' else r['original_qubits'],
            source_support=r['source_sparse_support'],selected=r['policy']['selected'],baseline=r['policy']['baseline'],
            coefficient_slots_original=r['coefficient_slots_original'],restricted_candidate_slots=r['coefficient_slots_restricted'],
            optimized_variables_before_and_after=0,independent_parameter_rank=None,
            check_reused_from_G4=r['G5_prefix_evidence_reused'])
        for name in ['NATIVE','RUSTIQ','PROJECTED']:
            for k in FIELDS: row[name+'_'+k]=r['resources'][name][k]
        for name,key in [('CONTROL','baseline_resources'),('SELECTED','selected_resources')]:
            for k in FIELDS: row[name+'_'+k]=r['policy'][key][k]
        flat.append(row)
    db.close()
    if recounts!=9396 or len(members)!=3132: raise RuntimeError('All actual circuits required')
    cases=[]; joined=[]
    for source in frozen['sources']:
        case=source['case']; group=sorted([r for r in members if r['case']==case],key=lambda x:x['index'])
        if len(group)!=783 or [r['index'] for r in group]!=list(range(783)):
            raise RuntimeError('Missing/repeated source member')
        with np.load(source['source_counts_path'],allow_pickle=False) as a:
            if int(a['counts'].sum())!=100000: raise RuntimeError('Original shot count')
        if sum(r['original_member_shot_weight'] for r in group)!=100000:
            raise RuntimeError('Member mixture weight normalization')
        def mean(name,k):
            return sum(r['original_member_shot_weight']*(r['resources'][name][k] if name in r['resources'] else
                r['policy']['baseline_resources' if name=='CONTROL' else 'selected_resources'][k]) for r in group)/100000
        means={name:{k:mean(name,k) for k in FIELDS} for name in ['NATIVE','RUSTIQ','PROJECTED','CONTROL','SELECTED']}
        old=next(x for x in r2 if x['case']==case and x['method']=='C5_D_SQDRIFT_N20')
        if old['status']!='COMPLETED' or int(old['R_cap'])!=128 or int(old['final_model_shots'])!=100000:
            raise RuntimeError('Old R2 readout context')
        for key,k in zip(['CNOT','CNOT_depth','total_depth'],FIELDS):
            if abs(float(old[key])-means['NATIVE'][k])>1e-9: raise RuntimeError('Historical native counter regression')
        p=next(x for x in t2['plans'] if x['source']['case']==case and x['source']['method_id']=='C5_D_SQDRIFT_N20')
        ts=p['source']
        if ts['payload_SHA256']!=source['source_counts_SHA256'] or ts['source_result_SHA256']!=source['source_result_SHA256'] or ts['expected_problem_id']!=source['problem_id']:
            raise RuntimeError('T2 observation/problem mismatch')
        if sha256(Path(ts['R2_baseline']).read_bytes())!=ts['R2_baseline_SHA256']:
            raise RuntimeError('Original R2 evaluation changed')
        baseline=load(ts['R2_baseline'])
        endpoint=ROOT/'runs/T2-POOLED-SELECTION-v1'/case/'C5_D_SQDRIFT_N20'/'terminal.json'
        t=load(endpoint)
        if t['frozen_selection_SHA256']!=p['selection_SHA256']: raise RuntimeError('Original T2 selected basis changed')
        g=next(x for x in t['rows'] if x['arm']=='GLOBAL_CI_MARGINAL')
        if g['status']!='COMPLETED' or g['CI_strings']!=p['selection']['arms']['GLOBAL_CI_MARGINAL']:
            raise RuntimeError('T2 original energy/basis mismatch')
        if abs(g['energy_Ha']+.1*g['S2']-g['penalized_energy_Ha'])>1e-8:
            raise RuntimeError('Finite spin-penalty energy mismatch')
        hf,ref=reference(case)
        summary=dict(case=case,members=783,model_shot_weight=100000,mean_resources=means,
            selection=dict(Counter(r['policy']['selected'] for r in group)),
            reductions_vs_native_pct={k:100*(1-means['SELECTED'][k]/means['NATIVE'][k]) for k in FIELDS},
            additional_reductions_vs_control_pct={k:100*(1-means['SELECTED'][k]/means['CONTROL'][k]) for k in FIELDS},
            active_qubits_range=[min(r['active_qubits'] for r in group),max(r['active_qubits'] for r in group)],
            source_support_range=[min(r['source_sparse_support'] for r in group),max(r['source_sparse_support'] for r in group)],
            single_configuration_member_shot_fraction=sum(r['original_member_shot_weight'] for r in group if r['source_sparse_support']==1)/100000,
            max_MPS_final_bond=max(c['max_final_bond_dimension'] for r in group for c in r['independent_check']['circuits'].values()),
            max_probability_L1=max(c['probability_L1_with_leakage_bound'] for r in group for c in r['independent_check']['circuits'].values()),
            original_R2_energy_Ha=baseline['energy_Ha'],original_R2_CI_dimension=baseline['actual_CI_dimension'],
            Global_T2_energy_Ha=g['energy_Ha'],Global_T2_recovery_pct=100*(hf-g['energy_Ha'])/(hf-ref),
            Global_T2_CI_dimension=g['actual_CI_dimension'],Global_T2_S2=g['S2'],
            original_R2_CI_CPU_seconds=baseline['incremental_CPU_seconds'],
            Global_T2_accounted_CI_CPU_seconds=g['accounted_operational_CI_CPU_seconds_not_cold_total'],
            T2_result_SHA256=sha256(endpoint.read_bytes()),counts_SHA256=source['source_counts_SHA256'],
            physical_problem_id=source['problem_id'],energy_and_counts_reuse='CERTIFICATE_DERIVED_OLD_OBSERVATIONS_NOT_NEW_MEASUREMENT')
        cases.append(summary)
        for label,resources,energy,spin,dimension,cost,origin in [
            ('N20_ORIGINAL_NATIVE_OLD_R2_FREQUENCY',means['NATIVE'],baseline['energy_Ha'],baseline['spin_square'],baseline['actual_CI_dimension'],baseline['incremental_CPU_seconds'],ts['R2_baseline']),
            ('N20_STRONG_COMPILER_SAME_GLOBAL',means['CONTROL'],g['energy_Ha'],g['S2'],g['actual_CI_dimension'],g['accounted_operational_CI_CPU_seconds_not_cold_total'],str(endpoint)),
            ('N20_COMPRESSED_SAME_OLD_R2_FREQUENCY',means['SELECTED'],baseline['energy_Ha'],baseline['spin_square'],baseline['actual_CI_dimension'],baseline['incremental_CPU_seconds'],ts['R2_baseline']),
            ('N20_COMPRESSED_SAME_GLOBAL',means['SELECTED'],g['energy_Ha'],g['S2'],g['actual_CI_dimension'],g['accounted_operational_CI_CPU_seconds_not_cold_total'],str(endpoint))]:
            joined.append(dict(case=case,method=label,status='CERTIFIED_OLD_EVALUATION_REUSE',energy_Ha=energy,
                recovery_pct=100*(hf-energy)/(hf-ref),S2=spin,CNOT=resources['cnot_count'],
                CNOT_depth=resources['cnot_depth'],total_depth=resources['total_unitary_depth'],
                total_model_shot_budget=100000,final_model_shots=100000,R_cap=128,actual_CI_dimension=dimension,
                VQE_optimized_variables=0,independent_parameter_rank=None,CI_numeric_CPU_seconds_not_cold_total=cost,
                additional_G5_compiler_campaign_CPU_seconds_all4=terminal['CPU_seconds_including_children'],
                compiler_cost_field_is_shared_campaign_not_per_row=True,evaluation_origin=origin,
                new_model_shots=0,new_energy_CI_evaluations=0,independent_quantum_confirmation=False))
    result=dict(status='ALL4_WHOLE_N20_BANKS_AUDITED_OLD_CI_REUSE_NOT_INDEPENDENT_CONFIRMATION',cases=cases,
        members=3132,actual_QPY_recounts=recounts,new_model_shots=0,new_QPU_shots=0,new_energy_CI_evaluations=0,
        fullbank_CPU_seconds_including_children=terminal['CPU_seconds_including_children'],fullbank_wall_seconds=terminal['wall_seconds'],
        prefix_evidence_reused_members=terminal['reused_G4_members'],new_independent_circuit_checks=terminal['new_MPS_circuit_checks'],
        database_bytes=path.stat().st_size,recount_analysis_CPU_seconds=process_time()-cpu,
        recount_analysis_wall_seconds=perf_counter()-wall,strongest_paper_superiority=False,goal_complete=False)
    save(ROOT/'analysis/G5-whole-N20-summary.json',result)
    write(ROOT/'analysis/G5-all-members.csv',rows_csv(flat))
    write(ROOT/'analysis/G5-energy-resource-joined.csv',rows_csv(joined))
    lines=['# G5：強いSqDRIFT20全回路の資源・精度を結び付ける','',
        '**4条件×783回路＝3,132件すべてを確認し、測定分布を保存する固定compiler policyの全bank資源を得た。** 旧countsと同じCI評価を証明由来の再利用として結び付ける。新しい独立測定・新しいCI精度実験ではない。最強論文への総合優越も未達。','',
        '## 1．全回路の資源：同じ100k model Shot・同じ物理問題','',
        '| 条件 | CNOT 元→圧縮 | CNOT Depth 元→圧縮 | Total Depth 元→圧縮 |',
        '|---|---:|---:|---:|']
    for c in cases:
        m=c['mean_resources'];p=c['reductions_vs_native_pct']
        cells=[f"{m['NATIVE'][k]:.2f} → {m['SELECTED'][k]:.2f}（−{p[k]:.1f}%）" for k in FIELDS]
        lines.append('| '+c['case']+' | '+' | '.join(cells)+' |')
    lines+=['','全783回路を元Shot配分で重み付けした平均。小数個のゲートという意味ではない。各回路のCNOT・CNOT Depth・Total Depthすべてが元nativeより非増加になるguardを固定し、同じ条件で全4caseに適用した。選択はenergyやCI係数を使わず、projectedを採れない場合は検証済みの対照へfallbackする。','',
        '## 2．既存Rustiqにも同じ入力を与えた強化対照からの追加効果','',
        '| 条件 | 強化対照CNOT→圧縮 | 追加CNOT削減 | 追加CNOT Depth削減 | 追加Total削減 | projected採用/783 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        m=c['mean_resources'];p=c['additional_reductions_vs_control_pct']
        lines.append(f"| {c['case']} | {m['CONTROL']['cnot_count']:.2f} → {m['SELECTED']['cnot_count']:.2f} | {p['cnot_count']:.1f}% | {p['cnot_depth']:.1f}% | {p['total_unitary_depth']:.1f}% | {c['selection'].get('PROJECTED',0)} |")
    lines+=['','元nativeからの大きな差の一部は既存コンパイラでも得られる。第1表だけで提案処理の独自効果と解釈せず、第2表の追加効果を分けて読む。Counterは共通all-to-all logical CX、opt1、同seedのZ-sampling規約。原著hardware最適化level3/routing/native2Qと同一ではない。','',
        '## 3．精度とCI費用：圧縮は精度を改善する処理ではない','',
        '| 条件 | 読出し | CI Energy Ha（旧評価再利用） | Recovery % | 実CI次元 | CI数値CPU秒 | S² |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in cases:
        for r in [x for x in joined if x['case']==c['case'] and x['method'] in ['N20_COMPRESSED_SAME_OLD_R2_FREQUENCY','N20_COMPRESSED_SAME_GLOBAL']]:
            name='元全観測の頻度順' if 'FREQUENCY' in r['method'] else 'T2 Global CI marginal'
            lines.append(f"| {r['case']} | {name} | {r['energy_Ha']:.10f} | {r['recovery_pct']:.3f} | {r['actual_CI_dimension']} | {r['CI_numeric_CPU_seconds_not_cold_total']:.3f} | {r['S2']:.5f} |")
    lines+=['','同じ読出し同士なら、元回路と圧縮回路のenergy/Recoveryは同じ旧値を参照する。Globalの精度改善は古典側の配置選択によるもので、回路圧縮の効果にしない。公平な対照にも同じGlobalを認める。Globalは必要な4screenのCI費用も計上し、頻度順に不要なscreen費用を押し付けない。NAPHは全候補を残す同一basisのため、Global優位の例ではない。','',
        '最大R128はspin stringの上限で、同じ実CI次元や同じ実CPUを意味しない。有限0.1Ha S²罰則、OOなし。値はCI出力のphysical energyであり、厳密S=0でも、圧縮sampling回路がそのCI状態を量子準備した証拠でもない。Recoveryは同じ既存HF/参照値で算出し、energyと独立な成功指標とは数えない。','',
        '## 4．現在の対照との位置づけ','',
        '| 条件 | 圧縮N20＋Global Ha | 既存R2の観測最小 Ha | その方法 | 差 mHa（負が今回低い） |',
        '|---|---:|---:|---|---:|']
    for c in cases:
        available=[r for r in r2 if r['case']==c['case'] and r['status']=='COMPLETED']
        best=min(available,key=lambda x:float(x['energy_Ha']))
        lines.append(f"| {c['case']} | {c['Global_T2_energy_Ha']:.10f} | {float(best['energy_Ha']):.10f} | {best['method']} | {1000*(c['Global_T2_energy_Ha']-float(best['energy_Ha'])):.3f} |")
    lines+=['','これは既存の全観測対照の事後envelopeであり、独立評価や完全な原著最高設定ではない。全対照68行の状態・資源・CI費用・欠測は[R2包括表](R2_COMMON_CI_RESULTS_JA.md)へ保持。古典CISDには量子sampling回路がなく、0 CNOTで相関状態を量子準備したと解釈しない。TNは限定調整であり原著のfull200 subproblem/500 iteration、native DICE/HCIも未再現。したがってこの表の最小更新だけで「最良論文に勝った」としない。','',
        '## 5．検算・複雑さ・Parameter・全費用','',
        '| 条件 | active qubit範囲 | 元member支持配置数 | 最大最終MPS bond | 最大確率L1誤差 |',
        '|---|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['active_qubits_range']} | {c['source_support_range']} | {c['max_MPS_final_bond']} | {c['max_probability_L1']:.3g} |")
    qual=load(ROOT/'qualification/G5-batched-prefix-v1/result.json')
    lines+=['',f"全9,396実QPYを再計数し、元nativeの過去資源平均も再現した。うち192回路の独立確率証拠はG4を明示参照し、新たな独立MPS検算は9,204回路。全支持確率と全ノルムの漏れを含む検査、閾値1e−8、MPS打切りなし。MPS/巨大statevectorを保存していない。一般的40q状態の古典simulationや量子優位性の証拠ではない。",'',
        f"G5本実行は{terminal['CPU_seconds_including_children']:.2f} CPU秒（子process含む）/{terminal['wall_seconds']:.2f} wall秒、DB{path.stat().st_size/2**20:.2f}MiB。一括化の工程資格試験は全64件で回路の順序付き中身/資源/確率を照合し{qual['CPU_seconds_including_children']:.2f}CPU秒。旧G4費用47.66CPU秒や新開発/テスト費用も消さない。再計数/この解析は別費用としてsummaryへ記録。これらは全3compilerと検算を含むcampaign費用で、選択された1回路の運用費用やcold-start総費用ではない。",'',
        '元source構築・CCSD/TN初期化・昔のShot/CIは元manifestと評価originへ残す。未知の構築費用を0にしない。全量子sourceの元予算100k model Shot、今回新規model/QPU Shotとも0。ハードウェア測定時間や量子ノイズ下の利益は未測定。','',
        'Parameterについて、元SqDRIFTはVQE最適化変数0の固定乱数回路。Pauli係数slotを独立Ansatz自由度と呼ばず、独立rankは未認証/未知としてCSVにnullを残す。restricted_candidate_slotsは候補のslot数であり、fallback時の選択回路の独立自由度ではない。','',
        '## 6．次の研究判断','',
        'N20全bankで精度を保持し、論理資源を下げるcompiler処理は保持する。ただし既知の初期状態依存最適化であり、同じcompiler機会は競合にも与えるべき。これだけに投資を固定しない。次は固定Global配置選択を、全4条件/全16sourceへ同じ別classical subsampling seedsで確認し、R2頻度順・同union頻度順と費用をそろえて比較する。同じcountsを使うので独立Quantum replicateとはしない。','',
        '同時に原著の強いTN/HCI/source setupに達しているかを優先して点検する。特に原著自体がidle mode除去の可能性を明示し、Rustiq/order-preservingとbackend最高最適化を使っている。SDK教程のdiagonal filteringやtrivial filteringはsource分布を変える別条件なので、既存bankへ無断変更せず新versionの対照へ切り出す。大削減率だけを新規性に読み替えない。','',
        '資料：[SqDRIFT原著v3 III.2/III.3](https://arxiv.org/html/2508.02578v3)、[公式SqDRIFT tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/sqdrift)、[全T2結果](T2_POOLED_SELECTION_RESULTS_JA.md)、[G4](G4_STRONG_PROJECTION_RESULTS_JA.md)。', '',
        '生データ：G5-all-members.csv、G5-energy-resource-joined.csv、G5-whole-N20-summary.json。元freeze/result/countsは変更していない。']
    write(ROOT/'analysis/G5_WHOLE_N20_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print('G5_ALL3132_AUDITED_9396_QPY_RECOUNTS_OLD_CONTEXT_BOUND_READOUT_REUSE',flush=True)


if __name__=='__main__': analyze()
