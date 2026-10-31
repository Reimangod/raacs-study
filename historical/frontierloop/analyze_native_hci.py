"""Audit all frozen N4 cells and distinguish classical proposals from sampling."""
import csv,gzip,math
from collections import Counter
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .native_hci_bootstrap import LOCAL
from .native_hci_plan import frequency_strings,marginal_strings
from .native_dice_binary import decode
from .analyze import reference


def audit_root(root,case,strings,observations):
    r=root['row'];key=(case,tuple(strings));entry=observations.get(key)
    if entry is None or root['evaluation_origin']!=entry['origin'] or r!=entry['row']:
        raise ValueError('Root/cache origin or actual basis mismatch')
    if r['CI_strings']!=strings or r['actual_CI_dimension']!=len(strings)**2:
        raise ValueError('Root physical basis/dimension mismatch')
    if r['common_equation']!='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO':
        raise ValueError('Spin/energy contract mismatch')
    required=['energy_Ha','penalized_energy_Ha','spin_square','residual_norm','CPU_seconds','wall_seconds']
    if not all(math.isfinite(r[k]) for k in required):raise ValueError('Nonfinite root')
    if abs(r['penalized_energy_Ha']-r['energy_Ha']-.1*r['spin_square'])>1e-8:
        raise ValueError('Physical and penalized energies confused')
    if r['residual_norm']>1e-8 or max(r['random_bilinear_hermiticity_errors'])>1e-10 or max(r['independent_RDM_expectation_errors'])>1e-8:
        raise ValueError('Saved physical/eigensolver tests failed')
    return True


def run():
    spec=load(LOCAL/'protocol.json');terminal=load(LOCAL/'terminal.json');bundle=load(LOCAL/'evidence-collection.json')
    if bundle['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Frozen protocol drift')
    impl=spec['task_identity_implementation']
    for name,key in [('native_hci_worker.py','worker_SHA256'),('native_hci_plan.py','proposal_rule_SHA256'),('native_dice_binary.py','decoder_SHA256')]:
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=impl[key]:raise RuntimeError('Frozen production implementation drift')
    plans=spec['plans'];states=terminal['states'];byid={x['task_id']:x for x in states}
    if len(plans)!=20 or len(states)!=20 or len(byid)!=20 or set(byid)!={p['task_id'] for p in plans}:
        raise RuntimeError('All20 frozen cells, including failures, required')
    observations=load(LOCAL/'common-ci-observations.json');attempts=load(LOCAL/'common-ci-attempts.json')
    lookup={(o['case'],tuple(o['strings'])):o for o in observations}
    if len(lookup)!=len(observations):raise RuntimeError('Duplicate actual CI observation')
    successful=[a for a in attempts if a['status']=='COMPLETED']
    if len(successful)!=len(observations) or terminal['new_unique_common_CI_solves']!=len(observations):
        raise RuntimeError('Attempt/observation accounting mismatch')
    if {(a['case'],tuple(a['strings']),a['origin']) for a in successful}!={(o['case'],tuple(o['strings']),o['origin']) for o in observations}:
        raise RuntimeError('Actual successful attempts not bound to saved observation origins')
    for a in attempts:
        if a['status'] not in ['COMPLETED','FAILED_PRESERVED'] or a['CPU_seconds']<0:raise RuntimeError('Unresolved actual attempt')
        if a['status']=='FAILED_PRESERVED' and a['matvec_calls'] is not None:raise RuntimeError('Failed unknown work fabricated')
    rows=[];filemap={x['task_id']:x['files'] for x in bundle['cells']};seen=set()
    def recorded_call(root,case,strings):
        key=(case,tuple(strings))
        if root['new_CI_solve']!=(key not in seen):raise RuntimeError('Actual solve/reuse event miscounted')
        audit_root(root,case,strings,lookup);seen.add(key)
    for p in plans:
        task=p['task_id'];s=byid[task];source=p['source'];case=p['case'];definition=spec['physical'][case]
        folder=LOCAL/'collected'/task
        for name,item in filemap[task].items():
            if name!='dets.bin' and sha256((folder/name).read_bytes())!=item['SHA256']:raise RuntimeError('Collected evidence drift')
        if load(folder/'result.json')!=s or s['source']!=source or s['method']!=p['method'] or s['case']!=case:
            raise RuntimeError('Per-cell terminal identity mismatch')
        if source.get('payload'):
            payload=Path(source['payload'])
            if sha256(payload.read_bytes())!=source['payload_SHA256']:raise RuntimeError('Source counts changed')
            with np.load(payload,allow_pickle=False) as a:expected=frequency_strings(a['blocked_ids'],a['counts'],definition['norb'],p['HF_string'])
            if expected!=p['initial_strings']:raise RuntimeError('Frequency plan not reconstructed from original counts')
        elif p['initial_strings']!=[p['HF_string']]:raise RuntimeError('HF-only plan mismatch')
        proposal=s.get('proposal');final=s.get('final');baseline=s.get('baseline')
        support={k:None for k in ['native_unique_alpha_strings','native_unique_beta_strings',
            'native_weight_retained_by_final_cartesian_basis','native_joint_determinants_retained']}
        if baseline:recorded_call(baseline,case,p['initial_strings'])
        if proposal:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes())
            if sha256(raw)!=proposal['raw_determinants_SHA256'] or sha256(raw)!=filemap[task]['dets.bin']['SHA256']:
                raise RuntimeError('Lossless native coefficient archive mismatch')
            n,pairs=decode(raw);selected,norm=marginal_strings(pairs,n,tuple(definition['nelec']),p['HF_string'])
            if n!=definition['norb'] or selected!=proposal['selected_strings'] or len(pairs)!=proposal['actual_joint_determinants'] or abs(norm-proposal['coefficient_norm'])>1e-12:
                raise RuntimeError('Native marginal policy not reproduced')
            chosen=set(selected)
            support=dict(native_unique_alpha_strings=len({a for a,b,v in pairs}),native_unique_beta_strings=len({b for a,b,v in pairs}),
                native_weight_retained_by_final_cartesian_basis=math.fsum(v*v for a,b,v in pairs if a in chosen and b in chosen),
                native_joint_determinants_retained=sum(a in chosen and b in chosen for a,b,v in pairs))
        if final:
            if not proposal:raise RuntimeError('Final without native proposal')
            recorded_call(final,case,proposal['selected_strings'])
            if abs(s['standalone_native_plus_common_numeric_CPU_proxy']-s['native_attempt_CPU_seconds']-final['row']['CPU_seconds'])>1e-8:
                raise RuntimeError('Standalone cost lost canonical final solve')
        if s['status']=='COMPLETED_NATIVE_PROPOSAL_AND_COMMON_FINAL' and not final:
            raise RuntimeError('Completed native/final cell missing final endpoint')
        if s['new_model_shots']!=0 or s['new_QPU_shots']!=0:raise RuntimeError('New shots not authorized by N4')
        hf,ref=reference(case);b=baseline['row'] if baseline else {};f=final['row'] if final else {}
        resource=source.get('sampler_resources') or {}
        rows.append(dict(case=case,method=p['method'],status=s['status'],task_id=task,
            baseline_energy_Ha=b.get('energy_Ha'),final_energy_Ha=f.get('energy_Ha'),
            energy_change_mHa=(f['energy_Ha']-b['energy_Ha'])*1000 if f and b else None,
            recovery_pct=100*(hf-f['energy_Ha'])/(hf-ref) if f else None,spin_square=f.get('spin_square'),
            actual_final_dimension=f.get('actual_CI_dimension'),actual_native_joint_dimension=proposal['actual_joint_determinants'] if proposal else None,
            native_diagnostic_spinfree_energy_Ha=proposal['native_physical_energy_Ha'] if proposal else None,
            native_CPU_seconds=s.get('native_attempt_CPU_seconds'),baseline_canonical_CI_CPU_seconds=b.get('CPU_seconds'),
            standalone_native_final_numeric_CPU_proxy=s.get('standalone_native_plus_common_numeric_CPU_proxy'),
            actual_cell_CPU_seconds=s.get('actual_cell_CPU_seconds_including_children'),actual_cell_wall_seconds=s.get('actual_cell_wall_seconds'),
            final_new_actual_solve=final['new_CI_solve'] if final else None,
            source_total_model_budget=source.get('source_total_budget'),source_final_model_shots=source.get('source_final_model_shots'),
            source_classical_draws=source.get('classical_draws'),source_construction_CPU_seconds=source.get('source_construction_CPU_seconds'),
            CNOT=resource.get('cnot_count'),CNOT_Depth=resource.get('cnot_depth'),Total_Depth=resource.get('total_unitary_depth',resource.get('total_depth')),
            coefficient_slots=source.get('sampler_coefficient_slots_mean'),independent_parameter_count=None,**support,
            baseline_error=s.get('baseline_error',{}).get('error'),error=s.get('error')))
    if seen!=set(lookup):raise RuntimeError('Unattributed completed CI observations')
    cases={}
    for case in spec['physical']:
        subset=[r for r in rows if r['case']==case];hfrow=next(r for r in subset if r['method']=='CLASSICAL_HF_ONLY')
        for r in subset:r['energy_minus_HF_HCI_mHa']=(r['final_energy_Ha']-hfrow['final_energy_Ha'])*1000 if r['final_energy_Ha'] is not None and hfrow['final_energy_Ha'] is not None else None
        cases[case]=dict(available_final=sum(r['final_energy_Ha'] is not None for r in subset),
            distinct_final_bases=len({tuple(s['final']['row']['CI_strings']) for s in states if s['case']==case and s.get('final')}),
            HF_HCI_energy_Ha=hfrow['final_energy_Ha'],
            best_observed_method=min((r for r in subset if r['final_energy_Ha'] is not None),key=lambda r:r['final_energy_Ha'])['method'] if any(r['final_energy_Ha'] is not None for r in subset) else None)
    summary=dict(status='ALL20_FROZEN_NATIVE_HCI_SOURCE_CELLS_AUDITED',rows=rows,cases=cases,
        status_counts=dict(Counter(r['status'] for r in rows)),actual_new_common_CI_solves=len(observations),
        available_frequency_baselines=sum(r['baseline_energy_Ha'] is not None for r in rows),
        available_final_endpoints=sum(r['final_energy_Ha'] is not None for r in rows),
        failed_common_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        actual_campaign_CPU_seconds=terminal['CPU_seconds_including_children'],actual_campaign_wall_seconds=terminal['wall_seconds'],
        standalone_cost_is_numeric_proxy_not_cold_end_to_end=True,source_construction_and_old_search_costs_separate=True,
        parent_threads_not_explicitly_fixed=True,Slurm_allocated_logical_CPUs=2,GPU_allocations=0,
        new_model_shots=0,new_QPU_shots=0,independent_quantum_confirmation=False,original_paper_highest_HCI_reproduction=False,
        final_CI_state_not_prepared_by_sampler=True,independent_parameters_unproven=True,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((LOCAL/'terminal.json').read_bytes()),
        collector_manifest_SHA256=sha256((LOCAL/'evidence-collection.json').read_bytes()))
    save(ROOT/'analysis/N4-native-HCI-summary.json',summary)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N4-native-HCI-all20.csv',stream.getvalue().encode('utf-8-sig'))
    lines=['# Native HCIを加えた共通CI比較：全20条件','',
        '全4条件×5開始方法を同じ設定で実行。新しい量子Shotは0。同じ保存サンプルを使った開発試験で、独立した性能確認や原著最高性能の再現ではない。',
        '',f"状態：{summary['status_counts']}。最終endpoint{summary['available_final_endpoints']}/20、native探索から分離したfrequency control{summary['available_frequency_baselines']}/20。新しい共通CI根{len(observations)}、共通CI失敗{summary['failed_common_CI_attempts']}。",
        f"実測campaign CPU {summary['actual_campaign_CPU_seconds']:.2f}秒、wall {summary['actual_campaign_wall_seconds']:.2f}秒。GPU割当0。",'']
    names={'CLASSICAL_HF_ONLY':'古典HFからHCI','C5_C_LUCJ_COMPRESSED_L1':'圧縮LUCJサンプルからHCI','C5_D_SQDRIFT_N20':'SqDRIFT20サンプルからHCI','RAACS_B4':'RAACSサンプルからHCI','CLASSICAL_CISD_100K_DRAWS':'古典CISD drawsからHCI'}
    for case in cases:
        lines += ['## '+case,'','| 開始方法 | 最終energy Ha | Recovery % | HF-HCIとの差 mHa（負が良い） | native＋最終CI CPU proxy 秒 | 状態 |','|---|---:|---:|---:|---:|---|']
        def fmt(v,d=3):return '未取得' if v is None else f'{v:.{d}f}'
        for r in [x for x in rows if x['case']==case]:
            lines.append('| '+' | '.join([names[r['method']],fmt(r['final_energy_Ha'],10),fmt(r['recovery_pct']),fmt(r['energy_minus_HF_HCI_mHa'],6),fmt(r['standalone_native_final_numeric_CPU_proxy']),r['status']])+' |')
        lines += ['', '読み方：量子回路のサンプルを使わないHF-HCIが同等なら、最終精度の主な改善を量子回路の優位性とは説明できない。小差は実測値として残すが、単一の旧サンプルに基づく差を一般的な勝利としない。','']
        interpretation={
            'NAPH20-BLOCK-BOYS':'SqDRIFT20開始が最小energyだが、HF開始との差は0.0413mHa、古典CISD開始との差は0.0126mHa。source構築やShotsまで含む実用的優越は、この単一試験からは決められない。',
            'N2-32-R1p2-P2':'5方式が同じ精度に到達した。HF開始で十分に低いenergyが得られ、この条件では量子sourceを追加した精度利益を確認できない。',
            'N2-32-R2p4-P2':'HF・SqDRIFT20・RAACS開始が同等。圧縮LUCJと古典CISD開始は約1.011mHa高い。良い初期sourceなら必ず良い最終結果になるわけではなく、選択された最終空間の差を残している。',
            'FE2S2-40-P2':'古典CISD開始が最小energy。SqDRIFT20はHF開始より0.0405mHa低いが、古典CISD開始より0.0506mHa高い。RAACSの浅さは維持されるものの、native処理後の精度利益を示してはいない。'}
        lines += [interpretation[case],f"最終basisの種類は5開始方法中{cases[case]['distinct_final_bases']}種類。同じbasis・同じ評価を再利用した点は独立の成功例ではない。",'']
        lines += ['| 開始方法 | 元sampler CNOT | CNOT Depth | Total Depth | 係数slot平均 | 旧model総予算 / final Shots | 頻度順control energy Ha |','|---|---:|---:|---:|---:|---:|---:|']
        for r in [x for x in rows if x['case']==case]:
            logical=[fmt(r[k],2) if r[k] is not None else '該当なし' for k in ['CNOT','CNOT_Depth','Total_Depth','coefficient_slots']]
            shots=str(r['source_total_model_budget'])+' / '+str(r['source_final_model_shots'])
            lines.append('| '+' | '.join([names[r['method']],*logical,shots,fmt(r['baseline_energy_Ha'],10)])+' |')
        lines += ['', 'この表の資源は元のsamplerに対する値で、最終CI状態の準備費用ではない。N4自体は回路を変更しない。G5の追加compiler削減は別に認証済みだが、ここへ無条件に混ぜていない。古典CISDの100k drawsも量子Shotではない。係数slotと独立Parameter rankは異なり、後者は未確定。','']
    lines += ['## 比較の境界と費用','',
        '- native HCIはspin-free物理Hamiltonianによる配置提案だけ。表の最終energyは全方式で共通の有限0.1 Ha S²罰則CIから得た物理期待値。native診断energyを混ぜていない。',
        '- 最終は最大128 spin stringsのCartesian積。native途中のjoint determinant数は別の量で、途中探索量やCPUは等しくない。',
        '- 最終CI状態を元のサンプル回路が準備するわけではない。HF/CISD行を「最終相関状態を0 CNOTで準備した」と解釈してはいけない。',
        '- CPU proxyはnative探索＋最終CIのcanonical数値費用。cache再利用でも単独実行に必要な最終CI費用を戻しており、実際の新規solve数とは別。parser、archive、cold startup、旧構築費用を含む完全なmatched-workではない。',
        '- campaign実測には頻度順controlと失敗も含む。各行へ同じcampaign費用を足し直さない。親BLAS/OpenMPのthread設定は明示固定されず、速度の厳密な再現性には限界がある。',
        '- Recoveryは既存参照に対する相関energy回収率。参照のsector/精度認証範囲は過去の制約を継承する。energyと独立した証拠ではない。',
        '- 独立パラメータrank、実機routing/noise、QPU実証、原著最強条件、独立量子サンプリングの確認は未達。研究全体の総合優越は主張しない。','']
    missing=[r for r in rows if r['baseline_energy_Ha'] is None or r['final_energy_Ha'] is None]
    lines += ['## 欠測・失敗の全リスト','']
    if not missing:lines.append('予定したcontrol/final endpointの欠測なし。')
    for r in missing:lines.append('- '+r['case']+' / '+r['method']+': baseline='+str(r['baseline_error'])+'; final='+str(r['error']))
    write(ROOT/'analysis/N4_NATIVE_HCI_RESULTS_JA.md','\n'.join(lines).encode())
    print(summary['status'],summary['status_counts'],'CPU',summary['actual_campaign_CPU_seconds'],flush=True)


if __name__=='__main__':run()
