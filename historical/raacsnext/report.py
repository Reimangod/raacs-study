"""Complete bounded-pilot accounting and stopping report, never tunes policies."""
import csv
from collections import Counter,defaultdict
from io import StringIO
from math import comb
from pathlib import Path
import sqlite3
from time import time
import numpy as np
from .paths import ROOT,OLD
from .io import save,load,write_bytes,sha256,verify_previous_completion

LABEL={'NAPH20-BLOCK-BOYS':'ナフタレン20q（block-Boys）',
    'N2-32-R1p2-P2':'N₂ 32q・1.2 Å','N2-32-R2p4-P2':'N₂ 32q・2.4 Å','FE2S2-40-P2':'Fe₂S₂ cluster40q'}
METHOD={'D0':'D0 元RAACS','D1':'D1 停止前の再確認','D2':'D2 CIを揃えた選択',
    'D3':'D3 最大4本の回路bank','LUCJ_COMPRESSED_L1':'Compressed LUCJ L1'}


def csv_file(name,rows):
    if not rows:return
    s=StringIO();keys=list(dict.fromkeys(k for r in rows for k in r))
    writer=csv.DictWriter(s,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    write_bytes(ROOT/'report'/name,s.getvalue().encode('utf-8-sig'))


def reference(case):
    r=load(OLD/'data/evaluation-only'/(case+'.json'))
    m=load(OLD/'data/physical'/case/'problem.json')
    if r['problem_id']!=m['problem_id']:raise RuntimeError('Evaluation-only problem mismatch')
    energies=r['energies_Ha'];value=energies.get('fci_energy')
    kind='FCI'
    if value is None:value=energies['sci_energy'];kind='公開SCI'
    return energies['hf_energy'],value,kind,r


def load_results():
    records=[];rows=[]
    for path in sorted((ROOT/'runs/DEVELOPMENT').glob('*/*/*/result.json')):
        r=load(path);hf,ref,kind,rr=reference(r['case_id'])
        if r['ledger']['known_actual_shots']!=r['budget']or r['ledger']['unknown_or_inflight_draws']:
            raise RuntimeError('Completed run has unreconciled shots')
        if r['pilot_model_shots']+r['final_model_shots']!=r['budget']or r['final_model_shots']<r['budget']/2:
            raise RuntimeError('Independent final budget mismatch')
        for c in r['final_CI']:
            if not c['converged']or c['valid_sector_fraction']!=1:raise RuntimeError('Final CI/sector not qualified')
            rows.append({'case_id':r['case_id'],'method':r['variant'],'seed':r['seed'],'total_model_shots':r['budget'],
                'pilot_model_shots':r['pilot_model_shots'],'independent_final_model_shots':r['final_model_shots'],
                'r_cap':c['r_cap'],'actual_d':c['actual_d'],'energy_Ha':c['energy_Ha'],
                'correlation_recovery_pct':100*(hf-c['energy_Ha'])/(hf-ref),'reference_kind':kind,
                'reference_precision_independently_certified':rr['sector_and_reference_precision_verified'],
                'spin_square':c['spin_square'],'CI_residual_norm':c['residual_norm'],'CI_CPU_seconds':c['CPU_seconds'],
                **r['sampling_resources'],'coefficient_slots_member_mean':r['coefficient_slots_shot_weighted_mean'],
                'slots_distinct_member_sum':r['coefficient_slots_distinct_member_sum'],'bank_size':r['bank_size'],
                'construction_CPU_seconds':r.get('construction_CPU_seconds'),
                'construction_stop':r.get('construction_stop_reason','FIXED_OPERATOR_NO_ADAPTIVE_CONSTRUCTION'),
                'all_model_CNOT_applications':r['all_model_CNOT_applications'],
                'source_result':str(path.relative_to(ROOT)),'source_result_SHA256':sha256(path.read_bytes())})
        records.append(r)
    return records,rows


def grouped(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['case_id'],r['method'],r['r_cap'])].append(r)
    answer=[]
    for key,rs in sorted(groups.items()):
        if len(rs)!=3:raise RuntimeError('Do not pick incomplete seed groups')
        group={'case_id':key[0],'method':key[1],'r_cap':key[2],'n_readouts':3}
        for field in('energy_Ha','correlation_recovery_pct','actual_d','spin_square','cnot_count','cnot_depth','total_depth',
            'coefficient_slots_member_mean','slots_distinct_member_sum','bank_size','pilot_model_shots',
            'independent_final_model_shots','all_model_CNOT_applications','CI_CPU_seconds'):
            values=np.array([r[field]for r in rs],float)
            group['mean_'+field]=float(np.mean(values));group['SD_'+field]=float(np.std(values,ddof=1))
        costs=[r['construction_CPU_seconds']for r in rs]
        group['mean_construction_CPU_seconds']=float(np.mean(costs))if all(x is not None for x in costs)else None
        answer.append(group)
    return answer


def historical():
    data=load(OLD/'table-comparison-v2/comparison-data.json');out=[]
    for r in data['groups']:
        if r['case_id']not in('NAPH20-P1',*LABEL)or int(r['budget'])!=100000 or int(r['r_cap'])!=64:continue
        if int(r['observed_runs'])!=int(r['planned_runs']):continue
        out.append(r)
    csv_file('historical-100k-r64-not-new-confirmation.csv',out)
    return out


def stopping(groups):
    index={(r['case_id'],r['method'],r['r_cap']):r for r in groups}
    witness_case='NAPH20-BLOCK-BOYS';control=index[(witness_case,'LUCJ_COMPRESSED_L1',64)]
    comparisons=[]
    for method in('D1','D2','D3'):
        candidate=index[(witness_case,method,64)]
        delta=candidate['mean_energy_Ha']-control['mean_energy_Ha']
        comparisons.append({'method':method,'case':witness_case,'control':'LUCJ_COMPRESSED_L1',
            'candidate_mean_energy_Ha':candidate['mean_energy_Ha'],'control_mean_energy_Ha':control['mean_energy_Ha'],
            'candidate_minus_control_mHa':1000*delta,'pilot_precision_condition_pass':delta<=0,
            'four_case_continuation_possible':delta<=0,'formal_statistical_superiority_claimed':False})
    if any(x['pilot_precision_condition_pass']for x in comparisons):
        raise RuntimeError('No preapproved early-futility witness; continue remaining registered controls')
    return {'decision':'STOP_PILOT_TARGET_NOT_MET','scientific_scope':'These three frozen successors under their registered budgets, not all RAACS or all bank methods',
        'rule':'S1: one common successor must pass ALL four cases; cheap-control early stop allowed',
        'failure_witness':comparisons,'precision_margin_Ha':0,'new_optimization_or_threshold_tuning_after_observation':False,
        'S2':'NOT_EXECUTED_PILOT_STOP_RULE','S3':'NOT_EXECUTED_PILOT_STOP_RULE','S3_planned_cells_not_executed':600,
        'world_best_or_paper_superiority_claimed':False,'full_confirmatory_study_completed':False}


def plots(groups):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'D0':'#626262','D1':'#2274A5','D2':'#E0902C','D3':'#5D9F54','LUCJ_COMPRESSED_L1':'#8D4E9F'}
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(case,label)in zip(axes.flat,LABEL.items()):
        for r in groups:
            if r['case_id']!=case or r['r_cap']!=64:continue
            method=r['method'];ax.errorbar(r['mean_cnot_count'],r['mean_correlation_recovery_pct'],
                xerr=r['SD_cnot_count'],yerr=r['SD_correlation_recovery_pct'],fmt='o',color=colors[method],capsize=3,label=method)
        ax.set_title(case,fontsize=10);ax.set_xlabel('Mean logical sampling CNOT count');ax.set_ylabel('Correlation recovery (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Development pilot: 100,000 total model shots; r=64; 3 readouts (SD, not CI)',fontsize=11)
    fig.savefig(ROOT/'report'/'pilot-frontier.png',dpi=200);fig.savefig(ROOT/'report'/'pilot-frontier.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(case,label)in zip(axes.flat,LABEL.items()):
        for method in('D0','D1','D2','D3','LUCJ_COMPRESSED_L1'):
            rs=sorted([r for r in groups if r['case_id']==case and r['method']==method],key=lambda r:r['r_cap'])
            if rs:ax.plot([r['r_cap']for r in rs],[r['mean_correlation_recovery_pct']for r in rs],'-o',label=method,color=colors[method],markersize=3)
        ax.set_title(case,fontsize=10);ax.set_xlabel('Spin-string cap r (same counts reused)');ax.set_ylabel('Correlation recovery (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Classical CI-capacity sensitivity: no additional model shots',fontsize=11)
    fig.savefig(ROOT/'report'/'CI-capacity.png',dpi=200);fig.savefig(ROOT/'report'/'CI-capacity.svg');plt.close(fig)


def main():
    if not(ROOT/'RAACS-pilot-complete.json').exists():raise RuntimeError('Wait for all four cases and all RAACS variants')
    records,rows=load_results();groups=grouped(rows)
    if len(records)!=57:raise RuntimeError('Expected48 RAACS and9 inexpensive strong control readouts; disclose any additions')
    decision=stopping(groups)
    with sqlite3.connect(ROOT/'queue.sqlite')as db:
        if db.execute("SELECT COUNT(*)FROM cells WHERE status='RUNNING'").fetchone()[0]:raise RuntimeError('Live cells remain')
        db.execute("UPDATE cells SET status='NOT_EXECUTED_STOP_RULE',reason='Registered cheap-control pilot futility: no common successor can pass all four cases' WHERE status='NOT_STARTED'")
        db.commit();planned=[dict(zip(('id','stage','case_id','method','budget','seed','status','reason','result_path'),r))
            for r in db.execute('SELECT id,stage,case_id,method,budget,seed,status,reason,result_path FROM cells ORDER BY case_id,method,seed')]
    statuses=dict(Counter(x['status']for x in planned))
    csv_file('all-planned-cells.csv',planned);csv_file('all-new-readouts.csv',rows);csv_file('all-new-group-statistics.csv',groups)
    history=historical();save(ROOT/'pilot-decision.json',decision)
    plots(groups)
    dispatch=[load(p)for p in(ROOT/'dispatch').glob('*/terminal.json')]
    actual_shots=sum(r['ledger']['known_actual_shots']for r in records)
    costs=[]
    for r in records:
        costs.append({'case_id':r['case_id'],'method':r['variant'],'seed':r['seed'],'total_model_shots':r['budget'],
            'pilot_shots':r['pilot_model_shots'],'final_shots':r['final_model_shots'],
            'construction_CPU_seconds':r.get('construction_CPU_seconds'),
            'run_body_CPU_seconds':r['attempt_CPU_seconds'],
            'all_CNOT_applications':r['all_model_CNOT_applications'],
            'cap_CPU_seconds':r.get('construction_CPU_cap'),
            'cap_boundary_overshoot_seconds':max(0,r['construction_CPU_seconds']-r['construction_CPU_cap'])
                if r.get('construction_CPU_cap')is not None else None,
            'run_body_includes_CI_not_additive_to_CI_cost':True})
    csv_file('paid-work-once-per-run.csv',costs)
    overshoots=[c for c in costs if(c['cap_boundary_overshoot_seconds']or 0)>0]
    history_control_cost=[]
    for p in(ROOT/'controls').glob('*/*/source.json'):
        r=load(p);c=r['source_construction'];init=r['upstream_initialization']
        # Basis transformation and upstream CCSD are separate for naphthalene.
        cold=(c['initialization_cost'].get('amplitude_creation_CPU_seconds')or 0)+c['initialization_cost']['compressed_initialization_CPU_seconds']
        upstream=init.get('upstream_CCSD_CPU_seconds',0);cold+=upstream
        history_control_cost.append({'case_id':r['case_id'],'one_time_known_initialization_CPU_seconds':cold,
            'reuse10_per_use_CPU_seconds':cold/10,'reuse100_per_use_CPU_seconds':cold/100,
            'upstream_CCSD_CPU_seconds_if_distinct':upstream,'new_cost_this_campaign':False,
            'all_initializer_stages_converged':c['initialization_cost']['all_initializer_stages_converged'],
            'compilation_and_simulation_separate':True})
    save(ROOT/'report'/'cost-summary.json',{'actual_model_shots':actual_shots,'actual_QPU_shots':0,
        'RAACS_children_including_process_startup_CPU_seconds':sum(x['child_CPU_seconds']for x in dispatch),
        'known_run_body_CPU_seconds_including_embedded_CI':sum(x['run_body_CPU_seconds']for x in costs),
        'run_body_not_added_to_child_process_times':True,
        'shared_LUCJ_simulation_CPU_seconds':sum(load(p)['CPU_seconds']for p in(ROOT/'controls').glob('*/*/shared-simulation.json')),
        'one_time_historical_initialization_costs':history_control_cost,
        'GPU_allocation_hours_this_campaign':0,'new_Slurm_jobs':0,
        'wall_clock_start':load(ROOT/'campaign-start.json')['started_unix'],'wall_clock_report_unix':time(),
        'RAACS_boundary_overshoot_count':len(overshoots),'RAACS_boundary_overshoots':overshoots,
        'strict_hard_CPU_cap_proved':False,'new_shared_control_process_startup_and_full_end_to_end_CPU_seconds':None,
        'unknown_cost_not_zero':True,'coldstart_cost_reused_historical_evidence_not_new_creation':True})
    lines=['# RAACS改良pilot：実施結果と投資判断','',
        '**結論：登録した3種類の改良では、全4条件を通過する目標は未達です。予定どおりpilotで停止し、600件の確認実験へは進めません。**','',
        '回路を小さくする能力は残りました。しかし、同じ総100,000 model Shot・同じr=64で高精度な対照以上にする条件を満たしません。「小さい＝勝ち」とは判断していません。これは今回の固定方策・予算に対する投資判断であり、RAACS全体や回路bank一般が無効という証明ではありません。','',
        '## 1. 何を実際に行ったか','',
        f'予定168件のうち{len(records)}件を完了。RAACSは4条件×4版×3seedの48件すべて、Compressed LUCJは安価に評価できる3条件×3独立readoutの9件です。残り111件は事前の早期停止ルールで未実行。全57件の実消費は{actual_shots:,} model Shot、実QPU Shotは0です。r8/16/32/64の再解析をShotへ重複加算していません。','',
        '| 段階 | 状態 |','|---|---|','| S0 入力・小系検算 | 完了、13 tests passed |',
        '| S1 開発pilot | 57/168完了、111/168は登録損切りにより未実行 |',
        '| S2 強いTN対照の補完・最終方策freeze | pilot条件未達につき未実行 |',
        '| S3 600件・141M Shot確認試験 | pilot条件未達につき未実行・追加費用0 |','',
        '独立readoutのseedは51001/51011/51019。LUCJの3回は同じ固定operatorの測定を3回行ったもので、独立なCCSD/optimizerを3回実行したものではありません。','',
        '## 2. 4条件の新しい数値結果','',
        '全行は総100,000 model Shot、r=64（CI上限4,096）、3回の平均。Parameters欄は係数スロットのmember平均です。D3のbank全体のスロット総数は別CSVへ保存。非冗長な独立パラメータ数は未証明であり、固定係数を0パラメータとは数えていません。資源は共通all-to-allのZ-sampling用です。','']
    for case,label in LABEL.items():
        lines+=['### '+label,'','| 手法 | CI energy Ha ↓ | Recovery % ↑ | CNOT ↓ | CNOT Depth ↓ | Total Depth ↓ | 係数スロット※ ↓ | 実CI次元 d |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        rs=[r for r in groups if r['case_id']==case and r['r_cap']==64]
        for r in rs:
            lines.append('| '+' | '.join([METHOD[r['method']],f"{r['mean_energy_Ha']:.8f}",f"{r['mean_correlation_recovery_pct']:.2f}",
                *[f"{r['mean_'+k]:.2f}"for k in('cnot_count','cnot_depth','total_depth','coefficient_slots_member_mean','actual_d')]])+' |')
        if case=='FE2S2-40-P2':
            lines+=['','Fe₂S₂の新しいLUCJ/SqDRIFT対照は未実行です。ここから対照に勝った／負けたとは結論しません。Recoveryは公開SCI基準でありFCI errorの保証ではありません。']
        else:
            control=next(r for r in rs if r['method']=='LUCJ_COMPRESSED_L1');variants=[r for r in rs if r['method']in('D1','D2','D3')]
            best=min(variants,key=lambda r:r['mean_energy_Ha'])
            delta=1000*(best['mean_energy_Ha']-control['mean_energy_Ha'])
            lines+=['',f"改良3版のうち平均energyが最も低いのは{METHOD[best['method']]}です。Compressed LUCJ L1との差は{delta:+.3f} mHa（正ならRAACSの方が高いenergy）。回路資源が小さくても、この差を精度維持の成功には読み替えません。"]
        if case!='NAPH20-BLOCK-BOYS':lines+=['','N₂/Feの公開参照はsector・参照精度の独立再認証が未完了です。Recoveryは参照付きの補助表示に留め、同じHamiltonianに対するCI energy差を主に見ます。Nα/Nβ保存はsinglet純度の保証ではありません。']
        lines.append('')
    lines+=['## 3. なぜ損切りするのか','',
        'ナフタレンだけで、D1/D2/D3の全版が新しく測定したCompressed LUCJ L1の平均energyへ届きません。どれか一つの共通版が全4条件に通過するという継続条件は、他条件の対照結果を待たなくても満たせなくなりました。したがって高価なTNやFe対照を追加する必要はありません。','',
        '| 版 | ナフタレンでの対照との差 mHa（正が悪化） | 継続の精度条件 |','|---|---:|---|']
    for c in decision['failure_witness']:lines.append(f"| {c['method']} | {c['candidate_minus_control_mHa']:+.3f} | 未達 |")
    lines+=['','これは3回平均による開発の投資判断であり、統計的優越・全手法の不可能性の証明ではありません。正式な20seed確認・多重比較区間は実施していません。','',
        '## 4. 何が効き、何がボトルネックだったか','',
        '- **元の強みは小回路。** RAACSの少数QEでCNOTと係数座標を小さくできる性質は残っています。','- **中心的な不足は、少数の測定から集める配置の質・広がり。** 小さな実CI次元で止まった回路は、rを64へ上げても候補のmarginal support自体が不足し、精度が伸びません。','- **再確認だけでは十分でない。** D1は延長できるrunがありますが、CPU上限で止まるrunもあり、精度不足を全条件で解消してはいません。','- **小CIへのalignmentは自動的な改善ではない。** D2はr8での局所評価を良くするため、最終r64に必要な探索方向を全て保護する保証はありません。','- **4本bankも常に多様性を増やすとは限らない。** D3は親をround-robinで選びますが、選んだ一つの親で確認後の利益がなくなると全体停止します。他の親を全て試してから止まる設計ではありません。この停止の粗さとHFを含むbankの薄いsupportは、次の検討対象にはなります。ただし今回の結果を見て別停止規則へ変更した実験はしていません。','',
        '上記はコードと観測履歴に基づく原因候補です。3seedしかなく、方策ごとに異なる乱数列なので「D2の要素だけが悪化を引き起こした」といった単一原因の断定はしません。','',
        '## 5. 計算費用とCPU上限の注意','',
        '| 条件・版 | 構築CPU秒 平均 | pilot Shot 平均 | 全CNOT適用数 平均 |','|---|---:|---:|---:|']
    for r in groups:
        if r['r_cap']!=64:continue
        cpu='新規構築なし・旧費用別記'if r['mean_construction_CPU_seconds']is None else f"{r['mean_construction_CPU_seconds']:.3f}"
        lines.append(f"| {LABEL[r['case_id']]}・{r['method']} | {cpu} | {r['mean_pilot_model_shots']:,.0f} | {r['mean_all_model_CNOT_applications']:,.0f} |")
    lines+=['',f'CPU計測はprimitive境界です。{len(overshoots)}件で登録CPU値を最後のprimitive実行分だけ超過しました。全件を費用CSVへ記録し、厳密なhard CPU capを証明したとはしていません。これを理由に後から上限を引き上げたり、超過時間を除外したりしていません。精度条件の未達はこの問題とは別に成立します。','',
        'LUCJは既存の物理operatorを再利用して新しいcountsを取りました。初期化を今回0秒で作ったのではありません。旧CCSD/CISD・compressed factorizationの作成費用をsource.jsonへ結合し、単独／10回／100回再利用の既知費用をcost-summary.jsonへ保存しました。共有シミュレーションは一回払いであり3runへ重複加算しません。未知のprocess-startup等を0としません。','',
        'A100×2は他チャットの実行中jobで使用されていました。今回の小support RAACS・安価なLUCJ比較はM4で数値的に実施できたため、新しいGPU jobは投入せず、他研究のjobは停止していません。GPU割当0は高速化・実機優位の主張ではありません。','',
        '## 6. 過去データと論文との差','',
        '過去の同Shot/r64の全設定平均をhistorical-100k-r64-not-new-confirmation.csvに読み取り専用で整理しました。canonicalとblock-Boys、主10seedと追加5seed、limited TN、資源compilerの差を統合して都合のよい勝者にはしていません。旧685runの性能値・コード・原始countsは変更していません。','',
        '[SqDRIFT v3](https://arxiv.org/abs/2508.02578v3)の収束保証を本RAACSへ移植したとは主張しません。今回のSqDRIFT対照はnative SDKの共通予算adaptationで、原著experiment driverそのものの完全再現ではありません。','',
        '[Compressed/TN LUCJ論文](https://arxiv.org/abs/2511.22476v1)の強いTN対照は、旧bounded TN controlと同じではありません。χ50、surrogate10k、optimizer500評価など本文・公開code差分は旧TN_SOURCE_CROSSWALK_JA.mdにあります。今回そこへ勝った結果はありません。','',
        '「現在一番良い論文」に一つの総合順位はありません。同じ物理入力、軌道、Shot、CI、sector、logical compiler、前処理費用を揃えた実在点への勝利から確かめる必要があります。今回はそれより弱い安価な対照の精度も越せず、全面的な原著再現へ進む費用を節約しました。','',
        '## 7. 今後の判断','',
        '**自動で改良を増やすことは止めます。** 次に投資するなら、精度の不足を狙った探索・停止規則を一つだけ改め、小系で「多様な配置が増える」「高精度CIにつながる」ことを先に確かめるのが順序です。さらに回路だけ小さくする改良やShotを増やすだけの延長は、今回の主ボトルネックに直接答えません。新方策の設計・追加分子・新ML・600件確認は本scopeでは未実行です。','',
        '## 8. データの場所','',
        '- all-planned-cells.csv：168件の完了／未実行理由','- all-new-readouts.csv：全57run×4Rのenergy・Recovery・sector・資源','- all-new-group-statistics.csv：全群の平均・SD','- paid-work-once-per-run.csv：Shot・構築CPU・上限超過・CNOT適用総数','- pilot-frontier.png / SVG：小回路と精度の交換関係','- CI-capacity.png / SVG：同countsでrだけ増やした効果','- cost-summary.json：既知費用・共有費用・cold/reuse・未知','',
        '図の誤差棒は3readoutの標準偏差で、信頼区間ではありません。混合回路を一つの物理的に準備したCI波動関数とは扱っていません。']
    write_bytes(ROOT/'report'/'REPORT_JA.md',('\n'.join(lines)+'\n').encode())
    old=verify_previous_completion()
    coverage={'bounded_pilot_protocol_completed':True,'full_confirmatory_study_completed':False,
        'planned_pilot_cells':168,'status_counts':statuses,'new_completed_runs':len(records),'new_model_shots':actual_shots,
        'four_RAACS_conditions_all_completed':True,'all_registered_RAACS_variants_completed':True,
        'no_paper_superiority_claim':True,'stop_decision':decision['decision'],'old_evidence_check':old,
        'scientific_code_SHA256':load(ROOT/'qualification/scientific-code.json')['code_SHA256'],
        'remaining_conditions_not_scientific_failures':True,'strict_hard_CPU_cap_proved':False,
        'strong_TN_control':'NOT_EXECUTED_STOP_RULE','large_FCI_reference_recertification':'NOT_EXECUTED_STOP_RULE',
        '52q_estimate_only':{'naphthalene_series26e26o_complex128_number_sector_state_GiB':16*comb(26,13)**2/2**30,
            'two_A100_memory_not_automatically_shared':True,'runtime_prediction_seconds':None,'new_52q_jobs':0}}
    save(ROOT/'report/requirement-coverage.json',coverage)
    paths=[p for p in ROOT.rglob('*')if p.is_file()and p.suffix in('.json','.md','.py','.csv','.png','.svg')
        and '__pycache__'not in p.parts and 'draws'not in p.parts and p.name!='FINAL_COMPLETION.json']
    save(ROOT/'FINAL_COMPLETION.json',{'bounded_plan_terminated_per_registered_stop':True,
        'scientific_superiority_objective_achieved':False,'full_confirmation_executed':False,'finished_unix':time(),
        'decision':decision['decision'],'coverage':coverage,'evidence_SHA256':{str(p.relative_to(ROOT)):sha256(p.read_bytes())for p in paths}})
    print({'completed':len(records),'model_shots':actual_shots,'statuses':statuses,'decision':decision['decision']},flush=True)


if __name__=='__main__':main()
