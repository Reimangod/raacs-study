"""TN optimization expense and independent readout quality, case by case."""
import csv,json
import numpy as np
from .io import ROOT,atomic_bytes,save_json
from .analysis_products import write_csv

CASES=('NAPH20-P1','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2')


def load(p):return json.loads(p.read_text())if p.exists()else None


def successor_lower_bounds(root):
    """Preserved counts include completed AND failed bounded objectives."""
    remote=root/'terminal-evidence/remote';paths=list(remote.glob('surrogate-evaluations/*-counts.npz'))
    draws=0
    for path in paths:
        with np.load(path,allow_pickle=False)as a:draws+=sum(map(int,a['counts']))
    progress=load(remote/'progress.json')or{}
    return {'successor_preserved_surrogate_draws_lower_bound':draws if paths else None,
        'successor_last_completed_progress_CPU_seconds_lower_bound':progress.get('CPU_seconds'),
        'successor_inflight_or_unpreserved_work_not_zero':True}


def run():
    rows=[];baseline_rows=[]
    group_file=ROOT/'analysis-v1/all-group-statistics.csv'
    if group_file.exists():
        with group_file.open()as f:baseline_rows=list(csv.DictReader(f))
    for case in('NAPH20-P1','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'):
        original=ROOT/'TN-controls-v1'/case;successor=ROOT/'TN-controls-v2'/case
        evidence=load(original/'terminal-evidence-v1/manifest.json')or{}
        effective=load(original/'effective-source-v2.json')
        if effective:
            source=successor/'selected-source-result.json';selected=load(source)
        else:source=original/'optimization-result.json';selected=load(source)
        final=list((ROOT/'common-runs/S_TN_INDEPENDENT_FINAL'/case/'E_LUCJ_TN_COMMON_L1').glob('*/result.json'))
        energies=[];capacity=[]
        for path in final:
            r=load(path);energies.append(next(c['energy_Ha']for c in r['final_CI']if c['r_cap']==32))
            side=ROOT/'control-CI-extension-v1/S_TN_INDEPENDENT_FINAL'/case/r['method_id']/(str(r['budget'])+'-'+str(r['seed']))/'CI-r64.json'
            if side.exists():capacity.append(load(side)['CI']['energy_Ha'])
        row={'case_id':case,'independent_final_readouts_completed':len(final),'planned_independent_final_readouts':5,
            'optimizer_replicates':1,'engineering_successor_submitted':(successor/'job.json').exists(),
            'original_terminal_evidence':evidence.get('actual_Slurm_accounting'),
            'original_surrogate_samples_preserved_lower_bound':evidence.get('preserved_classical_surrogate_draws_lower_bound'),
            'original_completed_objectives':evidence.get('completed_surrogate_objectives'),
            'original_failed_objectives':evidence.get('failed_surrogate_objectives'),
            'selected_source_status':None if not selected else selected['status'],
            'selected_source_metadata':str(source.relative_to(ROOT))if selected else None,
            'selected_optimizer_CPU_seconds':None if not selected else selected.get('CPU_seconds'),
            'selected_optimizer_wall_seconds':None if not selected else selected.get('wall_seconds'),
            'selected_completed_surrogate_samples':None if not selected else selected.get('surrogate_samples'),
            'surrogate_selected_energy_Ha':None if not selected else selected.get('best_surrogate_energy_Ha',selected.get('checkpoint',{}).get('surrogate_energy_Ha')),
            'mean_independent_final_r32_energy_Ha':float(np.mean(energies))if energies else None,
            'SD_final_readouts_not_training_replicates':float(np.std(energies,ddof=1))if len(energies)>1 else None,
            'r64_final_readouts_available':len(capacity),'mean_independent_final_r64_energy_Ha':float(np.mean(capacity))if capacity else None,
            'model_shots_per_final_readout':100000,'actual_QPU_shots':0,'paper_native_TN_reproduction':False,
            'failed_original_cost_not_erased':True,'failed_or_unknown_draws_not_free':True}
        row.update(successor_lower_bounds(successor))
        if row['surrogate_selected_energy_Ha']is not None and energies:
            row['independent_mean_minus_selected_surrogate_mHa']=1000*(np.mean(energies)-row['surrogate_selected_energy_Ha'])
        else:row['independent_mean_minus_selected_surrogate_mHa']=None
        baseline=next((r for r in baseline_rows if r['lane']=='C_PRIMARY'and r['case_id']==case
            and r['method_id']=='C_LUCJ_COMPRESSED_L1'and r['budget']=='100000'and r['r_cap']=='32'
            and int(r['observed_runs'])==10),None)
        request=load(original/'request.json')or{}
        construction=load(ROOT/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/construction.json')or{}
        initial_match=request.get('input_sha256',{}).get('initial-operator.npz')==construction.get('operator_sha256')
        row['same_registered_initial_C_operator_verified']=bool(initial_match and construction.get('operator_sha256'))
        row['initial_C_readout_replicates']=None if not baseline else 10
        row['initial_C_mean_r32_energy_Ha']=None if not baseline else float(baseline['mean_energy_Ha'])
        row['TN_mean_minus_initial_C_mean_mHa']=(1000*(np.mean(energies)-float(baseline['mean_energy_Ha']))
            if baseline and len(energies)==5 and row['same_registered_initial_C_operator_verified']else None)
        row['before_after_comparison_is_descriptive_not_paired_significance']=True
        rows.append(row)
    write_csv(ROOT/'analysis-v1/TN-optimization-cost-and-independent-quality.csv',rows)
    lines=['# TN対照：古典で係数を調整し、別の測定で精度を確認する','',
        'TN対照は、MPSの近似samplingとCIを使ってLUCJの係数を古典最適化した後、その回路を厳密シミュレーションで独立に測定します。最適化中に一番低かったenergyを、そのまま最終結果にはしていません。','',
        '| 条件 | 採用回路の状態 | 独立final / 5 | 元attemptの保存draws下限 | 修復attemptの保存draws下限 | 元の完了/失敗objective | 採用optimizer CPU分 | 独立final平均energy Ha（r32） |',
        '|---|---|---:|---:|---:|---|---:|---:|']
    def val(v,digits=3):return '不明/未完了'if v is None else f'{v:.{digits}f}'
    for r in rows:
        cpu=r['selected_optimizer_CPU_seconds']
        status=('時間切れ・完了評価の回路を回収済み'if r['selected_source_status']=='TN_COMMON_READOUT_BOUNDED_CHECKPOINT_RECOVERED'else
                '正常return回路あり・収束未認証'if r['selected_source_status']else'計算中/未回収')
        lines.append('| '+' | '.join([r['case_id'],status,str(r['independent_final_readouts_completed'])+' / 5',
            str(r['original_surrogate_samples_preserved_lower_bound'])if r['original_surrogate_samples_preserved_lower_bound']is not None else'未回収',
            str(r['successor_preserved_surrogate_draws_lower_bound'])if r['successor_preserved_surrogate_draws_lower_bound']is not None else'修復未使用/未回収',
            str(r['original_completed_objectives'])+' / '+str(r['original_failed_objectives']),val(None if cpu is None else cpu/60),
            val(r['mean_independent_final_r32_energy_Ha'],8)])+' |')
    lines+=['','CPU欄は採用されたoptimizer attemptの実測値です。元の失敗attemptや新しい修復attemptの費用を消した総額ではありません。全attemptのSlurm割当時間と、最終sampling・CI費用は別表へ残しています。',
        '保存されたdraws下限は完了objectiveだけでなく、countsを残して失敗した評価も含みます。未保存の作業があり得る場合は下限のまま扱います。これらはclassical surrogate drawsで、新規QPU shotsではありません。',
        '修復attemptが外部時間切れで終わった場合、正常returnの総CPU値は不明です。最後の完了評価時点のCPU下限をCSVへ保存し、Slurm割当時間とは別に表示します。同じjobの費用へ両方を足しません。',
        '5回の最終測定は、一つの調整済み回路の測定揺らぎを調べたものです。5回の独立した学習・最適化成功ではありません。',
        'r64感度は同じcountsを再利用し、発動caseの全対照に適用します。surrogateとの差にはMPS近似・選択バイアス・測定揺らぎが混ざるため、その差だけをMPS誤差の厳密な推定とはしません。',
        '原論文とはQSCI objective・一部NOMAD設定・runtimeが違います。詳しくはTN_SOURCE_CROSSWALK_JA.mdを参照してください。原論文の最強TN-LUCJより優れているという主張はしません。']
    lines+=['','## 係数調整は、独立最終評価でも良くなったか','',
        '| 条件 | 調整前Cの平均energy Ha（10seed） | TN後の平均energy Ha（5seed） | TN後−調整前 mHa | 読み方 |',
        '|---|---:|---:|---:|---|']
    for r in rows:
        delta=r['TN_mean_minus_initial_C_mean_mHa']
        reading=('未回収/同じ初期回路の照合未完了'if delta is None else
                 'この限定試験では平均energyが低下'if delta<0 else'この限定試験では平均energyが上昇')
        lines.append('| '+' | '.join([r['case_id'],val(r['initial_C_mean_r32_energy_Ha'],8),
            val(r['mean_independent_final_r32_energy_Ha'],8),val(delta),reading])+' |')
    lines+=['','負の差は良化、正の差は悪化です。r32・100k final readoutでの記述的比較で、独立seed数は10対5、TNには別のclassical surrogate最適化費用があります。同じ総CPU予算のpaired検定や因果効果の推定ではありません。',
        'TN後が悪ければ、この登録された近似・最適化予算・選択規則で独立finalの改善を確認できなかった、と読みます。元論文のTN法全体の否定でも、RAACSが原著の最強LUCJを上回った証明でもありません。MPS近似、有限sampling、選択、最適化上限のどれが主因かは、この表だけでは特定しません。']
    if(ROOT/'TN-high-final-v1/spec.json').exists():
        lines+=['','## 当初計画の高予算TN読出しの補完','',
            '当初の科学計画5.2には「高予算相当」とありますが、上の20件は原論文の最終評価に合わせた100k版です。これを1M比較の完了とは数えず、同じ4個の採用済みoperatorを固定し、1M×5seed×4条件の20件も別laneへ補完します。新しい最適化や良いcaseの選別ではありません。',
            '既存100k版はその費用とともに残します。これは実験開始後の範囲補完で、独立に事前登録された確認的結果ではありません。各予算内の5seedは独立読出しですが、予算間では同じ数値seedを使うため乱数列のprefixを共有する可能性があり、40個の独立optimizerや40個の独立構造とは数えません。','',
            '| 条件 | 1M読出し完了 / 5 | r32平均energy Ha | r64完了 / 5 | r64平均energy Ha |',
            '|---|---:|---:|---:|---:|']
        for case in CASES:
            high=[json.loads(p.read_text())for p in ROOT.glob('common-runs/S_TN_HIGH_BUDGET_COMPLETION/'+case+'/*/*/result.json')]
            r32=[c['energy_Ha']for result in high for c in result['final_CI']if c['r_cap']==32];r64=[]
            for result in high:
                path=ROOT/'control-CI-extension-v1'/result['lane']/case/result['method_id']/(str(result['budget'])+'-'+str(result['seed']))/'CI-r64.json'
                if path.exists():r64.append(json.loads(path.read_text())['CI']['energy_Ha'])
            lines.append('| '+' | '.join([case,str(len(high)),val(float(np.mean(r32))if len(r32)==5 else None,8),
                str(len(r64)),val(float(np.mean(r64))if len(r64)==5 else None,8)])+' |')
        lines.append('同じ最適化sourceの訓練費用をもう一度加算せず、新しいexact状態準備・sampling・CIの実費を別記録します。100万Shotを使ったから古典費用まで同じという主張はしません。')
    diagnostic=load(ROOT/'TN-MPS-diagnostic-v1/output/result.json')
    lines+=['','## 固定20q回路でのMPS近似の検算','',
        'ナフタレンの採用済みTN回路を固定し、厳密ffsim状態とMPS状態を比較します。係数の再最適化や追加測定はしません。最適化中の最良値と独立finalの差を、そのままMPS誤差と呼ぶことも避けます。',
        'これは主実験開始後の近似診断です。40qを全状態展開せず、この20q結果からN₂/Fe₂S₂のMPS精度や原論文のTN結果を認証しません。','']
    if diagnostic:
        lines+=['| MPS最大bond χ | 状態 | 位相を合わせた状態L2差 | Z確率のL1差 | 正規化state infidelity |',
                '|---:|---|---:|---:|---:|']
        for r in diagnostic['rows']:
            def number(k):return '未取得'if r.get(k)is None else f"{r[k]:.3e}"
            lines.append('| '+' | '.join([str(r['chi']),r['status'],number('global_phase_aligned_L2'),
                number('normalized_Z_probability_L1'),number('normalized_state_infidelity')])+' |')
        lines.append('norm、粒子数sectorの確率、各χのCPU時間も元JSONへ保存しています。小さなbit順検算は独立Qiskit状態と照合しています。値はこの固定回路についての実測で、全回路の誤差上界ではありません。')
        lines.append('χは許容したbondの上限であり、実際に常にその大きさを使ったという意味ではありません。CPUはχごと1回の計時で、実行順・cacheの影響を分離していないため、χを増やすと高速化するという証拠には使いません。')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        complete=[r for r in diagnostic['rows']if r['status']=='COMPLETED']
        if complete:
            fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained')
            for field,label in(('normalized_Z_probability_L1','Z-probability L1 difference'),
                                ('normalized_state_infidelity','Normalized state infidelity')):
                axes[0].semilogy([r['chi']for r in complete],[max(r[field],1e-16)for r in complete],'o-',label=label)
            axes[0].set_xlabel('MPS maximum bond dimension');axes[0].set_ylabel('Measured difference, fixed 20q circuit')
            axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
            axes[1].plot([r['chi']for r in complete],[r['CPU_seconds']for r in complete],'o-')
            axes[1].set_xlabel('MPS maximum bond dimension');axes[1].set_ylabel('CPU seconds, one timing / chi');axes[1].set_ylim(bottom=0);axes[1].grid(alpha=.2)
            fig.suptitle('Fixed selected naphthalene TN circuit; no new shots or training; not a 32q/40q certificate',fontsize=9)
            for ext in('png','svg'):fig.savefig(ROOT/('analysis-v1/TN-fixed20q-MPS-error.'+ext),dpi=180)
            plt.close(fig)
            lines+=['','![20q固定回路のMPS近似差と費用](TN-fixed20q-MPS-error.png)']
    else:lines.append('固定状態の検算は未回収です。未実施・上限到達を誤差0で埋めません。')
    atomic_bytes(ROOT/'analysis-v1/TN_CONTROL_JA.md',('\n'.join(lines)+'\n').encode())
    save_json(ROOT/'analysis-v1/TN-control-report-status.json',{'planned_cases':4,'complete_final_cases':sum(r['independent_final_readouts_completed']==5 for r in rows),
        'whole_experiment_complete':False,'independent_optimizer_replicates_per_case':1})
    print(json.dumps({'final_readouts':sum(r['independent_final_readouts_completed']for r in rows),'planned':20}),flush=True)


if __name__=='__main__':run()
