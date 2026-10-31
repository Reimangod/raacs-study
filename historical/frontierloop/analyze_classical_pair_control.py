"""N26 saved-root audit and comparison at N25's actual dimensions."""
from collections import Counter
from pathlib import Path
from time import process_time,perf_counter
from io import StringIO
import csv,statistics
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest
from .classical_pair_control import LOCAL,PARENT,rank_pair_amplitudes,requested_plans
from .joint_ci_sparse import matrices
from paperstudy.lucj import amplitudes_from_saved


def run():
    cpu,wall=process_time(),perf_counter();p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json')
    if t['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or sha256((ROOT/'frontierloop/classical_pair_control.py').read_bytes())!=p['code_SHA256']:raise RuntimeError('Frozen control/context differs')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256'] or any(sha256((ROOT/'frontierloop'/name).read_bytes())!=h for name,h in p['helper_SHA256'].items()):raise RuntimeError('Qualified math changed')
    if sha256((OLD/'paperstudy/lucj.py').read_bytes())!=p['initializer_helper_SHA256']:raise RuntimeError('Initializer helper changed')
    parent=load(PARENT/'terminal.json');summary=load(ROOT/'analysis/N25-lowshot-pair-summary.json')
    if sha256((PARENT/'terminal.json').read_bytes())!=p['parent_terminal_SHA256'] or sha256((ROOT/'analysis/N25-lowshot-pair-summary.json').read_bytes())!=p['parent_summary_SHA256']:raise RuntimeError('Actual parent/summary changed')
    expected=[r for s in p['sources'] for r in requested_plans(s['case'],parent['draws'])]
    if expected!=p['plans'] or len(t['rows'])!=16 or len({(r['case'],r['arm'],r['replicate']) for r in t['rows']})!=16:raise RuntimeError('All frozen control points required')
    if load(LOCAL/'rankings.json')!=t['rankings']:raise RuntimeError('Saved rankings differ')
    rankings={};roots={};rows=[];pairwise=[]
    for s in p['sources']:
        case=s['case'];d=s['definition'];n=d['norb'];k=d['nelec'][0]
        for name,hname in [('amplitudes','amplitudes_SHA256'),('initialization_record_path','initialization_record_SHA256')]:
            if sha256(Path(s[name]).read_bytes())!=s[hname]:raise RuntimeError('Same physical initialization required')
        _,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k);rank,scores=rank_pair_amplitudes(n,k,t2)
        actual=next(r for r in t['rankings'] if r['case']==case)
        if actual['rank']!=rank or actual['scores']!=scores or actual['amplitude_kind']!=kind:raise RuntimeError('Classical input-only ranking changed')
        rankings[case]=actual
        file=Path(s['local_integrals'])
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
        with np.load(file,allow_pickle=False) as a:h=a['h1'];g=a['eri']
        for r in [r for r in t['rows'] if r['case']==case]:
            expected_plan=next(a for a in p['plans'] if (a['case'],a['arm'],a['replicate'])==(case,r['arm'],r['replicate']))
            if any(r[k]!=v for k,v in expected_plan.items()):raise RuntimeError('Plan/result identity differs')
            R=expected_plan['R'] if r['arm']=='MATCHED_N25_COMMON_R' else min(32,len(rank))
            if r['actual_R']!=R or r['quantum_resources'] is not None:raise RuntimeError('Actual R or non-applicable quantum resource differs')
            row={k:v for k,v in r.items() if k!='result'}
            if r['status']=='COMPLETED':
                strings=sorted(rank[:R]);expected_pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                file=Path(r['root_path']);blob=file.read_bytes()
                if sha256(blob)!=r['root_SHA256']:raise RuntimeError('Actual root changed')
                with np.load(file,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
                key=digest([case,pairs.tolist()])
                if not np.array_equal(pairs,expected_pairs) or key!=r['basis_id'] or len(pairs)!=R*R or r['actual_D']!=len(pairs):raise RuntimeError('Wrong source-to-basis or actual CI dimension')
                if key not in roots:
                    H,S,info=matrices(h,g,tuple(d['nelec']),pairs,p['library'],seconds=30)
                    e=float(v@(H@v)+d['core_energy']);ss=float(v@(S@v));obj=e+.1*ss
                    residual=float(np.linalg.norm((H+.1*S)@v-(obj-d['core_energy'])*v))
                    if abs(v@v-1)>1e-10 or residual>1e-8:raise RuntimeError('Actual saved-root parity failed')
                    roots[key]=dict(energy_Ha=e,spin_square=ss,penalized_energy_Ha=obj,residual=residual)
                if max(abs(roots[key][field]-r['result'][field]) for field in ('energy_Ha','spin_square','penalized_energy_Ha'))>1e-8:raise RuntimeError('Reported physical/root consistency differs')
                row.update(roots[key],CI_CPU_seconds=r['result']['CPU_seconds'])
                if r['arm']=='MATCHED_N25_COMMON_R':
                    old=[x for x in summary['rows'] if (x['case'],x['replicate'],x['arm'],x['status'])==(case,r['replicate'],'COMMON_R','COMPLETED')]
                    if not old or any(x['actual_D']!=row['actual_D'] for x in old):raise RuntimeError('Unequal actual dimension in primary contrast')
                    for x in old:pairwise.append(dict(case=case,replicate=r['replicate'],comparator=x['method'],
                        actual_D=row['actual_D'],classical_energy_Ha=row['energy_Ha'],comparator_energy_Ha=x['energy_Ha'],
                        classical_minus_comparator_mHa=1000*(row['energy_Ha']-x['energy_Ha']),same_actual_D=True,
                        same_raw_observations=False,same_classical_amplitude_information_available=True,cold_matched_work=False))
            else:row.update(energy_Ha=None,spin_square=None,penalized_energy_Ha=None)
            from .analyze import reference
            hf,ref=reference(case);meta=load(OLD/'data/evaluation-only'/(case+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Reporting reference problem differs')
            row.update(recovery_pct=100*(hf-row['energy_Ha'])/(hf-ref) if row['energy_Ha'] is not None else None,
                reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED')
            rows.append(row)
    attempts=load(LOCAL/'CI-attempts.json')
    if len(roots)!=t['actual_new_CI_roots'] or {a['basis_id'] for a in attempts if a['status']=='COMPLETED'}!=set(roots) or len(attempts)!=len({a['basis_id'] for a in attempts}):raise RuntimeError('Actual unique root accounting differs')
    if sum(r.get('new_CI_solve') is True for r in rows)!=len(roots):raise RuntimeError('Alias accounting differs')
    comparisons=[]
    for case in rankings:
        good=[r for r in rows if r['case']==case and r['arm']=='MATCHED_N25_COMMON_R' and r['status']=='COMPLETED']
        contrast=[r for r in pairwise if r['case']==case];means={}
        for method in {r['comparator'] for r in contrast}:means[method]=statistics.mean(r['classical_minus_comparator_mHa'] for r in contrast if r['comparator']==method)
        oldgroups=[g for g in summary['groups'] if g['case']==case and g['arm']=='COMMON_R' and g['method'].startswith('C5_') and g['completed']==3]
        best=min(oldgroups,key=lambda g:g['energy_Ha_mean'])
        comparisons.append(dict(case=case,matched_R_values=[r['actual_R'] for r in good],actual_D_values=[r['actual_D'] for r in good],
            classical_energy_Ha_mean=statistics.mean(r['energy_Ha'] for r in good) if good else None,
            classical_minus_pair_mHa=means.get('PAIR_T2'),classical_minus_best_LUCJ_mHa=means.get(best['method']),best_LUCJ=best['method'],
            classical_minus_HCI_oracle_mHa=means.get('CLASSICAL_HF_HCI_MARGINAL_ORACLE'),
            distinct_classical_roots=len({r['basis_id'] for r in good}),independent_stochastic_replicates=False))
    out=dict(status='N26_ALL16_INPUT_ONLY_CLASSICAL_RANKINGS_AND_SAVED_ROOTS_AUDITED',rows=rows,pairwise=pairwise,comparisons=comparisons,
        status_counts=dict(Counter(r['status'] for r in rows)),actual_new_CI_roots=len(roots),failed_CI_attempts=t['failed_CI_attempts'],
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],CI_attempt_CPU_seconds=t['CI_attempt_CPU_seconds'],ranking_CPU_seconds=sum(r['CPU_seconds'] for r in t['rankings']),
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        initial_classical_costs_retained=[s['initialization_record'] for s in p['sources']],
        no_new_Shot_QPU_GPU_HCI_optimizer_work=True,quantum_resources=None,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((LOCAL/'terminal.json').read_bytes()),
        independent_confirmation=False,statistical_or_causal_necessity_proof=False,goal_complete=False)
    save(ROOT/'analysis/N26-classical-pair-control-summary.json',out)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N26-all16-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x,n=3:'未取得' if x is None else f'{x:.{n}f}'
    lines=['# N26：同じt2係数を古典的に使った場合','',
        f"全16点：{out['status_counts']}。実際の新CI根{len(roots)}、CI失敗{t['failed_CI_attempts']}。新Shot/QPU/GPU/HCI探索/optimizerは0。",'',
        '| 条件 | 同じ実DのR列 | 古典t2 energy平均 Ha | 古典t2 − pair mHa | 古典t2 − 最良LUCJ mHa | 古典t2 − HCI oracle mHa |',
        '|---|---|---:|---:|---:|---:|']
    for r in comparisons:lines.append('| '+' | '.join([r['case'],str(r['matched_R_values']),fmt(r['classical_energy_Ha_mean'],9),fmt(r['classical_minus_pair_mHa']),fmt(r['classical_minus_best_LUCJ_mHa']),fmt(r['classical_minus_HCI_oracle_mHa'])])+' |')
    lines+=['','負は古典t2選抜のenergyが低い。比較対象の全3反復のRへ結合した記述比較。同じRの古典結果は再利用され、3つの独立な古典成功とはしない。',
        '同じ実CI次元、同じ初期係数情報だが、同じ観測データではない。古典法は既知t2を直接使い、pair/LUCJは有限countsを使う。この違いを隠さず、同じcold総費用や量子優位の証明とはしない。','',
        '| 条件 | arm | 対応反復 | 実R | 実D | energy Ha | Recovery % | 新CI根か |','|---|---|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append('| '+' | '.join([r['case'],r['arm'],str(r['replicate']),str(r['actual_R']),str(r.get('actual_D')),fmt(r['energy_Ha'],9),fmt(r['recovery_pct']),str(r.get('new_CI_solve'))])+' |')
    lines+=['','## 費用・限界','',
        f"実行CPU {t['CPU_seconds']:.6f}秒、wall {t['wall_seconds']:.6f}秒、CI attempt {t['CI_attempt_CPU_seconds']:.6f} CPU秒、ranking {out['ranking_CPU_seconds']:.6f} CPU秒。監査費用はJSONへ別記。", 
        '既存CCSD/CISD初期化と失敗費用は別途必要。N25との総時間差にはCI数・入力数の違いもあるので、guard改善だけのmatched speedupとはしない。',
        '古典選抜に量子回路はない。CNOT/Depth/Parametersは未適用であり、古典CI解を0ゲートで準備できた意味ではない。',
        'N₂/FeのCISD fallback、ナフタレンblock-Boys、有限H+0.1S²、SCI/FCI archive参照の精度限界はN25と共通。Recoveryはenergy換算で独立指標ではない。',
        'CLASSICAL_R32_CAPのナフタレンsupportは最大26。元LUCJ R32=1024と等しいCI次元とは呼ばない。',
        '既知の振幅に基づく古典対照であり、新しい量子ansatz・原著最高予算再現・独立確認・最強論文への勝利は主張しない。',
        '[前回の全LUCJ/pair/uniform/古典oracle結果](N25_PAIR_LOW_SHOT_RESULTS_JA.md)を残し、欠測・悪い結果を除外しない。']
    write(ROOT/'analysis/N26_CLASSICAL_PAIR_CONTROL_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['status_counts'],len(roots),flush=True)


if __name__=='__main__':run()
