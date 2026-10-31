"""N25 all-outcome audit/report; no optimization, new draw, or numerical rescue."""
from collections import Counter
from io import BytesIO,StringIO
from pathlib import Path
from time import process_time,perf_counter
import csv,statistics
import numpy as np
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest
from .paired_lowshot import LOCAL,paired_sector,thin_counts,ranked_half,seed
from .joint_ci_sparse import matrices
from paperstudy.lucj import amplitudes_from_saved


def run():
    cpu,wall=process_time(),perf_counter();p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json')
    if t['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or sha256((ROOT/'frontierloop/paired_lowshot.py').read_bytes())!=p['code_SHA256']:raise RuntimeError('Frozen N25 implementation/context changed')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256'] or any(sha256((ROOT/'frontierloop'/name).read_bytes())!=h for name,h in p['helper_SHA256'].items()):raise RuntimeError('Qualified numerical/resource helper changed')
    if len(t['draws'])!=54 or len(t['states'])!=144 or t['actual_new_model_equivalent_samples']!=12000 or t['actual_new_uniform_classical_samples']!=12000 or t['retrospective_old_counts_subsets']!=30:raise RuntimeError('All planned draw/readout accounting required')
    if load(LOCAL/'draws.json')!=t['draws'] or load(LOCAL/'resources.json')!=t['resources']:raise RuntimeError('Terminal/actual file linkage differs')
    if len({(r['case'],r['method'],r['replicate'],r['arm']) for r in t['states']})!=144:raise RuntimeError('Duplicate/missing state identity')
    expected={(o['case'],o['source'],p['replicates'][o['replicate']-160101],arm) for o in p['originals'] for arm in ('R32_CAP','COMMON_R')}
    expected|={(s['case'],m,rep,arm) for s in p['sources'] for m in ('PAIR_T2','UNIFORM_SECTOR','CLASSICAL_HF_HCI_MARGINAL_ORACLE') for rep in p['replicates'] for arm in ('R32_CAP','COMMON_R')}
    if expected!={(r['case'],r['method'],r['replicate'],r['arm']) for r in t['states']}:raise RuntimeError('All sources incl missing required')
    sources={s['case']:s for s in p['sources']};paired={};audit=[]
    for case,s in sources.items():
        for path,hkey in [('amplitudes','amplitudes_SHA256'),('initialization_record_path','initialization_record_SHA256')]:
            if sha256(Path(s[path]).read_bytes())!=s[hkey]:raise RuntimeError('Saved initialization changed')
        n=s['definition']['norb'];k=s['definition']['nelec'][0];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
        ids,v,angles=paired_sector(n,k,t2)
        file=LOCAL/'sources'/case/'pair-state.npz'
        with np.load(file,allow_pickle=False) as z:
            if not np.array_equal(ids,z['half_strings']) or not np.array_equal(v,z['amplitudes']):raise RuntimeError('Actual pair sector differs')
        if load(LOCAL/'sources'/case/'angles.json')['angles']!=angles:raise RuntimeError('Actual initialized angles differ')
        paired[case]=(ids,v,sha256(file.read_bytes()))
    for r in t['draws']:
        case=r['case'];s=sources[case];n=s['definition']['norb'];k=s['definition']['nelec'][0];file=Path(r['path'])
        if sha256(file.read_bytes())!=r['SHA256']:raise RuntimeError('Actual new/subset counts changed')
        with np.load(file,allow_pickle=False) as z:actual_ids=z['blocked_ids'];actual_counts=z['counts']
        if 'RETROSPECTIVE' in r['sample_role']:
            old=r['old_source'];oldpath=Path(old['payload'])
            if sha256(oldpath.read_bytes())!=old['payload_SHA256']:raise RuntimeError('Original100k counts changed')
            expectedseed=seed(case,r['method'],r['replicate'],old['payload_SHA256'])
            with np.load(oldpath,allow_pickle=False) as z:ids,counts=thin_counts(z['blocked_ids'],z['counts'],1000,expectedseed)
        else:
            half,v,state_sha=paired[case];is_pair=r['method']=='PAIR_T2'
            expectedseed=seed(case,r['method'],r['replicate'],state_sha if is_pair else s['definition']['problem_id']);rng=np.random.default_rng(expectedseed)
            a=rng.choice(half,size=1000,p=v*v if is_pair else None);b=a if is_pair else rng.choice(half,size=1000)
            ids,counts=np.unique(a|(b<<np.uint64(n)),return_counts=True)
        if expectedseed!=r['seed'] or not np.array_equal(actual_ids,ids) or not np.array_equal(actual_counts,counts):raise RuntimeError('Counts not deterministic from frozen context')
        if int(counts.sum())!=1000 or r['rank']!=ranked_half(ids,counts,n,(1<<k)-1):raise RuntimeError('Wrong observed pool/ranking')
        if any((int(x)&((1<<n)-1)).bit_count()!=k or (int(x)>>n).bit_count()!=k for x in ids):raise RuntimeError('Actual sample wrong sector')
    all_resources={}
    for r in t['resources']:
        for o in r['compiler_points']:
            for role,f in o['circuits'].items():
                blob=Path(f['path']).read_bytes()
                if sha256(blob)!=f['SHA256']:raise RuntimeError('Actual QPY changed')
                q=qpy.load(BytesIO(blob))[0]
                if q.num_qubits!=r['n_qubits']:raise RuntimeError('Encoding changed')
                if role=='sampling':
                    z=dict(CNOT=q.count_ops().get('cx',0),CNOT_Depth=q.depth(filter_function=lambda i:i.operation.name=='cx'),Total_Depth=q.depth())
                    if any(o['resources'][k]!=v for k,v in z.items()):raise RuntimeError('Actual resource recount differs')
        all_resources[r['case'],r['view']]=r
    control_ranks={}
    for c in p['classical_controls']:
        path=Path(c['vector'])
        if sha256(path.read_bytes())!=c['vector_SHA256']:raise RuntimeError('Classical oracle root changed')
        s=sources[c['case']];n=s['definition']['norb'];k=s['definition']['nelec'][0]
        with np.load(path,allow_pickle=False) as a:
            pairs=a['joint_pairs'];v=a['coefficients']
            if abs(v@v-1)>1e-10:raise RuntimeError('Classical root norm changed')
            control_ranks[c['case']]=ranked_half(pairs[:,0]|(pairs[:,1]<<np.uint64(n)),v*v,n,(1<<k)-1)
    ranked={(r['case'],r['method'],r['replicate']):r['rank'] for r in t['draws']}
    for case,rank in control_ranks.items():
        for rep in p['replicates']:ranked[case,'CLASSICAL_HF_HCI_MARGINAL_ORACLE',rep]=rank
    missing={(o['case'],o['source'],p['replicates'][o['replicate']-160101]) for o in p['originals'] if o['status']!='COMPLETED'}
    roots={};cost=load(LOCAL/'CI-attempts.json');rows=[]
    for case,s in sources.items():
        d=s['definition'];file=Path(s['local_integrals'])
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical input changed')
        with np.load(file,allow_pickle=False) as a:h=a['h1'];g=a['eri']
        for r in [r for r in t['states'] if r['case']==case]:
            row=dict(r);row.pop('result',None)
            ident=case,r['method'],r['replicate']
            if ident in missing:
                if r['status']!='MISSING_SAME_PROBLEM_SOURCE':raise RuntimeError('Scientific missingness changed')
            else:
                rank=ranked[ident]
                common=min(32,*(len(v) for (c,m,rep),v in ranked.items() if c==case and rep==r['replicate']))
                requested=min(32,len(rank)) if r['arm']=='R32_CAP' else common
                if r['R']!=requested or r['half_strings']!=sorted(rank[:requested]):raise RuntimeError('CI basis not derived from its frozen source ranking')
            if r['status']=='COMPLETED':
                file=Path(r['root_path']);blob=file.read_bytes()
                if sha256(blob)!=r['root_SHA256']:raise RuntimeError('Saved CI root changed')
                with np.load(file,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
                expected_pairs=np.array([(a,b) for a in r['half_strings'] for b in r['half_strings']],np.uint64)
                if not np.array_equal(pairs,expected_pairs) or digest([case,pairs.tolist()])!=r['basis_id'] or len(pairs)!=r['actual_D'] or len(pairs)!=r['R']**2:raise RuntimeError('Exact CI dimension/basis identity differs')
                if r['basis_id'] not in roots:
                    H,S,info=matrices(h,g,tuple(d['nelec']),pairs,p['library'],seconds=30)
                    e=float(v@(H@v)+d['core_energy']);ss=float(v@(S@v));obj=e+.1*ss
                    residual=float(np.linalg.norm((H+.1*S)@v-(obj-d['core_energy'])*v))
                    if abs(v@v-1)>1e-10 or residual>1e-8 or abs(e-r['result']['energy_Ha'])>1e-8 or abs(obj-r['result']['penalized_energy_Ha'])>1e-8:raise RuntimeError('Actual root physical consistency failed')
                    roots[r['basis_id']]=dict(energy_Ha=e,spin_square=ss,residual=residual)
                if abs(roots[r['basis_id']]['energy_Ha']-r['result']['energy_Ha'])>1e-8 or abs(roots[r['basis_id']]['spin_square']-r['result']['spin_square'])>1e-8:raise RuntimeError('Alias result not equal to actual shared root')
                row.update(energy_Ha=r['result']['energy_Ha'],penalized_energy_Ha=r['result']['penalized_energy_Ha'],spin_square=r['result']['spin_square'],CI_CPU_seconds=r['result']['CPU_seconds'])
            else:row['energy_Ha']=None
            # Reporting-only reference access after source/CI work has finished.
            from .analyze import reference
            hf,ref=reference(case);meta=load(OLD/'data/evaluation-only'/(case+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Reporting reference problem differs')
            row.update(recovery_pct=100*(hf-row['energy_Ha'])/(hf-ref) if row['energy_Ha'] is not None else None,
                reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED')
            rows.append(row)
        for replica in p['replicates']:
            group=[r for r in rows if r['case']==case and r['replicate']==replica and r['arm']=='COMMON_R' and r['status']!='MISSING_SAME_PROBLEM_SOURCE']
            if len({r['actual_D'] for r in group})!=1:raise RuntimeError('COMMON_R did not match actual dimension')
    if len(roots)!=t['actual_new_CI_roots'] or sum(r.get('new_CI_solve') is True for r in rows)!=len(roots) or sum(a['status']=='FAILED_PRESERVED' for a in cost)!=t['failed_CI_attempts']:raise RuntimeError('Actual CI work/dedup counts differ')
    if len({a['basis_id'] for a in cost})!=len(cost) or {a['basis_id'] for a in cost if a['status']=='COMPLETED'}!=set(roots):raise RuntimeError('CI attempts not bound to actual unique root set')
    # Recount every archived original LUCJ opt1/opt3 point, not just its winner.
    from .audit_prefix_sampling import all_control_outputs
    baseline_resources=[]
    for case,method in sorted({(o['case'],o['source']) for o in p['originals']}):
        old=next(o for o in p['originals'] if (o['case'],o['source'])==(case,method))
        if old['status']!='COMPLETED':baseline_resources.append(dict(case=case,method=method,status='MISSING_SAME_PROBLEM_SOURCE',points=[]));continue
        points=all_control_outputs(case,method,None)
        for point in points:
            field='original_opt1_resources' if point['resources']['optimization_level']==1 else 'original_opt3_resources'
            if point['resources']!=old[field]:raise RuntimeError('Actual original LUCJ compiler point differs')
        baseline_resources.append(dict(case=case,method=method,status='RECOUNTED_ARCHIVED_QPYS',points=[o['resources'] for o in points],declared_parameters=old['declared_parameters']))
    groups=[]
    for key in sorted({(r['case'],r['method'],r['arm']) for r in rows}):
        rs=[r for r in rows if (r['case'],r['method'],r['arm'])==key];good=[r for r in rs if r['energy_Ha'] is not None]
        if len(rs)!=3:raise RuntimeError('Every group needs all3 planned replicate states')
        mean=lambda field:statistics.mean(r[field] for r in good) if good else None
        groups.append(dict(case=key[0],method=key[1],arm=key[2],planned=3,completed=len(good),
            energy_Ha_mean=mean('energy_Ha'),energy_SD_mHa=1000*statistics.stdev(r['energy_Ha'] for r in good) if len(good)>1 else None,
            recovery_pct_mean=mean('recovery_pct'),actual_Ds=[r.get('actual_D') for r in rs],R_values=[r.get('R') for r in rs],
            distinct_roots=len({r['basis_id'] for r in good}),actual_CI_CPU_mean=mean('CI_CPU_seconds'),
            independent_confirmation=False,statistical_superiority_certified=False))
    comparison=[]
    for case in sources:
        for arm in ('COMMON_R','R32_CAP'):
            pair=next(g for g in groups if (g['case'],g['method'],g['arm'])==(case,'PAIR_T2',arm))
            lc=[g for g in groups if g['case']==case and g['method'].startswith('C5_') and g['arm']==arm and g['completed']==3]
            oracle=next(g for g in groups if (g['case'],g['method'],g['arm'])==(case,'CLASSICAL_HF_HCI_MARGINAL_ORACLE',arm))
            best=min(lc,key=lambda g:g['energy_Ha_mean'])
            complete=pair['completed']==oracle['completed']==3
            comparison.append(dict(case=case,arm=arm,pair_energy_Ha_mean=pair['energy_Ha_mean'],best_LUCJ=best['method'],
                pair_minus_best_LUCJ_mHa=1000*(pair['energy_Ha_mean']-best['energy_Ha_mean']) if complete else None,
                pair_minus_classical_oracle_mHa=1000*(pair['energy_Ha_mean']-oracle['energy_Ha_mean']) if complete else None,
                all_available_strong_LUCJ_controls_included=True,matched_actual_D=arm=='COMMON_R',cold_matched_work=False))
    out=dict(status='N25_ALL144_STATES54_DRAW_VIEWS_AND_SAVED_ROOTS_AUDITED',rows=rows,groups=groups,comparisons=comparison,
        CI_status_counts=dict(Counter(r['status'] for r in rows)),actual_new_CI_roots=len(roots),failed_CI_attempts=t['failed_CI_attempts'],
        actual_new_pair_model_equivalent_samples=12000,actual_new_uniform_classical_samples=12000,retrospective_1000_of100000_subsets=30,
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],CI_attempt_CPU_seconds=sum(a['CPU_seconds'] for a in cost),
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,deterministic_reconstructed_counts_not_new_experimental_draws=True,
        initialization_costs_additional=[s['initialization_record'] for s in p['sources']],old_HCI_costs_additional=p['classical_controls'],
        resources=t['resources'],all_original_LUCJ_compiler_points=baseline_resources,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((LOCAL/'terminal.json').read_bytes()),
        new_QPU_shots=0,GPU_allocations=0,original_paper_reproduction=False,independent_confirmation=False,goal_complete=False)
    save(ROOT/'analysis/N25-lowshot-pair-summary.json',out)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N25-all144-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    fmt=lambda x,n=3:'未取得' if x is None else f'{x:.{n}f}'
    lines=['# N25：低Shotのpair sourceと、同じ実CI次元での比較','',
        f"全144予定readout：{out['CI_status_counts']}。実際の新CI根{len(roots)}、失敗CI {t['failed_CI_attempts']}。",'各sourceのreadout入力は1,000 counts。LUCJは既存100,000からのsubsetで、旧実費を消した1,000-shot実行ではない。',
        f"新pair12,000 model-equivalent draws、新uniform12,000古典draws。QPU/GPU0。実行CPU {t['CPU_seconds']:.3f}秒、wall {t['wall_seconds']:.3f}秒。",'',
        '| 条件 | 方法 | arm | 完了/3 | R各反復 | CI実D各反復 | energy平均 Ha | SD mHa | Recovery % |',
        '|---|---|---|---:|---|---|---:|---:|---:|']
    for g in groups:lines.append('| '+' | '.join([g['case'],g['method'],g['arm'],str(g['completed']),str(g['R_values']),str(g['actual_Ds']),fmt(g['energy_Ha_mean'],9),fmt(g['energy_SD_mHa']),fmt(g['recovery_pct_mean'])])+' |')
    lines+=['','## 全利用可能L1/L2/L4を残した比較','',
        '| 条件 | arm | 最良LUCJ平均のsource | pair − LUCJ mHa | pair − 古典oracle mHa |',
        '|---|---|---|---:|---:|']
    for r in comparison:lines.append('| '+' | '.join([r['case'],r['arm'],r['best_LUCJ'],fmt(r['pair_minus_best_LUCJ_mHa']),fmt(r['pair_minus_classical_oracle_mHa'])])+' |')
    lines+=['','負はpairのenergyが低い。これは事後の記述比較で、統計的優越・独立確認ではない。bestという表示で他のLUCJ行を隠さない。',
        'COMMON_Rは条件/反復内で全法の実Dが同一。ただし弱いsourceの小poolでRが下がる。R32_CAPも残し、低次元だけでの勝利を主張しない。',
        '古典oracleは既存HCI根を使い、旧HCI費用と大きい空間の情報を保持する。cold matched-workではない。新しいHCI探索は行っていない。','',
        '## pair回路の実係数資源（合成N24ではない）','',
        '| 条件 | 表現 | qubit | opt | CNOT | CX Depth | Total Depth | 宣言P | 非ゼロ初期角 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in t['resources']:
        for o in r['compiler_points']:
            z=o['resources'];lines.append(f"| {r['case']} | {r['view']} | {r['n_qubits']} | {z['optimization_level']} | {z['CNOT']} | {z['CNOT_Depth']} | {z['Total_Depth']} | {r['declared_parameters']} | {r['nonzero_initialized_angles']} |")
    lines+=['','## 元LUCJの全compiler点（保存回路を再計数）','',
        '| 条件 | source | opt | CNOT | CX Depth | Total Depth | 宣言P |','|---|---|---:|---:|---:|---:|---:|']
    for r in baseline_resources:
        if not r['points']:lines.append(f"| {r['case']} | {r['method']} | 欠測 | — | — | — | — |");continue
        for z in r['points']:lines.append(f"| {r['case']} | {r['method']} | {z['optimization_level']} | {z['CNOT']} | {z['CNOT_Depth']} | {z['Total_Depth']} | {r['declared_parameters']} |")
    lines+=['','## 読み方と限界','',
        'ナフタレン/近N₂はCCSD系、遠N₂/Feは登録CISD fallback。t1とbroken-pair t2を落とす、軌道依存の既知pair近似であり、初期化後のVQE最適化はしていない。',
        'paired stateは最大15,504振幅までの限定sectorで計算した。汎用40qシミュレーションでも量子優位でもない。',
        'CI根はclassical readout後の状態。pair回路がbroken-pairを含む最終CI状態を準備した意味ではない。古典対照のCNOT/Depth/Pを0としない。',
        '宣言Pは登録したpair edge変数の個数。非ゼロ初期角、数値gate slots、optimizer変数0とは別。最小Jacobian rankを証明していない。',
        '有限H+0.1S²ペナルティを使う。physical Hとpenalized Eを分け、厳密singletとは呼ばない。Fe SCI参照とN₂ archive参照の精度は未認証。',
        'Recoveryはenergy換算で独立な性能指標ではない。既存全100k sampling/初期化・oracle作成・監査費用は追加。原著TN最適化/SqDRIFT最高予算の再現・論文優越・全goal達成は未証明。']
    write(ROOT/'analysis/N25_PAIR_LOW_SHOT_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['CI_status_counts'],out['actual_new_CI_roots'],flush=True)


if __name__=='__main__':run()
