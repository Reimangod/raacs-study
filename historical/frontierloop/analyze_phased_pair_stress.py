"""N28 all-candidate/circuit/root review; no molecular winner is hidden."""
from io import BytesIO,StringIO
from pathlib import Path
from time import process_time,perf_counter
from collections import Counter
import csv
import numpy as np
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest
from .phased_pair_stress import LOCAL,PARENT,angles_for,evolve
from .dephased_pair_diagnostic import dephased_distribution,top_half
from .joint_ci_sparse import matrices


def run():
    cpu,wall=process_time(),perf_counter();p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json')
    if t['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or sha256((ROOT/'frontierloop/phased_pair_stress.py').read_bytes())!=p['code_SHA256']:raise RuntimeError('Frozen N28 code/context changed')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256'] or any(sha256((ROOT/'frontierloop'/n).read_bytes())!=h for n,h in p['helper_SHA256'].items()):raise RuntimeError('Frozen math/resource helper changed')
    expected={(s['case'],scale,m) for s in p['sources'] for scale in p['scales'] for m in ('COHERENT_REAL','COHERENT_FLUX','MARKOV')}
    if len(t['rows'])!=36 or {(r['case'],r['scale'],r['method']) for r in t['rows']}!=expected or len(t['sources'])!=24 or sum(len(s['all_compiler_points']) for s in t['sources'])!=96:raise RuntimeError('All36states24sources96resourcepoints required')
    if load(LOCAL/'sources.json')!=t['sources']:raise RuntimeError('Actual source manifest differs')
    old=load(ROOT/'analysis/N27-dephased-pair-summary.json')
    if sha256((ROOT/'analysis/N27-dephased-pair-summary.json').read_bytes())!=p['parent_summary_SHA256']:raise RuntimeError('Audited parent context differs')
    baseline=load(ROOT/'analysis/N25-lowshot-pair-summary.json');roots={};rows=[];diagnostics=[];parities=[]
    sources={(s['case'],s['scale'],s['pattern']):s for s in t['sources']}
    for original in p['sources']:
        case=original['case'];d=original['definition'];n=d['norb'];k=d['nelec'][0];hf=(1<<k)-1
        for field,hkey in [('state','state_SHA256'),('angles','angles_SHA256')]:
            if sha256(Path(original[field]).read_bytes())!=original[hkey]:raise RuntimeError('Original angle/state input changed')
        with np.load(original['state'],allow_pickle=False) as z:ids=z['half_strings'];oldv=z['amplitudes']
        base=load(original['angles'])['angles'];file=Path(original['integrals'])
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integral identity changed')
        with np.load(file,allow_pickle=False) as z:h=z['h1'];g=z['eri']
        for scale in p['scales']:
            probs={};markov=dephased_distribution(ids,k,angles_for(base,k,scale,'REAL'))
            for pattern in p['patterns']:
                source=sources[case,scale,pattern];angles=angles_for(base,k,scale,pattern);v=evolve(ids,k,angles);prob=abs(v)**2;file=Path(source['source_file'])
                if source['angles']!=angles or source['declared_real_pair_variables']!=k*(n-k) or sha256(file.read_bytes())!=source['source_SHA256']:raise RuntimeError('Registered complex source differs')
                with np.load(file,allow_pickle=False) as z:
                    if not np.array_equal(ids,z['half_strings']) or not np.array_equal(v,z['amplitudes']) or not np.array_equal(prob,z['probabilities']):raise RuntimeError('Actual complex amplitude/probability differs')
                if scale==1 and pattern=='REAL' and np.max(abs(v-oldv))>1e-12:raise RuntimeError('Original coherent amplitude parity failed')
                tv=float(np.sum(abs(prob-markov)))/2;hfp=float(prob[list(map(int,ids)).index(hf)])
                if abs(tv-source['single_draw_TV_vs_same_scale_Markov'])>1e-14 or abs(hfp-source['HF_probability'])>1e-14:raise RuntimeError('Source probability diagnostic differs')
                for point in source['all_compiler_points']:
                    for role,f in point['circuits'].items():
                        blob=Path(f['path']).read_bytes()
                        if sha256(blob)!=f['SHA256']:raise RuntimeError('Actual compiled QPY changed')
                        q=qpy.load(BytesIO(blob))[0];expected_n=2*n if point['encoding']=='FULL_SPIN_COPY' else n
                        if q.num_qubits!=expected_n or q.num_qubits!=point['n_qubits']:raise RuntimeError('Compiler encoding mismatch')
                        if role=='sampling':
                            measured=dict(CNOT=q.count_ops().get('cx',0),CNOT_Depth=q.depth(filter_function=lambda i:i.operation.name=='cx'),Total_Depth=q.depth())
                            if any(point['resources'][key]!=value for key,value in measured.items()):raise RuntimeError('Actual compiler resource recount differs')
                probs['COHERENT_'+pattern]=prob
            probs['MARKOV']=markov
            for method,prob in probs.items():
                r=next(r for r in t['rows'] if (r['case'],r['scale'],r['method'])==(case,scale,method));row={key:value for key,value in r.items() if key!='result'}
                file=Path(r['source_file'])
                if sha256(file.read_bytes())!=r['source_SHA256']:raise RuntimeError('Readout bound to changed source')
                with np.load(file,allow_pickle=False) as z:
                    if not np.array_equal(z['half_strings'],ids) or not np.array_equal(z['probabilities'],prob):raise RuntimeError('Readout source probabilities differ')
                strings,margin=top_half(ids,prob,hf,p['R'])
                if strings!=r['half_strings'] or margin!=r['probability_rank_margin'] or r['near_tie']!=(margin<=p['near_tie_margin']):raise RuntimeError('Source ranking/tie semantics differ')
                if r['status']=='COMPLETED':
                    file=Path(r['root_path'])
                    if sha256(file.read_bytes())!=r['root_SHA256']:raise RuntimeError('Actual root bytes changed')
                    with np.load(file,allow_pickle=False) as z:pairs=z['joint_pairs'];c=z['coefficients']
                    expected_pairs=np.array([(a,b) for a in strings for b in strings],np.uint64);key=digest([case,pairs.tolist()])
                    if not np.array_equal(pairs,expected_pairs) or key!=r['basis_id'] or len(pairs)!=1024 or r['actual_D']!=1024:raise RuntimeError('Actual matched dimension/basis differs')
                    if key not in roots:
                        H,S,info=matrices(h,g,tuple(d['nelec']),pairs,p['library'],seconds=20)
                        e=float(c@(H@c)+d['core_energy']);ss=float(c@(S@c));obj=e+.1*ss
                        residual=float(np.linalg.norm((H+.1*S)@c-(obj-d['core_energy'])*c))
                        if abs(c@c-1)>1e-10 or residual>1e-8:raise RuntimeError('Actual CI root norm/residual failure')
                        roots[key]=dict(energy_Ha=e,spin_square=ss,penalized_energy_Ha=obj,residual=residual)
                    if max(abs(roots[key][f]-r['result'][f]) for f in ('energy_Ha','spin_square','penalized_energy_Ha'))>1e-8:raise RuntimeError('Actual root physical consistency differs')
                    row.update(roots[key])
                    if scale==1 and method in ('COHERENT_REAL','MARKOV'):
                        parentmethod='COHERENT_EXACT_PROB' if method=='COHERENT_REAL' else 'DEPHASED_EXACT_PROB'
                        previous=next(x for x in old['rows'] if (x['case'],x['method'],x['R'])==(case,parentmethod,32))
                        if row['basis_id']!=previous['basis_id'] or abs(row['energy_Ha']-previous['energy_Ha'])>1e-8:raise RuntimeError('Old same-source R32 parity failed')
                        parities.append(dict(case=case,method=method,absolute_energy_error_Ha=abs(row['energy_Ha']-previous['energy_Ha']),identical_basis=True))
                else:row['energy_Ha']=None
                from .analyze import reference
                hf_energy,ref=reference(case);meta=load(OLD/'data/evaluation-only'/(case+'.json'))
                if meta['problem_id']!=d['problem_id']:raise RuntimeError('Reporting reference mismatch')
                row.update(recovery_pct=100*(hf_energy-row['energy_Ha'])/(hf_energy-ref) if row['energy_Ha'] is not None else None,
                    reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_NOT_FCI',reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') else 'UNVERIFIED')
                quantum=method!='MARKOV';source=sources[case,scale,method.removeprefix('COHERENT_')] if quantum else None
                point=next(x for x in source['all_compiler_points'] if x['encoding']=='FULL_SPIN_COPY' and x['resources']['optimization_level']==1) if quantum else None
                row.update(CNOT=point['resources']['CNOT'] if quantum else None,CNOT_Depth=point['resources']['CNOT_Depth'] if quantum else None,
                    Total_Depth=point['resources']['Total_Depth'] if quantum else None,declared_parameters=k*(n-k) if quantum else None,
                    HF_probability=float(prob[list(map(int,ids)).index(hf)]),single_draw_TV_vs_Markov=float(np.sum(abs(prob-markov)))/2)
                rows.append(row)
    attempts=load(LOCAL/'CI-attempts.json')
    if len(roots)!=t['actual_new_CI_roots'] or sum(r.get('new_CI_solve') is True for r in rows)!=len(roots) or {a['basis_id'] for a in attempts if a['status']=='COMPLETED'}!=set(roots):raise RuntimeError('Actual CI work/alias accounting differs')
    comparisons=[];context=[]
    for original in p['sources']:
        case=original['case'];oldgroups=[g for g in baseline['groups'] if g['case']==case and g['arm']=='R32_CAP' and g['method'].startswith('C5_') and g['completed']==3]
        best=min(oldgroups,key=lambda g:g['energy_Ha_mean']);classical=next(g for g in baseline['groups'] if (g['case'],g['arm'],g['method'])==(case,'R32_CAP','CLASSICAL_HF_HCI_MARGINAL_ORACLE'))
        for row in [r for r in rows if r['case']==case]:
            context.append(dict(case=case,scale=row['scale'],method=row['method'],same_actual_D=True,
                exact_candidate_minus_finite_LUCJ_mean_mHa=1000*(row['energy_Ha']-best['energy_Ha_mean']),best_finite_LUCJ=best['method'],
                candidate_minus_HCI_marginal_oracle_mHa=1000*(row['energy_Ha']-classical['energy_Ha_mean']),information_parity_with_finite_LUCJ=False,cold_matched_work=False))
        for scale in p['scales']:
            mark=next(r for r in rows if (r['case'],r['scale'],r['method'])==(case,scale,'MARKOV'))
            real=next(r for r in rows if (r['case'],r['scale'],r['method'])==(case,scale,'COHERENT_REAL'))
            flux=next(r for r in rows if (r['case'],r['scale'],r['method'])==(case,scale,'COHERENT_FLUX'))
            comparisons.append(dict(case=case,scale=scale,real_minus_Markov_mHa=1000*(real['energy_Ha']-mark['energy_Ha']),
                flux_minus_Markov_mHa=1000*(flux['energy_Ha']-mark['energy_Ha']),flux_minus_real_mHa=1000*(flux['energy_Ha']-real['energy_Ha']),
                real_Markov_same_basis=real['basis_id']==mark['basis_id'],flux_Markov_same_basis=flux['basis_id']==mark['basis_id']))
    out=dict(status='N28_ALL36_READOUTS24_SOURCES96_COMPILER_POINTS_AUDITED',rows=rows,comparisons=comparisons,context=context,
        all_source_compiler_points=t['sources'],parent_R32_parities=parities,status_counts=dict(Counter(r['status'] for r in rows)),
        actual_new_CI_roots=len(roots),failed_CI_attempts=t['failed_CI_attempts'],CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],CI_CPU_seconds=t['CI_CPU_seconds'],
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((LOCAL/'terminal.json').read_bytes()),
        historical_LUCJ_context_SHA256=sha256((ROOT/'analysis/N25-lowshot-pair-summary.json').read_bytes()),
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,finite_shot_performance=False,independent_confirmation=False,goal_complete=False)
    save(ROOT/'analysis/N28-source-stress-summary.json',out)
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N28-all36-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    f=lambda v,n=3:'未適用/未取得' if v is None else f'{v:.{n}f}'
    lines=['# N28：位相と共通scaleを変えた全source試験','',
        f"全36readout {out['status_counts']}、実際の新CI根{len(roots)}、失敗{t['failed_CI_attempts']}。24coherent source、96資源点を保存/再計数。", 
        '全法exact probability、R32/実D1024。新Shot/QPU/GPU0。以下のenergyは全て古典CI後であり、sampler自身の期待値ではない。','',
        '| 条件 | scale | 方法 | energy Ha | Recovery % | HF確率 % | TV vs Markov | CNOT | CX Depth | Total Depth | 宣言P | near-tie |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append('| '+' | '.join([r['case'],str(r['scale']),r['method'],f(r['energy_Ha'],9),f(r['recovery_pct']),f(100*r['HF_probability']),f(r['single_draw_TV_vs_Markov'],6),f(r['CNOT'],0),f(r['CNOT_Depth'],0),f(r['Total_Depth'],0),f(r['declared_parameters'],0),str(r['near_tie'])])+' |')
    lines+=['','## 同じscale・同じ実Dでの情報が揃った対照','',
        '| 条件 | scale | real − Markov mHa | flux − Markov mHa | flux − real mHa |','|---|---:|---:|---:|---:|']
    for r in comparisons:lines.append('| '+' | '.join([r['case'],str(r['scale']),f(r['real_minus_Markov_mHa'],6),f(r['flux_minus_Markov_mHa'],6),f(r['flux_minus_real_mHa'],6)])+' |')
    lines+=['','## 元のLUCJ・古典HCIとの文脈比較（情報条件は不一致）','',
        '| 条件 | scale | 方法 | exact候補 − finite LUCJ平均 mHa | 候補 − HCI marginal oracle mHa |','|---|---:|---|---:|---:|']
    for r in context:lines.append('| '+' | '.join([r['case'],str(r['scale']),r['method'],f(r['exact_candidate_minus_finite_LUCJ_mean_mHa']),f(r['candidate_minus_HCI_marginal_oracle_mHa'])])+' |')
    lines+=['','## 全compiler点','',
        '| 条件 | scale | pattern | encoding | opt | CNOT | CX Depth | Total Depth |','|---|---:|---|---|---:|---:|---:|---:|']
    for s in t['sources']:
        for a in s['all_compiler_points']:
            z=a['resources'];lines.append(f"| {s['case']} | {s['scale']} | {s['pattern']} | {a['encoding']} | {z['optimization_level']} | {z['CNOT']} | {z['CNOT_Depth']} | {z['Total_Depth']} |")
    lines+=['','## 限界と費用','',
        f"実行CPU {t['CPU_seconds']:.6f}秒、wall {t['wall_seconds']:.6f}秒、CI {t['CI_CPU_seconds']:.6f} CPU秒。検証・旧初期化・旧source生成は追加費用。", 
        '同じscaleのMarkov対照にもexact probabilityを与えた。一方で元LUCJは有限1,000 countsの歴史的subsetなので、同じ実Dでも新候補とのinformation parityはなく、原著への勝利ではない。',
        'best-found候補は探索結果で、事前固定policy・独立確認・未知分子一般化に読み替えない。',
        '古典Markovの量子CNOT/Depth/Pは未適用。CI後状態をsamplerが準備した意味ではない。pair変数は登録real-angle chartで、fixed phase/scaleは今回の離散探索条件。最小Jacobian rankは未証明。',
        '資源は同じbasisでのlogical all-to-all値。実deviceのrouting・native接続・noiseを含まない。full-spin copyの長距離CNOTもroutingしていない。n-qubit pair encodingと2n-qubit表現を別々に残した。',
        '有限H+0.1S²で選んだ根のphysical Hを報告。厳密singlet保証ではない。Fe SCI/N₂ archive参照の精度は未認証、Recoveryはenergy換算。',
        '数値CIはsingle-thread設定。compiler内部の完全なthread使用数は未計測で、CPUとwallの差を速度優越に使わない。',
        '最大15,504のpaired-sector内の限定計算。汎用40q simulationでも新ansatz/量子優位でもない。強い原著native TN/QSCI最適化と独立有限Shot評価は未完了。']
    write(ROOT/'analysis/N28_SOURCE_STRESS_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['status_counts'],len(roots),flush=True)


if __name__=='__main__':run()
