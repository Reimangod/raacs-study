"""N27 all-root and distribution audit; finite-shot performance not inferred."""
from pathlib import Path
from time import process_time,perf_counter
from collections import Counter
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .dephased_pair_diagnostic import LOCAL,dephased_distribution,top_half
from .paired_lowshot import paired_sector
from .joint_ci_sparse import matrices


def run():
    cpu,wall=process_time(),perf_counter();p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json')
    if t['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or sha256((ROOT/'frontierloop/dephased_pair_diagnostic.py').read_bytes())!=p['code_SHA256']:raise RuntimeError('Frozen context/code changed')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256'] or any(sha256((ROOT/'frontierloop'/name).read_bytes())!=h for name,h in p['helper_SHA256'].items()):raise RuntimeError('Frozen helper/library changed')
    expected={(s['case'],m,R) for s in p['sources'] for m in p['methods'] for R in p['R_values']}
    if len(t['rows'])!=16 or {(r['case'],r['method'],r['R']) for r in t['rows']}!=expected:raise RuntimeError('All16fixed outcomes required')
    roots={};rows=[];diagnostics=[];contrasts=[]
    for s in p['sources']:
        case=s['case'];d=s['definition'];n=d['norb'];k=d['nelec'][0];hf=(1<<k)-1
        for field,hfield in [('state','state_SHA256'),('angles','angles_SHA256')]:
            if sha256(Path(s[field]).read_bytes())!=s[hfield]:raise RuntimeError('Registered source changed')
        with np.load(s['state'],allow_pickle=False) as z:ids=z['half_strings'];v=z['amplitudes']
        angles=load(s['angles'])['angles'];t2=np.zeros((k,k,n-k,n-k))
        for e in angles:t2[e['i'],e['i'],e['a']-k,e['a']-k]=e['theta']
        check_ids,check_v,check_angles=paired_sector(n,k,t2)
        if not np.array_equal(ids,check_ids) or not np.array_equal(v,check_v) or check_angles!=angles:raise RuntimeError('Actual coherent source not reproduced from frozen angles')
        coh=v*v;mark=dephased_distribution(ids,k,angles);original=next(x for x in t['diagnostics'] if x['case']==case)
        file=Path(original['probabilities_path'])
        if sha256(file.read_bytes())!=original['probabilities_SHA256']:raise RuntimeError('Actual probability payload changed')
        with np.load(file,allow_pickle=False) as z:
            if not np.array_equal(ids,z['half_strings']) or not np.array_equal(coh,z['coherent']) or not np.array_equal(mark,z['dephased']):raise RuntimeError('Exact probability reconstruction differs')
        high=np.array([k-(int(x)&hf).bit_count()>=2 for x in ids]);hfindex=list(map(int,ids)).index(hf)
        values=dict(single_draw_total_variation=float(np.sum(abs(coh-mark)))/2,
            coherent_HF_probability=float(coh[hfindex]),dephased_HF_probability=float(mark[hfindex]),
            coherent_multiple_pair_mass=float(coh[high].sum()),dephased_multiple_pair_mass=float(mark[high].sum()))
        if any(abs(original[name]-v)>1e-14 for name,v in values.items()):raise RuntimeError('Reported distribution diagnostic differs')
        diagnostics.append(dict(original,distribution_normalizations=[float(coh.sum()),float(mark.sum())]))
        file=Path(s['integrals'])
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
        with np.load(file,allow_pickle=False) as z:h=z['h1'];g=z['eri']
        for r in [r for r in t['rows'] if r['case']==case]:
            row={key:value for key,value in r.items() if key!='result'}
            prob=coh if r['method']=='COHERENT_EXACT_PROB' else mark;strings,margin=top_half(ids,prob,hf,r['R'])
            if strings!=r['half_strings'] or margin!=r['probability_rank_margin'] or r['near_tie']!=(margin<=p['near_tie_margin_design_flag']):raise RuntimeError('Frozen ranking or near-tie flag differs')
            if r['status']=='COMPLETED':
                file=Path(r['root_path'])
                if sha256(file.read_bytes())!=r['root_SHA256']:raise RuntimeError('Actual root changed')
                with np.load(file,allow_pickle=False) as z:pairs=z['joint_pairs'];c=z['coefficients']
                want=np.array([(a,b) for a in strings for b in strings],np.uint64);key=digest([case,pairs.tolist()])
                if not np.array_equal(pairs,want) or key!=r['basis_id'] or len(pairs)!=r['actual_D'] or len(pairs)!=r['R']**2:raise RuntimeError('Actual dimension or source basis differs')
                if key not in roots:
                    H,S,info=matrices(h,g,tuple(d['nelec']),pairs,p['library'],seconds=20)
                    energy=float(c@(H@c)+d['core_energy']);s2=float(c@(S@c));obj=energy+.1*s2
                    residual=float(np.linalg.norm((H+.1*S)@c-(obj-d['core_energy'])*c))
                    if abs(c@c-1)>1e-10 or residual>1e-8:raise RuntimeError('Actual root norm/residual failed')
                    roots[key]=dict(energy_Ha=energy,spin_square=s2,penalized_energy_Ha=obj,residual=residual)
                if max(abs(roots[key][field]-r['result'][field]) for field in ('energy_Ha','spin_square','penalized_energy_Ha'))>1e-8:raise RuntimeError('Actual root physical consistency failed')
                row.update(roots[key])
            else:row['energy_Ha']=None
            rows.append(row)
        for R in p['R_values']:
            a=next(r for r in rows if (r['case'],r['method'],r['R'])==(case,'COHERENT_EXACT_PROB',R))
            b=next(r for r in rows if (r['case'],r['method'],r['R'])==(case,'DEPHASED_EXACT_PROB',R))
            contrasts.append(dict(case=case,R=R,actual_D=R*R,coherent_energy_Ha=a['energy_Ha'],dephased_energy_Ha=b['energy_Ha'],
                dephased_minus_coherent_mHa=1000*(b['energy_Ha']-a['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None,
                identical_basis=a['basis_id']==b['basis_id'],half_pool_overlap_fraction=len(set(a['half_strings'])&set(b['half_strings']))/R,
                either_near_tie=a['near_tie'] or b['near_tie'],finite_shot_performance=False))
    attempts=load(LOCAL/'CI-attempts.json')
    if len(roots)!=t['actual_new_CI_roots'] or {a['basis_id'] for a in attempts if a['status']=='COMPLETED'}!=set(roots) or sum(r.get('new_CI_solve') is True for r in rows)!=len(roots):raise RuntimeError('Actual unique CI count differs')
    out=dict(status='N27_ALL16_ROOTS4_DISTRIBUTIONS_AUDITED',rows=rows,diagnostics=diagnostics,contrasts=contrasts,
        status_counts=dict(Counter(r['status'] for r in rows)),actual_new_CI_roots=len(roots),failed_CI_attempts=t['failed_CI_attempts'],
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],CI_CPU_seconds=t['CI_CPU_seconds'],
        audit_CPU_seconds=process_time()-cpu,audit_wall_seconds=perf_counter()-wall,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((LOCAL/'terminal.json').read_bytes()),
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,finite_shot_independent_confirmation=False,quantum_advantage=False,goal_complete=False)
    save(ROOT/'analysis/N27-dephased-pair-summary.json',out)
    f=lambda v,n=3:'未取得' if v is None else f'{v:.{n}f}'
    lines=['# N27：複数pairを残し、gate間のcoherenceを外した対照','',
        f"全16readout {out['status_counts']}、実際の新CI根{len(roots)}、CI失敗{t['failed_CI_attempts']}。新Shot/QPU/GPU0。",'',
        '| 条件 | 単一draw TV | coherentの複数pair質量 % | dephasedの複数pair質量 % |','|---|---:|---:|---:|']
    for d in diagnostics:lines.append(f"| {d['case']} | {d['single_draw_total_variation']:.9f} | {100*d['coherent_multiple_pair_mass']:.6f} | {100*d['dephased_multiple_pair_mass']:.6f} |")
    lines+=['','| 条件 | R | 実D | coherent E Ha | dephased E Ha | dephased − coherent mHa | 同一CI基底 | half-pool一致率 % | near-tie |','|---|---:|---:|---:|---:|---:|---|---:|---|']
    for r in contrasts:lines.append('| '+' | '.join([r['case'],str(r['R']),str(r['actual_D']),f(r['coherent_energy_Ha'],9),f(r['dephased_energy_Ha'],9),f(r['dephased_minus_coherent_mHa'],6),str(r['identical_basis']),f(100*r['half_pool_overlap_fraction']),str(r['either_near_tie'])])+' |')
    lines+=['','## 読み方・費用','',
        '負のenergy差はdephased対照が低い。両者へexact probabilityを与えた同じ実Dの診断で、有限Shotの実用成績とは違う。単一drawのTVから1,000 drawsの同等性は証明しない。',
        'per-gate完全dephasingという数学的介入であり、現実のhardware noiseを再現していない。小さいCI基底を保った例から一般的な量子計算不要の定理へ拡張しない。',
        '複数pairの移動は両側に残る。従ってN26のsingle-pair古典対照との差を、coherence固有の利益と短絡しない。',
        f"実行CPU {t['CPU_seconds']:.6f}秒、wall {t['wall_seconds']:.6f}秒、CI {t['CI_CPU_seconds']:.6f} CPU秒。初期化・旧source生成・今回の監査は別費用。", 
        '最大15,504のpaired-sector内の計算。汎用40q simulatorではない。古典対照の量子CNOT/Depth/Pは未適用。古典CI後の状態を0ゲートで準備した意味でもない。',
        '有限H+0.1S²で選ぶ根のphysical Hを報告。厳密singlet・per-root RDM検証とはしない。原著最高予算再現・独立確認・最強論文優越・全goal達成は未証明。']
    write(ROOT/'analysis/N27_DEPHASED_PAIR_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['status_counts'],len(roots),flush=True)


if __name__=='__main__':run()
