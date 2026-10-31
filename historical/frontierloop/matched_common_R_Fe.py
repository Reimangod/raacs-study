"""N59: largest common source-supported R per paired seed, capped at 32.

The choice uses counts/support only, never an energy. Posthoc diagnostic,
not a prospective algorithm or independent confirmation.
"""
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,arrays,digest,space
from .filtered_SqDRIFT_plan import LOCAL,verified
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit

OUT=ROOT/'runs/N61-FE-COMMON-R-DIAGNOSTIC-v1'


def run():
    p=verified();space()
    if OUT.exists():raise RuntimeError('Do not repeat diagnostic')
    plans=[]
    for c in p['cases'][-1:]:
        meta=load(c['meta']);n=meta['norb'];ne=(meta['nalpha'],meta['nbeta'])
        for seed in p['seeds']:
            sources=[]
            for arm in ['FULL_N20','FILTERED_SHORT']:
                d=next(d for d in p['draws'] if d['case']==c['case'] and d['seed']==seed and d['arm']==arm)
                f=LOCAL/'draws'/d['draw_id'];rank=None
                if (f/'result.json').exists():
                    r=load(f/'result.json');assert r['counts_SHA256']==sha256((f/'counts.npz').read_bytes())
                    with np.load(f/'counts.npz',allow_pickle=False) as a:rank=list(map(int,ranked_half(a['blocked_ids'],a['counts'],n,(1<<ne[0])-1)))
                sources.append(dict(draw_id=d['draw_id'],arm=arm,rank=rank))
            common=min([32]+[len(s['rank']) for s in sources]) if all(s['rank'] is not None for s in sources) else None
            plans.append(dict(case=c['case'],seed=seed,actual_R=common,sources=sources))
    save(OUT/'protocol.json',dict(plans=plans,rule='min(32,available_full_half_strings,available_filtered_half_strings)',
        source_counts_only_not_energy=True,exploratory_posthoc=True,implementation_SHA256=sha256(Path(__file__).read_bytes()),
        audit_SHA256=sha256((ROOT/'frontierloop/contiguous_CI_audit.py').read_bytes()),
        N57_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),CPU_limit_seconds=120,root_wall_seconds=15,
        prior_terminal_SHA256=sha256((LOCAL/'partial-CI-terminal-v3.json').read_bytes()),new_model_samples=0))
    old=load(LOCAL/'partial-CI-terminal-v3.json')['states'];cache={r['basis_id']:r['root'] for r in old if r['status']=='COMPLETED'}
    cpu,wall=process_time(),perf_counter();states=[];attempts=[];contrasts=[]
    for plan in plans:
        c=next(c for c in p['cases'] if c['case']==plan['case']);meta=load(c['meta']);ne=(meta['nalpha'],meta['nbeta'])
        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
        rows=[]
        for source in plan['sources']:
            row=dict(case=plan['case'],seed=plan['seed'],arm=source['arm'],draw_id=source['draw_id'],actual_R=plan['actual_R'],
                actual_D=plan['actual_R']**2 if plan['actual_R'] is not None else None,energy_Ha=None,status='SOURCE_MISSING')
            if plan['actual_R'] is not None:
                strings=sorted(source['rank'][:plan['actual_R']]);key=digest([meta['problem_id'],strings,p['common_equation'],p['library_SHA256']])
                new=key not in cache
                if new:
                    remaining=120-(process_time()-cpu)
                    if remaining<=0:raise TimeoutError('Finite N61 Fe CPU limit')
                    pairs=np.asarray([(a,b) for a in strings for b in strings],np.uint64);tick=process_time()
                    result,v=solve(h,g,meta['core_energy'],ne,pairs,Path(p['library']),seconds=min(15,remaining),triangle_cap=8000000)
                    check=audit(h,g,meta['core_energy'],ne,strings,v,result)
                    file=OUT/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v);write(file,blob)
                    cache[key]=dict(status='COMPLETED',result=result,independent_RDM_audit=check,root_file=str(file),root_SHA256=sha256(blob))
                    attempts.append(dict(basis_id=key,CPU_seconds=process_time()-tick));save(OUT/'roots'/(key+'.json'),cache[key])
                entry=cache[key]
                assert sha256(Path(entry['root_file']).read_bytes())==entry['root_SHA256']
                row.update(status='COMPLETED',energy_Ha=entry['result']['energy_Ha'],basis_id=key,root=entry,new_CI_root=new)
            rows.append(row);states.append(row);save(OUT/'progress.json',dict(states=states,attempts=attempts))
        contrasts.append(dict(case=plan['case'],seed=plan['seed'],actual_R=plan['actual_R'],
            delta_energy_mHa=1000*(rows[1]['energy_Ha']-rows[0]['energy_Ha']) if plan['actual_R'] is not None else None))
    terminal=dict(status='ALL6_FE_STATUSES_RECORDED',states=states,contrasts=contrasts,actual_new_CI_roots=len(attempts),
        reused_roots=sum(r.get('new_CI_root') is False for r in states),
        completed_endpoints=sum(r['status']=='COMPLETED' for r in states),source_missing_endpoints=sum(r['status']=='SOURCE_MISSING' for r in states),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,
        independent_confirmation=False,goal_complete=False)
    save(OUT/'terminal.json',terminal)
    small=load(ROOT/'runs/N58-MATCHED-R4-SOURCE-DIAGNOSTIC-v1/terminal.json')
    lines=['# N61 Fe：両sourceで可能な最大の同一CIサイズでの診断','',
        'R=min(32,元sourceの利用可能な半配置数,filtered sourceの利用可能な半配置数)。energyを使わず選択したが、N57結果後の追加診断である。',
        '新しいShotなし。各seed内の両armで同じ実R、D=R²、同じH+0.1S²とRDM検査を使う。','',
        '| 条件 | 完了ペア | 共通R（各seed） | filtered差 mHa、共通最大R | filtered差 mHa、R4 |',
        '|---|---:|---|---:|---:|']
    for c in p['cases'][-1:]:
        rr=[r for r in contrasts if r['case']==c['case']];ok=[r for r in rr if r['delta_energy_mHa'] is not None]
        r4=[r['matched_R4_delta_energy_mHa'] for r in small['contrasts'] if r['case']==c['case'] and r['both_numeric']]
        fmt=lambda x:'欠測' if not x else f'{np.mean(x):+.6f}'
        lines.append(f"| {c['case']} | {len(ok)}/3 | {[r['actual_R'] for r in rr]} | {fmt([r['delta_energy_mHa'] for r in ok])} | {fmt(r4)} |")
    lines+=['','負の差はfilteredが良い。R4だけで勝敗を判断せず、全利用可能サイズの結果と元R32上限の結果を併記する。',
        '同じ実Rでも全古典費用・誤差・ensembleの量子構築費用は同一ではない。CI空間は異なる。',
        f"新CI根{len(attempts)}、保存根再利用{terminal['reused_roots']}。body CPU {terminal['CPU_seconds']:.6f}秒。Fe6 drawはgate-level frame契約で取得済み。N58のR4診断はFe欠測時の履歴であり、右欄は未計算。",
        'filterと長さ変更は未分離。古典HCIと強いLUCJ/TNを上回った事実はまだない。','']
    write(ROOT/'analysis/N61_FE_MATCHED_COMMON_R_RESULTS_JA.md','\n'.join(lines).encode())
    print('N59_TERMINAL',terminal['completed_endpoints'],len(attempts),flush=True)


if __name__=='__main__':run()
