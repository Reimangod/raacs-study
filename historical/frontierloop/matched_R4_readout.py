"""Exploratory N58: fixed actual R=4/D=16 source-quality diagnostic.

Reuses N57 model observations. Not an independent confirmation or new sampler.
All four cases retained; missing Fe is never assigned an energy of zero.
"""
from pathlib import Path
from time import process_time,perf_counter
import traceback
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,arrays,digest,space
from .filtered_SqDRIFT_plan import LOCAL,verified
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit

OUT=ROOT/'runs/N58-MATCHED-R4-SOURCE-DIAGNOSTIC-v1'


def run():
    p=verified();space()
    if OUT.exists(): raise RuntimeError('Existing diagnostic; do not repeat')
    plan=dict(status='FROZEN_BEFORE_NEW_R4_READOUTS',actual_R=4,actual_D=16,
        scientific_scope='POSTHOC_EXPLORATORY_READOUT_DIMENSION_CONTROL_NOT_INDEPENDENT_CONFIRMATION',
        original_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        implementation_SHA256=sha256(Path(__file__).read_bytes()),
        audit_SHA256=sha256((ROOT/'frontierloop/contiguous_CI_audit.py').read_bytes()),
        input_draws=[dict(draw_id=d['draw_id'],case=d['case'],arm=d['arm'],seed=d['seed'],
            counts_SHA256=sha256((LOCAL/'draws'/d['draw_id']/'counts.npz').read_bytes()) if (LOCAL/'draws'/d['draw_id']/'result.json').exists() else None) for d in p['draws']],
        total_CPU_limit_seconds=120,per_root_wall_limit_seconds=15,
        same_equation=p['common_equation'],source_policies_and_resources_unchanged=True,
        new_model_samples=0,new_QPU_shots=0,Fe_missing_preserved=True)
    save(OUT/'protocol.json',plan)
    cpu,wall=process_time(),perf_counter();states=[];cache={}
    for c in p['cases']:
        meta=load(c['meta']);n=meta['norb'];ne=(meta['nalpha'],meta['nbeta'])
        with np.load(c['integrals'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
        for d in [x for x in p['draws'] if x['case']==c['case']]:
            row={k:d[k] for k in ['draw_id','case','arm','seed']}
            row.update(status='SOURCE_MISSING',actual_R=None,actual_D=None,energy_Ha=None,basis_id=None)
            f=LOCAL/'draws'/d['draw_id']
            if (f/'result.json').exists():
                source=load(f/'result.json');assert source['counts_SHA256']==sha256((f/'counts.npz').read_bytes())
                with np.load(f/'counts.npz',allow_pickle=False) as a:rank=ranked_half(a['blocked_ids'],a['counts'],n,(1<<ne[0])-1)
                if len(rank)<4: raise ValueError('Cannot claim matched actual R')
                strings=sorted(rank[:4]);key=digest([d['problem_id'],strings,p['common_equation'],p['library_SHA256']])
                row.update(actual_R=4,actual_D=16,basis_id=key)
                if key not in cache:
                    t=process_time();space();remaining=120-(process_time()-cpu)
                    if remaining<=0: raise TimeoutError('No automatic global cap increase')
                    try:
                        pairs=np.array([(x,y) for x in strings for y in strings],dtype=np.uint64)
                        result,v=solve(h,g,meta['core_energy'],ne,pairs,Path(p['library']),seconds=min(15,remaining),triangle_cap=8000000)
                        independent=audit(h,g,meta['core_energy'],ne,strings,v,result)
                        file=OUT/'roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v);write(file,blob)
                        cache[key]=dict(status='COMPLETED',result=result,independent_RDM_audit=independent,root_file=str(file),root_SHA256=sha256(blob))
                    except Exception as exc:
                        cache[key]=dict(status='FAILED_PRESERVED',energy_Ha=None,error=repr(exc),traceback=traceback.format_exc())
                    cache[key]['CPU_seconds']=process_time()-t
                    save(OUT/'roots'/(key+'.json'),cache[key])
                row.update(status=cache[key]['status'],root=cache[key])
                if row['status']=='COMPLETED':row['energy_Ha']=cache[key]['result']['energy_Ha']
            states.append(row);save(OUT/'progress.json',dict(states=states,CPU_seconds=process_time()-cpu))
    contrasts=[]
    old=load(LOCAL/'partial-CI-terminal-v2.json')['states']
    for c in p['cases']:
        for seed in p['seeds']:
            a=next(r for r in states if r['case']==c['case'] and r['seed']==seed and r['arm']=='FULL_N20')
            b=next(r for r in states if r['case']==c['case'] and r['seed']==seed and r['arm']=='FILTERED_SHORT')
            olda=next(r for r in old if r['draw_id']==a['draw_id'] and r['R_cap']==32)
            oldb=next(r for r in old if r['draw_id']==b['draw_id'] and r['R_cap']==32)
            both=a['energy_Ha'] is not None and b['energy_Ha'] is not None
            contrasts.append(dict(case=c['case'],seed=seed,both_numeric=both,
                matched_R4_delta_energy_mHa=1000*(b['energy_Ha']-a['energy_Ha']) if both else None,
                prior_R32cap_delta_energy_mHa=1000*(oldb['energy_Ha']-olda['energy_Ha']) if both else None))
    out=dict(status='ALL24_STATUSES_RECORDED',states=states,contrasts=contrasts,
        completed_endpoints=sum(r['status']=='COMPLETED' for r in states),source_missing_endpoints=sum(r['status']=='SOURCE_MISSING' for r in states),
        actual_new_CI_roots=sum(r['status']=='COMPLETED' for r in cache.values()),failed_attempts=sum(r['status']=='FAILED_PRESERVED' for r in cache.values()),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        new_model_samples=0,new_QPU_shots=0,independent_confirmation=False,goal_complete=False)
    save(OUT/'terminal.json',out)
    lines=['# N58：実CIサイズをそろえた追加診断','',
        'N57の取得済み18 drawを再利用。全て実R=4、D=16に固定し、同じH+0.1S²の根を独立RDM照合。新Shotなし。Fe6件は欠測のまま。',
        '結果を見た後に追加した機構診断であり、事前登録された独立確認ではない。','',
        '| 条件 | 数値ペア | 同じ実R=4でのfiltered energy差 mHa | 以前のR32上限での差 mHa |',
        '|---|---:|---:|---:|']
    for c in p['cases']:
        rows=[r for r in contrasts if r['case']==c['case'] and r['both_numeric']]
        means=[float(np.mean([r[k] for r in rows])) if rows else None for k in ['matched_R4_delta_energy_mHa','prior_R32cap_delta_energy_mHa']]
        fmt=lambda x:'欠測' if x is None else f'{x:+.6f}'
        lines.append(f"| {c['case']} | {len(rows)}/3 | {fmt(means[0])} | {fmt(means[1])} |")
    lines+=['','負の差はfilteredが低いenergy。実サイズをそろえても差が残れば、配置数の増加だけでは説明できない。ただしfilterと短縮長の効果は分離していない。',
        '実Rの統一は完全な計算費用の統一ではない。元source生成・失敗検査の費用はN57に属し、ここで消去しない。',
        f"新規CI根{out['actual_new_CI_roots']}、失敗{out['failed_attempts']}、body CPU {out['CPU_seconds']:.6f}秒。",
        '最高論文・強いLUCJ/TN・古典HCIへの総合優越は未証明。','']
    write(ROOT/'analysis/N58_MATCHED_ACTUAL_R_RESULTS_JA.md','\n'.join(lines).encode())
    print('N58_TERMINAL',out['completed_endpoints'],out['actual_new_CI_roots'],out['failed_attempts'],flush=True)


if __name__=='__main__':run()
