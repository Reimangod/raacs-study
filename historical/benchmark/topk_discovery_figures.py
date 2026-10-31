"""Additive all-K Fig4 supplement using saved final-checkpoint expectations.

No new state, sampling, optimization or CI solve. Never rerender old figures.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
KS=(3,5,10,20,50)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(inventory, metrics, planned, shots):
    import pandas as pd
    primary=inventory[inventory.primary].copy()
    if primary.duplicated(['case_id','method','step']).any(): raise ValueError('Ambiguous primary checkpoint')
    final=primary.loc[primary.groupby(['case_id','method']).step.idxmax()]
    lookup=final.set_index(['case_id','method'])
    values=metrics[metrics.primary_record].set_index(['checkpoint_id','shots'])
    if values.index.duplicated().any(): raise ValueError('Duplicate metric row')
    if planned.duplicated(['case_id','method']).any(): raise ValueError('Duplicate planned cell')
    rows=[]
    for cell in planned.sort_values(['case_id','method']).itertuples(index=False):
        key=(cell.case_id,cell.method); available=key in lookup.index
        cp=lookup.loc[key] if available else None
        for M in shots:
            record=values.loc[(cp.checkpoint_id,M)] if available else None
            for k in KS:
                value=record[f'recall_top{k}'] if available else None
                n=int(record[f'recall_top{k}_actual_targets']) if available else None
                if available:
                    if not 0<=n<=k: raise ValueError('Invalid actual target count')
                    if n and (not math.isfinite(float(value)) or not 0<=value<=1): raise ValueError('Invalid recall')
                    if not n and not pd.isna(value): raise ValueError('No-target value is not unknown')
                rows.append(dict(case_id=cell.case_id,method=cell.method,source_terminal_status=cell.status,
                    status=('ANALYTIC_EXPECTED_RECALL' if n else 'NO_NONHF_TARGETS') if available else 'MISSING_SOURCE_PRESERVED',
                    step=int(cp.step) if available else None,checkpoint_id=cp.checkpoint_id if available else None,
                    shots=M,K_requested=k,actual_targets=n,
                    expected_recall=None if value is None or pd.isna(value) else float(value),
                    selection_scope='LAST_PRIMARY_SAVED_CHECKPOINT_NOT_BEST',oracle_evaluation_only=True))
    return pd.DataFrame(rows)


def build(output):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .study_figures import COLORS
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<128*2**20: raise RuntimeError('Insufficient figure headroom')
    src=ROOT/'results/sampler-study-standard-v1'
    paths=[src/'manifest.json',src/'checkpoint-inventory.json',src/'analysis/discovery.parquet',
           ROOT/'results/standard-v1/all-cell-status.csv',Path(__file__)]
    inputs={str(p):sha(p) for p in paths}
    old=output/'manifest.json'
    if old.exists():
        saved=json.loads(old.read_text())
        if saved['inputs']!=inputs or any(sha(output/p)!=h for p,h in saved['outputs'].items()):
            raise ValueError('Existing supplement drift; use a successor')
        return saved
    cfg=json.loads(paths[0].read_text()); inv=pd.DataFrame(json.loads(paths[1].read_text()))
    metrics=pd.read_parquet(paths[2]); planned=pd.read_csv(paths[3])
    table=project(inv,metrics,planned,cfg['M_grid'])
    table.to_csv(output/'all-planned-topK.csv',index=False)
    plt.rcParams['svg.hashsalt']='all-K-Fig4-supplement-v1'
    for case,rows in table.groupby('case_id',sort=True):
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
        handles={}
        for k,ax in zip(KS,axes.flat):
            part=rows[rows.K_requested==k]
            target_counts=part.actual_targets.dropna().unique()
            if len(target_counts)>1: raise ValueError('Reference target count changed across methods/M')
            n=str(int(target_counts[0])) if len(target_counts) else 'unknown'
            for method,g in part.groupby('method',sort=True):
                good=g[g.expected_recall.notna()].sort_values('shots')
                if good.empty: continue
                line,=ax.plot(good.shots,100*good.expected_recall,'o-',ms=2.5,lw=1,
                    label=method,color=COLORS[method]);handles[method]=line
            ax.set(xscale='log',ylim=(-2,102),xlabel='Model shots M',ylabel='Expected target recall (%)',
                   title=f'Requested K={k}; actual targets={n}')
            ax.grid(alpha=.18)
        axes.flat[-1].axis('off')
        axes.flat[-1].legend(handles.values(),handles.keys(),loc='center',fontsize=9)
        missing=sorted(set(rows.loc[rows.status=='MISSING_SOURCE_PRESERVED','method']))
        fig.suptitle(f'{case} | last primary saved checkpoints | all five Top-K diagnostics')
        fig.text(.015,.018,'Mean per-target discovery probability, NOT probability of finding every target. FCI target identities are evaluation-only.\n'
            +'Last SAVED step can be capped, not converged; no new draws/solves. Missing: '+(', '.join(missing) or 'none')+'.',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,.95))
        for ext in ('png','svg','pdf'):
            fig.savefig(output/f'{case}_Fig04_allK.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    if any(sha(p)!=h for p,h in inputs.items()): raise ValueError('Input drift during figure generation')
    outputs={p.name:sha(p) for p in sorted(output.iterdir()) if p.suffix in ('.csv','.png','.svg','.pdf')}
    manifest=dict(schema='all-K-final-checkpoint-discovery-figure-supplement-v1',inputs=inputs,outputs=outputs,
        planned_cells=len(planned),cases=int(planned.case_id.nunique()),rows=len(table),K_requested=list(KS),
        missing_cells=int(table.loc[table.status=='MISSING_SOURCE_PRESERVED',['case_id','method']].drop_duplicates().shape[0]),
        new_samples=0,new_CI_solves=0,new_FCI_solves=0,old_figures_modified=False,whole_goal_complete=False)
    old.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    (output/'README.md').write_text('# All-K discovery supplement\n\n'
        'Five panels per case: K=3/5/10/20/50. Values are analytic expected mean per-target recall, '
        'NOT joint discovery probability or finite-shot QSCI energy. FCI is used only to identify diagnostic targets. '
        'All planned source failures and last saved (possibly capped) checkpoints are retained. '
        'Original figures/data are untouched. Independent raw-probability validation is required separately.\n')
    return manifest


def review(output):
    """Recompute every displayed value from saved probabilities/reference weights."""
    import numpy as np
    import pandas as pd
    output=Path(output); manifest=json.loads((output/'manifest.json').read_text())
    if any(sha(p)!=h for p,h in manifest['inputs'].items()): raise ValueError('Source drift')
    if any(sha(output/p)!=h for p,h in manifest['outputs'].items()): raise ValueError('Figure/CSV drift')
    inventory=json.loads((ROOT/'results/sampler-study-standard-v1/checkpoint-inventory.json').read_text())
    byid={r['checkpoint_id']:r for r in inventory if r['primary']}
    table=pd.read_csv(output/'all-planned-topK.csv'); source=ROOT/'results/standard-v1'
    planned=pd.read_csv(source/'all-cell-status.csv'); references={}; inputs={}; maximum=0.; checked=0
    if table.duplicated(['case_id','method','shots','K_requested']).any(): raise ValueError('Duplicate exported identity')
    for (case,method),g in table.groupby(['case_id','method'],sort=True):
        cpids=set(g.checkpoint_id.dropna())
        actual=[r for r in byid.values() if r['case_id']==case and r['method']==method]
        last=max(actual,key=lambda r:r['step']) if actual else None
        if not actual:
            if cpids or set(g.status)!={'MISSING_SOURCE_PRESERVED'} or g.expected_recall.notna().any():
                raise ValueError('Changed missingness')
            continue
        if cpids!={last['checkpoint_id']}: raise ValueError('Not the last primary checkpoint')
        if case not in references:
            rp=source/'references'/case/'reference.json'; ap=rp.with_suffix('.npz')
            meta=json.loads(rp.read_text()); inputs[str(rp)]=sha(rp);inputs[str(ap)]=sha(ap)
            with np.load(ap,allow_pickle=False) as a:
                ids=list(map(int,a['ids'])); weights=list(map(float,a['weights']))
            ranked=sorted((x,w) for x,w in zip(ids,weights) if x!=meta['hf_id'] and w>1e-16)
            ranked=sorted(ranked,key=lambda t:(-t[1],t[0]))
            references[case]=[x for x,w in ranked]
        path=source/last['distribution_path']; inputs[str(path)]=sha(path)
        if inputs[str(path)]!=last['distribution_sha']: raise ValueError('Distribution identity drift')
        with np.load(path,allow_pickle=False) as a: probability=a['probability']
        for row in g.itertuples(index=False):
            targets=references[case][:row.K_requested]
            if int(row.actual_targets)!=len(targets): raise ValueError('Wrong target count')
            expected=sum(-math.expm1(int(row.shots)*math.log1p(-float(probability[x])))
                if probability[x]<1 else 1. for x in targets)/len(targets) if targets else None
            if expected is None:
                if pd.notna(row.expected_recall): raise ValueError('No-target result fabricated')
            else:
                delta=abs(row.expected_recall-expected);maximum=max(maximum,delta)
                if not math.isfinite(delta) or delta>1e-13: raise ValueError('Raw-probability parity failed')
            checked+=1
    shots=json.loads((ROOT/'results/sampler-study-standard-v1/manifest.json').read_text())['M_grid']
    expected={(r.case_id,r.method,M,k) for r in planned.itertuples() for M in shots for k in KS}
    observed=set(table[['case_id','method','shots','K_requested']].itertuples(index=False,name=None))
    if observed!=expected: raise ValueError('Incomplete planned grid')
    if any(sha(p)!=h for p,h in inputs.items()): raise ValueError('Raw input changed during review')
    record=dict(status='ALL_PLANNED_ROWS_AND_RAW_FINAL_PROBABILITY_EXPECTATIONS_RECONSTRUCTED',
        planned_rows=len(table),available_values_checked=checked,maximum_recall_difference=maximum,
        manifest_sha256=sha(output/'manifest.json'),review_code_sha256=sha(__file__),raw_inputs=inputs,
        new_samples=0,new_CI_solves=0,new_FCI_solves=0,whole_goal_complete=False)
    path=output/'independent-value-review-v1.json'; text=json.dumps(record,sort_keys=True,indent=2)+'\n'
    if path.exists() and path.read_text()!=text: raise ValueError('Review drift; preserve existing evidence')
    if not path.exists():path.write_text(text)
    return {k:v for k,v in record.items() if k!='raw_inputs'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/TopK-discovery-final-checkpoint-figures-v1')
    p.add_argument('--review',action='store_true');a=p.parse_args()
    record=review(a.output) if a.review else build(a.output)
    print(json.dumps({k:v for k,v in record.items() if k not in ('inputs','outputs')},sort_keys=True))
