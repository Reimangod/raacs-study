"""Independent synthesis-table audit and direct RAACS paired comparisons.

Does NOT subtract means drawn from different available geometry subsets.
No producer helpers, molecular solver, new circuits, samples or references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
PRIMARY='RAACS:full_cost-qsci_local_grid-last_only-exact_small'
CEO='B1:source_final_ceo_mvp'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def close(a,b):
    if b is None:assert pd.isna(a)
    else:assert np.isclose(a,b,atol=1e-8,rtol=1e-10),(a,b)


def direct_comparisons(g,cases):
    methods=sorted(g.family.unique());lookup=g.set_index(['case_id','family','scope','M']);rows=[]
    assert not lookup.index.duplicated().any()
    for case in cases:
        for method in methods:
            for scope in ('readout_only','known_build_plus_readout'):
                for M in (1000,10000,100000):
                    key=(case,PRIMARY,scope,M);other=(case,method,scope,M)
                    r=dict(case_id=case,molecule=case.split('_R')[0],baseline_family=method,scope=scope,M=M,R=64)
                    if key not in lookup.index:r['status']='PRIMARY_NO_BUDGET_ENDPOINT'
                    elif other not in lookup.index:r['status']='BASELINE_NO_BUDGET_ENDPOINT'
                    else:
                        a=lookup.loc[key];b=lookup.loc[other]
                        valid=bool(a.all_registered_seeds_available and b.all_registered_seeds_available)
                        if scope=='known_build_plus_readout':valid=bool(valid and a.full_model_budget_known and b.full_model_budget_known)
                        r['status']='AVAILABLE_COMPLETE_SEED_PAIR' if valid else 'INCOMPLETE_OR_UNKNOWN_BUDGET_PRESERVED'
                        if valid:
                            r.update(RAACS_minus_baseline_mHa=a.mean_error_mHa-b.mean_error_mHa,
                                RAACS_error_mHa=a.mean_error_mHa,baseline_error_mHa=b.mean_error_mHa,
                                RAACS_minus_baseline_CNOT=a.worst_registered_component_cnot-b.worst_registered_component_cnot,
                                RAACS_minus_baseline_depth=a.worst_registered_component_depth-b.worst_registered_component_depth,
                                RAACS_total_model_shots=a.total_model_shots_seed_mean,
                                baseline_total_model_shots=b.total_model_shots_seed_mean,
                                actual_CI_dimension_matched=False)
                    rows.append(r)
    return pd.DataFrame(rows)


def review(root):
    root=Path(root);p=root/'source-evidence.json';parent=json.loads(p.read_text())
    destination=root/'review-v1';destination.mkdir(exist_ok=True)
    validation=destination/'validation.json'
    if validation.exists():
        old=json.loads(validation.read_text());assert old['reviewer_sha256']==sha(__file__)
        for path,s in old['input_sha256'].items():assert sha(Path(path) if Path(path).is_absolute() else ROOT/path)==s
        for path,s in old['output_sha256'].items():assert sha(destination/path)==s
        return old
    inputs={}
    def bind(path):
        path=Path(path);inputs[str(path) if not path.is_relative_to(ROOT) else str(path.relative_to(ROOT))]=sha(path);return path
    def table(path):return pd.read_csv(bind(path))
    bind(p);bind(root/'REPORT_JA.md')
    producer='benchmark/research_synthesis_v2.py' if parent['schema']=='scalable-study-evidence-synthesis-v2' else 'benchmark/research_synthesis.py'
    assert sha(ROOT/producer)==parent['script_sha256']
    for path,s in parent['source_sha256'].items():assert sha(bind(Path(path) if Path(path).is_absolute() else ROOT/path))==s
    for path,s in parent['table_sha256'].items():assert sha(bind(root/path))==s
    assert len(parent['questions'])==15 and len(parent['primary_metrics'])==10 and not parent['whole_goal_complete']
    assert [q['number'] for q in parent['questions']]==list(range(1,16))
    assert all((root/x).exists() for q in parent['questions'] for x in q['evidence'])
    costs=table(ROOT/'results/final-cost-frontiers-standard-v2/all-seed-family-frontiers.csv')
    costs=costs[costs.case_id.isin(parent['cases'])]
    primary=costs[(costs.scope=='readout_only')&(costs.M==1000)].set_index(['case_id','family'])
    pairs=table(root/'all-case-primary-pairs.csv')
    checked=0
    for row in pairs.itertuples():
        b=(row.case_id,CEO);q=(row.case_id,row.family)
        complete=b in primary.index and q in primary.index
        if complete:complete=bool(primary.loc[b].all_registered_seeds_available and primary.loc[q].all_registered_seeds_available)
        assert (row.status=='AVAILABLE_PAIR')==complete
        if complete:
            close(row.delta_error_mHa,primary.loc[q].mean_error_mHa-primary.loc[b].mean_error_mHa)
            close(row.candidate_worst_CNOT,primary.loc[q].worst_registered_component_cnot)
            close(row.candidate_worst_depth,primary.loc[q].worst_registered_component_depth)
        checked+=1
    # Independent direct group means, not a difference of unpaired cohort means.
    means=table(root/'equal-molecule-primary-comparisons.csv')
    for row in means.itertuples():
        q=pairs[(pairs.family==row.family)&(pairs.status=='AVAILABLE_PAIR')]
        mol=[float(x.delta_error_mHa.to_numpy().sum()/len(x)) for _,x in q.groupby('molecule')]
        close(row.equal_molecule_delta_error_mHa,sum(mol)/len(mol));assert row.molecules==len(mol) and row.paired_conditions==len(q)
    curve=table(ROOT/'results/final-relationships-standard-v1/growth-curves.csv')
    cover=table(ROOT/'results/final-relationships-standard-v1/HF-vs-nonHF-coverage.csv')
    joined=curve.merge(cover[['checkpoint_id','nonhf_weighted_coverage']],on='checkpoint_id',validate='one_to_one')
    associations=table(root/'within-trajectory-coverage-energy-associations.csv')
    for row in associations.itertuples():
        q=joined[(joined.case_id==row.case_id)&(joined.method==row.method)]
        assert len(q)==row.checkpoints
        if len(q)>=3 and q.mean_error_mHa.nunique()>1 and q.nonhf_weighted_coverage.nunique()>1:
            close(row.spearman_coverage_vs_QSCI_error,spearmanr(q.mean_error_mHa,q.nonhf_weighted_coverage).statistic)
        else:assert pd.isna(row.spearman_coverage_vs_QSCI_error)
    c=table(root/'actual-FCI-free-primary-stopping.csv')
    assert np.max(np.abs(c[['regret_mHa','cnot_count_reduction_percent','cnot_depth_reduction_percent']].to_numpy()))<1e-12
    direct=direct_comparisons(costs,parent['cases'])
    valid=direct[direct.status=='AVAILABLE_COMPLETE_SEED_PAIR']
    mol=valid.groupby(['baseline_family','scope','M','molecule']).agg(conditions=('case_id','size'),
        RAACS_minus_baseline_mHa=('RAACS_minus_baseline_mHa','mean'),RAACS_minus_baseline_CNOT=('RAACS_minus_baseline_CNOT','mean'),
        RAACS_minus_baseline_depth=('RAACS_minus_baseline_depth','mean')).reset_index()
    aggregate=mol.groupby(['baseline_family','scope','M']).agg(molecular_clusters=('molecule','size'),available_conditions=('conditions','sum'),
        equal_molecule_RAACS_minus_baseline_mHa=('RAACS_minus_baseline_mHa','mean'),
        equal_molecule_RAACS_minus_baseline_CNOT=('RAACS_minus_baseline_CNOT','mean'),
        equal_molecule_RAACS_minus_baseline_depth=('RAACS_minus_baseline_depth','mean')).reset_index()
    # Bring actual-d controls alongside common-cap results. Registered baseline
    # B2/qDRIFT seed101 stays seed101, NOT silently relabelled a3-seed estimate.
    path=ROOT/'results/RAACS-independent-comparison-standard-sharded-v1/derived-report-v1/all-cell-actual-d-pairs.csv'
    chunks=[];bind(path)
    for chunk in pd.read_csv(path,chunksize=20000):
        q=chunk[(chunk.family==PRIMARY)&(chunk.R==64)&chunk.case_id.isin(parent['cases'])]
        if len(q):chunks.append(q)
    actual=pd.concat(chunks,ignore_index=True)
    actual['registered_RAACS_seeds_available']=actual.groupby(['case_id','baseline','scope','M']).method.transform('nunique')
    actual=actual[actual.registered_RAACS_seeds_available==3].copy()
    acase=actual.groupby(['case_id','molecule','baseline','scope','M']).agg(
        mean_RAACS_minus_baseline_mHa=('mean_RAACS_minus_baseline_mHa','mean'),
        mean_paired_actual_d=('mean_paired_actual_d','mean')).reset_index()
    amol=acase.groupby(['molecule','baseline','scope','M']).agg(conditions=('case_id','size'),
        mean_RAACS_minus_baseline_mHa=('mean_RAACS_minus_baseline_mHa','mean'),mean_paired_actual_d=('mean_paired_actual_d','mean')).reset_index()
    files={'RAACS-direct-all-planned-pairs.csv':direct,'RAACS-direct-paired-molecule-means.csv':mol,
        'RAACS-direct-equal-molecule-means.csv':aggregate,'RAACS-registered-baselines-actual-d-paired-case-means.csv':acase,
        'RAACS-registered-baselines-actual-d-molecule-means.csv':amol}
    for name,frame in files.items():frame.to_csv(destination/name,index=False)
    text=['# Independent synthesis review and direct RAACS comparisons','',
        f'Parent{parent["mode"]}: {checked}case-pair rows and{len(associations)}coverage-energy associations reconstructed; source/table hashes checked.',
        'This audits aggregation of completed results, not a new molecular experiment or a final whole-study certificate.','',
        '## Why direct pairing is necessary',
        'Do not subtract two whole-cohort means that have different missing geometries. Each RAACS/baseline difference is formed on the SAME physical case first; all construction seeds within that case precede molecule averaging.',
        'Negative RAACS-minus-baseline energy error means RAACS is more accurate at that common cap. Readout-only is not end-to-end-budget matching: training/pilots can differ or be unknown.',
        'All registered family/budget combinations, including unavailable/partial/unknown endpoints, remain in RAACS-direct-all-planned-pairs.csv.','',
        '## Two CI-dimension views',
        'Common-R tables compare the operational capR64. Actual-d tables use the original independently audited frequency-selected paired subspaces, truncate to pair actuald, then average all3 RAACS construction seeds. This changes which discovered configurations are retained, not the quantum draws.',
        'The registered actual-d B2/qDRIFT baselines use their originally specified seed101; they are NOT the3-seed baseline-family average from the common-R supplement. Do not subtract these views as a pure dimension-only causal effect.',
        'No best seed or molecular geometry was selected. Distances are not independent molecule replications. Means are descriptive; use source cluster-uncertainty reports within their exact registered comparator scope, not new significance claims.',
        '## Scope',
        'No new draw, optimizer, CI/root solve, FCI reference, hardware shot or production restart. Total classical wall and Fig18 remain missing. E/F/HCI pending scope remains unchanged.']
    (destination/'REVIEW.md').write_text('\n\n'.join(text)+'\n')
    result={'status':'SYNTHESIS_PAIRED_MEANS_AND_ASSOCIATIONS_RECONSTRUCTED_DIRECT_RAACS_SUPPLEMENT',
        'parent_snapshot':str(root),'reviewer_sha256':sha(__file__),'input_sha256':dict(sorted(inputs.items())),
        'case_pairs_checked':checked,'associations_checked':len(associations),'direct_registered_comparisons':len(direct),
        'actual_d_case_means':len(acase),'new_draws':0,'new_solver_calls':0,'whole_goal_complete':False,
        'output_sha256':{p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name!='validation.json'}}
    validation.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    r=review(a.snapshot);print(json.dumps({k:v for k,v in r.items() if k not in ('input_sha256','output_sha256')}))
