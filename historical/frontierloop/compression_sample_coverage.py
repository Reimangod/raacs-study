"""N14: retrospective sample-loss diagnostic on the existing 16 D2 draws.

Classical without-replacement resampling is not new quantum data. Compare
equal 50k subsets against the same held-out 50k parent observations. The
parent/parent split provides a finite-observation baseline, not exact support.
"""
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256

LOCAL=ROOT/'runs/N14-D2-SAMPLE-COVERAGE-DIAGNOSTIC-v1'


def split_counts(counts,n,seed):
    counts=np.asarray(counts,dtype=np.int64)
    if counts.ndim!=1 or np.any(counts<=0) or int(counts.sum())<n:raise ValueError('Invalid count population')
    selected=np.random.default_rng(seed).multivariate_hypergeometric(counts,n)
    return selected,counts-selected


def coverage(reference_ids,reference_counts,candidate_ids,candidate_counts,norb):
    reference_ids=np.asarray(reference_ids,dtype=np.uint64);candidate_ids=np.asarray(candidate_ids,dtype=np.uint64)
    mask=(1<<norb)-1
    candidate={int(x) for x,c in zip(candidate_ids,candidate_counts) if c>0};half=set()
    for x in candidate:half.update([x&mask,x>>norb])
    denom=int(np.sum(reference_counts))
    if not denom:raise ValueError('Empty heldout reference')
    full=cartesian=0
    for x,c in zip(reference_ids,reference_counts):
        x=int(x);full+=int(c)*(x in candidate);cartesian+=int(c)*((x&mask) in half and (x>>norb) in half)
    def occupations(ids,counts):
        return np.array([sum(int(c)*((int(x)>>i)&1) for x,c in zip(ids,counts))/int(np.sum(counts)) for i in range(2*norb)])
    delta=np.abs(occupations(reference_ids,reference_counts)-occupations(candidate_ids,candidate_counts))
    return dict(heldout_exact_configuration_coverage_pct=100*full/denom,
        heldout_half_pool_cartesian_coverage_pct=100*cartesian/denom,candidate_distinct_joint_observed=len(candidate),
        candidate_distinct_half_observed=len(half),occupation_max_abs_difference=float(delta.max()),occupation_mean_abs_difference=float(delta.mean()))


def run():
    parent=ROOT/'runs/D2-COMPRESSION-SCREEN';manifest=load(parent/'manifest.json');protocol=load(ROOT/'protocol-D2-COMPRESSION-SCREEN.json')
    # The analysis is exploratory after D2; do not describe this as pre-outcome
    # confirmation. The frozen analysis rule never uses energy/FCI for selection.
    spec=dict(version='N14_D2_REUSED_OBSERVATION_COVERAGE_V1',parent_manifest_SHA256=sha256((parent/'manifest.json').read_bytes()),
        subsets=50000,seed=2026102601,reference='EXACT_RANDOM_HALF_B',baseline='EXACT_DISJOINT_HALF_A',
        candidate='WITHOUT_REPLACEMENT_HALF_OF_EACH_EXISTING_F25_F50_F75_DRAW',
        implementation_SHA256=sha256(Path(__file__).read_bytes()),new_CI_solves=0,new_model_shots=0,new_QPU_shots=0,
        exploratory_retrospective=True,exact_distribution_or_fidelity_claim=False)
    LOCAL.mkdir(exist_ok=True);freeze=LOCAL/'analysis-rule.json'
    if freeze.exists() and load(freeze)!=spec:raise RuntimeError('Existing analysis rule changed')
    if not freeze.exists():save(freeze,spec)
    n13=load(ROOT/'runs/N13-SAMPLE-CONSTRAINED-ALL-SOURCES-v1/protocol.json');rows=[]
    for case in protocol['cases']:
        n=n13['physical'][case]['norb'];population={};proof={}
        for point in protocol['points']:
            folder=parent/case/point;request=load(folder/'request.json');draw=load(folder/'draw-result.json');file=folder/'final-counts.npz'
            if request not in manifest['cells'] or draw['input_id']!=request['execution_input_id'] or sha256(file.read_bytes())!=draw['counts_SHA256'] or request['problem_id']!=n13['physical'][case]['problem_id']:raise RuntimeError('Source/input/count/problem binding mismatch')
            with np.load(file,allow_pickle=False) as z:ids=z['blocked_ids'].copy();counts=z['counts'].copy()
            if len(ids)!=len(counts) or len(set(map(int,ids)))!=len(ids) or np.any(counts<=0) or int(counts.sum())!=100000 or draw['actual_model_shots']!=100000:raise RuntimeError('Wrong draw population')
            ne=n13['physical'][case]['nelec'];mask=(1<<n)-1
            if any((int(x)&mask).bit_count()!=ne[0] or (int(x)>>n).bit_count()!=ne[1] for x in ids):raise RuntimeError('Wrong particle sector')
            population[point]=(ids,counts);proof[point]=dict(counts_SHA256=draw['counts_SHA256'],request_SHA256=sha256((folder/'request.json').read_bytes()))
        ids,counts=population['EXACT'];a,b=split_counts(counts,50000,spec['seed'])
        for index,point in enumerate(protocol['points']):
            ci,cc=population[point]
            selected=a if point=='EXACT' else split_counts(cc,50000,spec['seed']+index)[0]
            r=coverage(ids,b,ci,selected,n);request=load(parent/case/point/'request.json');res=request['resources']['Z_sampling_equivalent']
            rows.append(dict(case=case,point=point,**r,CNOT=res['cnot_count'],CNOT_Depth=res['cnot_depth'],Total_Depth=res['total_unitary_depth'],
                original_source_counts_SHA256=proof['EXACT']['counts_SHA256'],candidate_counts_SHA256=proof[point]['counts_SHA256'],candidate_request_SHA256=proof[point]['request_SHA256']))
    if len(rows)!=16:raise RuntimeError('All16 existing draws required')
    for case in protocol['cases']:
        baseline=next(r for r in rows if r['case']==case and r['point']=='EXACT')
        for r in [x for x in rows if x['case']==case]:
            r['half_pool_coverage_minus_parent_baseline_pp']=r['heldout_half_pool_cartesian_coverage_pct']-baseline['heldout_half_pool_cartesian_coverage_pct']
            r['joint_coverage_minus_parent_baseline_pp']=r['heldout_exact_configuration_coverage_pct']-baseline['heldout_exact_configuration_coverage_pct']
    result=dict(status='ALL16_D2_REUSED_DRAW_SAMPLE_COVERAGE_DIAGNOSTICS_COMPLETED',rows=rows,analysis_rule_SHA256=sha256(freeze.read_bytes()),
        new_CI_solves=0,new_model_shots=0,new_QPU_shots=0,old_counts_unchanged=True,goal_complete=False)
    save(ROOT/'analysis/N14-D2-sample-coverage.json',result)
    lines=['# N14：古いLUCJ圧縮で失われたsample情報の診断','',
        '新Shot/CI0。D2全16既存drawを再利用した探索的な分析。各候補から50kを非復元抽出し、同じEXACT側holdout50kに対するcoverageを測定した。EXACT側の別50kを有限観測baselineとする。',
        'この割合は**実際に観測されたparent配置**の再発見率。厳密な波動関数fidelity、真の分布距離、ground-state配置の回収率ではない。有限sample・seedの不確かさがある。F25等は圧縮時の回転mode上限設定で、CNOT削減率そのものではない。','',
        '| 条件 | 設定 | CNOT | CNOT Depth | parent観測のjoint再発見 % | parent観測のhalf-pool内所属 % | half-pool差 pp（parent分割比） | 占有率最大差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['point']} | {r['CNOT']} | {r['CNOT_Depth']} | {r['heldout_exact_configuration_coverage_pct']:.3f} | {r['heldout_half_pool_cartesian_coverage_pct']:.3f} | {r['half_pool_coverage_minus_parent_baseline_pp']:+.3f} | {r['occupation_max_abs_difference']:.5f} |")
    lines+=['','単一粒子unitaryの近似度やCNOTだけでは、この観測配置の保持を直接評価していなかった。今後state/sampling-aware目的を検討する際は、元回路の観測と新しい圧縮回路の観測を同一視せず、独立drawと同じCI条件で最終精度を確かめる。元D2の数値結果・失敗・費用は変更していない。']
    write(ROOT/'analysis/N14_D2_SAMPLE_COVERAGE_JA.md',('\n'.join(lines)+'\n').encode());print(result['status'],flush=True)


if __name__=='__main__':run()
