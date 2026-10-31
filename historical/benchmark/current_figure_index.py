"""Read-only semantic/source index for all18 requested figure relationships.

Representative rendered PNGs were manually inspected2026-10-23. This module
verifies the referenced files/three-format coverage, binds every indexed file
and the supporting numerical/audit sources, and emits explicit claim limits.
It does not claim visual inspection of every export or independently recompute
every plotted numerical value. It NEVER invokes an old renderer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from .current_evidence import ROOT, canonical, csv_text, save_once, sha

A='results/sampler-study-standard-v1'
S='results/final-relationships-standard-v1'
K='results/TopK-discovery-final-checkpoint-figures-v1'
F14='results/Recovery-energy-Fig14-supplement-v1'
COST='results/total-workflow-standard-v1/report-presentation-v2'

# number, title, PNG pattern, inspected representative, data scope, numerical evidence
CONTRACTS=[
 (1,'QSCI energy error vs shots',A+'/figures/*/Fig01_shots.png',A+'/figures/LiH_R1.5/Fig01_shots.png',
  'All30conditions; final PRIMARY saved circuits; R32,9M,20draws, pointwise90% intervals. Negative interval tails are not negative exact ground-state errors. Later optimizedB1 not plotted here.',
  [A+'/analysis/readout-summary.parquet',A+'/milestone1-audit.json']),
 (2,'QSCI energy error vs CNOT',A+'/figures/*/Fig02_cnot.png',A+'/figures/LiH_R1.5/Fig02_cnot.png',
  'All30cases/all5pools; M1000/R32; growth trajectories and fixed controls. Z-distribution sampling cost, not full-unitary VQE compression; independent readout means, not growth-policy validation.',
  [A+'/analysis/readout-summary.parquet',A+'/checkpoint-inventory.json']),
 (3,'QSCI energy error vs CNOT depth',A+'/figures/*/Fig03_depth.png',A+'/figures/LiH_R1.5/Fig03_depth.png',
  'Same cohorts/M/R asFig2; logical Z-sampling depth, not calibrated native-device depth. SameR is not equal actuald; no same-energy resource guarantee.',
  [A+'/analysis/readout-summary.parquet',A+'/checkpoint-inventory.json']),
 (4,'Non-HF Top-K discovery vs shots',K+'/*_Fig04_allK.png',K+'/LiH_R1.5_Fig04_allK.png',
  'All30cases, K3/5/10/20/50 and9M. Mean marginal target recall, NOT joint all-target discovery probability. Actual available target count shown; FCI targets evaluation-only; final saved not best.',
  [K+'/manifest.json',K+'/independent-value-review-v1.json',K+'/all-planned-topK.csv']),
 (5,'VQE vs QSCI during ADAPT growth',S+'/*_Fig05_growth.png',S+'/LiH_R1.5_Fig05_growth.png',
  'All30cases/all5pools; original preparation VQE vs independent A M1000/R32 QSCI.5231primary points; FCI subtraction reporting-only; not an online early-stop policy.',
  [S+'/growth-curves.csv',S+'/presentation-v2/validation-and-style-v2.json']),
 (6,'Early-stopping ratio distribution',A+'/figures/Fig06_retrospective_ratios_tol_*.png',A+'/figures/Fig06_retrospective_ratios_tol_0.1.png',
  'All150trajectories, tolerance0/.1/.5/1mHa. Same-sample historical M1000 retrospective selection relative to last SAVED endpoint, not unknown true convergence. Descriptive geometry boxplots, not independent molecules.',
  [A+'/retrospective-stops.parquet',A+'/analysis/new-draw-retrospective-stops.csv']),
 (7,'Classical construction time vs qubits','results/molecular-MPS-scaling-standard-v1/derived-report-v1/01_construction_scaling.png',
  'results/molecular-MPS-scaling-standard-v1/derived-report-v1/01_construction_scaling.png',
  '16real physical problems/64cells,16–44q; original3RAACSscore constructors. Wall scope is recorded task, not given-integrals Fig18 or hardware. Distinct chemical systems do not establish asymptotic exponent. E26strong-control results separately linked.',
  ['results/molecular-MPS-scaling-standard-v1/validation.json','results/molecular-MPS-scaling-standard-v1/derived-report-v1/manifest.json']),
 (8,'Shots vs construction cost / depth frontier','results/final-cost-frontiers-standard-v2/*_Fig08_known_model_budget_frontier.png',
  'results/final-cost-frontiers-standard-v2/LiH_R1.5_Fig08_known_model_budget_frontier.png',
  'All30cases. Known build+final MODEL budget, R64, allregisteredseeds before observed90% feasibility. Unknown historical VQE budgets excluded/notzero; partialseed families retained/ineligible. Recorded-time panel scope-stratified; NOT complete classicalwall or QPUtime.',
  ['results/final-cost-frontiers-standard-v2/all-seed-family-frontiers.csv','results/final-cost-frontiers-standard-v2/audit-v1/validation.json']),
 (9,'CEO / LUCJ / QEB / qubit / GSD heatmap',S+'/Fig09_unified_method_heatmap.png',S+'/Fig09_unified_method_heatmap.png',
  '30cases,6families, commonM1000/R64 final circuits. Bounded CCSD-VQE LUCJ, not optimal LUCJ. Actuald/RNGnamespaces differ; N/Aretained. Colorfloor only for display, signed raw roundoff labels retained. H4R2.2qubit Szproblem not useful convergence.',
  [S+'/unified-method-heatmap.csv',S+'/presentation-v2/validation-and-style-v2.json']),
 (10,'Held-out molecule generalization','results/PhaseC-replay-standard-v1/derived-report-v1/figures/LOMO-cluster-M*.png',
  'results/PhaseC-replay-standard-v1/derived-report-v1/figures/LOMO-cluster-M1000.png',
  'Training-only LOMO90folds,20independentreplay draws; molecular-cluster descriptive t90. Allselected rulesgradient => zeroresource saving/regret. Replaygraph is not avoided-growth proof; freshgeometry/AS actual112growth separately evaluated. No unseen-molecule win.',
  ['results/PhaseC-replay-standard-v1/LOMO-v1/independent-validation.json','results/PhaseC-actual-growth-standard-v1/derived-report-v1/primary-equal-molecule-summary.csv']),
 (11,'Important determinant probability vs iteration',A+'/figures/*/Fig11_probability_tracks.png',A+'/figures/LiH_R1.5/Fig11_probability_tracks.png',
  '30case pairsCEO-OVP/qubit top5oracle-important traces; complete perconfiguration/operator/threshold records retained inParquet. Oracleidentity diagnostic only. Operatoraddition and fullreoptimization inseparable here.',
  [A+'/operator-tracking/LiH_R1.5/important-probabilities.parquet',A+'/milestone1-audit.json']),
 (12,'Operator addition vs QSCI improvement',S+'/*_Fig12_operator_gain.png',S+'/LiH_R1.5_Fig12_operator_gain.png',
  '4907adjacent actualsaved transitions across30cases, M1000/R32. Positive gainmeans lowererror; negativevalues retained. Delta samplingCNOT/depth, no missing-step bridging, reoptimization confounded. OperatorIDs inrawtable.',
  [S+'/addition-reoptimization-transitions.csv',S+'/presentation-v2/validation-and-style-v2.json']),
 (13,'HF probability vs non-HF coverage',S+'/*_Fig13_HF_coverage.png',S+'/LiH_R1.5_Fig13_HF_coverage.png',
  '5231primarycheckpoint analytic M1000 expectations. X=HF measurementprobability, NOT FCI HFweight. Y=FCI-weighted nonHF discovery, oracleevaluation only. Savedtrajectory lines not causal/monotonic fit.',
  [S+'/HF-vs-nonHF-coverage.csv',S+'/presentation-v2/validation-and-style-v2.json']),
 (14,'Recovery before/after energy',F14+'/*_Fig14_QSCI_fullSQD_energy_pairs.png',F14+'/LiH_Fig14_QSCI_fullSQD_energy_pairs.png',
  '720planned/708complete/12missing. Same M1000acquisition, QSCI vs selected-ground fullSQD, ideal/readout1%, commonR64 andactuald. Recovery/spinproduct/batch/carryover jointlychange; not recovery-only causal effect. FCIouterlabel. Smallnegative roundoffretained; symloglinearregion.01mHa.',
  [F14+'/manifest.json',F14+'/all-planned-energy-pairs.csv','results/full-fermionic-SQD-projected-standard-v1/selected-ground-validation.json']),
 (15,'Random / trained / oracle sampler',S+'/presentation-v2/*_Fig15_random_trained_oracle.png',S+'/presentation-v2/LiH_R1.5_Fig15_random_trained_oracle.png',
  '30cases, all3randomseeds mean/range(NOTconfidenceinterval), trainedlength/orderretrospective. FCI Born analytic hypothetical oracle; no practicalpreparation or discovery upperbound. Color/rangematchfixedv2. Some curvesoverlap legitimately.',
  [S+'/oracle-random-trained-discovery.csv',S+'/presentation-v2/validation-and-style-v2.json']),
 (16,'Single vs multiple shallow samplers',A+'/figures/*/Fig16_multi_*.png',A+'/figures/LiH_R1.5/Fig16_multi_ceo_mvp.png',
  'All30cases/all5pools: final, retrospectivebest, equal3, diversitypilot, QSCIenergypilot. Same totalevaluationM includespilots; historicalgrowth/calibrationnotfree. FixedR32 andactualdcontrols separate. Fixed-bankD6supportsaturation is different scope.',
  [A+'/mixtures/LiH_R1.5/readouts.parquet',A+'/milestone1-audit.json','results/fixed-bank-allocation-standard-v1/support-saturation-v1/per-source-support-diagnostic.csv']),
 (17,'CI dimension sensitivity',A+'/figures/*/Fig17_R_cap.png',A+'/figures/LiH_R1.5/Fig17_R_cap.png',
  '30cases,M1000,R8/16/32/64/128/256/512, finalprimary saved. Nested samecount spaces, actualdseparate; frequency vs equal-d randomobserved selection inanalysis. No FCIpadding; observed-only+knownHF.',
  [A+'/analysis/readout-summary.parquet',A+'/analysis/final-capacity-effects.csv']),
 (18,'Total classical / quantum resource frontier',COST+'/Fig18_given_integrals_total_workflow_*.png',COST+'/Fig18_given_integrals_total_workflow_LiH.png',
  '90planned/86complete/4CCSDinitfailures,3methods/30conditions. Given-integrals fresh-process totalwall vs all-paidCNOT×modelsamples, R64errorcolor,2000paidtotal. Imports/init/construction/pilots/finalCI/persistincluded; priorRHF/integrals/sharedexport/audit/plots excluded. One realization, not QPUtime, not historicalCEO totalcost. v2layoutonly.',
  [COST+'/manifest.json','results/total-workflow-standard-v1/validation.json',COST+'/all-planned-workflows.csv']),
]


def validate_contracts(contracts):
    if [r[0] for r in contracts]!=list(range(1,19)):raise ValueError('All18 named figure types required')
    for row in contracts:
        if len(row)!=6 or not all(row[1:]):raise ValueError('Incomplete semantic/source contract')


def build(output=None,mode='standard'):
    if mode not in ('quick','standard'):raise ValueError('Unknown mode')
    validate_contracts(CONTRACTS)
    inputs={};files={};rows=[]
    for number,title,pattern,representative,scope,evidence in CONTRACTS:
        pngs=sorted(ROOT.glob(pattern))
        if not pngs or ROOT/representative not in pngs:raise ValueError('Missing rendered relationship: '+title)
        for p in pngs:
            for ext in ('png','svg','pdf'):
                sibling=p.with_suffix('.'+ext)
                if not sibling.exists():raise ValueError('Missing format: '+str(sibling))
                files[str(sibling.relative_to(ROOT))]=sha(sibling)
        for rel in evidence:
            inputs[rel]=sha(ROOT/rel)
        rows.append(dict(figure=number,title=title,three_format_sets=len(pngs),representative=representative,
            status='SCOPED_RELATIONSHIP_AND_REPRESENTATIVE_RENDER_REVIEWED',scope=scope,
            evidence=';'.join(evidence),all_export_files_visually_reviewed=False,
            all_plotted_values_recomputed_by_this_index=False,
            source_audits_reused=True,unsupported_general_superiority=False))
    inputs['benchmark/current_figure_index.py']=sha(__file__)
    # Bind the actual science datasets/audits, not a mutableLATEST pointer.
    record=dict(schema='current18figure-semantic-source-index-v1',mode=mode,rows=rows,
                input_sha256=inputs,figure_sha256=files,figure_types=18,
                representative_visual_review_date='2026-10-23',
                review_scope='Representative axes/units/legend/scope plus all referenced file hashes; not every-render visual QA or new independent numerical reanalysis',
                no_renderers_called=True,new_experiments=0,new_solver_calls=0,whole_goal_complete=False)
    identity=hashlib.sha256(canonical(record).encode()).hexdigest()
    out=Path(output or ROOT/'results/current18figure-index-v1')/'snapshots'/identity
    out.mkdir(parents=True,exist_ok=True)
    lines=['#18 requested figures — current semantic/source index','',
           'All18 requested relationships have actual PNG/SVG/PDF exports. This is scoped figure coverage, not proof of scientific superiority or full-objective completion.',
           'Representative PNGs were inspected for axes, units, legend, comparison population and caveats. '
           'Every indexed export is hash-bound; this is not visual inspection of every file or an independent recomputation of every plotted number.','',
           '|Fig|Relationship|3-format sets|','|---|---|---:|']
    lines += [f'|{r["figure"]}|{r["title"]}|{r["three_format_sets"]}|' for r in rows]
    for r in rows:
        path=ROOT/r['representative']
        lines += ['',f'## Fig{r["figure"]}: {r["title"]}','',
          ' / '.join(f'[{ext.upper()}]({os.path.relpath(path.with_suffix("."+ext),out)})' for ext in ('png','svg','pdf')),'',
          r['scope'],'','Numerical/audit evidence: '+', '.join(f'[{Path(p).name}]({os.path.relpath(ROOT/p,out)})' for p in r['evidence'].split(';'))]
    lines += ['','## Additional completed evidence (not new figure-number substitutes)','',
       '[Synthetic noise6molecules]('+os.path.relpath(ROOT/'results/gate-noise-full-standard-v1/report-v1/REPORT_JA.md',out)+')',
       '[E26strong controls]('+os.path.relpath(ROOT/'results/molecular-strong-controls-standard-v2/report-v1/REPORT.md',out)+')',
       '[ThresholdHCI]('+os.path.relpath(ROOT/'results/integral-threshold-HCI-standard-v1/report-v1/REPORT.md',out)+')',
       '[HCI same-d review]('+os.path.relpath(ROOT/'results/HCI-paired-diagnostic-standard-v1-review-v2/README.md',out)+')','',
       'Fig18 is now a real completed three-method given-integrals study, NOT all-historical-method total classical/QPU cost. '
       'Fig10 records a zero-saving selected policy; it is not a positive generalization claim. '
       'Fig14 changes more than Recovery alone. Fig7 excludes unsupported Fe–S extrapolation.']
    save_once(out/'index.json',canonical(record));save_once(out/'requested18figures.csv',csv_text(rows));save_once(out/'INDEX.md','\n'.join(lines)+'\n')
    (out.parents[1]/'LATEST.md').write_text(f'# Current18figure index\n\n[Index](snapshots/{identity}/INDEX.md)\n\nNo experiments or old renderings changed.\n')
    return dict(snapshot=str(out),requested_forms=18,scoped_relationships_available=18,
                indexed_files=len(files),new_experiments=0,no_renderers_called=True,whole_goal_complete=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true')
    p.add_argument('--mode',choices=['quick','standard'],default='standard');p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(build(a.output,a.mode),sort_keys=True))


if __name__=='__main__':main()
