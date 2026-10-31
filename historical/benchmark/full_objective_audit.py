"""Final full-objective evidence audit, not another experimental execution gate.

Checks sealed source preservation, actual required data fields, current
cohort/figure/report reconciliation and conditional applicability. Retains
unresolved scientific claims instead of converting them to fabricated wins.
Never executes an optimizer, samples a circuit or regenerates an old plot.
"""
from __future__ import annotations
import argparse
import csv
from io import StringIO
import importlib.util
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from .current_evidence import ROOT, OBJECTIVE, canonical, csv_text, save_once, sha

REPORT=ROOT/'results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1'
FIGURES=ROOT/'results/current18figure-index-standard-v1/snapshots/d89264228c0c90ea9eeeae60c3bd75f5306e3dbbbd3505dab1180c17ee0079e3'

# These are evidence judgments, not requirements rewritten to fit existing code.
# Positive superiority is an outcome, not a prerequisite for honest research.
JUDGMENTS=[
 ('0','Source inventory/preservation/named artifacts','EXECUTED_VERIFIED',
  ['results/standard-v1/REPRODUCIBILITY.json','results/standard-v1/execution-manifest-final.json'],
  'All40944 original artifacts and original code capsule freshly SHA/size checked; original330cells/6failures preserved. New work in siblings. A derivative presentation-binding incident remains disclosed, not claimed restored.'),
 ('A1','R grid, actuald, observed-only selection, success, time and memory','EXECUTED_SCOPED',
  ['results/sampler-study-standard-v1/milestone1-audit.json','results/sampler-study-standard-v1/analysis/readout-summary.parquet','results/sampler-study-standard-v1/analysis/final-ranking-control-effects.csv'],
  'R8–512, common raw counts and equal-d frequency/random-observed contrast; HF insertion not FCIpadding. Count/selection checks exhaustive;150independent numerical spot checks, not every eigenvalue recomputed.'),
 ('A2','Low shots / allK / coverage / unique / duplication / individual and joint M90 / finite energy','EXECUTED_SCOPED',
  ['results/sampler-study-standard-v1/analysis/discovery.parquet','results/sampler-study-standard-v1/analysis/individual-target-discovery.parquet','results/sampler-study-standard-v1/analysis/joint-m90.parquet','results/sampler-study-standard-v1/analysis/shots-to-observed-p90.parquet'],
  'M10–100000; all5K; joint categorical waiting-time MC not product of marginal probabilities; oracle evaluation separated. Tested-grid observed90 differs from confidence-lower-bound90; nonattainment retained.'),
 ('A3','Every trajectory and near-best sensitivity / molecule-pool summaries','EXECUTED_SCOPED',
  ['results/sampler-study-standard-v1/retrospective-stops.parquet','results/sampler-study-standard-v1/analysis/new-draw-retrospective-stops.csv','results/sampler-study-standard-v1/analysis/historical-choices-independent-readout.csv'],
  'All30cases/5pools/0,.1,.5,1mHa tolerances. Saved-final denominators andMAX_STEPS not true convergence; post-hoc choices not prospective savings.'),
 ('A4','Operator addition / probability / perconfiguration lifecycle','EXECUTED_SCOPED',
  ['results/sampler-study-standard-v1/operator-tracking/LiH_R1.5/important-probabilities.parquet','results/final-relationships-standard-v1/addition-reoptimization-transitions.csv','results/sampler-study-standard-v1/interpretation-status.json'],
  'All150trajectories/4907transitions; thresholds1e-12/1e-8/1e-4; lifecycle/excitation/HF/operator records. Addition and fullreoptimization confounded; not isolated causality.'),
 ('A5','Single/final/equal/diversity/energy mixtures, pilots/switches, equal totalshots','EXECUTED_SCOPED',
  ['results/sampler-study-standard-v1/analysis/mixture-summary.csv','results/sampler-study-standard-v1/analysis/mixture-cluster-diagnostics.csv','results/sampler-study-standard-v1/milestone1-audit.json'],
  '135000readouts/6750groups. Pilots included, historical calibration/growth not free, switching count and unknown hardware latency separate. Oracle-best is retrospective.'),
 ('B1','All6 named LUCJ initialization/optimization families','EXECUTED_SCOPED',
  ['results/sampler-baselines-B1-standard-v1/REPRODUCIBILITY.json','study-v1/BASELINES_B1.md'],
  '13actual constructions/390planned/380available/10CCSDmissing, independent final readouts. Bounded VQE/QSCI search and multiseed random/MP2/CCSD/truncation; not optimizer-independent optimalLUCJ.'),
 ('B2','Five native pools / ADAPT-QSCI / HF separation','EXECUTED_SCOPED',
  ['results/sampler-baselines-B2-standard-v1/validation-available-v2.json','results/sampler-baselines-B2-exactprep-standard-v2/validation.json','study-v1/BASELINES_B2.md','study-v1/BASELINES_B2_EXACT_INIT.md'],
  '720originalattempts/96greedyinitfailures preserved;360exactprep successor trajectories/6900CP,660fresh QPY checks. Native CI-gradient/new-angle objectives; explicitMP initializer/settings, not exact author figures or pool-only proxy.'),
 ('B3','TE / qDRIFT / SqDRIFT / Givens / orbital / multiple basis','EXECUTED_SCOPED',
  ['results/nonvariational-B3-standard-v1/validation.json','results/SqDRIFT-ensemble-standard-v2/validation.json','results/multibasis-NOCI-standard-v2/validation.json'],
  '540nonvariationalcells;360nativeSqDRIFTensembles/6480members;4320multibasis tasks/4248complete. Magnitude-refined author-SDK partitionvariant, fullprep before rotation, no optimal CDF/NOCI reproduction claim.'),
 ('B3optional','k-UpCCGSD/tUPS/QNP/GQE-QSCI optional additions','OPTIONAL_NOT_IMPLEMENTED',
  ['study-v1/BASELINES_B3_PAULI_TE.md','study-v1/BASELINES_B3_SQDRIFT.md','study-v1/MULTIBASIS_NOCI_V1.md'],
  'Explicit additional candidates, not required priority families. Not implemented, not substituted by unrelated methods; no results claimed.'),
 ('B4','HF/uniform/random Pauli/QEB/CEO/trained-order random-angles/LUCJ/orbital controls','EXECUTED_SCOPED',
  ['results/random-B4-standard-v1/validation.json','results/sampler-baselines-B1-standard-v1/validation.json','results/nonvariational-B3-standard-v1/validation.json'],
  '390random controls with3seeds plus B1randomLUCJ/B3orbital. Classical uniform proposals not QPUshots. Trained topology length/order costs not free.'),
 ('B4classical','HCI/CIPSI-style classical references / quality and work','EXECUTED_SCOPED',
  ['results/classical-SCI-standard-v2/summary.csv','results/integral-threshold-HCI-standard-v1/validation.json','results/HCI-paired-diagnostic-standard-v1-review-v2/validation.json'],
  '90cardinality controls and168thresholdHCI cells/464spaces;280transition audits. Same-d1872planned withmissingness/HF-only. Not optimized SHCI/PT2/DMRG, grounderror or matchedwork; those claims not made.'),
 ('C1','FCI-free current features / residual / PT2 / cost','EXECUTED_SCOPED',
  ['study-v1/PHASE_C_STREAMING_CONTRACT.md','results/PhaseC-pilot-bank-standard-v1/validation.json','results/PhaseC-actual-growth-standard-v1/validation.json'],
  '3000episodes/20920frames, rawcounts-rebuilt paid pilots; currentenergy/newconfigs/HF/dimension/internal/external residual/cost. Screened external residual not full-space convergence.'),
 ('C2','All registered stopping-rule families / simple baselines','EXECUTED_NEGATIVE_PRIMARY_RESULT',
  ['results/PhaseC-replay-standard-v1/validation.json','results/PhaseC-actual-growth-standard-v1/derived-report-v1/primary-equal-molecule-summary.csv'],
  '63000replay decisions, actual112growth cells. Native gradient/energyplateau/discovery/cost/residual/PT2/combined; fixed saved percentages oracle only. Primary selectedgradient =>0%resource savings, not a failed-to-run result.'),
 ('C3','LOMO, fresh distances/AS, independentseeds, pilots, cluster uncertainty','EXECUTED_SCOPED',
  ['results/PhaseC-replay-standard-v1/LOMO-v1/independent-validation.json','results/PhaseC-actual-growth-standard-v1/evaluation-validation.json'],
  '90training-onlyLOMO folds and8freshgeometry/AS physicalproblems; separatefinal draws. One actualconstructionseed andknownmolecularidentities limit generalization. All pilotscharged; no unsupported unseen-identity claim.'),
 ('D1','Actual FCI/full-vector-independent practical RAACS','EXECUTED_SCOPED',
  ['results/RAACS-quick-v1/validation.json','results/RAACS-independent-comparison-standard-sharded-v1/validation.json','results/molecular-MPS-scaling-standard-v1/validation.json'],
  'HF/counts/observedCI/namedintegrals/localproposals/paidpilots/independentfinal, stop and all5scores. MPS practical route demonstrated to44q; smallstatevector qualification separate.'),
 ('D2','Regularized external importance / nonzero denominator / limited scope','EXECUTED_SCOPED',
  ['benchmark/resource_adaptive_sampler.py','benchmark/integral_connections.py','study-v1/RAACS_PROTOTYPE.md'],
  'Registered finite EN denominator floor and bounded target/parent kernels. Local residual proxy, not exact full-space importance or free CI-vector physical preparation.'),
 ('D3','Bounded local operator bank / no full-pool each iteration','EXECUTED_SCOPED',
  ['benchmark/local_sampler_primitives.py','study-v1/RAACS_STANDARD_EXECUTION.md','results/RAACS-standard-v1/manifest.json'],
  'Qualified shallow QE/exchange/generator transformations, bankbounds persisted. Local CIgradient not relabelled global ADAPT; other native baseline separately available.'),
 ('D4','Five score ablations / resource and pilot accounting','EXECUTED_SCOPED',
  ['results/RAACS-independent-comparison-standard-sharded-v1/validation.json','study-v1/RAACS_INDEPENDENT_COMPARISON.md'],
  '840constructorcells, independent comparison164320drawrecords/3868200actual-d pairs. Same-M/R andknownconstructor+final scopes distinct; mode-boundRNG/path interaction not isolatedcausal law.'),
 ('D5','Fixed / oneparameter / block / fullVQE / QSCI local angles','EXECUTED_SCOPED',
  ['results/parameter-controls-standard-v1/validation.json','results/online-parameter-standard-v1/validation.json'],
  '540conditional and450actualonline cells, pairedphysical/circuit/decision audits. FullVQE worsens meanQSCI vsgrid in this protocol; not universalanglecausal theorem. VQE quantumobjective measurement budget unknown.'),
 ('D6','Greedy/uniform/discovery/energy multishallow / same totalshots','EXECUTED_SCOPED',
  ['results/fixed-bank-allocation-standard-v1/validation.json','results/fixed-bank-allocation-standard-v1/support-saturation-v1/per-source-support-diagnostic.csv'],
  '540allocationcells/90inheritedbanks; paidpilots andswitches.90/90prefixunions add0support beyondlast, conditionalbank result not allshallowmixtures.'),
 ('E1AB','Named small/intermediate molecules / largerN2/conjugated systems','EXECUTED_SCOPED',
  ['results/molecular-strong-controls-standard-v2/manifest.json','results/molecular-strong-controls-standard-v2/validation.json'],
  '26actualphysicalproblems8–44q, H8/10/12/16/20/22,H2O,C2,CH2,C4H4,C6H6,extendedN2.234terminal146complete88preserved engineering/init/approximation failures, not88physical defeats.'),
 ('E1C','Conditional [2Fe–2S]/[4Fe–4S] availability','CONDITIONAL_NOT_EXECUTED_UNQUALIFIED_INPUT_AND_ROUTE',
  ['benchmark/molecular_problem_no_fci.py','study-v1/MOLECULAR_MPS_SCALING.md','results/molecular-strong-controls-standard-v2/physical-layout-validation-v2.json'],
  'No defined Fe-bearing geometry/charge/spin/active-space integral problem or qualified Fe-specific simulation route in the project. Generic MPS code/44qHchain does not activate this conditional requirement; no QPU executed. Not a claim that FeS simulation is mathematically impossible.'),
 ('E2','NoFCI/fullFock; bounded named-H/MPS/local candidates/memory/CI/durable evidence','EXECUTED_SCOPED',
  ['results/molecular-MPS-scaling-standard-v1/manifest.json','results/molecular-MPS-scaling-standard-v1/validation.json','results/molecular-strong-controls-standard-v2/physical-layout-validation-v2.json'],
  'Runtime nofull-vector/FCI guards, boundedCI<=64/integral cap, perdraw layout/norm qualification andsamecircuitbond checks. Missingrequired H10CCSD initializer leavesqualificationINCOMPLETE, not silentlyPASS. GroundreferenceUNKNOWN explicitly allowed by objective.'),
 ('E3','Qubit/orbitals/electrons vs time/RSS/candidates/gates/native/shots/d/energy','EXECUTED_SCOPED',
  ['results/molecular-MPS-scaling-standard-v1/derived-report-v1/manifest.json','results/molecular-strong-controls-standard-v2/report-v1/common-R-results-and-resources.csv','results/molecular-strong-postprocess-standard-v1-review-v1/validation.json'],
  'Measured16/26problem scaling and240nativecompile/bond rows, not merely a circuitfactory. Grounderror, chemicalaccuracy andhardwareVRAM/QPU time unknown/notapplicable, not0. No asymptotic exponent/generalchemicalscaling claim.'),
 ('F','QSCI / officialRecovery / fullfermionicSQD / readout/depolarizing/T1T2/routing','EXECUTED_SCOPED',
  ['results/full-fermionic-SQD-projected-standard-v1/selected-ground-validation.json','results/gate-noise-full-standard-v1/validation.json','results/gate-noise-full-standard-v1/report-v1/REPORT_JA.md'],
  '720selected-ground/708complete and3600noise/3537complete. QSCI/recovery/directmeasured/classicallygenerated configurations distinct, independentselectedroots. gate-localT1T2(noidle) andsynthetictargets;3capacitycensors/60sourceCCSDmissing retained. Trim optionalnotrequired. No realhardware/calibrationavailable in executed study.'),
 ('Foptional','TrimSQD if necessary','OPTIONAL_NOT_IMPLEMENTED',
  ['study-v1/FULL_FERMIONIC_SQD.md','study-v1/FULL_SQD_PROJECTED_GROUND.md'],
  'TrimSQD is optional ifneeded, not implemented or presented as the bounded selected-ground fullfermionicSQD solver.'),
 ('7','All10primarymetrics andauxiliarydata','EXECUTED_WITH_EXPLICIT_UNKNOWNS',
  ['results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1/primary-metrics.json','results/total-workflow-standard-v1/validation.json'],
  '10records: exactsmallerror/success/tested90, logical/nativegates, scopedconstruction/totalwall/RSS/modelshots. Unknown historicaltrainingmeasurement cost/largegrounderror retained; actualhardware0. Gate×shotproxy notQPUtime.'),
 ('8','All18namedrelationships / realdata / PNGSVG PDF','EXECUTED_SCOPED',
  ['results/current18figure-index-standard-v1/snapshots/d89264228c0c90ea9eeeae60c3bd75f5306e3dbbbd3505dab1180c17ee0079e3/index.json'],
  'All18semanticrelationships;1503indexedexports,18representativesinspected andsourceauditsbound. Fig4allK,Fig10LOMO,Fig14energy pairs,Fig18actual3methodworkflow. Noteveryexportvisuallyreviewed orallplotvaluesnewlyrecomputed. No placeholder.'),
 ('9','Fairness / physicalidentity / reald / FCIisolation / seeds/clusters/missingness/cost','EXECUTED_SCOPED',
  ['study-v1/SCIENCE.md','study-v1/ATTRIBUTION_AND_COST_INTERPRETATION.md','results/HCI-paired-diagnostic-standard-v1-review-v2/validation.json','results/shared-source-final-draw-audit-v4/validation.json'],
  'SharedHamiltonians/sector/MO, commonR andactuald, noFCItraining/stop; oracle separate, nofreepastgrowth. Molecularmeans/clusteruncertainty andallfailed/capped groups preserved. Sharedcounts/anglepath/RNGdependencies explicit. No universaloptimalLUCJ/hardware advantage claims.'),
 ('10','All6milestones / runnableCLI / quickqualification / configcaps / tests/resume','EXECUTED_SCOPED',
  ['benchmark/run.py','benchmark/report.py','benchmark/plot.py','results/full-sampler-regression-2026-10-23-v6.xml','study-v1/CURRENT_DELIVERY_V3.md'],
  'ActualA→B→C→D→E→Fcohorts withquickbeforestandard, boundedworkers/configs/durablecounts. Legacycommands preserved; --scalable-study --current-evidence currentreport/index.401tests pass, notalonefullscopeproof. GPUoriginalqualification/source evidence separate, no newremotejobs.'),
 ('11','All15questions / success/failure/uncertainty / cause andcost distinction','EXECUTED_SCOPED',
  ['results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1/questions.json','results/RAACS-direct-configuration-fork-v1/report-v2/validation.json'],
  'All15fromfullobjective answered withactualresults or unresolvedscientificclaim. Same-initial-info classicalcontrol4identicalspaces shrinks quantumnecessityclaim, notwholeprojectworthless. Addition instructions do not require genericquantum advantage.'),
 ('12','Workingcode/dataset/mainfigures/ablations/scaling/limitations; no fabricated win','EXECUTED_WITH_CONDITIONAL_LIMITS',
  ['study-v1/CURRENT_DELIVERY_V3.md','study-v1/ATTRIBUTION_AND_COST_INTERPRETATION.md','results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1/REPORT_JA.md'],
  'Research pipeline/results/report available. GeneralRAACS/quantum/cost/generalization superiority not established; objective explicitly does not assume every-condition success. Data-supportednegative/conditionaloutcomes not replaced by plan-onlyNoGo.'),
 ('12target','Ideal scientific performance targets','EVALUATED_NOT_UNIVERSALLY_ESTABLISHED',
  ['results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1/questions.json','study-v1/ATTRIBUTION_AND_COST_INTERPRETATION.md'],
  'These are scientific goals, not observed universal wins. Primaryearlystopping saves0%; broadRAACS/quantum/cost/unseenmolecule advantage remainsunestablished. No success manufactured from implementation or conditional subset.'),
 ('addition','Focused reuse / quality vs cost / classical fork / fullprep+rank+residual correctness','EXECUTED_SCOPED',
  ['study-v1/OBJECTIVE_ADDITION_SCOPE.md','study-v1/ATTRIBUTION_AND_COST_INTERPRETATION.md','results/RAACS-direct-configuration-fork-v1/report-v2/validation.json','results/multibasis-NOCI-standard-v2/validation.json'],
  'Minimal3classicalforks/9reusedtrajectories, originalbounds/source kept. Fullprep beforebasisrotation, inputvectors/retainedrank/cutoff distinct, partialexternalresidual notglobalcertificate. No newunnecessaryfamilies/noiserobustnessclaim, unsupportedquantumnecessity shrunk.'),
]


def extract_items(text):
    """All objective headings/list items/named CLI commands, with original line IDs."""
    rows=[];main='0';sub='0';addition=False
    for number,raw in enumerate(text.splitlines(),1):
        line=raw.replace('\\','').strip()
        if '追加方針：研究範囲' in line:addition=True;sub='addition'
        if not addition:
            top=re.match(r'^# (\d+)\.',line)
            if top:main=top.group(1);sub=main
            inner=re.match(r'^## ([A-E]\d)\.',line)
            if inner:sub=inner.group(1)
        kind=('heading' if re.match(r'^#{1,3} ',line) else
              'list_item' if re.match(r'^(?:[-*] |\d+\. )',line) else
              'named_figure' if re.match(r'^Fig \d+\s*[：:]',line) else
              'named_command' if line.startswith('python -m benchmark.') else None)
        if kind:
            group='addition' if addition else sub
            group={'1':'A1','2':'B1','3':'C1','4':'D1','5':'E1AB','6':'F'}.get(group,group)
            if group=='B3' and any(value in line for value in ('k-UpCCGSD','tUPS','QNP','GQE-QSCI')):group='B3optional'
            if group=='F' and 'Trim SQD' in line:group='Foptional'
            if group=='E1':group='E1AB' # Fe entries override below
            if 'Fe–' in line or 'Fe−' in line or 'Fe–' in raw:group='E1C'
            if group=='12' and kind=='list_item':group='12target'
            rows.append(dict(line=number,kind=kind,original_text=raw,group=group))
    return rows


def verify_sealed(folder):
    folder=Path(folder);pointer=json.loads((folder/'REPRODUCIBILITY.json').read_text())
    manifest_path=folder/pointer['manifest']
    if sha(manifest_path)!=pointer['manifest_sha256']:raise ValueError('Sealed manifest drift')
    manifest=json.loads(manifest_path.read_text());total=0
    for i,(name,expected) in enumerate(manifest['artifacts'].items(),1):
        path=folder/name
        if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise ValueError('Sealed source drift: '+str(path))
        total+=path.stat().st_size
        if i%10000==0:print(folder.name,i,'artifacts checked',flush=True)
    capsule=folder/manifest['code_capsule']
    if manifest.get('code_capsule_sha256') and sha(capsule)!=manifest['code_capsule_sha256']:
        raise ValueError('Code capsule drift')
    return dict(folder=str(folder),dataset_id=pointer['dataset_id'],manifest_sha256=sha(manifest_path),
                artifacts=len(manifest['artifacts']),bytes=total,all_match=True,
                capsule_sha256=sha(capsule),old_files_modified=False)


def availability():
    modules={}
    for name in ('block2','pyblock2','pyscf.dmrgscf','pyci','dice'):
        try:modules[name]=importlib.util.find_spec(name) is not None
        except (ModuleNotFoundError,ValueError):modules[name]=False
    queue=json.loads((ROOT/'results/molecular-strong-controls-standard-v2/manifest.json').read_text())['queue']
    return dict(reference_modules=modules,reference_binaries={n:shutil.which(n) for n in ('block2main','block.spin_adapted','Dice','shci')},
        actual_physical_specs=[r['physical_spec'] for r in queue],
        registered_Fe_bearing_problems=sum('Fe' in r['physical_spec']['molecule'] for r in queue),
        qualified_Fe_cluster_input_and_backend_available=False,
        scope='Current configured project only. No inference of generic impossibility or all external resource availability.',
        reason='No defined Fe geometry/charge/spin/active-space integrals or Fe-specific approximation qualification; small-memory genericMPS/44qHchains do not certify it. No QPU runs configured/executed by this study.')


def audit(output=None):
    from .current_evidence_review import review
    out=Path(output or ROOT/'results/full-objective-audit-2026-10-23-v1');out.mkdir(parents=True,exist_ok=True)
    before={str(p):sha(p) for p in (OBJECTIVE,ROOT/'AGENTS.md',ROOT/'study-v1/AGENTS.md')}
    sealed=[verify_sealed(ROOT/'results'/folder) for folder in
            ('standard-v1','sampler-study-standard-v1','sampler-baselines-B1-standard-v1')]
    current=review(REPORT,FIGURES)
    # Actual data schemas/rows are checked, not inferred from figure existence.
    import pyarrow.parquet as pq
    a=ROOT/'results/sampler-study-standard-v1'
    needed={
      'analysis/readout-summary.parquet':['R_cap','shots','mean_dimension','success_probability_estimate','actual_lookup_or_solve_seconds_mean','process_peak_rss_bytes'],
      'analysis/discovery.parquet':['hf_probability','entropy_nats','expected_unique','nonhf_weighted_coverage','recall_top3','recall_top5','recall_top10','recall_top20','recall_top50'],
      'analysis/individual-target-discovery.parquet':['id','probability','m90_decimal','shots','discovery_probability'],
      'analysis/joint-m90.parquet':['K_requested','actual_targets','trials','m90_estimate','m90_mc90_low','m90_mc90_high','scope'],
      'analysis/shots-to-observed-p90.parquet':['smallest_tested_M_with_observed_p90','smallest_tested_M_with_ci_lower90','maximum_M_tested','status'],
      'operator-tracking/LiH_R1.5/important-probabilities.parquet':['configuration_id','step','probability','fci_weight'],
    }
    schemas=[]
    for path,fields in needed.items():
        p=a/path;table=pq.ParquetFile(p)
        if not set(fields)<=set(table.schema.names) or table.metadata.num_rows==0:raise ValueError('Missing required dataset columns: '+path)
        schemas.append(dict(file=str(p.relative_to(ROOT)),rows=table.metadata.num_rows,required_columns=fields,sha256=sha(p)))
    # Verify every named original file and each manual evidence pointer exists.
    named=['STUDY_PROTOCOL.md','report.md','REQUIREMENTS_AUDIT.md','execution-manifest-final.json','summary.csv','qsci-summary.csv','all-cell-status.csv','analysis/checkpoint_optima.csv','analysis/analytic_metrics.csv','analysis/construction_costs.csv','analysis/shots_to_chemical_accuracy.csv','sqd-matched-dimension/summary.csv']
    named_files={}
    for name in named:
        path=(ROOT/name if name=='STUDY_PROTOCOL.md' else ROOT/'results/standard-v1'/name)
        named_files[str(path.relative_to(ROOT))]=sha(path)
    references={};judgments=[]
    for key,title,status,paths,scope in JUDGMENTS:
        for path in paths:references[path]=sha(ROOT/path)
        judgments.append(dict(group=key,title=title,status=status,evidence=';'.join(paths),interpretation_scope=scope))
    items=extract_items(OBJECTIVE.read_text());keys={j['group'] for j in judgments}
    for item in items:
        if item['group'] not in keys:raise ValueError('Unmapped original objective item: '+str(item))
        j=next(j for j in judgments if j['group']==item['group'])
        item['evidence']=j['evidence'];item['assessment']=j['status']
        item['scope']='Group-level evidence judgment, not a claim that text extraction alone proves every mathematical property.'
    tree=ET.parse(ROOT/'results/full-sampler-regression-2026-10-23-v6.xml').getroot()
    suites=list(tree) if tree.tag=='testsuites' else [tree]
    tests=sum(int(s.attrib['tests']) for s in suites);fail=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    if tests!=401 or fail:raise ValueError('Actual full regression mismatch')
    practical=json.loads((ROOT/'results/molecular-MPS-scaling-standard-v1/manifest.json').read_text())
    if practical['FCI_allowed'] or practical['full_Fock_vector_allowed']:raise ValueError('Practical noFCI/fullvector boundary changed')
    available=availability()
    result=dict(schema='full-objective-evidence-audit-v1',status='FULL_OBJECTIVE_ITEMS_MAPPED_PRESERVATION_AND_CURRENT_SCOPED_EVIDENCE_CHECKED',
        original_objective_sha256=sha(OBJECTIVE),sealed_cohorts=sealed,named_original_files=named_files,
        current_review=current,actual_required_data_fields=schemas,evidence_sha256=references,
        judgments=len(judgments),extracted_original_items=len(items),unmapped_items=0,
        tests=dict(passed=tests,failed=fail),conditional_applicability=available,
        new_experiments=0,new_eigensolver_calls=0,hardware_shots=0,
        scientific_claims_not_established=['UniversalRAACS/CEO advantage','Quantum sampling necessity','HistoricalCEO end-to-end quantum-cost advantage','Large-system ground-state accuracy','Previouslyunseen molecular-identity superiority','Calibratedhardware advantage'],
        optional_not_implemented=['k-UpCCGSD','tUPS/oo-tUPS','QNP','GQE-QSCI','TrimSQD'],
        whole_goal_complete=False,
        completion_scope='This audit supplies full item/evidence mapping, not an automatic goal-completion decision. ConditionalFe and unknown costs remain explicit; scientific wins are not fabricated.',
        script_sha256=sha(__file__))
    for p,value in before.items():
        if sha(p)!=value:raise ValueError('Authority source changed during audit')
    save_once(out/'audit.json',canonical(result));save_once(out/'requirement-judgments.csv',csv_text(judgments))
    save_once(out/'original-objective-items.csv',csv_text(items))
    lines=['# Full-objective evidence audit','',
      'Original scope retained. This is a requirement/evidence reconciliation, not a positive scientific result or an automatic completion certificate.','',
      f'{len(judgments)} explicit judgments, {len(items)} original headings/list items/named commands, zero unmapped items. '
      'Text extraction is not a proof of semantics; the judgments below state the inspected sources and limitations.','',
      '## Fresh sealed preservation','', '|Cohort|Artifacts|Bytes|Allhashes/sizes match|','|---|---:|---:|---|']
    lines += [f'|{Path(s["folder"]).name}|{s["artifacts"]}|{s["bytes"]}|{s["all_match"]}|' for s in sealed]
    lines += ['','## Requirement judgments','', '|Group|Requirement|Evidence judgment|Boundary|','|---|---|---|---|']
    lines += [f'|{j["group"]}|{j["title"]}|{j["status"]}|{j["interpretation_scope"]}|' for j in judgments]
    lines += ['','Evidence paths and original line numbers are in the two CSVs. All named original artifacts were read and SHA-bound. '
       'Source/current review and field inventories are in audit.json.','',
       '## Conditional scope, not erased','',
       'Fe–S has no specified physical input or qualified Fe-specific route in this project. '
       'GenericMPS availability does not establish valid Fe-sector/active-space simulation. No global impossibility claim. '
       'ExtendedN2, benzene and44qHchains remain actual scaling outcomes, not Fe substitutes. '
       'No trusted large ground reference or hardware-calibrated job was available in the executed study; unknown labels andhardware0 remain.','',
       'The H10 fixedCCSD source did not converge. Its status and physicalqualificationINCOMPLETE remain unchanged; '
       'otherqualified/capped/failed routes have their own evidence. The objective requires preserving missingness, not rerunning until success.','',
       'Historical VQE quantum trainingcost is unknown. The given-integrals3method wholeworkflow trial does not backfill it. '
       'Reporting the unknown is required; it is not an observedzero or a cost-advantage proof.','',
       '## Final review decision still separate','',
       'No remaining gap may be dismissed merely because tests or this mapping are green. '
       'A final reader must inspect the conditional/limited judgments against the unchanged objective and distinguish '
       'unachieved scientific advantage from an unexecuted deliverable. Goal remains active in this artifact.']
    save_once(out/'AUDIT.md','\n'.join(lines)+'\n')
    return {k:result[k] for k in ('status','judgments','extracted_original_items','unmapped_items','new_experiments','whole_goal_complete')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    print(json.dumps(audit(parser.parse_args().output),sort_keys=True))
