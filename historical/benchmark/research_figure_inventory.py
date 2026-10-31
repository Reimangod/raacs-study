"""Pure-IO inventory of the actual18 requested figure types; no fake plots.

Existence is only candidate evidence, never semantic completion certification.
Does not import numerical backends or touch an executing study's outputs.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A='results/sampler-study-standard-v1/figures'
O='results/standard-v1/figures'
S='results/final-relationships-standard-v1'
ITEMS=[
 (1,'QSCI Energy Error vs Shots',[A+'/*/Fig01_shots.png'],'Same registered source final checkpoints; R32; simulated draws.'),
 (2,'QSCI Energy Error vs CNOT Count',[A+'/*/Fig02_cnot.png'],'M1000/R32; checkpoint trajectories and fixed controls.'),
 (3,'QSCI Energy Error vs CNOT Depth',[A+'/*/Fig03_depth.png'],'M1000/R32; sampling circuit, not preparation-state compression.'),
 (4,'Non-HF Top-K Discovery vs Shots',[A+'/*/Fig04_top3_discovery.png'],'Top3 oracle-target diagnostic; all K metrics retained separately.'),
 (5,'VQE vs QSCI during ADAPT Growth',[S+'/*_Fig05_growth.png'],'All30 cases, all5 pools; original preparation VQE and new A M1000/R32 QSCI means. Full plotted tables independently checked separately.'),
 (6,'Early-Stopping Ratio Distribution',[A+'/Fig06_retrospective_ratios_tol_*.png'],'Ratios to final SAVED checkpoint, not unknown convergence; retrospective.'),
 (7,'Classical Construction Time vs Qubit Count',['results/molecular-MPS-scaling-standard-v1/derived-report-v1/01_construction_scaling.png'],'Original16 physical problems; stronger controls still in progress.'),
 (8,'Shots vs Classical Cost / Depth Pareto',['results/final-cost-frontiers-standard-v2/*_Fig08_known_model_budget_frontier.png'],'All30 planned cases, R64 known construction-plus-final model budgets, all registered seeds grouped before empirical90% feasibility. Unknown VQE/trained-topology budgets retained in tables, excluded from known-cost frontier; constructor time scoped separately, NOT full classical wall. This is not Fig18.'),
 (9,'CEO vs LUCJ / QEB / qubit / GSD Heatmap',[S+'/Fig09_unified_method_heatmap.png'],'All30 planned conditions, common M1000/R64; actual d can differ. A fivepool/B1 bounded CCSD-VQE LUCJ different independent RNG namespaces, descriptive not paired inference.'),
 (10,'Held-out Molecule Generalization',['results/PhaseC-replay-standard-v1/derived-report-v1/figures/*.png'],'LOMO diagnostic; new-geometry actualgrowth separate; no unseen-identity superiority.'),
 (11,'Important Determinant Probability vs Iteration',[A+'/*/Fig11_probability_tracks.png'],'Oracle target identity only; operator addition and reoptimization confounded.'),
 (12,'Operator Addition vs QSCI Energy Improvement',[S+'/*_Fig12_operator_gain.png'],'All4907 adjacent changes; actual addition IDs, new A mean-energy differences, no bridging missing steps. Addition and reoptimization remain confounded.'),
 (13,'HF Probability vs Non-HF Coverage',[S+'/*_Fig13_HF_coverage.png'],'All5231 primary checkpoint analytic M1000 values; sampling HF probability is NOT FCI HF weight.'),
 (14,'Recovery Before/After Energy',['dashboard/figures/*recovery*.png'],'Candidate UI figures; recovery alone vs full fermionic SQD scope must be checked.'),
 (15,'Random vs Trained vs Oracle Sampler',[S+'/presentation-v2/*_Fig15_random_trained_oracle.png'],'Saved FCI Born-distribution analytic oracle vs trained CEO and B4 all3 random seeds; NOT practical oracle preparation, NOT discovery upper bound. v2 line/range colors matched; no oracle finite-shot energy comparison.'),
 (16,'Single vs Multiple Shallow Samplers',[A+'/*/Fig16_multi_*.png','results/fixed-bank-allocation-standard-v1/report-v1/*.png'],'Historical checkpoint bank vs fixed-bank diagnostics are different scopes, not avoided-growth proof.'),
 (17,'CI Dimension Sensitivity',[A+'/*/Fig17_R_cap.png'],'Common capR, actual dimension retained separately.'),
 (18,'Total Classical + Quantum Resource Frontier',[],'Unified cost-scope aware frontier still required; do not sum incompatible units.'),
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output=None):
    output=Path(output or ROOT/'results/final-figure-inventory-v1');output.mkdir(parents=True,exist_ok=True)
    rows=[];files={}
    for number,title,patterns,scope in ITEMS:
        pngs=sorted({p for pattern in patterns for p in ROOT.glob(pattern)})
        complete=[];incomplete=[]
        for png in pngs:
            siblings=[png.with_suffix('.'+ext) for ext in ('png','svg','pdf')]
            if all(p.exists() for p in siblings):complete.append(str(png.relative_to(ROOT)))
            else:incomplete.append(str(png.relative_to(ROOT)))
            for p in siblings:
                if p.exists():files[str(p.relative_to(ROOT))]=sha(p)
        rows.append({'figure':number,'requested_title':title,'candidate_png_svg_pdf_sets':len(complete),
            'status':'CANDIDATES_EXIST_SEMANTIC_AUDIT_PENDING' if complete else 'REQUIRED_FORM_NOT_YET_GENERATED',
            'candidate_pngs':complete,'incomplete_format_sets':incomplete,'scope':scope,
            'completed_by_this_inventory':False})
    assert len(rows)==18 and sorted(r['figure'] for r in rows)==list(range(1,19))
    record={'schema':'requested18figure-existence-inventory-v1','rows':rows,'source_file_hashes':files,
        'script_sha256':sha(Path(__file__)),'no_scientific_values_generated':True,'whole_goal_complete':False,
        'file_count_is_not_experiment_or_sample_count':True}
    text=json.dumps(record,sort_keys=True,indent=2)+'\n';identity=hashlib.sha256(text.encode()).hexdigest()
    snapshot=output/'snapshots'/identity;snapshot.mkdir(parents=True,exist_ok=True)
    path=snapshot/'inventory.json'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    lines=['#18 requested figure types — actual file inventory','',
        'This is not final semantic certification or study completion. No placeholder plots were made.',
        'Missing requested forms remain explicit even when supporting tables or differently scoped figures exist.','',
        '| Fig | Requested relationship | Existing3-format candidates | Status |',
        '|---|---|---:|---|']
    for r in rows:
        lines.append(f"|{r['figure']}|{r['requested_title']}|{r['candidate_png_svg_pdf_sets']}|{r['status']}|")
        if r['candidate_pngs']:
            p=ROOT/r['candidate_pngs'][0]
            lines.append(f"\nRepresentative: [PNG]({p}) · [SVG]({p.with_suffix('.svg')}) · [PDF]({p.with_suffix('.pdf')})\n")
        lines.append('\nScope: '+r['scope']+'\n')
    (snapshot/'INDEX.md').write_text('\n'.join(lines)+'\n')
    (output/'LATEST.md').write_text(f'# Latest read-only inventory\n\n[18-figure checklist](snapshots/{identity}/INDEX.md)\n\nWhole study remains incomplete.\n')
    return {'snapshot':str(snapshot),'requested_types':18,'candidate_types':sum(bool(r['candidate_pngs']) for r in rows),
        'still_missing_required_forms':[r['figure'] for r in rows if not r['candidate_pngs']],'whole_goal_complete':False}


if __name__=='__main__':print(json.dumps(build()))
