"""Report only independently audited all30-case true multi-basis experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,file_digest,save_json
from .multibasis_study import unpack,ARMS


def report(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text());audit=json.loads((root/'validation.json').read_text())
    if audit['status']!='ALL_STANDARD_MULTIBASIS_SPACES_INDEPENDENTLY_VALIDATED' or file_digest(root/'events.sqlite')!=audit['database_sha256']:raise RuntimeError('Full independent standard audit required')
    dest=root/'report-v1';dest.mkdir(exist_ok=True);db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True)
    rows=[];statuses=[];basisrows=[];resource_rows=[]
    for case in cfg['cases']:
        meta=json.loads((Path(cfg['source'])/'references'/case/'reference.json').read_text())
        basis=unpack(db.execute('SELECT payload FROM bases WHERE case_id=?',(case,)).fetchone()[0]);b=basis['construction']
        basisrows.append({'case_id':case,'objective_evaluations':b['basis_objective_evaluations'],
            'basis_construction_seconds':b['basis_construction_seconds'],'basis_pair_requests':basis['basis_pair_requests'],
            'classical_anchor_error_mHa':1000*(b['classical_anchor_ground']['energy_Ha']-meta['fci_energy']),
            'anchor_rank':b['classical_anchor_ground']['rank'],'scope':'shared inherited or new basis work, not repeated per source/arm'})
        for tid,method,arm,M,rep,status,payload in db.execute('SELECT id,method,arm,M,rep,status,payload FROM tasks WHERE case_id=?',(case,)).fetchall():
            task=unpack(payload);statuses.append({'case_id':case,'method':method,'arm':arm,'M':M,'seed':rep,'status':status})
            for R,blob in db.execute('SELECT R,payload FROM readouts WHERE task=?',(tid,)).fetchall():
                r=unpack(blob);rows.append({'case_id':case,'molecule':meta['case']['molecule'],'distance_A':meta['case']['distance'],
                    'method':method,'arm':arm,'M':M,'seed':rep,'R':R,'energy_Ha':r['energy_Ha'],
                    'error_mHa':1000*(r['energy_Ha']-meta['fci_energy']),'actual_rank':r['rank'],'raw_vectors':r['candidate_vectors'],
                    'retained_condition_number':r['retained_condition_number'],'acquisition_id':task['acquisition_id'],
                    'anchor_gain_mHa':1000*(task['classical_anchor_energy_Ha']-r['energy_Ha'])})
        for method,b,blob in db.execute('SELECT method,basis,payload FROM circuits WHERE case_id=?',(case,)).fetchall():
            r=unpack(blob);resource_rows.append({'case_id':case,'method':method,'basis':b,
                **{k:r['resources'][k] for k in ('cnot_count','cnot_depth','total_depth','parameter_count')},
                'original_source_result':r['origin']['source_result'],'historical_quantum_objective_shots':None,
                'orbital_variables_per_nonidentity_basis':basis['construction']['orbital_parameters_per_nonidentity_basis']})
    table=pd.DataFrame(rows);table.to_csv(dest/'all-common-R-readouts.csv',index=False)
    pd.DataFrame(statuses).to_csv(dest/'all-planned-task-status.csv',index=False)
    pd.DataFrame(basisrows).to_csv(dest/'shared-basis-construction-and-anchor-cost.csv',index=False)
    pd.DataFrame(resource_rows).to_csv(dest/'all-physical-measurement-resources.csv',index=False)
    means=table.groupby(['case_id','molecule','distance_A','method','arm','M','R']).agg(
        error_mHa=('error_mHa','mean'),draw_SD_mHa=('error_mHa','std'),actual_rank=('actual_rank','mean'),
        raw_vectors=('raw_vectors','mean'),anchor_gain_mHa=('anchor_gain_mHa','mean'),seeds=('seed','size')).reset_index()
    means.to_csv(dest/'case-method-descriptive-means.csv',index=False)
    rank=pd.read_csv(root/'actual-d-secondary-diagnostic.csv')
    rank.to_csv(dest/'actual-d-secondary-diagnostic.csv',index=False)
    if len(rank):
        rank.groupby(['case_id','method','M','R','arm','status'],dropna=False).agg(error_mHa=('error_mHa','mean'),
            target_rank=('target_rank','mean'),pairs=('rep','size')).reset_index().to_csv(dest/'actual-d-descriptive-means.csv',index=False)
    colors=['#3574b9','#8667b4','#d57928','#419061'];figures=[]
    for molecule in sorted(table.molecule.unique()):
        fig,axes=plt.subplots(2,2,figsize=(12,8),squeeze=False)
        for ax,method in zip(axes.flat,cfg['methods']):
            for arm,color in zip(ARMS,colors):
                p=means[(means.molecule==molecule)&(means.method==method)&(means.arm==arm)&(means.M==1000)&(means.R==64)].sort_values('distance_A')
                ax.errorbar(p.distance_A,p.error_mHa,yerr=p.draw_SD_mHa,color=color,marker='o',markersize=3,linewidth=1.2,label=arm)
            ax.axhline(0,color='black',lw=.5);ax.set_xlabel('Distance / equal spacing (angstrom)');ax.set_ylabel('Energy error (mHa)')
            ax.set_title(method.split(':')[-1],fontsize=8);ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
        fig.suptitle(molecule+' | M=1000, vector cap64; bars=3-draw SD, NOT CI',fontsize=11);fig.tight_layout(rect=(0,.09,1,.96))
        for ext in ('png','svg','pdf'):
            path=dest/f'{molecule}-multibasis.{ext}';fig.savefig(path,dpi=160);figures.append(path.name)
        plt.close(fig)
    primary=means[(means.M==1000)&(means.R==64)].pivot(index=['case_id','method'],columns='arm',values='error_mHa')
    contrasts=[]
    for arm in ('multibasis','uniform_multibasis'):
        d=primary[arm]-primary['canonical_plus_anchors']
        # Conditional descriptive case/source average, not independent samples
        # or a population-level significance test. Keep missingness in status.
        contrasts.append({'arm':arm,'control':'canonical_plus_anchors','available_case_source_pairs':int(d.notna().sum()),
            'mean_error_difference_mHa':float(d.mean()),'scope':'descriptive available-pair means; no independent molecular-population CI'})
    save_json(dest/'descriptive-contrasts.json',contrasts)
    text=f'''# All30-case multiple orbital-basis measurement study

{audit['planned_tasks']} planned tasks; {audit['tasks_by_status']}. All
{audit['all_projected_readouts_checked']} stored projected readouts checked by
independent ffsim sector vectors + saved sparse Hamiltonian + physical-space SVD;
{audit['physical_measurement_circuit_checks']} physical measurement circuits
independently verified. Maximum energy disagreement {audit['max_energy_error_Ha']:.3g}Ha.

## Method and limits

This is independently specified bounded greedy real-restricted exact-integral
NOCI basis construction, not a reproduction of arXiv2608.11569v2's optimized
CDF/selection/global-refinement/full-recovery protocol. All30 conditions and
four saved sources remain; missing original CCSD-LUCJ is NOT CEO/NOCI success.
Primary endpoints use common raw quantum shots M and vector caps R8/32/64.
True retained rank differs from vector count. The actual-d table is a labelled
POST-HOC secondary diagnostic, preserving the same3 anchors and choosing first
frequency-ranked prefix reaching common minimum rank without energy selection.
No anchor is removed to force a match. Different extra classical information
in canonical-only control is explicit; main measurement comparison is against
canonical_plus_anchors. Classical-anchor-only energies/costs are reported.

No FCI selection, new FCI solve, whole-vector adaptive-policy input, hardware
execution or population-level superiority claim. The count-only simulator here
is qualified small <=16q, not a scalable quantum emulator. Source construction
costs are inherited. Quantum VQE objective budgets remain UNKNOWN, not0.
Combined parameter binding is un-reconstructed/null, not zero; original source
parameters and orbital optimized-variable counts are distinct. CNOT/depth
include actual compiled basis rotations, without assuming free orbital fusion.

## Unique costs, not readout-row sums

Linked quantum model shots {audit['linked_unique_quantum_model_shots']};
inherited quick shots {audit['inherited_quick_quantum_model_shots']};
new standard model shots {audit['new_standard_quantum_model_shots']}.
Classical uniform samples {audit['classical_uniform_samples']}; hardware shots0.
Uniform controls are classical baseline data, not quantum shots. R variants and
anchor controls share underlying acquisitions; uniform data are shared across
saved-source controls. Do not count these as independent samples.

Basis objective counts are the ACTUAL per-case counts in the cost CSV, not the
maximum96 for every case. Reporting construction seconds from CPU/saved sources
does not establish controlled hardware speedup. All resource/source limits and
missing tasks remain visible. Figures use3-draw SD, not confidence intervals;
shots-to90% success is not identifiable from this isolated grid/three repeats.

Descriptive M1000/R64 available-pair contrasts: {json.dumps(contrasts)}.
These are not a proof of general superiority or of failure of the published
optimized NOCI method. All6 phases and final synthesis remain incomplete.
'''
    (dest/'REPORT.md').write_text(text);db.close()
    record={'status':'ALL30_MULTIBASIS_AUDITED_REPORTED','figures':figures,'readouts':len(table),'tasks':len(statuses),
        'audit_sha256':file_digest(root/'validation.json'),'report_code_sha256':file_digest(__file__),
        'artifacts':{p.name:file_digest(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'},'whole_goal_complete':False}
    save_json(dest/'manifest.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/multibasis-NOCI-standard-v2'))
    print(json.dumps(report(p.parse_args().output)))
