"""Descriptive fixed-R and labelled post-hoc actual-rank diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,save_json,file_digest
from .multibasis_study import unpack,ARMS
from .resource_adaptive_sampler import integral_problem
from .nonorthogonal_ci import NonorthogonalHamiltonian,occupied_columns,generalized_ground


def report(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text());dest=root/'report-v1';dest.mkdir(exist_ok=True)
    if cfg['mode']=='quick':
        audit=json.loads((root/'validation.json').read_text())
        if audit['status']!='ALL_QUICK_SPACES_INDEPENDENTLY_VALIDATED' or file_digest(root/'events.sqlite')!=audit['database_sha256']:raise RuntimeError('Actual quick audit required')
    else:raise NotImplementedError('Standard report needs the bounded standard independent auditor, not a quick-only claim')
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True);rows=[];basisrows=[];costrows=[];sources={}
    for case,blob in db.execute('SELECT case_id,payload FROM bases').fetchall():
        b=unpack(blob);meta=json.loads((Path(cfg['source'])/'references'/case/'reference.json').read_text());sources[case]=(b,meta)
        basisrows.append({'case':case,'basis_construction_seconds':b['construction']['basis_construction_seconds'],
            'basis_objective_evaluations':b['construction']['basis_objective_evaluations'],'basis_pair_requests':b['basis_pair_requests'],
            'classical_anchor_energy_Ha':b['construction']['classical_anchor_ground']['energy_Ha'],
            'classical_anchor_error_mHa':1000*(b['construction']['classical_anchor_ground']['energy_Ha']-meta['fci_energy']),
            'classical_anchor_rank':b['construction']['classical_anchor_ground']['rank']})
    for case,method,b,blob in db.execute('SELECT case_id,method,basis,payload FROM circuits').fetchall():
        r=unpack(blob);costrows.append({'case':case,'method':method,'basis':b,**{k:r['resources'][k] for k in ('cnot_count','cnot_depth','total_depth')},
            'combined_parameter_count':None,'source_parameter_binding':'see '+r['origin']['source_result'],
            'historical_quantum_objective_shots':None})
    for case,method,arm,M,rep,status,taskblob,tid in db.execute('SELECT case_id,method,arm,M,rep,status,payload,id FROM tasks').fetchall():
        task=unpack(taskblob)
        for R,payload in db.execute('SELECT R,payload FROM readouts WHERE task=?',(tid,)).fetchall():
            out=unpack(payload);meta=sources[case][1]
            rows.append({'case':case,'method':method,'arm':arm,'M':M,'seed':rep,'R':R,'status':status,
                'energy_Ha':out['energy_Ha'],'error_mHa':1000*(out['energy_Ha']-meta['fci_energy']),
                'rank':out['rank'],'raw_vectors':out['candidate_vectors'],'condition_number':out['retained_condition_number'],
                'anchor_improvement_mHa':1000*(task['classical_anchor_energy_Ha']-out['energy_Ha']),
                'acquisition_id':task['acquisition_id'],'selected_nodes_json':json.dumps(out['selected_nodes']),
                'not_independent_of_other_R_or_anchor_controls':True})
    table=pd.DataFrame(rows);table.to_csv(dest/'common-R-readouts.csv',index=False)
    summary=table.groupby(['case','method','arm','M','R'],sort=True).agg(error_mHa=('error_mHa','mean'),mean_rank=('rank','mean'),
        mean_raw_vectors=('raw_vectors','mean'),seeds=('seed','size'),anchor_improvement_mHa=('anchor_improvement_mHa','mean')).reset_index()
    summary.to_csv(dest/'common-R-descriptive-means.csv',index=False)
    pd.DataFrame(basisrows).to_csv(dest/'classical-basis-cost-and-anchors.csv',index=False)
    pd.DataFrame(costrows).to_csv(dest/'physical-measurement-circuit-resources.csv',index=False)
    raw_shots=inherited_shots=classical=0
    for blob, in db.execute("SELECT payload FROM draws WHERE status='COMPLETED'"):
        x=unpack(blob);shots=x['metadata']['raw_simulated_quantum_shots'];raw_shots+=shots
        inherited_shots+=shots if 'inherited_quick_raw_draw_id' in x else 0
        classical+=x['metadata']['classical_uniform_samples']
    # Post-outcome, secondary rank control: preserve all3 identical classical
    # anchors in all3 arms. Choose first registered-frequency prefix at the
    # common minimum rank; no energy-dependent vector choice. Never discard
    # anchors merely to fabricate a match.
    tick=perf_counter();matched=[];diagnostic_pairs=0
    arms=['canonical_plus_anchors','multibasis','uniform_multibasis']
    for case,(b,meta) in sources.items():
        physical,info=integral_problem(Path(cfg['source'])/'references'/case)
        backend=NonorthogonalHamiltonian(physical.h1,physical.eri,physical.core)
        rotations=[np.asarray(u) for u in b['rotations']];pair_cache={};col={}
        def matrices(nodes):
            n=len(nodes);h=np.empty((n,n),complex);s=np.empty_like(h)
            for i,a in enumerate(nodes):
                for j,bb in enumerate(nodes[:i+1]):
                    key=(tuple(a),tuple(bb))
                    if key not in pair_cache:
                        for node in key:
                            if node not in col:col[node]=occupied_columns(rotations[node[0]],node[1])
                        pair_cache[key]=backend.pair(col[key[0]],col[key[1]])
                    hij,sij=pair_cache[key];h[i,j],s[i,j]=hij,sij;h[j,i],s[j,i]=hij.conjugate(),sij.conjugate()
            return h,s
        for (method,M,seed,R),group in table[table.case==case].groupby(['method','M','seed','R']):
            group=group[group.arm.isin(arms)]
            if len(group)!=3:continue
            target=int(group['rank'].min())
            for row in group.itertuples():
                nodes=json.loads(row.selected_nodes_json);hh,ss=matrices(nodes);found=None
                for prefix in range(3,len(nodes)+1):
                    out,_=generalized_ground(hh[:prefix,:prefix],ss[:prefix,:prefix])
                    if out['rank']==target:found=(out,prefix);break
                matched.append({'case':case,'method':method,'M':M,'seed':seed,'R':R,'arm':row.arm,'target_rank':target,
                    'status':'MATCHED' if found else 'NOT_MATCHABLE_WITH_ALL_ANCHORS',
                    'retained_rank':found[0]['rank'] if found else None,'prefix_vectors':found[1] if found else None,
                    'energy_Ha':found[0]['energy_Ha'] if found else None,
                    'error_mHa':1000*(found[0]['energy_Ha']-meta['fci_energy']) if found else None,
                    'selection_scope':'POST_HOC_SECONDARY_RANK_ONLY_PREFIX; same3anchors retained; not independent acquisitions'})
        diagnostic_pairs+=backend.pair_requests
    pd.DataFrame(matched).to_csv(dest/'actual-d-secondary-diagnostic.csv',index=False)
    diagnostic_seconds=perf_counter()-tick
    colors=['#3574b9','#8667b4','#d57928','#419061'];labels=['Canonical + HF','Canonical + same3 anchors','3 measurement bases + same anchors','Classical uniform / 3 bases']
    figure_files=[]
    for case in cfg['cases']:
        fig,axes=plt.subplots(4,len(cfg['M']),figsize=(11,12),squeeze=False)
        for i,method in enumerate(cfg['methods']):
            for j,M in enumerate(cfg['M']):
                ax=axes[i,j]
                for arm,color,label in zip(ARMS,colors,labels):
                    part=summary[(summary.case==case)&(summary.method==method)&(summary.M==M)&(summary.arm==arm)].sort_values('R')
                    ax.plot(part.R,part.error_mHa,'o-',color=color,label=label,linewidth=1.3,markersize=3)
                ax.axhline(0,color='black',linewidth=.5);ax.set_yscale('symlog',linthresh=.01)
                ax.set_title(method.split(':')[-1]+f' / M={M}',fontsize=8);ax.set_xlabel('Registered vector cap R');ax.set_ylabel('QSCI/NOCI error (mHa)')
                ax.grid(alpha=.18)
        handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=2,fontsize=8)
        fig.suptitle(case+' | quick qualification, 2 seeds; same counts across R / anchors',fontsize=11)
        fig.tight_layout(rect=(0,.06,1,.97))
        for ext in ('png','svg','pdf'):
            p=dest/f'{case}-multibasis.{ext}';fig.savefig(p,dpi=160);figure_files.append(p.name)
        plt.close(fig)
    text=f'''# Multiple orbital measurement bases: actual quick result

128/128 tasks completed,256 projected readouts,24 full preparation/basis circuits
independently checked. Maximum independent energy disagreement
{audit['max_energy_error_Ha']:.3g}Ha. This is engineering/small-system evidence,
not the complete30-condition comparison or proof of improved scalability.

## Accounting and information controls

- Unique model quantum shots {raw_shots}; inherited {inherited_shots}; newly
  acquired in this output {raw_shots-inherited_shots}. Classical uniform samples
  {classical}. Actual hardware shots0. Do not sum readout-row occurrences.
- Basis construction:96 classical objective evaluations/case in <=3 Slater
  space, shared by sources/arms. All3 classical anchors consume R. Their energy
  appears in the anchor CSV; improvements already present there are not quantum
  discovery advantages. Source VQE quantum objective budgets remain UNKNOWN.
- Uniform controls are classical samples, not simulated quantum acquisitions.
- Rank differs from raw vector count in a nonorthogonal basis. The separate
  actual-d table is a POST-HOC secondary diagnostic preserving all3 anchors,
  with first frequency-ranked prefix achieving common minimum rank. It never
  selects vectors by FCI/energy. Nonmatchable groups stay explicit.
- The resource-helper parameter/operator0 fields in quick v1 are unconfigured
  placeholders, NOT scientific counts. Additive parameter-cost interpretation
  masks them as unknown. CNOT/depth are independently rebuilt actual gates.
- Basis rotations are applied to original full preparation, not the old
  phase-stripped Z sampler; rotation/fusion cost is in the circuit CSV.
- This is bounded greedy exact-integral NOCI, NOT reproduction of the optimized
  CDF/residual-selection/global-refinement/recovery workflow of2608.11569v2.
- Repeated seeds are only2 quick draws, not a statistically established90%
  success threshold or molecular-population claim. No universal CEO/NOCI win.

## Outputs

common-R-readouts.csv / common-R-descriptive-means.csv;
actual-d-secondary-diagnostic.csv;
classical-basis-cost-and-anchors.csv;
physical-measurement-circuit-resources.csv;2figures each PNG/SVG/PDF.

Secondary rank diagnostic classical work: {diagnostic_pairs} Slater-pair requests,
{diagnostic_seconds:.3f}s. This is extra classical analysis, not free quantum work.
'''
    (dest/'REPORT.md').write_text(text)
    files=[p for p in dest.iterdir() if p.is_file() and p.name!='manifest.json']
    record={'status':'QUICK_AUDITED_DESCRIPTIVE_REPORT','database_sha256':file_digest(root/'events.sqlite'),
            'common_R_rows':len(table),'actual_d_rows':len(matched),'actual_d_matched':sum(x['status']=='MATCHED' for x in matched),
            'unique_quantum_model_shots':raw_shots,'inherited_quantum_model_shots':inherited_shots,'new_quantum_model_shots':raw_shots-inherited_shots,
            'classical_uniform_samples':classical,'rank_diagnostic_pair_requests':diagnostic_pairs,'rank_diagnostic_seconds':diagnostic_seconds,
            'report_code_sha256':file_digest(__file__),'figures':figure_files,'artifacts':{p.name:file_digest(p) for p in files},'whole_goal_complete':False}
    save_json(dest/'manifest.json',record);db.close();return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/multibasis-NOCI-quick-v1'))
    print(json.dumps(report(p.parse_args().output)))
