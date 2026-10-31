"""Descriptive molecular scaling figures from audited actual records only."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from .common import file_digest,save_json


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);validation=root/'validation.json';a=json.loads(validation.read_text())
    assert 'AUDITED' in a['status'] or a['status']=='MOLECULAR_STRONG_QUICK_INDEPENDENTLY_VALIDATED'
    assert not a['unexpected_failures']
    folder=root/'report-v1';folder.mkdir(exist_ok=True)
    binding={'validation_sha256':file_digest(validation),'implementation_sha256':file_digest(__file__)}
    manifest=folder/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text());assert old['binding']==binding
        for p,sha in old['files'].items():assert file_digest(folder/p)==sha
        return old
    records=[];status=[];dependencies={}
    for r in a['rows']:
        status.append({k:v for k,v in r.items() if k not in ('readouts','sampling_resources','initialization')})
        for q in r.get('readouts',[]):
            records.append({'case':r['case'],'method':r['method'],'n_qubits':r['n_qubits'],'status':r['status'],
                'R':q['R'],'actual_d':q['actual_dimension'],'energy_Ha':q['energy_Ha'],
                'energy_gain_from_HF_mHa':1000*(r['HF_energy_Ha']-q['energy_Ha']),
                **(r['sampling_resources'] or {'cnot_count':None,'cnot_depth':None,'parameter_count':None,'total_depth':None}),
                'minimum_retained_MPS_norm_squared':r['minimum_retained_MPS_norm_squared'],
                'ground_state_error_Ha':None,'chemical_accuracy':None,
                'linked_model_shots':r['known_committed_model_shots'],'new_model_shots':r['new_model_shots'],
                'unknown_inflight_draws':r['unknown_inflight_draws'],
                'wall_seconds_this_invocation':r['wall_seconds_this_invocation'],
                'inherited_construction_seconds':r['inherited_construction_seconds']})
        init=r.get('initialization') or {}
        if init.get('dependency'):
            path=Path(init['dependency']);assert file_digest(path)==init['dependency_sha256']
            dependencies[str(path)]={'dependency':str(path),'sha256':file_digest(path),
                'standalone_classical_seconds_once':init['dependency_standalone_seconds'],
                'quantum_shots':0,'shared_not_each_cell_new_work':True}
    write_csv(folder/'all-planned-status.csv',status);write_csv(folder/'common-R-results-and-resources.csv',records)
    write_csv(folder/'matched-d-contrasts.csv',a['comparisons']);write_csv(folder/'unique-amplitude-dependencies.csv',list(dependencies.values()))
    cases=list(dict.fromkeys(r['case'] for r in a['rows']));methods=list(dict.fromkeys(r['method'] for r in a['rows']))
    palette=plt.get_cmap('tab10');colors={m:palette(i) for i,m in enumerate(methods)}
    for case in cases:
        obs=[r for r in records if r['case']==case and r['R']==64]
        if not obs:continue
        fig,axs=plt.subplots(1,3,figsize=(15,4.4))
        for r in obs:
            if r['method']=='classical-SCI64':
                for i,ax in enumerate(axs):ax.axhline(r['energy_gain_from_HF_mHa'],color=colors[r['method']],ls=':',lw=1,
                    label='bounded classical SCI64' if i==0 else None)
                continue
            for ax,key in zip(axs[:2],('cnot_count','cnot_depth')):
                ax.scatter(r[key],r['energy_gain_from_HF_mHa'],color=colors[r['method']],label=r['method'],s=32)
            t=r['wall_seconds_this_invocation'] if r['wall_seconds_this_invocation'] is not None else r['inherited_construction_seconds']
            if t is not None:axs[2].scatter(t,r['energy_gain_from_HF_mHa'],color=colors[r['method']],s=32)
        for ax,x in zip(axs,('Sampling CNOT count','Sampling CNOT depth','Recorded invocation seconds (not matched hardware)')):
            ax.set_xlabel(x);ax.set_ylabel('QSCI energy gained from HF (mHa)');ax.grid(alpha=.2)
        axs[2].set_xscale('symlog',linthresh=1)
        fig.suptitle(case+' | M=2000 including pilots; R cap64; one final draw; approximate MPS')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.tight_layout(rect=(0,.24,1,.93))
        for ext in ('png','svg','pdf'):fig.savefig(folder/(case+'.'+ext),dpi=180)
        plt.close(fig)
    counts=Counter(r['status'] for r in a['rows'])
    text=f'''# Native molecular strong-control comparison

{a['planned']} planned cells; status distribution: {dict(counts)}.
Independent projected solves/checks {a['independent_projected_solves']}; maximum
saved/rebuilt energy difference {a['maximum_energy_parity_error_Ha']:.3g} Ha.
Known linked model shots {a['known_linked_model_shots']}; new model shots
{a['known_new_model_shots']}; unknown in-flight draw records {a['unknown_inflight_draws']}.
Hardware shots0, no FCI or large full-Fock vectors.

## Interpretations and boundaries

Figures show energy improvement relative to the SAME physical HF reference,
not ground-state error, recovered ground-state correlation percentage or
chemical accuracy. No trustworthy larger-system ground reference was computed.
All methods share saved Hamiltonian/orbitals/active space. R8/32/64 and matched
realized-d readouts are reported; final-d matching is not full work matching.
All registered cases and missing/capped methods stay in the status table.

LUCJ circuits are native balanced/unbalanced ffsim UCJ with local pair constraints.
The four-point QSCI coordinate search pays256 pilot shots and1744 final shots.
Fixed circuits receive2000 final shots; RAACS pays its own pilot/final split
within2000. Final draw is independent from pilot selection. Three random
initialization seeds are separate protocols, not three independent repetitions
of every fixed method. There are no population confidence intervals or general
superiority conclusions from these single final draws.

Same MPS bond64 does NOT guarantee equal approximation error across shallow
RAACS and deep LUCJ. Retained norm is reported and identical-circuit bond/routing
diagnostics are separate, never policy inputs. A norm close to1 is not a global
error certificate. Unqualified/capped MPS comparisons cannot prove physical
quantum sampler superiority. Inspect same-circuit-bond-and-routing-v1 separately.
Native targets are synthetic, not calibrated devices; no large noise results.

Invocation timing includes classical simulation/readout and can reuse an
amplitude dependency. Unique dependency costs are exposed once, not summed for
every consumer. Inherited RAACS work is linked and charged, not newly acquired
or free. Different devices, scope and approximation prevent a controlled speedup
claim. Quantum VQE construction cost is unknown; no classical seconds-to-shots
conversion. Classical SCI64 is bounded screened cardinality SCI, not converged
SHCI/CIPSI or a trusted exact reference.

CH2 is a registered planar Ms=1 test, not spin-pure equilibrium geometry.
C4H4/C6H6 π selection is projector-verified; π-only and expanded spaces are
different physical problems. None of this completes all Phase E/F or the full
six-phase scientific study. Code/spec/tests and all failed records are preserved.
'''
    (folder/'REPORT.md').write_text(text)
    files={str(p.relative_to(folder)):file_digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='manifest.json'}
    out={'binding':binding,'files':files,'hardware_shots':0,'whole_goal_complete':False}
    save_json(manifest,out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(report(a.root)))
