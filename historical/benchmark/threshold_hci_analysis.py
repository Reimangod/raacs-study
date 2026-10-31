"""Independent selected-space and small exact-enumeration HCI audits/report."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT,file_digest,save_json
from .resource_adaptive_sampler import integral_problem
from .integral_connections import _enumerate
from .molecular_mps_scaling import no_full_state_guards


def exhaustive_union(backend,ids,coefficients,epsilon):
    selected=set(ids);external=set();examined=0
    for y,c in zip(ids,coefficients):
        if c==0:continue
        ii,vv,n=_enumerate(np.uint64(y),backend.n_qubits,backend.h1,backend.eri,backend.core,0.)
        examined+=int(n);external.update(int(x) for x,v in zip(ii,vv) if int(x) not in selected and abs(v*c)>epsilon)
    return external,examined


def audit(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text())
    status=json.loads((root/'execution-status.json').read_text())
    assert status['status']=='THRESHOLD_HCI_VARIATIONAL_COHORT_EXECUTED_PENDING_AUDIT'
    for p,sha in cfg['implementation'].items():assert file_digest(ROOT/'benchmark'/p)==sha
    assert file_digest(ROOT/'study-v1/INTEGRAL_THRESHOLD_HCI_V1.md')==cfg['spec_sha256']
    checks=selection_checks=examined=0;maximum=0.;rows=[]
    with threadpool_limits(limits=1),no_full_state_guards():
        for item in cfg['queue']:
            case=root/'cases'/item['id'];sourcepath=case/'source.json'
            if sourcepath.exists():
                src=json.loads(sourcepath.read_text());problem=Path(src['path'])
                metadata=problem/('problem.json' if (problem/'problem.json').exists() else 'reference.json')
                archive=problem/('integrals.npz' if (problem/'problem.json').exists() else 'reference.npz')
                assert file_digest(metadata)==src['metadata_sha256'] and file_digest(archive)==src['archive_sha256']
                backend,info=integral_problem(problem);assert info==src['problem']
            for epsilon in cfg['thresholds']:
                p=case/('epsilon-'+str(epsilon))/'result.json';r=json.loads(p.read_text())
                row={'case':item['id'],'epsilon1_Ha':epsilon,'status':r['status'],'source_result_sha256':file_digest(p)}
                if r['status']=='SOURCE_PHYSICAL_INPUT_FAILURE_PRESERVED':rows.append(row);continue
                assert r['quantum_shots']==r['hardware_shots']==0
                history=r.get('history',[])
                for i,h in enumerate(history):
                    ids=h['selected_ids'];assert len(ids)==len(set(ids))==h['dimension']<=512 and info['hf_id'] in ids
                    matrix=backend.matrix(ids,ids);energy=float(np.linalg.eigvalsh(matrix)[0]);coef=np.asarray(h['coefficients'])
                    error=abs(energy-h['variational_energy_Ha']);assert error<1e-8;maximum=max(maximum,error)
                    assert np.linalg.norm(matrix@coef-energy*coef)<1e-8 and abs(np.linalg.norm(coef)-1)<1e-8;checks+=1
                    if i:
                        assert set(history[i-1]['selected_ids'])<=set(ids)
                        assert energy<=history[i-1]['variational_energy_Ha']+1e-8
                    # Full independent connection enumeration for every small
                    # transition; predetermined HF transition only for larger
                    # cases. No all-large-iteration independence claim.
                    if i+1<len(history) and (backend.n_qubits<=16 or i==0):
                        ext,n=exhaustive_union(backend,ids,coef,epsilon);examined+=n
                        assert sorted(set(ids)|ext)==history[i+1]['selected_ids'];selection_checks+=1
                row.update(n_qubits=info['n_qubits'],HF_energy_Ha=info['HF_energy_Ha'],
                    final_dimension=history[-1]['dimension'] if history else None,
                    final_energy_Ha=history[-1]['variational_energy_Ha'] if history else None,
                    criterion_numeric_seconds=r.get('wall_seconds'),named_entry_requests=r.get('named_matrix_entry_requests'),
                    peak_process_RSS_hwm=r.get('peak_process_RSS_hwm'),PT2_energy_Ha=None,ground_error_Ha=None,
                    logical_CNOT=None,quantum_shots=0)
                rows.append(row)
    assert len(rows)==len(cfg['queue'])*3
    failures=[r for r in rows if r['status']=='ENGINEERING_FAILURE_PRESERVED']
    out={'status':('THRESHOLD_HCI_QUICK_INDEPENDENTLY_VALIDATED' if cfg['mode']=='quick' and not failures else
                   'THRESHOLD_HCI_STANDARD_INDEPENDENTLY_AUDITED' if not failures else 'THRESHOLD_HCI_ENGINEERING_FAILURES_PRESERVED'),
        'planned':len(rows),'all_selected_space_eigensolves_checked':checks,'independent_threshold_transition_checks':selection_checks,
        'exhaustive_audit_entries_examined':examined,'maximum_energy_parity_error_Ha':maximum,'rows':rows,'engineering_failures':failures,
        'all_large_parent_selections_independently_regenerated':False,'manifest_sha256':file_digest(root/'manifest.json'),
        'analysis_code_sha256':file_digest(__file__),'quantum_shots':0,'hardware_shots':0,'whole_goal_complete':False}
    save_json(root/'validation.json',out);return out


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);v=json.loads((root/'validation.json').read_text());folder=root/'report-v1';folder.mkdir(exist_ok=True)
    manifest=folder/'manifest.json';binding={'validation_sha256':file_digest(root/'validation.json'),'analysis_code_sha256':file_digest(__file__)}
    if manifest.exists():
        old=json.loads(manifest.read_text());assert old['binding']==binding
        for p,sha in old['files'].items():assert file_digest(folder/p)==sha
        return
    keys=list(dict.fromkeys(k for r in v['rows'] for k in r))
    with (folder/'all-planned-status-costs.csv').open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(v['rows'])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for epsilon in (1e-2,3e-3,1e-3):
        selected=[r for r in v['rows'] if r['epsilon1_Ha']==epsilon and r.get('final_energy_Ha') is not None]
        axes[0].scatter([r['n_qubits'] for r in selected],[r['final_dimension'] for r in selected],s=15,label=str(epsilon)+' Ha')
        axes[1].scatter([r['n_qubits'] for r in selected],[r['criterion_numeric_seconds'] for r in selected],s=15,label=str(epsilon)+' Ha')
    axes[0].set(xlabel='Qubits',ylabel='Last solved HCI variational dimension')
    axes[1].set(xlabel='Qubits',ylabel='Criterion execution seconds (setup separate)',yscale='symlog')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Threshold HCI variational stage | all caps/missing in table | no PT2/ground-reference claim')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(folder/('HCI-dimension-work-scaling.'+ext),dpi=180)
    plt.close(fig)
    counts=dict(Counter(r['status'] for r in v['rows']))
    (folder/'REPORT.md').write_text(f'''# Integral threshold HCI variational comparison

{v['planned']} planned cells. Statuses: {counts}.
All {v['all_selected_space_eigensolves_checked']} saved selected-space energies
independently rebuilt; maximum disagreement {v['maximum_energy_parity_error_Ha']:.3g}Ha.
Threshold transitions independently enumerated {v['independent_threshold_transition_checks']};
every small-system saved transition and large-system first transition, NOT all
parents of every large iteration. Exact generator unit parity is separate.

This implements the2016 HCI VARIATIONAL epsilon criterion with presorted doubles,
all selected parents and entire threshold-set addition. Dimension cap is tested
before expansion; no top-cardinality replacement. Last solved states after
caps are not convergence. Shared table builds and each actual rebuild are saved
under cases/*/table-builds, separately from per-criterion numeric work. Do not
sum standalone setup proxies as newly executed work in every threshold arm.

No full Fock/sector vector or FCI solve/label is used. Quantum/hardware shots0;
CNOT/depth N/A, not a fictitious free quantum circuit. This is not optimized
Dice/SHCI, no semistochastic/deterministicPT2, no certified ground reference,
no chemical accuracy claim. Per-cell complete selected IDs and coefficients
remain in original result.json; statuses/costs include failures and caps.
Different epsilon arms do not automatically have equal dimension, accuracy or
cost. Their retrospective envelope is not an online parameter-selection rule.
Comparison with quantum R64 must retain the actual dimensions and total costs.
The full six-phase study remains incomplete.

Primary method: https://arxiv.org/html/1606.07453 , II andIII.1. No author code
copied or vendor/package edited. The cardinality-ranked B4 controls remain a
different experiment; their outcomes were not silently replaced.
''')
    files={str(p.relative_to(folder)):file_digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='manifest.json'}
    save_json(manifest,{'binding':binding,'files':files,'whole_goal_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--component',choices=['audit','report'],default='audit');a=p.parse_args()
    if a.component=='audit':
        v=audit(a.root);print(json.dumps({k:v[k] for k in ('status','planned','all_selected_space_eigensolves_checked','maximum_energy_parity_error_Ha')}))
    else:report(a.root)
