"""Successor-specific audit of simulator bitmapping, approximation and costs.

Adds evidence; never mutates execution ledgers or v1 results. Generic named-CI
audit remains a different validation scope from physical-layout qualification.
"""
import argparse
import json
from pathlib import Path
from .common import ROOT,file_digest,save_json
from .molecular_strong_analysis import audit as named_audit,final_evidence
from .mps_layout_qualification import logical_id,spin_blocked_mapping


def cached(root):
    """Verify bound raw evidence without rewriting timing-bearing audits."""
    root=Path(root);out=json.loads((root/'physical-layout-validation-v2.json').read_text())
    assert file_digest(__file__)==out['audit_code_sha256']
    assert file_digest(root/'manifest.json')==out['manifest_sha256']
    assert file_digest(root/'validation.json')==out['generic_validation_sha256']
    a=json.loads((root/'validation.json').read_text());cfg=json.loads((root/'manifest.json').read_text())
    assert file_digest(ROOT/'benchmark/molecular_strong_analysis.py')==a['analysis_code_sha256']
    for p,sha in cfg['implementation'].items():assert file_digest(ROOT/'benchmark'/p)==sha
    for row in a['rows']:
        if 'result_sha256' not in row:continue
        folder=root/'cases'/row['case']/row['method'];p=folder/'result.json'
        assert file_digest(p)==row['result_sha256'];r=json.loads(p.read_text())
        if r.get('event_database_sha256'):assert file_digest(folder/'events.sqlite')==r['event_database_sha256']
        if r.get('native_events_sha256'):
            native=Path(r['native_result']);native=native if native.is_absolute() else folder/native
            assert file_digest(native.parent/'events.sqlite')==r['native_events_sha256']
        source=json.loads((folder.parent/'physical-source.json').read_text());physical=Path(source['path'])
        assert file_digest(physical/'problem.json')==source['meta_sha256']
        assert file_digest(physical/'integrals.npz')==source['integrals_sha256']
    return out


def audit(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text())
    assert cfg['schema']=='molecular-strong-controls-v2-spin-blocked'
    assert file_digest(ROOT/'study-v1/MOLECULAR_STRONG_CONTROLS_V2.md')==cfg['successor_spec_sha256']
    qroot=ROOT/'results/molecular-MPS-layout-qualification-v1'
    assert file_digest(qroot/'manifest.json')==cfg['qualification_manifest_sha256']
    assert file_digest(qroot/'execution-status.json')==cfg['qualification_result_sha256']
    a=named_audit(root);checked=0;qualified=[];censored=[];allnorms=[]
    required=[]
    for item in cfg['queue']:
        if item['id'].startswith(('H10_','C6H6_pi_sigma12_')):
            required.extend((item['id'],m) for m in ('LUCJ-MP2-2-fixed','LUCJ-CCSD-2-fixed'))
        for method in cfg['methods']:
            path=root/'cases'/item['id']/method/'result.json'
            if not path.exists():continue
            r,counts,rows,qc,physical=final_evidence(root,item,method)
            inherited=item['kind']=='inherited' and method=='RAACS-full-cost'
            for row in rows:
                meta=row['metadata']
                if meta is None or inherited:continue
                assert meta['layout_engine']=='spin-blocked-simulator-only-v1'
                mapping=meta['layout_evidence']['logical_to_simulator_qubit']
                assert mapping==spin_blocked_mapping(r['problem']['n_qubits'])
                rawids=meta['layout_evidence']['backend_ids'];rawcounts=meta['layout_evidence']['backend_counts']
                assert len(set(rawids))==len(rawids)==len(rawcounts)
                rebuilt={logical_id(x,mapping):n for x,n in zip(rawids,rawcounts)}
                assert rebuilt==row['counts']
                assert meta['layout_evidence']['physical_gate_count_unchanged']
                norm=meta['original_build_metadata']['retained_norm_squared']
                assert cfg['minimum_retained_norm_squared']<=norm<=1+1e-7
                assert meta['minimum_required_retained_norm_squared']==cfg['minimum_retained_norm_squared']
                checked+=1;allnorms.append(norm)
            if r['status']=='COMPLETED':qualified.append((item['id'],method))
            if 'BEFORE_SAMPLING' in r.get('error',''):
                # Engine explicitly rejected before sample_mps; this reconciles
                # that operation alone, not other unrelated in-flight work.
                e=r['error'];raw=json.JSONDecoder().raw_decode(e[e.index('{'):])[0]
                assert raw['confirmed_new_model_shots']==0
                assert raw['build']['retained_norm_squared']<cfg['minimum_retained_norm_squared']
                assert len([x for x in rows if x['status']=='STARTED'])==1
                censored.append({'case':item['id'],'method':method,'classification':'APPROXIMATION_CENSORED_NOT_PHYSICAL_METHOD_LOSS',
                    'rejected_operation_new_model_shots':0,'other_committed_model_shots':sum(x['raw'] for x in rows if x['raw'] is not None),
                    'confirmed_pre_sampling_build_evidence':raw,'original_result_sha256':file_digest(path)})
    missing=[x for x in required if x not in qualified]
    good=not a['unexpected_failures'] and not missing
    out={'status':('SPIN_BLOCKED_NATIVE_QUICK_QUALIFIED' if cfg['mode']=='quick' and good else
                  'SPIN_BLOCKED_NATIVE_STANDARD_AUDITED' if cfg['mode']=='standard' and good else 'PHYSICAL_LAYOUT_QUALIFICATION_INCOMPLETE'),
        'generic_validation_sha256':file_digest(root/'validation.json'),'manifest_sha256':file_digest(root/'manifest.json'),
        'audit_code_sha256':file_digest(__file__),'mapped_draws_reconstructed':checked,
        'minimum_sampled_retained_norm_squared':min(allnorms) if allnorms else None,
        'required_fixed_controls_missing_or_censored':missing,'approximation_censored':censored,
        'not_global_error_certificate':True,'ground_state_reference':None,'hardware_shots':0,'whole_goal_complete':False}
    save_json(root/'physical-layout-validation-v2.json',out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    out=cached(a.root) if a.cached else audit(a.root)
    print(json.dumps({k:v for k,v in out.items() if k!='approximation_censored'}))
