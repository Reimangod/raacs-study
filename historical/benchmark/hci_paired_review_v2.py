"""Independent paired-d review with explicit molecular/integral ID namespaces.

Preserves v1 (which rejected metadata-builder IDs vs integral IDs). No producer
or frozen numerical code changes, new sampling, matrix solves, or FCI calls.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

from .hci_paired_review import ROOT, sha, require, finite, identity, check_body, final_counts


def integral_identity(meta, e_source, h_source, sampler, hci_id, metadata_sha, archive_sha):
    """Different ID schemas are allowed; different physical inputs are not."""
    require(e_source['meta_sha256'] == h_source['metadata_sha256'] == metadata_sha,
            'Physical metadata bytes differ')
    require(e_source['integrals_sha256'] == h_source['archive_sha256'] == archive_sha,
            'Physical integral archive bytes differ')
    p = h_source['problem']
    for key in ('problem_id', 'physical_integral_binding', 'n_qubits', 'nalpha', 'nbeta', 'hf_id'):
        require(key in sampler and key in p and sampler[key] == p[key], 'Sampler/HCI physical mismatch: '+key)
    require(p['problem_id'] == hci_id, 'HCI result/source integral identity mismatch')
    binding = p['physical_integral_binding']
    expected = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(p['problem_id'] == expected, 'Malformed content-addressed integral identity')
    for key in ('n_qubits', 'nalpha', 'nbeta', 'hf_id'):
        require(type(meta[key]) is int and meta[key] == p[key] == binding[key], 'Metadata/sector mismatch: '+key)
    require(finite(meta['hf_energy_Ha']) and finite(p['HF_energy_Ha']) and finite(sampler['HF_energy_Ha']) and
            abs(meta['hf_energy_Ha']-p['HF_energy_Ha']) <= 1e-9 and
            abs(sampler['HF_energy_Ha']-p['HF_energy_Ha']) <= 1e-9, 'Physical HF energy mismatch')
    require(finite(binding['core']) and finite(meta['core_constant_Ha']) and
            abs(binding['core']-meta['core_constant_Ha']) <= 1e-8, 'Physical core constant mismatch')
    require(isinstance(meta['problem_id'], str) and len(meta['problem_id']) == 64,
            'Missing molecular-builder identity')
    return dict(metadata_builder_problem_id=meta['problem_id'], integral_problem_id=p['problem_id'],
                identifiers_use_different_schemas=meta['problem_id'] != p['problem_id'],
                metadata_sha256=metadata_sha, integral_archive_sha256=archive_sha,
                n_qubits=p['n_qubits'], nalpha=p['nalpha'], nbeta=p['nbeta'], hf_id=p['hf_id'])


def review(folder):
    folder = Path(folder).resolve(); dependencies = {}
    def bind(path):
        path = Path(path); dependencies[str(path)] = sha(path); return path
    def read(path):
        return json.loads(bind(path).read_text())
    config = read(folder/'manifest.json'); result = read(folder/'result.json')
    require(result['manifest_sha256'] == sha(folder/'manifest.json'), 'Manifest drift')
    require(sha(ROOT/'study-v1/HCI_PAIRED_DIMENSION_DIAGNOSTIC_V1.md') == config['spec_sha256'], 'Specification drift')
    endpoint = ROOT/'results/HCI-native-sampler-comparison-v1/b05725f001503ef867a1de37f332b7274762aa2ab97ceeeefbedf3f6208811eb/comparison.json'
    require(sha(endpoint) == config['source_endpoint_sha256'], 'Endpoint source drift')
    original = read(endpoint)
    require(config['source_input_hashes'] == original['inputs'], 'Source closure mismatch')
    require(all(sha(ROOT/'benchmark'/p) == value for p,value in config['implementation'].items()), 'Bound producer drift')
    require(all(sha(p) == value for p,value in config['source_input_hashes'].items()), 'Source drift')
    require(all(sha(p) == value for p,value in result['readout_database_hashes'].items()), 'Readout drift')
    e = ROOT/'results/molecular-strong-controls-standard-v2'
    hci = ROOT/'results/integral-threshold-HCI-standard-v1'
    all_cases = [item['id'] for item in read(e/'manifest.json')['queue']]
    require(config['mode'] in ('quick','standard'), 'Unregistered mode')
    cases = [all_cases[0],all_cases[-1]] if config['mode']=='quick' else all_cases
    require(config['cases'] == cases, 'Changed planned cases')
    require(config['rows'] == [r for r in original['rows'] if r['case'] in cases and r['method_kind']=='SAMPLER'], 'Changed planned rows')
    qrows = {(r['case'],r['method']): r for r in read(e/'validation.json')['rows']}
    sources = {}; counts_cache = {}; identities = {}; physics_cache = {}; native_cache = {}
    for old in config['rows']:
        if old['comparison_status'] != 'AVAILABLE_DESCRIPTIVE_PAIR': continue
        case,method,rcap,eps = identity(old)
        method_folder = e/'cases'/case/method
        raw = read(method_folder/'result.json')
        h = read(hci/'cases'/('E-'+case)/('epsilon-'+str(eps))/'result.json')
        if case not in physics_cache:
            es = read(e/'cases'/case/'physical-source.json')
            hs = read(hci/'cases'/('E-'+case)/'source.json')
            meta_path = bind(Path(es['path'])/'problem.json'); meta = json.loads(meta_path.read_text())
            archive = bind(Path(es['path'])/'integrals.npz')
            # If two paths are used, both actual files must independently match.
            hmeta = bind(Path(hs['path'])/'problem.json'); harchive = bind(Path(hs['path'])/'integrals.npz')
            require(sha(hmeta) == sha(meta_path) and sha(harchive) == sha(archive), 'Different source bytes')
            physics_cache[case] = (es,hs,meta,dependencies[str(meta_path)],dependencies[str(archive)])
        es,hs,meta,meta_sha,archive_sha = physics_cache[case]
        problem = raw.get('problem')
        if problem is None:
            require(method == 'RAACS-full-cost' and 'native_result' in raw, 'No sampler physical input evidence')
            path = Path(raw['native_result'])
            if not path.is_absolute(): path = method_folder/path
            if str(path) not in native_cache: native_cache[str(path)] = read(path)
            problem = native_cache[str(path)]['problem']
        record = integral_identity(meta,es,hs,problem,h['problem_id'],meta_sha,archive_sha)
        if case in identities: require(record == identities[case], 'Per-case physical identity drift')
        else: identities[case] = record
        if (case,method) not in counts_cache:
            counts_cache[(case,method)] = final_counts(method_folder,method,raw,dependencies)
        sources[identity(old)] = dict(problem_id=record['integral_problem_id'],hf_id=meta['hf_id'],
            sampler=next(r for r in qrows[(case,method)]['readouts'] if r['R']==rcap),
            hci=h['history'][-1],counts=counts_cache[(case,method)])
    rows, statuses = check_body(config,result,sources)
    for p in (Path(__file__), ROOT/'benchmark/hci_paired_review.py', ROOT/'benchmark/resource_adaptive_sampler.py',
              ROOT/'benchmark/molecular_problem_no_fci.py'):
        bind(p)
    require(all(sha(p)==value for p,value in dependencies.items()), 'Source changed during review')
    require(set(identities) == {r['case'] for r in config['rows'] if r['comparison_status']=='AVAILABLE_DESCRIPTIVE_PAIR'},
            'Incomplete physical identity coverage')
    summary = dict(status='ALL_PLANNED_PAIRS_AND_NAMESPACED_PHYSICAL_IDENTITIES_REVIEWED',
        planned_pairs=len(rows),statuses=statuses,unique_spaces=len(result['unique_spaces']),
        trivial_HF_pairs=sum(r['trivial_HF_space'] is True for r in rows),
        nontrivial_available_pairs=sum(r['matched_d'] is not None and r['matched_d']>1 for r in rows),
        physical_cases_checked=len(identities),physical_identity_records=identities,
        source_manifest_sha256=sha(folder/'manifest.json'),source_result_sha256=sha(folder/'result.json'),
        review_code_sha256=sha(__file__),v1_review_code_sha256=sha(ROOT/'benchmark/hci_paired_review.py'),
        dependencies=dependencies,new_CI_solves=0,new_quantum_acquisitions=0,new_FCI_solves=0,
        independent_matrix_or_energy_recomputation=False,old_numerical_outputs_modified=False,
        whole_goal_complete=False)
    stream=StringIO(); writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    readme = ('# HCI / sampler same-d diagnostic — independent review v2\n\n'
        f'{len(rows)} all-planned display pairs; {summary["nontrivial_available_pairs"]} nontrivial available; '
        f'{summary["trivial_HF_pairs"]} trivial HF-only. All source missingness remains.\n\n'
        'V1 rejected molecular-builder IDs compared with integral/HF/sector IDs. '
        'V2 keeps both namespaces and checks source metadata/archive SHA, actual sampler physical input '
        '(including native RAACS input), HCI source/result integral identity, core, sector and HF energy. '
        'This is not accepting a mismatched Hamiltonian or editing either source ID.\n\n'
        'Selections are independently rebuilt from actual final counts and native HCI coefficients. '
        'Saved numerical rechecks/arithmetic/cache accounting are validated; no new eigenvalues are computed here. '
        'Positive sampler_minus_truncated_HCI_mHa favors coefficient-truncated HCI, not its original native endpoint. '
        'Same d is not matched work or a ground-state accuracy certificate. R/epsilon rows reuse sources and '
        'are not independent molecular experiments. No new quantum/FCI and no producer/result changes.\n')
    outputs={'validation.json':json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n',
             'all-planned-pairs.csv':stream.getvalue(),'README.md':readme}
    target=folder.with_name(folder.name+'-review-v2');target.mkdir(exist_ok=True)
    for name,body in outputs.items():
        path=target/name
        require(not path.exists() or path.read_text()==body,'Existing review differs; use successor')
    for name,body in outputs.items():
        path=target/name
        if not path.exists(): path.write_text(body)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('dependencies','physical_identity_records')},sort_keys=True))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    review(parser.parse_args().root)
