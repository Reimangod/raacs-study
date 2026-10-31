"""Read-only paired-d selection/arithmetic review and all-planned CSV export.

No producer imports, new projected solves, quantum draws or optimizer calls.
Run only after the isolated total-workflow timing cohort has finished.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import sqlite3
import zlib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def identity(row):
    return row['case'], row['method'], row['R_cap'], row['epsilon1_Ha']


def expected_subset(ids, hf, dimension, scores):
    require(type(dimension) is int and 1 <= dimension <= len(ids), 'Invalid dimension')
    require(len(ids) == len(set(ids)) and hf in ids, 'Invalid source IDs/HF')
    require(all(finite(scores[x]) for x in ids if x != hf), 'Invalid source scores')
    ordered = [hf] + sorted((x for x in ids if x != hf), key=lambda x: (-scores[x], x))
    return sorted(ordered[:dimension])


def check_body(config, result, sources):
    """Pure validation against independently loaded source data, not producer helpers."""
    require(result['status'] == 'ALL_PAIRED_DIAGNOSTIC_SPACES_RECHECKED', 'Incomplete result')
    require(all(type(result[k]) is int and result[k] == 0 for k in ('new_quantum_acquisitions','new_FCI_solves')), 'Unexpected new acquisition')
    require(result['whole_goal_complete'] is False, 'Not whole-study completion')
    expected = {identity(r): r for r in config['rows']}
    require(len(expected) == len(config['rows']) == config['planned_pairs'], 'Duplicate planned rows')
    require(len(result['rows']) == len(expected), 'Incomplete all-planned result')
    seen = set(); used_spaces = set(); output = []; statuses = Counter()
    for row in result['rows']:
        key = identity(row)
        require(key in expected and key not in seen, 'Unexpected/duplicate result row')
        seen.add(key); old = expected[key]
        require(row['sampler_status'] == old['sampler_status'] and row['HCI_status'] == old['HCI_status'], 'Changed source status')
        available = old['comparison_status'] == 'AVAILABLE_DESCRIPTIVE_PAIR'
        out = {k: row[k] for k in ('case', 'method', 'R_cap', 'epsilon1_Ha', 'sampler_status', 'HCI_status', 'status')}
        out.update(matched_d=None, trivial_HF_space=None, sampler_restricted_energy_Ha=None,
                   HCI_coefficient_truncated_energy_Ha=None, sampler_minus_truncated_HCI_mHa=None,
                   original_sampler_energy_Ha=old['sampler_energy_Ha'],
                   original_HCI_energy_Ha=old['HCI_last_solved_energy_Ha'],
                   work_matched=False, ground_state_error_Ha=None)
        if not available:
            require(row['status'] == 'MISSING_PAIR_PRESERVED', 'Missingness was changed')
            require(not any(k in row for k in ('matched_d', 'sampler_space_key', 'HCI_coefficient_truncated_space_key')), 'Invented missing result')
        else:
            require(row['status'] == 'PAIRED_DIMENSION_DIAGNOSTIC', 'Wrong available status')
            source = sources[key]; q, h = source['sampler'], source['hci']; hf = source['hf_id']
            d = min(q['actual_dimension'], h['dimension'])
            require(type(row['matched_d']) is int and row['matched_d'] == d <= 64, 'Wrong matched dimension')
            require(row['trivial_HF_space'] is (d == 1) and row['work_matched'] is False, 'Invalid comparison claim')
            counts = source['counts']
            require(all(type(counts.get(x)) is int and counts[x] > 0 for x in q['selected_ids'] if x != hf), 'Unobserved sampler ID')
            coefficients = dict(zip(h['selected_ids'], h['coefficients']))
            require(len(h['coefficients']) == len(h['selected_ids']), 'Coefficient shape')
            require(all(finite(c) for c in h['coefficients']), 'Invalid coefficients')
            wanted = (expected_subset(q['selected_ids'], hf, d, counts),
                      expected_subset(h['selected_ids'], hf, d, {x: abs(c) for x, c in coefficients.items()}))
            energies = []
            for field, ids in zip(('sampler_space_key', 'HCI_coefficient_truncated_space_key'), wanted):
                skey = row[field]; space = result['unique_spaces'][skey]; used_spaces.add(skey)
                require(space['problem_id'] == source['problem_id'] and space['selected_ids'] == ids, 'Wrong selected space/problem')
                expected_key = hashlib.sha256(json.dumps([source['problem_id'], ids]).encode()).hexdigest()
                require(skey == expected_key, 'Wrong content-addressed space key')
                require(finite(space['energy_Ha']) and finite(space['independent_recheck_energy_Ha']), 'Invalid space energy')
                require(abs(space['energy_Ha'] - space['independent_recheck_energy_Ha']) <= 1e-8, 'Saved energy recheck failed')
                require(finite(space['internal_residual']) and 0 <= space['internal_residual'] <= 1e-8, 'Bad saved residual')
                require(space['new_eigensolver_calls'] == 2 and finite(space['analysis_seconds']) and space['analysis_seconds'] >= 0, 'Unaccounted solve work')
                energies.append(space['energy_Ha'])
            qe, he = energies
            require(row['sampler_restricted_energy_Ha'] == qe and row['HCI_coefficient_truncated_energy_Ha'] == he, 'Row/space energy mismatch')
            require(finite(row['sampler_minus_truncated_HCI_mHa']) and abs(row['sampler_minus_truncated_HCI_mHa'] - 1000*(qe-he)) <= 1e-10, 'Wrong energy difference/sign')
            require(row['original_sampler_energy_Ha'] == old['sampler_energy_Ha'] and row['original_HCI_energy_Ha'] == old['HCI_last_solved_energy_Ha'], 'Changed native endpoint')
            # A restricted variational space cannot beat its untruncated source.
            require(qe >= old['sampler_energy_Ha'] - 1e-8 and he >= old['HCI_last_solved_energy_Ha'] - 1e-8, 'Restriction variational inequality failed')
            out.update({k: row[k] for k in ('matched_d', 'trivial_HF_space', 'sampler_restricted_energy_Ha',
                'HCI_coefficient_truncated_energy_Ha', 'sampler_minus_truncated_HCI_mHa')})
        statuses[row['status']] += 1; output.append(out)
    require(seen == set(expected), 'Missing planned identity')
    require(used_spaces == set(result['unique_spaces']), 'Unreferenced/missing cached spaces')
    require(result['new_eigensolver_calls'] == 2*len(used_spaces), 'Cached work double counting/omission')
    observed_maximum = max((abs(s['energy_Ha']-s['independent_recheck_energy_Ha']) for s in result['unique_spaces'].values()),default=0)
    require(finite(result['maximum_eigensolver_disagreement_Ha']) and abs(result['maximum_eigensolver_disagreement_Ha']-observed_maximum)<=1e-15, 'Wrong recheck summary')
    require(finite(result['analysis_wall_seconds']) and result['analysis_wall_seconds'] >= 0, 'Invalid analysis time')
    return output, dict(statuses)


def final_counts(folder, method, result, dependencies):
    # Numerical libraries are loaded ONLY when actual completed evidence is reviewed.
    import numpy as np
    if method == 'RAACS-full-cost':
        native_path = Path(result['native_result'])
        if not native_path.is_absolute(): native_path = folder/native_path
        native = json.loads(native_path.read_text()); dbpath = native_path.parent/'events.sqlite'
        require(sha(dbpath) == result['native_events_sha256'], 'Inherited database drift')
        dependencies[str(native_path)] = sha(native_path)
        wanted = set(native['final_draw_ids'])
        expected_total = native['independent_final_raw_model_shots']
        query = 'SELECT id,role,counts64 FROM draws'
        role = 'independent_final'
    else:
        dbpath = folder/'events.sqlite'; wanted = {result['final_draw_id']}
        expected_total = result['independent_final_model_shots']
        require(sha(dbpath) == result['event_database_sha256'], 'Native database drift')
        query = 'SELECT id,role,counts FROM draws'; role = 'independent-final'
    dependencies[str(dbpath)] = sha(dbpath); counts = {}; found = set()
    with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro', uri=True) as db:
        for did, kind, blob in db.execute(query):
            if did not in wanted: continue
            require(did not in found and kind == role and blob is not None, 'Non-final/duplicate/missing draw')
            found.add(did)
            with np.load(BytesIO(zlib.decompress(blob)), allow_pickle=False) as arrays:
                ids, ns = arrays['ids'], arrays['counts']
                require(ids.dtype.kind == ns.dtype.kind == 'u' and ids.dtype.itemsize == ns.dtype.itemsize == 8, 'Count codec type')
                require(len(ids) == len(ns) and len(set(map(int, ids))) == len(ids), 'Count codec shape')
                for x,n in zip(ids,ns):
                    require(int(n)>0, 'Nonpositive count'); counts[int(x)] = counts.get(int(x),0)+int(n)
    require(found == wanted, 'Missing independent-final draw')
    require(type(expected_total) is int and sum(counts.values()) == expected_total, 'Independent-final shot total mismatch')
    return counts


def review(folder):
    folder = Path(folder).resolve()
    config = json.loads((folder/'manifest.json').read_text())
    result = json.loads((folder/'result.json').read_text())  # Do not turn absent output into an empty success.
    require(result['manifest_sha256'] == sha(folder/'manifest.json'), 'Manifest drift')
    require(sha(ROOT/'study-v1/HCI_PAIRED_DIMENSION_DIAGNOSTIC_V1.md') == config['spec_sha256'], 'Specification drift')
    endpoint = ROOT/'results/HCI-native-sampler-comparison-v1/b05725f001503ef867a1de37f332b7274762aa2ab97ceeeefbedf3f6208811eb/comparison.json'
    require(sha(endpoint) == config['source_endpoint_sha256'], 'Endpoint source drift')
    original = json.loads(endpoint.read_text())
    require(config['source_input_hashes'] == original['inputs'], 'Source closure mismatch')
    require(all(sha(ROOT/'benchmark'/p) == value for p,value in config['implementation'].items()), 'Bound producer drift')
    require(all(sha(p) == value for p,value in config['source_input_hashes'].items()), 'Source drift')
    require(all(sha(p) == value for p,value in result['readout_database_hashes'].items()), 'Readout drift')
    e = ROOT/'results/molecular-strong-controls-standard-v2'
    hci = ROOT/'results/integral-threshold-HCI-standard-v1'
    all_cases = [item['id'] for item in json.loads((e/'manifest.json').read_text())['queue']]
    require(config['mode'] in ('quick','standard'), 'Unregistered mode')
    cases = [all_cases[0],all_cases[-1]] if config['mode']=='quick' else all_cases
    require(config['cases'] == cases, 'Changed planned cases')
    require(config['rows'] == [r for r in original['rows'] if r['case'] in cases and r['method_kind']=='SAMPLER'], 'Changed planned source rows')
    qrows = {(r['case'],r['method']): r for r in json.loads((e/'validation.json').read_text())['rows']}
    sources = {}; counts_cache = {}; dependencies = {}
    for old in config['rows']:
        if old['comparison_status'] != 'AVAILABLE_DESCRIPTIVE_PAIR': continue
        case,method,rcap,eps = identity(old)
        method_folder = e/'cases'/case/method
        raw_path = method_folder/'result.json'; raw = json.loads(raw_path.read_text())
        physical_path = e/'cases'/case/'physical-source.json'
        physical = json.loads(physical_path.read_text()); meta_path = Path(physical['path'])/'problem.json'
        meta = json.loads(meta_path.read_text())
        hpath = hci/'cases'/('E-'+case)/('epsilon-'+str(eps))/'result.json'
        h = json.loads(hpath.read_text())
        require(meta['problem_id'] == h['problem_id'], 'Cross-Hamiltonian comparison')
        require(sha(meta_path) == physical['meta_sha256'], 'Physical metadata drift')
        for p in (raw_path,physical_path,meta_path,hpath): dependencies[str(p)] = sha(p)
        if (case,method) not in counts_cache:
            counts_cache[(case,method)] = final_counts(method_folder,method,raw,dependencies)
        sources[identity(old)] = dict(problem_id=meta['problem_id'],hf_id=meta['hf_id'],
            sampler=next(r for r in qrows[(case,method)]['readouts'] if r['R']==rcap),
            hci=h['history'][-1],counts=counts_cache[(case,method)])
    rows, statuses = check_body(config,result,sources)
    require(all(sha(p)==value for p,value in dependencies.items()), 'Source changed during review')
    target = folder.with_name(folder.name+'-review-v1'); target.mkdir(exist_ok=True)
    stream = StringIO(); writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(status='ALL_PLANNED_SELECTIONS_SOURCE_COUNTS_DIMENSIONS_AND_SAVED_ENERGIES_REVIEWED',
        planned_pairs=len(rows),statuses=statuses,unique_spaces=len(result['unique_spaces']),
        trivial_HF_pairs=sum(r['trivial_HF_space'] is True for r in rows),
        nontrivial_available_pairs=sum(r['matched_d'] is not None and r['matched_d']>1 for r in rows),
        source_manifest_sha256=sha(folder/'manifest.json'),source_result_sha256=sha(folder/'result.json'),
        review_code_sha256=sha(__file__),dependencies=dependencies,new_CI_solves=0,new_quantum_acquisitions=0,
        independent_matrix_or_energy_recomputation=False,whole_goal_complete=False)
    readme = ('# HCI / sampler same-d readout diagnostic\n\n'
        f'{len(rows)} planned display pairs; {summary["nontrivial_available_pairs"]} nontrivial available; '
        f'{summary["trivial_HF_pairs"]} trivial HF-only. See all-planned-pairs.csv for missing rows.\n\n'
        'Positive sampler_minus_truncated_HCI_mHa means HCI has lower energy. '
        'HCI is coefficient-truncated then re-diagonalized, NOT its native endpoint. '
        'Same d is not matched construction work, quantum budget, or ground-state accuracy. '
        'R/epsilon rows reuse sources: they are not independent molecular experiments. '
        'This review independently checks final-count selection, HCI coefficient selection, '
        'identities, dimensions, saved rechecks and arithmetic; it does not repeat matrix solves. '
        'No reference energy or success-probability claim. Original endpoints and all missingness are retained.\n')
    outputs = {'validation.json':json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n',
               'all-planned-pairs.csv':stream.getvalue(),'README.md':readme}
    for name,value in outputs.items():
        path = target/name
        require(not path.exists() or path.read_bytes() == value.encode(), 'Existing review differs; use successor')
    for name,value in outputs.items():
        path = target/name
        if not path.exists(): path.write_bytes(value.encode())
    print(json.dumps({k:v for k,v in summary.items() if k!='dependencies'},sort_keys=True))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    review(parser.parse_args().root)
