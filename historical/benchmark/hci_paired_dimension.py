"""Bounded paired-d diagnostic, queued AFTER the isolated workflow timing run."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/HCI-native-sampler-comparison-v1/b05725f001503ef867a1de37f332b7274762aa2ab97ceeeefbedf3f6208811eb/comparison.json'
E = ROOT / 'results/molecular-strong-controls-standard-v2'
HCI = ROOT / 'results/integral-threshold-HCI-standard-v1'
SPEC = ROOT / 'study-v1/HCI_PAIRED_DIMENSION_DIAGNOSTIC_V1.md'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    temporary = path.with_suffix('.tmp'); temporary.write_text(text); temporary.replace(path)


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('Frozen diagnostic drift')
    else: save(path, value)


def select_counts(ids, counts, hf, d):
    if type(d) is not int or len(ids) != len(set(ids)) or hf not in ids or not 1 <= d <= len(ids):
        raise ValueError('Invalid saved sampler space/d')
    if any(type(counts.get(x)) is not int or counts[x] <= 0 for x in ids if x != hf):
        raise ValueError('Non-HF sampler IDs must be actually observed')
    order = [hf] + sorted((x for x in ids if x != hf), key=lambda x: (-counts[x], x))
    return sorted(order[:d])


def select_coefficients(ids, coefficients, hf, d):
    if type(d) is not int or len(ids) != len(coefficients) or len(ids) != len(set(ids)) or hf not in ids or not 1 <= d <= len(ids):
        raise ValueError('Invalid HCI space/d')
    if not all(type(c) in (int, float) and math.isfinite(c) for c in coefficients):
        raise ValueError('Invalid HCI coefficients')
    scores = dict(zip(ids, coefficients))
    order = [hf] + sorted((x for x in ids if x != hf), key=lambda x: (-abs(scores[x]), x))
    return sorted(order[:d])


def prepare(mode):
    source = json.loads(SOURCE.read_text())
    if any(sha(p) != digest for p, digest in source['inputs'].items()):
        raise ValueError('Audited endpoint inputs changed')
    em = json.loads((E / 'manifest.json').read_text())
    cases = [item['id'] for item in em['queue']]
    chosen = [cases[0], cases[-1]] if mode == 'quick' else cases
    rows = [r for r in source['rows'] if r['case'] in chosen and r['method_kind'] == 'SAMPLER']
    root = ROOT / ('results/HCI-paired-diagnostic-' + mode + '-v1')
    cfg = dict(schema='HCI-paired-diagnostic-v1', mode=mode, cases=chosen,
        planned_pairs=len(rows), source_endpoint_sha256=sha(SOURCE), spec_sha256=sha(SPEC),
        maximum_projected_dimension=64, free_floor_bytes=int(2.75*2**30),
        source_input_hashes=source['inputs'],
        implementation={p:sha(ROOT/'benchmark'/p) for p in ('hci_paired_dimension.py',
            'molecular_strong_analysis.py','resource_adaptive_sampler.py','integral_hamiltonian.py',
            'integral_connections.py','molecular_mps_scaling.py','local_sampler_primitives.py')},
        rows=rows, new_quantum_acquisitions=0, whole_goal_complete=False)
    root.mkdir(exist_ok=True); frozen(root/'manifest.json', cfg)
    return root, cfg


def execute(root, cfg):
    # Lazy numerical imports keep preparation/waiting free of quantum kernels.
    import numpy as np
    from threadpoolctl import threadpool_limits
    from .resource_adaptive_sampler import integral_problem
    from .molecular_strong_analysis import final_evidence
    from .molecular_mps_scaling import no_full_state_guards
    path = root/'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['manifest_sha256'] != sha(root/'manifest.json'): raise ValueError('Result binding drift')
        return result
    for name, digest in cfg['implementation'].items():
        if sha(ROOT/'benchmark'/name) != digest: raise ValueError('Diagnostic implementation drift')
    if sha(SPEC) != cfg['spec_sha256'] or sha(SOURCE) != cfg['source_endpoint_sha256']:
        raise ValueError('Diagnostic specification/source drift')
    if any(sha(p) != digest for p,digest in cfg['source_input_hashes'].items()):
        raise ValueError('Changed source inputs')
    if shutil.disk_usage(ROOT).free < cfg['free_floor_bytes']: raise RuntimeError('Capacity boundary')
    if cfg['mode'] == 'standard':
        quick = json.loads((ROOT/'results/HCI-paired-diagnostic-quick-v1/result.json').read_text())
        if quick['status'] != 'ALL_PAIRED_DIAGNOSTIC_SPACES_RECHECKED': raise ValueError('Actual quick required')
    started = time.perf_counter(); output = []; spaces = {}; count_cache = {}; hci_cache = {}; dependencies = {}
    maximum = 0.; new_solves = 0
    manifest = json.loads((E/'manifest.json').read_text())
    items = {item['id']:item for item in manifest['queue']}
    qvalidation = json.loads((E/'validation.json').read_text())
    qrows = {(x['case'],x['method']):x for x in qvalidation['rows']}
    with threadpool_limits(limits=1), no_full_state_guards():
        for case in cfg['cases']:
            physical = json.loads((E/'cases'/case/'physical-source.json').read_text())
            backend, info = integral_problem(physical['path'])
            def solve(ids):
                nonlocal new_solves, maximum
                key = hashlib.sha256(json.dumps([info['problem_id'], ids]).encode()).hexdigest()
                if key not in spaces:
                    tick = time.perf_counter(); matrix = backend.matrix(ids, ids)
                    values, vectors = np.linalg.eigh(matrix)
                    fresh = float(np.linalg.eigvalsh(matrix)[0]); new_solves += 2
                    delta = abs(float(values[0])-fresh); maximum = max(maximum, delta)
                    residual = float(np.linalg.norm(matrix@vectors[:,0]-values[0]*vectors[:,0]))
                    if len(ids)>64 or delta>1e-8 or residual>1e-8: raise ValueError('Projected solve qualification failed')
                    spaces[key] = dict(problem_id=info['problem_id'],selected_ids=ids,energy_Ha=float(values[0]),
                        independent_recheck_energy_Ha=fresh,internal_residual=residual,
                        analysis_seconds=time.perf_counter()-tick,new_eigensolver_calls=2)
                return key, spaces[key]['energy_Ha']
            for row in (r for r in cfg['rows'] if r['case']==case):
                result = {k:row[k] for k in ('case','method','R_cap','epsilon1_Ha','sampler_status','HCI_status')}
                if row['comparison_status'] != 'AVAILABLE_DESCRIPTIVE_PAIR':
                    output.append(dict(result,status='MISSING_PAIR_PRESERVED')); continue
                method = row['method']; epsilon = row['epsilon1_Ha']
                if (case,method) not in count_cache:
                    raw,counts,paid,qc,evidence = final_evidence(E, items[case], method)
                    if raw['status']!='COMPLETED' or counts is None: raise ValueError('Wrong final readout')
                    count_cache[(case,method)] = counts
                    dependencies[evidence['source_database']] = evidence['source_database_sha256']
                if (case,epsilon) not in hci_cache:
                    h = json.loads((HCI/'cases'/('E-'+case)/('epsilon-'+str(epsilon))/'result.json').read_text())
                    hci_cache[(case,epsilon)] = h['history'][-1]
                q = qrows[(case,method)]
                qspace = next(x for x in q['readouts'] if x['R']==row['R_cap'])
                h = hci_cache[(case,epsilon)]; d = min(qspace['actual_dimension'],h['dimension'])
                qi = select_counts(qspace['selected_ids'],count_cache[(case,method)],info['hf_id'],d)
                hi = select_coefficients(h['selected_ids'],h['coefficients'],info['hf_id'],d)
                qkey, qe = solve(qi); hkey, he = solve(hi)
                output.append(dict(result,status='PAIRED_DIMENSION_DIAGNOSTIC',matched_d=d,trivial_HF_space=(d==1),
                    sampler_space_key=qkey,HCI_coefficient_truncated_space_key=hkey,
                    sampler_restricted_energy_Ha=qe,HCI_coefficient_truncated_energy_Ha=he,
                    sampler_minus_truncated_HCI_mHa=1000*(qe-he),original_HCI_energy_Ha=row['HCI_last_solved_energy_Ha'],
                    original_sampler_energy_Ha=row['sampler_energy_Ha'],work_matched=False))
            save(root/'checkpoint.json',dict(completed_cases=case,rows=output,unique_spaces=spaces,new_eigensolver_calls=new_solves))
    if len(output)!=cfg['planned_pairs'] or not spaces: raise ValueError('Incomplete planned diagnostic')
    for p,digest in {**cfg['source_input_hashes'],**dependencies}.items():
        if sha(p)!=digest: raise ValueError('Source drift during diagnostic')
    result = dict(status='ALL_PAIRED_DIAGNOSTIC_SPACES_RECHECKED',manifest_sha256=sha(root/'manifest.json'),
        rows=output,unique_spaces=spaces,maximum_eigensolver_disagreement_Ha=maximum,
        new_eigensolver_calls=new_solves,analysis_wall_seconds=time.perf_counter()-started,
        readout_database_hashes=dependencies,new_quantum_acquisitions=0,new_FCI_solves=0,
        scope='Post-hoc coefficient-truncated HCI/frequency-truncated sampler; not native HCI endpoint or matched work.',
        whole_goal_complete=False)
    save(path,result);return result


def wait_and_run(pid):
    root = ROOT/'results/HCI-paired-diagnostic-sequence-v1';root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        modes = [prepare(mode) for mode in ('quick','standard')]
        while True:
            p = subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
            if p.returncode not in (0,1): raise RuntimeError('OS observation error, not terminal evidence')
            if p.returncode==1: break
            if ' -m benchmark.workflow_cost_sequence' not in p.stdout: raise ValueError('Foreign/reused dependency PID')
            save(root/'execution-status.json',dict(status='WAITING_ON_VERIFIED_TOTAL_COST_COORDINATOR',
                dependency_PID=pid,OS_command=p.stdout.strip(),numerical_started=False,whole_goal_complete=False))
            time.sleep(20)
        cost = json.loads((ROOT/'results/total-workflow-sequence-v1/execution-status.json').read_text())
        if cost['status']!='WORKFLOW_TOTAL_COST_COHORTS_AUDITED_REPORTED':
            save(root/'execution-status.json',dict(status='DEPENDENCY_TERMINAL_INCOMPLETE',cost_status=cost['status']));return
        try:
            for folder,cfg in modes:
                save(root/'execution-status.json',dict(status='ONE_NUMERICAL_DIAGNOSTIC_RUNNING',mode=cfg['mode'],whole_goal_complete=False))
                execute(folder,cfg)
            save(root/'execution-status.json',dict(status='PAIRED_DIMENSION_DIAGNOSTICS_EXECUTED',whole_goal_complete=False))
        except Exception as exc:
            save(root/'execution-status.json',dict(status='FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(exc),whole_goal_complete=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-cost-pid',type=int);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:
        for mode in ('quick','standard'):
            root,cfg=prepare(mode);print(mode,cfg['planned_pairs'],sha(root/'manifest.json'))
    elif a.wait_cost_pid is not None: wait_and_run(a.wait_cost_pid)
    else: p.error('Use --prepare-only or a verified --wait-cost-pid')
