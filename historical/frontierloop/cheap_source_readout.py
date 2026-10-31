"""N42 same physical CI for fixed cheap source and ALL strong existing controls."""
from pathlib import Path
from time import process_time, perf_counter
import os
import sqlite3
import sys
import numpy as np
from .paths import ROOT
from .io import load, save, write, arrays, sha256, digest, space
from . import cheap_source_sampling as s
from . import analytical_source_readout as old
from . import finite_source_readout as parent
from .joint_ci_sparse import solve

LOCAL = s.LOCAL/'readout-v1'
ARMS = ('R32_CAP', 'COMMON_R')


def inventory(p, sub):
    rows = list(p['control_inventory'])
    for cell in sub['cells']:
        rows.append(dict(case=cell['case'], method=s.METHOD, replicate=cell['replicate'], draw_id=cell['draw_id'],
                         cell_id=cell['cell_id'], circuit_id=cell['circuit_id'], available=True, origin='N42_NEW_FIXED_MP2'))
    if len(rows) != 72 or sum(r['available'] for r in rows) != 63:
        raise ValueError('Keep ALL72 planned63 available9 missing source cells')
    return rows


def new_counts(p, sub, row):
    with sqlite3.connect(s.SAMPLING/'draw-ledger.sqlite') as db:
        ledger = db.execute('SELECT status,raw_sha,decoded_sha FROM draws WHERE draw_id=?', (row['draw_id'],)).fetchone()
    if ledger is None:
        raise ValueError('Unregistered new draw')
    if ledger[0] != 'COMPLETED':
        return dict(status='WAITING_FOR_FROZEN_INPUTS')
    c = next(c for c in sub['circuits'] if c['circuit_id'] == row['circuit_id'])
    request = next(r for r in c['sampling_requests'] if r['id'] == row['draw_id'])
    folder = s.SAMPLING/('collected/circuits' if c['backend'] == 'A100' else 'circuits')/c['circuit_id']/'output'
    resultfile = folder/'result.json'; result = load(resultfile)
    expected = 'EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED' if c['backend'] == 'A100' else 'EXACT_MATERIALIZED_CIRCUIT_M4_COMPLETED'
    if result['status'] != expected or result['input_id'] != c['input_id'] or result['physical_problem_id'] != c['problem_id'] or result['source_QPY_SHA256'] != c['original_QPY_SHA256'] or result['sector_dimension'] != c['sector_dimension'] or abs(result['norm_squared']-1) > 1e-8 or result['actual_QPU_shots'] != 0:
        raise ValueError('Actual source/physical state/executor identity differs')
    if c['backend'] == 'A100':
        job = next(load(f) for f in s.SAMPLING.glob('lane*-job.json') if c['circuit_id'] in load(f)['circuit_ids'])
        if result['cupy'] != p['expected_CUDA_cupy'] or result['job_id'] != job['job_id'] or load(folder/'started.json')['job_id'] != job['job_id']:
            raise ValueError('Actual CUDA allocation differs')
        if result['primitive_semantics'] != 'HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP':
            raise ValueError('Wrong physical primitive')
    actual = next(r for r in result['sampling_results'] if r['id'] == request['id'])
    if any(actual[k] != v for k, v in request.items()) or actual['actual_model_shots'] != 1000 or load(folder/(request['id']+'-draw-started.json')) != request:
        raise ValueError('Actual RNG/Shot context differs')
    raw = folder/(request['id']+'.npz'); decoded = s.SAMPLING/'decoded'/(request['id']+'.npz'); evidence = load(decoded.with_suffix('.json'))
    if sha256(raw.read_bytes()) != ledger[1] or actual['counts_SHA256'] != ledger[1] or evidence['raw_counts_SHA256'] != ledger[1] or evidence['decoded_SHA256'] != ledger[2] or sha256(decoded.read_bytes()) != ledger[2] or evidence['result_SHA256'] != sha256(resultfile.read_bytes()):
        raise ValueError('Actual raw/result/ledger/hash mismatch')
    with np.load(raw, allow_pickle=False) as z:
        ids = s.legacy.decode(z['blocked_ids'], c['decoder'], c['norb'], c['nelec']); order = np.argsort(ids)
        rawids, rawcounts = ids[order], z['counts'][order]
    with np.load(decoded, allow_pickle=False) as z: ids, counts = z['blocked_ids'], z['counts']
    if not np.array_equal(ids, rawids) or not np.array_equal(counts, rawcounts):
        raise ValueError('Independent decoder differs')
    s.n29.validate_counts(ids, counts, c['norb'], c['nelec'])
    return dict(status='COMPLETED', **parent.rank_record(ids, counts, c['norb'], c['nelec'][0]),
                payload_SHA256=ledger[2], seed=request['seed'], result_SHA256=sha256(resultfile.read_bytes()),
                sampling_role='NEW_FIXED_MP2_L1_MODEL_EQUIVALENT', backend=c['backend'])


def records(p, sub, rows):
    controls = [r for r in rows if not r['origin'].startswith('N42_')]
    result = s.control_records(controls)
    for key, value in result.items():
        if digest(value) != p['control_record_digests'][key]:
            raise ValueError('Frozen old counts context changed')
    for row in rows:
        if row['origin'].startswith('N42_'):
            result[row['draw_id']] = new_counts(p, sub, row)
    return result


def prepare():
    p = s.verified(); space(); sub = load(s.SAMPLING/'protocol.json'); rows = inventory(p, sub)
    if LOCAL.exists():
        raise RuntimeError('Existing readout; no overwrite')
    save(LOCAL/'protocol.json', dict(version='N42_ALL6_METHOD_STRONG_CONTROL_READOUT_V1',
         tasks=[dict(row, arm=arm) for row in rows for arm in ARMS], max_R=32, max_D=1024,
         root_seconds=20, partition_wall_seconds=180, partition_CPU_seconds=180,
         source_protocol_SHA256=sha256((s.LOCAL/'protocol.json').read_bytes()),
         code_SHA256=sha256(Path(__file__).read_bytes()), library_SHA256=p['library_SHA256'],
         objective='PHYSICAL_H_PLUS_0p1S2_NOT_EXACT_S0',
         ranking='FREQUENCY_SPINHALF_SYMMETRIZED_HF_ADDED_COMMON_R_ALL_AVAILABLE_STRONG_CONTROLS',
         no_new_optimizer_FCI_QPU=True, old_costs_retained=True, goal_complete=False))
    print('N42_READOUT_FROZEN144_ENDPOINTS_126_AVAILABLE18_MISSING')


def verified():
    p = s.verified(); q = load(LOCAL/'protocol.json')
    if q['code_SHA256'] != sha256(Path(__file__).read_bytes()) or q['source_protocol_SHA256'] != sha256((s.LOCAL/'protocol.json').read_bytes()):
        raise ValueError('Frozen readout drift')
    if sha256(Path(p['library']).read_bytes()) != q['library_SHA256']:
        raise ValueError('Qualified solver changed')
    return p, q, load(s.SAMPLING/'protocol.json')


def common_R(task, rows, counts):
    return min(counts[r['draw_id']]['R32'] for r in rows if r['available'] and r['case'] == task['case'] and r['replicate'] == task['replicate'])


def run(partition):
    if partition not in ('M4', 'A100_INPUTS'):
        raise ValueError('Explicit disjoint partition required')
    p, q, sub = verified(); space(); cpu, wall = process_time(), perf_counter()
    tasks = [t for t in q['tasks'] if (t['case'] == 'NAPH20-BLOCK-BOYS') == (partition == 'M4')]
    cases = {t['case'] for t in tasks}; rows = [r for r in inventory(p, sub) if r['case'] in cases]
    counts = records(p, sub, rows)
    if any(c['status'] == 'WAITING_FOR_FROZEN_INPUTS' for c in counts.values()):
        print('N42_WAITING_NO_CI_INTENT', partition); return
    marker = LOCAL/(partition+'-started.json')
    if marker.exists():
        raise RuntimeError('Existing partial/terminal; no blind rerun')
    save(marker, dict(pid=os.getpid(), protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    states = []; attempts = []
    try:
        for task in tasks:
            if process_time()-cpu > q['partition_CPU_seconds'] or perf_counter()-wall > q['partition_wall_seconds']:
                raise TimeoutError('Finite readout partition cap')
            if sum(f.stat().st_size for f in s.LOCAL.rglob('*') if f.is_file()) > p['new_run_cap_bytes']:
                raise OSError('N42 new namespace cap')
            row = dict(task)
            if not task['available']:
                row.update(status='MISSING_SAME_PROBLEM_SOURCE', energy_Ha=None)
            else:
                record = counts[task['draw_id']]; d = p['physical'][task['case']]
                R = record['R32'] if task['arm'] == 'R32_CAP' else common_R(task, rows, counts)
                strings = sorted(record['rank'][:R]); pairs = np.array([(a, b) for a in strings for b in strings], np.uint64)
                key = parent.basis_key(d, pairs, q['library_SHA256']); root = LOCAL/'roots'/key; meta = root.with_suffix('.json')
                candidates = [meta, old.LOCAL/'roots'/(key+'.json'), old.oldread.LOCAL/'roots'/(key+'.json'), parent.LOCAL/'roots'/(key+'.json')]
                existing = next((f for f in candidates if f.exists()), None)
                if existing:
                    cache = load(existing); new = False; origin = 'N42_SAME_BASIS' if existing == meta else 'OLD_SAME_PHYSICAL_BASIS'
                    if cache['basis_id'] != key or cache['status'] not in ('COMPLETED', 'FAILED_PRESERVED'):
                        raise ValueError('Cache identity differs')
                else:
                    if root.with_suffix('.started.json').exists():
                        raise RuntimeError('Partial root; no repeated work')
                    f = Path(d['local_integrals'])
                    if sha256(f.read_bytes()) != d['integrals_SHA256']:
                        raise ValueError('Integral drift')
                    save(root.with_suffix('.started.json'), dict(basis_id=key, case=task['case'], pid=os.getpid())); tick = process_time()
                    try:
                        with np.load(f, allow_pickle=False) as z:
                            result, vec = solve(z['h1'], z['eri'], d['core_energy'], tuple(d['nelec']), pairs, p['library'], seconds=q['root_seconds'], triangle_cap=8000000)
                        f = root.with_suffix('.npz'); write(f, arrays(joint_pairs=pairs, coefficients=vec))
                        cache = dict(status='COMPLETED', basis_id=key, result=result, root_path=str(f), root_SHA256=sha256(f.read_bytes()))
                    except Exception as exc:
                        cache = dict(status='FAILED_PRESERVED', basis_id=key, result=None, error=repr(exc))
                    cache['attempt_CPU_seconds'] = process_time()-tick; save(meta, cache)
                    attempts.append(dict(basis_id=key, status=cache['status'], CPU_seconds=cache['attempt_CPU_seconds'])); new = True; origin = 'N42_NEW_ROOT'
                if cache['status'] == 'COMPLETED' and sha256(Path(cache['root_path']).read_bytes()) != cache['root_SHA256']:
                    raise ValueError('Saved root changed')
                row.update(cache, R=R, actual_D=R*R, half_strings=strings, source_evidence=record, new_CI_attempt=new, cache_origin=origin)
            states.append(row); save(LOCAL/(partition+'-progress.json'), dict(states=states, attempts=attempts))
        save(LOCAL/(partition+'-terminal.json'), dict(status='PARTITION_TERMINAL', partition=partition, states=states, attempts=attempts,
             CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
             protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()), goal_complete=False))
        print('N42_PARTITION_TERMINAL', partition, len(states), len(attempts), 'newroots')
    except Exception as exc:
        save(LOCAL/(partition+'-failed.json'), dict(error=repr(exc), completed=len(states), CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall))
        raise


if __name__ == '__main__':
    if sys.argv[1] == 'run': run(sys.argv[2])
    else: prepare()
