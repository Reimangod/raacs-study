"""N29 bounded readout of fresh counts, without changing frozen samplers."""
from pathlib import Path
from time import process_time, perf_counter
import importlib.metadata
import os
import xml.etree.ElementTree as ET
import numpy as np
from .io import load, save, write, arrays, sha256, digest, space
from .paths import ROOT
from . import finite_source_study as source
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve

LOCAL = source.LOCAL / 'readout-v1'
ARMS = ('R32_CAP', 'COMMON_R')
HELPERS = ('finite_source_study.py', 'paired_lowshot.py', 'joint_ci_sparse.py',
           'joint_ci_sparse.c', 'prefix_performance_sampling.py')


def inventory(p, sub):
    rows = [dict(r, source_kind='CPU', available=True) for r in p['cpu_requests']]
    circuits = {c['circuit_id']: c for c in sub['circuits']}
    for c in sub['cells']:
        available = c['status'] == 'PLANNED'
        rows.append(dict(case=c['case'], method=c['source'], replicate=c['replicate'],
            draw_id=c.get('draw_id'), available=available, source_kind='LUCJ',
            backend=circuits[c['circuit_id']]['backend'] if available else None,
            cell_id=c['cell_id'], circuit_id=c.get('circuit_id')))
    if len(rows) != 60 or sum(r['available'] for r in rows) != 54:
        raise RuntimeError('All60sourcecells/54available required')
    return rows


def owner(row, arm):
    if arm == 'COMMON_R':
        return 'M4' if row['case'] == 'NAPH20-BLOCK-BOYS' else 'A100_INPUTS'
    return 'M4' if row['source_kind'] == 'CPU' or row['backend'] != 'A100' else 'A100_INPUTS'


def rank_record(ids, counts, n, k):
    hf = (1 << k) - 1
    rank = ranked_half(ids, counts, n, hf)
    mask = (1 << n) - 1
    halves = {int(x) & mask for x in ids} | {int(x) >> n for x in ids}
    return dict(rank=rank, R32=min(32, len(rank)),
        HF_full_configuration_observed=bool(np.any(ids == (hf | (hf << n)))),
        HF_half_reference_added=hf not in halves)


def basis_key(d, pairs, library_sha):
    return digest(dict(problem_id=d['problem_id'], integrals_SHA256=d['integrals_SHA256'],
        nelec=d['nelec'], core=d['core_energy'], pairs=np.asarray(pairs).tolist(),
        equation='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
        library_SHA256=library_sha))


def audited_counts(p, sub, row):
    """Read only. Pending is not missing; deterministic replays are audit work."""
    d = p['physical'][row['case']]
    if not row['available']:
        return dict(status='MISSING_SAME_PROBLEM_SOURCE')
    if row['source_kind'] == 'CPU':
        request = next(r for r in p['cpu_requests'] if r['draw_id'] == row['draw_id'])
        folder = source.LOCAL / 'cpu-draws' / row['draw_id']
        if not (folder / 'result.json').exists():
            return dict(status='WAITING_FOR_FROZEN_INPUTS')
        result = load(folder / 'result.json')
        if load(folder / 'started.json') != request or any(result[k] != v for k, v in request.items()):
            raise RuntimeError('CPU actual request changed')
        file = Path(result['payload'])
        if file != folder / 'counts.npz' or sha256(file.read_bytes()) != result['payload_SHA256']:
            raise RuntimeError('CPU counts binding changed')
        if result['problem_id'] != d['problem_id'] or result['status'] != 'COMPLETED':
            raise RuntimeError('CPU physical context differs')
        with np.load(file, allow_pickle=False) as z:
            ids, counts = z['blocked_ids'], z['counts']
        s = next(s for s in p['cpu_sources'] if s['case'] == row['case'])
        if sha256(Path(s['source_file']).read_bytes()) != s['source_SHA256']:
            raise RuntimeError('Source input drift')
        rng = np.random.default_rng(request['seed'])
        with np.load(s['source_file'], allow_pickle=False) as z:
            half = rng.choice(z['half_strings'], size=1000, p=z['probabilities']) if row['method'] == 'PAIR_SCALE2_REAL' else source.markov_bits(d['norb'], d['nelec'][0], s['angles'], 1000, rng)
        full = half | (half << np.uint64(d['norb']))
        expected_ids, expected_counts = np.unique(full, return_counts=True)
        if not np.array_equal(ids, expected_ids) or not np.array_equal(counts, expected_counts):
            raise RuntimeError('CPU frozen RNG replay mismatch')
        proof = dict(payload_SHA256=result['payload_SHA256'], seed=request['seed'],
            result_SHA256=sha256((folder / 'result.json').read_bytes()),
            sampling_role=result['sample_role'], audit_replay_is_not_new_draw=True)
    else:
        c = next(c for c in sub['circuits'] if c['circuit_id'] == row['circuit_id'])
        decoded = source.LUCJ / 'decoded' / (row['draw_id'] + '.npz')
        with source.configured_legacy().db() as db:
            ledger = db.execute('SELECT status,raw_sha,decoded_sha FROM draws WHERE draw_id=?', (row['draw_id'],)).fetchone()
        if ledger is None:
            raise RuntimeError('Unregistered draw')
        if ledger[0] != 'COMPLETED':
            return dict(status='WAITING_FOR_FROZEN_INPUTS', ledger_status=ledger[0])
        evidence = load(decoded.with_suffix('.json'))
        folder = source.LUCJ / ('collected/circuits' if c['backend'] == 'A100' else 'circuits') / c['circuit_id'] / 'output'
        resultfile = folder / 'result.json'
        result = load(resultfile)
        request = next(r for r in c['sampling_requests'] if r['id'] == row['draw_id'])
        raw = folder / (row['draw_id'] + '.npz')
        actual = next(r for r in result['sampling_results'] if r['id'] == row['draw_id'])
        status = 'EXACT_MATERIALIZED_CIRCUIT_' + ('GPU' if c['backend'] == 'A100' else 'M4') + '_COMPLETED'
        if result['status'] != status or result['input_id'] != c['input_id'] or result['physical_problem_id'] != d['problem_id'] or result['source_QPY_SHA256'] != c['original_QPY_SHA256'] or abs(result['norm_squared'] - 1) > 1e-8:
            raise RuntimeError('LUCJ runtime binding differs')
        if result['actual_QPU_shots'] != 0 or result['sector_dimension'] != c['sector_dimension']:
            raise RuntimeError('Unexpected physical result')
        if any(actual[k] != v for k, v in request.items()) or actual['actual_model_shots'] != 1000 or load(folder / (row['draw_id'] + '-draw-started.json')) != request:
            raise RuntimeError('LUCJ seed/shot binding differs')
        if sha256(raw.read_bytes()) != actual['counts_SHA256'] or evidence['raw_counts_SHA256'] != ledger[1] or ledger[1] != actual['counts_SHA256'] or sha256(decoded.read_bytes()) != evidence['decoded_SHA256'] or ledger[2] != evidence['decoded_SHA256'] or sha256(resultfile.read_bytes()) != evidence['result_SHA256']:
            raise RuntimeError('LUCJ evidence/ledger mismatch')
        if c['backend'] == 'A100':
            if result['cupy'] != p['expected_GPU_cupy']:
                raise RuntimeError('Unqualified CUDA runtime')
            jobs = [load(path) for path in source.LUCJ.glob('lane*-job.json')]
            job = next(j for j in jobs if c['circuit_id'] in j['circuit_ids'])
            if result['job_id'] != job['job_id'] or load(folder / 'started.json')['job_id'] != job['job_id']:
                raise RuntimeError('GPU ownership mismatch')
        with np.load(raw, allow_pickle=False) as z:
            logical = source.legacy.decode(z['blocked_ids'], c['decoder'], c['norb'], c['nelec'])
            order = np.argsort(logical)
            expected_ids, expected_counts = logical[order], z['counts'][order]
        with np.load(decoded, allow_pickle=False) as z:
            ids, counts = z['blocked_ids'], z['counts']
        if not np.array_equal(ids, expected_ids) or not np.array_equal(counts, expected_counts):
            raise RuntimeError('Independent actual decoder audit failed')
        proof = dict(payload_SHA256=evidence['decoded_SHA256'], seed=request['seed'],
            result_SHA256=evidence['result_SHA256'], sampling_role='FRESH_FULL_LUCJ_MODEL_EQUIVALENT',
            backend=c['backend'])
    source.validate_counts(ids, counts, d['norb'], d['nelec'])
    return dict(status='COMPLETED', **rank_record(ids, counts, d['norb'], d['nelec'][0]), **proof)


def prepare():
    p, _ = source.verified()
    space()
    if LOCAL.exists():
        raise RuntimeError('Existing N29 CI protocol; no overwrite')
    q = ROOT / 'qualification-N29-readout-preexecution-v1-tests.xml'
    suites = list(ET.parse(q).getroot().iter('testsuite'))
    if sum(int(s.get('tests', 0)) for s in suites) != 4 or any(int(s.get('failures', 0)) or int(s.get('errors', 0)) for s in suites):
        raise RuntimeError('All4 readout qualification tests required')
    sub = load(source.LUCJ / 'protocol.json')
    rows = inventory(p, sub)
    tasks = [dict(row, arm=arm, owner=owner(row, arm)) for row in rows for arm in ARMS]
    if sum(t['owner'] == 'M4' for t in tasks) != 48:
        raise RuntimeError('Predefined M4 partition differs')
    save(LOCAL / 'protocol.json', dict(version='N29_FINITE_READOUT_V1', tasks=tasks,
        source_protocol_SHA256=sha256((source.LOCAL / 'protocol.json').read_bytes()),
        code_SHA256=sha256(Path(__file__).read_bytes()),
        helpers={name:sha256((ROOT / 'frontierloop' / name).read_bytes()) for name in HELPERS},
        packages={name:importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pyscf')},
        qualification_SHA256=sha256(q.read_bytes()), library_SHA256=p['library_SHA256'],
        max_R=32, max_D=1024, root_seconds=20, batch_seconds=180, new_CI_bytes_cap=16*2**20,
        task_partition='M4_CPU_PRIMARY_ALL_PLUS_ALL_NAPH;A100_INPUTS_REMAINING_PRIMARY_AND_COMMON',
        common_R='MIN_ALL_AVAILABLE_METHODS_WITHIN_CASE_AND_REPLICATE;PENDING_NOT_MISSING',
        independent_MC_draws_remain_independent_even_if_CI_basis_matches=True,
        independent_source_optimizations=False, QPU_shots=0, goal_complete=False))
    print('N29_CI_FROZEN120ENDPOINTS_M4_48_A100INPUTS72', flush=True)


def verified():
    p, _ = source.verified()
    q = load(LOCAL / 'protocol.json')
    if sha256(Path(__file__).read_bytes()) != q['code_SHA256'] or sha256((source.LOCAL / 'protocol.json').read_bytes()) != q['source_protocol_SHA256'] or sha256(Path(p['library']).read_bytes()) != q['library_SHA256']:
        raise RuntimeError('Frozen readout drift')
    for name, h in q['helpers'].items():
        if sha256((ROOT / 'frontierloop' / name).read_bytes()) != h:
            raise RuntimeError('Frozen helper drift')
    if {name:importlib.metadata.version(name) for name in q['packages']} != q['packages']:
        raise RuntimeError('Pinned package drift')
    return p, q, load(source.LUCJ / 'protocol.json')


def run(batch):
    p, q, sub = verified()
    if batch not in ('M4', 'A100_INPUTS'):
        raise ValueError('Registered partition only')
    out = LOCAL / batch
    if (out / 'started.json').exists():
        raise RuntimeError('Existing attempt; inspect actual process/terminal, never blind restart')
    rows = inventory(p, sub)
    wanted = [t for t in q['tasks'] if t['owner'] == batch]
    needed = {t['draw_id'] for t in wanted if t['available']}
    for t in wanted:
        if t['arm'] == 'COMMON_R':
            needed.update(r['draw_id'] for r in rows if r['case'] == t['case'] and r['replicate'] == t['replicate'] and r['available'])
    cpu, wall = process_time(), perf_counter()
    counts = {r['draw_id']:audited_counts(p, sub, r) for r in rows if r['available'] and r['draw_id'] in needed}
    waiting = [key for key, value in counts.items() if value['status'] != 'COMPLETED']
    if waiting:
        print('WAITING_FOR_FROZEN_INPUTS', batch, len(waiting), flush=True)
        return
    space()
    save(out / 'started.json', dict(pid=os.getpid(), protocol_SHA256=sha256((LOCAL / 'protocol.json').read_bytes()),
        input_audit_CPU_seconds=process_time()-cpu))
    states, attempts = [], []
    try:
        for task in wanted:
            if perf_counter() - wall > q['batch_seconds']:
                raise TimeoutError('Readout batch cap')
            space()
            if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file()) > q['new_CI_bytes_cap']:
                raise OSError('Bounded new CI storage cap')
            row = dict(task)
            if not task['available']:
                row.update(status='MISSING_SAME_PROBLEM_SOURCE', energy_Ha=None)
            else:
                record = counts[task['draw_id']]
                R = record['R32'] if task['arm'] == 'R32_CAP' else min(counts[r['draw_id']]['R32'] for r in rows if r['case'] == task['case'] and r['replicate'] == task['replicate'] and r['available'])
                strings = sorted(record['rank'][:R])
                pairs = np.array([(a,b) for a in strings for b in strings], np.uint64)
                d = p['physical'][task['case']]
                key = basis_key(d, pairs, q['library_SHA256'])
                root = LOCAL / 'roots' / key
                meta = root.with_suffix('.json')
                if meta.exists():
                    cached = load(meta)
                    if cached['basis_id'] != key or cached['status'] not in ('COMPLETED', 'FAILED_PRESERVED'):
                        raise RuntimeError('Root cache inconsistency')
                    new = False
                else:
                    if root.with_suffix('.started.json').exists():
                        raise RuntimeError('Partial root attempt; reconcile rather than duplicate')
                    file = Path(d['local_integrals'])
                    if sha256(file.read_bytes()) != d['integrals_SHA256']:
                        raise RuntimeError('Integral drift')
                    save(root.with_suffix('.started.json'), dict(basis_id=key, case=task['case'], batch=batch, pid=os.getpid()))
                    tick = process_time()
                    try:
                        with np.load(file, allow_pickle=False) as z:
                            result, v = solve(z['h1'], z['eri'], d['core_energy'], tuple(d['nelec']), pairs, p['library'], seconds=q['root_seconds'], triangle_cap=8000000)
                        vector = root.with_suffix('.npz')
                        write(vector, arrays(joint_pairs=pairs, coefficients=v))
                        cached = dict(status='COMPLETED', basis_id=key, result=result,
                            root_path=str(vector), root_SHA256=sha256(vector.read_bytes()))
                    except Exception as exc:
                        cached = dict(status='FAILED_PRESERVED', basis_id=key, result=None, error=repr(exc))
                    cached['attempt_CPU_seconds'] = process_time()-tick
                    save(meta, cached)
                    attempts.append(dict(basis_id=key, status=cached['status'], CPU_seconds=cached['attempt_CPU_seconds']))
                    save(out / 'attempts.json', attempts)
                    new = True
                if cached['status'] == 'COMPLETED' and sha256(Path(cached['root_path']).read_bytes()) != cached['root_SHA256']:
                    raise RuntimeError('Saved root hash differs')
                row.update(cached, R=R, actual_D=R*R, half_strings=strings,
                    source_evidence=record, new_CI_attempt=new)
            states.append(row)
            save(out / 'progress.json', dict(states=states))
        save(out / 'terminal.json', dict(status='N29_READOUT_BATCH_TERMINAL', batch=batch,
            states=states, attempts=attempts, CPU_seconds=process_time()-cpu,
            wall_seconds=perf_counter()-wall, protocol_SHA256=sha256((LOCAL / 'protocol.json').read_bytes()),
            new_QPU_shots=0, goal_complete=False))
        print('N29_READOUT_TERMINAL', batch, len(states), len(attempts), flush=True)
    except Exception as exc:
        save(out / 'failed.json', dict(error=repr(exc), completed_endpoints=len(states),
            CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall))
        raise


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'prepare':prepare()
    else:run(sys.argv[1])
