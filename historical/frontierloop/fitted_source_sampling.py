"""N45: ALL4 x fitted/same-amplitude-naive L1, 3 fresh1000 draws each; pinned executor."""
from pathlib import Path
from io import BytesIO
from time import process_time, perf_counter
import math
import sys
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load, save, write, arrays, sha256, digest
from . import fixed_rotation_source as init
from .fitted_source_profiles import profiles, METHOD, NAIVE_METHOD
from . import analytical_source_sampling as previous
from . import analytical_source_readout as previous_read
from . import finite_source_study as n29
from . import prefix_performance_sampling as legacy

LOCAL = ROOT/'runs/N45-FIXED-FIT-AND-NAIVE-FINITE-v1'
SAMPLING = LOCAL/'sampling'
REMOTE = '/share/reiwatanabe/raacs-frontier-loop-20261025/N45-fixed-fit-naive-finite-v1'
REPLICATES = (290101, 290102, 290103)
STRONG = ('C5_C_LUCJ_COMPRESSED_L1', 'C5_C_LUCJ_COMPRESSED_L2', 'C5_C_LUCJ_COMPRESSED_L4',
          'EXISTING_TN_COMMON_L1', 'ANALYTICAL_OPT_L1_8EVAL')


def seed(cid, rep):
    return int(digest(['N45_NEW_MODEL_RNG_V1', cid, rep])[:8], 16)


def configured():
    legacy.LOCAL = SAMPLING; legacy.REMOTE = REMOTE; legacy.SHOTS = 1000
    return legacy


def control_inventory():
    p = previous.verified(); sub = load(previous.SAMPLING/'protocol.json')
    return [r for r in previous_read.inventory(p, sub) if r['method'] in STRONG]


def control_records(rows):
    p = previous.verified(); sub = load(previous.SAMPLING/'protocol.json')
    return previous_read.records(p, sub, rows)


def helpers():
    names = ['fitted_source_sampling.py', 'fitted_source_profiles.py', 'fixed_rotation_source.py', 'fixed_rotation_ls.py', 'prefix_performance_sampling.py', 'collect_analytical_single_session.py',
             'analytical_source_readout.py', 'finite_source_readout.py', 'joint_ci_sparse.py', 'joint_ci_sparse.c']
    return {name: sha256((ROOT/'frontierloop'/name).read_bytes()) for name in names}


def prepare():
    storage = legacy.capacity(); ip = init.verified(); terminal = load(init.RUN/'terminal.json')
    if terminal['status'] != 'ALL4_FIXED_U_LINEAR_J_SOURCES_CONSTRUCTED' or len(terminal['rows']) != 4:
        raise ValueError('All4 qualified physical sources required')
    if LOCAL.exists():
        raise RuntimeError('Existing namespace; reconcile, never overwrite')
    oldrows = control_inventory(); oldrecords = control_records(oldrows)
    if len(oldrows) != 60 or sum(r['available'] for r in oldrows) != 51:
        raise ValueError('Full 17 actual control sources and3 missing, each3 replicates')
    if any(v['status'] != 'COMPLETED' for v in oldrecords.values() if v.get('status') != 'MISSING_SAME_PROBLEM_SOURCE'):
        # Missing entries have an explicit source-missing status; no new data fabrication.
        availablekeys = {r['draw_id'] for r in oldrows if r['available']}
        if any(oldrecords[k]['status'] != 'COMPLETED' for k in availablekeys):
            raise RuntimeError('Existing control counts are incomplete')
    old = load(n29.LUCJ/'protocol.json'); module = legacy.qualified_module()
    cpu, wall = process_time(), perf_counter(); circuits = []; cells = []; sources = []
    for name in ('worker.py', 'base_kernel.py', 'primitive_plan.py'):
        write(SAMPLING/name, (n29.LUCJ/name).read_bytes())
    for row in profiles(terminal['rows']):
        method_id = row['method_id']
        item = next(x for x in ip['catalog'] if x['source']['case'] == row['case']); d = item['physical']
        f = row['files']['logical.qpy']; blob = Path(f['path']).read_bytes()
        if sha256(blob) != f['SHA256']:
            raise ValueError('Actual source QPY changed')
        qc = qpy.load(BytesIO(blob))[0]; n, ne = row['norb'], row['nelec']
        cfg, vals = module.plan(qc, n, tuple(ne))
        cid = digest(dict(problem_id=d['problem_id'], QPY_SHA256=sha256(blob),
                          decoder=list(range(2*n)), qualified_plan_SHA256=old['plan_SHA256']))
        folder = SAMPLING/'circuits'/cid; payload = arrays(**vals); write(folder/'arrays.npz', payload)
        requests = [dict(id=digest(['N45', cid, rep]), replicate_label=rep, seed=seed(cid, rep), shots=1000)
                    for rep in REPLICATES]
        cfg.update(arrays_SHA256=sha256(payload), worker_SHA256=old['worker_SHA256'], base_kernel_SHA256=old['base_SHA256'],
                   source_QPY_SHA256=sha256(blob), physical_problem_id=d['problem_id'], sampling_requests=requests)
        cfg['input_id'] = digest(cfg); save(folder/'config.json', cfg)
        dim = math.prod(math.comb(n, e) for e in ne)
        circuit = dict(circuit_id=cid, case=row['case'], source=method_id, problem_id=d['problem_id'],
                       norb=n, nelec=ne, physical_SHA256=item['source']['physical_SHA256'], original_QPY=f['path'],
                       original_QPY_SHA256=sha256(blob), decoder=list(range(2*n)),
                       config_SHA256=sha256((folder/'config.json').read_bytes()), arrays_SHA256=sha256(payload),
                       input_id=cfg['input_id'], sampling_requests=requests, sector_dimension=dim,
                       primitive_count=len(cfg['ops']), estimated_work=dim*len(cfg['ops']), backend='M4' if n == 10 else 'A100')
        circuits.append(circuit); sources.append(row)
        for req in requests:
            c = dict(case=row['case'], source=method_id, method='ORIGINAL', problem_id=d['problem_id'],
                     replicate=req['replicate_label'], status='PLANNED', circuit_id=cid, draw_id=req['id'],
                     selected_resources=row['resources']['Z_sampling_equivalent'])
            c['cell_id'] = digest(['N45', c]); cells.append(c)
    prior_seeds = {r['seed'] for r in oldrecords.values() if r.get('seed') is not None}
    seeds = [r['seed'] for c in circuits for r in c['sampling_requests']]
    if len(seeds) != 24 or len(set(seeds)) != 24 or set(seeds) & prior_seeds:
        raise ValueError('Fresh independent draws required')
    lanes = [[], []]; loads = [0, 0]
    for c in sorted((c for c in circuits if c['backend'] == 'A100'), key=lambda c: (-c['estimated_work'], c['circuit_id'])):
        j = min(range(2), key=lambda a: loads[a]); lanes[j].append(c['circuit_id']); loads[j] += c['estimated_work']
    sub = dict(version='N45_ALL4_FIXED_FIT_AND_NAIVE_L1_1000_DRAWS_V1', cells=cells, circuits=circuits, GPU_lanes=lanes,
               code_SHA256=sha256(Path(legacy.__file__).read_bytes()),
               worker_SHA256=old['worker_SHA256'], base_SHA256=old['base_SHA256'], plan_SHA256=old['plan_SHA256'],
               shots_per_draw=1000, maximum_new_model_shots=24000, maximum_logical_model_shots=24000,
               maximum_GPU_workers=2, GPU_wall_cap_seconds_per_lane=3600, maximum_GPU_hours=2,
               replicate_labels=list(REPLICATES), new_QPU_shots=0, selection='ALL8_FIXED_FIT_NAIVE_NO_OUTCOME_SELECTION',
               readouts=['R32_CAP', 'COMMON_R_ALL_STRONG_CONTROLS'], independent_seed_per_distinct_circuit=True)
    legacy.init_ledger(SAMPLING/'draw-ledger.sqlite', cells, circuits); save(SAMPLING/'protocol.json', sub)
    p = dict(version='N45_FIXED_FIT_NAIVE_FINITE_V1', sources=sources, physical={i['source']['case']: i['physical'] for i in ip['catalog']},
             sampling_protocol_SHA256=sha256((SAMPLING/'protocol.json').read_bytes()), helpers=helpers(),
             N44_protocol_SHA256=sha256((init.RUN/'protocol.json').read_bytes()),
             N44_terminal_SHA256=sha256((init.RUN/'terminal.json').read_bytes()), control_inventory=oldrows,
             control_record_digests={k: digest(v) for k, v in oldrecords.items()},
             previous_readout_terminals={part: sha256((previous_read.LOCAL/(part+'-terminal.json')).read_bytes()) for part in ('M4', 'A100_INPUTS')},
             prepare_CPU_seconds=process_time()-cpu, prepare_wall_seconds=perf_counter()-wall,
             new_run_cap_bytes=32*2**20, new_model_shots_cap=24000, new_QPU_shots=0, new_optimizer_calls=0,
             expected_CUDA_cupy='13.6.0', library=load(n29.LOCAL/'protocol.json')['library'],
             library_SHA256=load(n29.LOCAL/'protocol.json')['library_SHA256'], storage_before=storage,
             plan_SHA256=sha256((ROOT/'analysis/N45_FIXED_FIT_FINITE_NEXT_JA.md').read_bytes()),
             old_draw_reuse_not_new_confirmation=True, cold_matched_work=False, goal_complete=False)
    save(LOCAL/'protocol.json', p)
    print('N45_FROZEN_ALL4_24_DRAWS_24000_MODEL_SAMPLES_17_ACTUAL_3_MISSING_CONTROLS')


def verified():
    init.verified(); p = load(LOCAL/'protocol.json')
    if p['helpers'] != helpers() or p['sampling_protocol_SHA256'] != sha256((SAMPLING/'protocol.json').read_bytes()):
        raise ValueError('Frozen helper/executor/input drift')
    if p['N44_terminal_SHA256'] != sha256((init.RUN/'terminal.json').read_bytes()) or p['N44_protocol_SHA256'] != sha256((init.RUN/'protocol.json').read_bytes()):
        raise ValueError('Source provenance drift')
    if p['control_inventory'] != control_inventory():
        raise ValueError('Control roster changed')
    if any(sha256((previous_read.LOCAL/(part+'-terminal.json')).read_bytes()) != v for part, v in p['previous_readout_terminals'].items()):
        raise ValueError('Old terminal changed')
    for s in p['sources']:
        for f in s['files'].values():
            if sha256(Path(f['path']).read_bytes()) != f['SHA256']:
                raise ValueError('Selected physical source changed')
    configured().verified_protocol()
    return p


def m4(): verified(); configured().m4()
def submit(): verified(); configured().submit()
def status(): verified(); configured().status()
def collect():
    verified(); configured()
    from . import collect_analytical_single_session as c
    c.s = sys.modules[__name__]  # New adapter/namespace only; frozen collector/worker bytes unchanged.
    c.run()


if __name__ == '__main__':
    {'prepare': prepare, 'm4': m4, 'submit': submit, 'status': status, 'collect': collect}[sys.argv[1]]()
