"""N45 exact pre-outcome completion after a plan-path typo. No input regeneration."""
from pathlib import Path
from time import process_time, perf_counter
import sqlite3
from . import fitted_source_sampling as s
from .io import load, save, sha256, digest
from .paths import ROOT


def run():
    cpu, wall = process_time(), perf_counter()
    if (s.LOCAL/'protocol.json').exists(): raise RuntimeError('No recovery overwrite')
    sub = s.configured().verified_protocol(); init = s.init.verified()
    with sqlite3.connect(s.SAMPLING/'draw-ledger.sqlite') as db:
        states = dict(db.execute('SELECT status,count(*) FROM draws GROUP BY status'))
        actual = db.execute('SELECT coalesce(sum(actual_shots),0) FROM draws').fetchone()[0]
    if states != {'REGISTERED': 24} or actual != 0 or list(s.SAMPLING.glob('lane*-job.json')):
        raise ValueError('Only exact pre-outcome/pre-dispatch recovery is allowed')
    rows = s.control_inventory(); records = s.control_records(rows); materialized = load(s.LOCAL/'profile-construction.json')
    sources = materialized['rows']
    if len(sources) != 8 or len(sub['circuits']) != 8 or len(sub['cells']) != 24: raise ValueError('Exact partial roster differs')
    for c in sub['circuits']:
        row = next(r for r in sources if (r['case'], r['method_id']) == (c['case'], c['source']))
        if row['problem_id'] != c['problem_id'] or row['files']['logical.qpy']['SHA256'] != c['original_QPY_SHA256']:
            raise ValueError('Existing source/plan binding differs')
        for f in row['files'].values():
            if sha256(Path(f['path']).read_bytes()) != f['SHA256']: raise ValueError('Staged source changed')
    save(s.LOCAL/'preparation-path-recovery.json', dict(status='EXACT_PRE_OUTCOME_PLAN_PATH_REPAIR',
         failed_code_SHA256=sha256((s.LOCAL/'prepare-v1-code-snapshot.py').read_bytes()),
         corrected_code_SHA256=sha256(Path(s.__file__).read_bytes()),
         reason='N45_FIXED_FIT_NAIVE_FINITE_TEST_NEXT_JA.md typo; canonical N45_FIXED_FIT_FINITE_NEXT_JA.md',
         existing_input_regeneration=0, new_model_samples=0, pre_repair_draw_states=states,
         old_profile_construction_CPU_seconds=materialized['CPU_seconds'],
         failed_full_preparation_CPU_unknown_not_zero=True, recovery_code_SHA256=sha256(Path(__file__).read_bytes())))
    p = dict(version='N45_FIXED_FIT_NAIVE_FINITE_V1_WITH_PRE_OUTCOME_PATH_REPAIR', sources=sources,
         physical={i['source']['case']: i['physical'] for i in init['catalog']},
         sampling_protocol_SHA256=sha256((s.SAMPLING/'protocol.json').read_bytes()), helpers=s.helpers(),
         N44_protocol_SHA256=sha256((s.init.RUN/'protocol.json').read_bytes()),
         N44_terminal_SHA256=sha256((s.init.RUN/'terminal.json').read_bytes()), control_inventory=rows,
         control_record_digests={k: digest(v) for k, v in records.items()},
         previous_readout_terminals={part: sha256((s.previous_read.LOCAL/(part+'-terminal.json')).read_bytes()) for part in ('M4', 'A100_INPUTS')},
         prepare_CPU_seconds=None, prepare_wall_seconds=None,
         prepare_cost_note='ORIGINAL_FAILURE_TOTAL_UNKNOWN_NOT_ZERO;profile_construction_and_recovery_recorded_separately',
         profile_construction_CPU_seconds=materialized['CPU_seconds'],
         new_run_cap_bytes=32*2**20, new_model_shots_cap=24000, new_QPU_shots=0, new_optimizer_calls=0,
         expected_CUDA_cupy='13.6.0', library=load(s.n29.LOCAL/'protocol.json')['library'],
         library_SHA256=load(s.n29.LOCAL/'protocol.json')['library_SHA256'], storage_before=s.legacy.capacity(),
         plan_SHA256=sha256((ROOT/'analysis/N45_FIXED_FIT_FINITE_NEXT_JA.md').read_bytes()),
         old_draw_reuse_not_new_confirmation=True, cold_matched_work=False, goal_complete=False,
         preparation_recovery_SHA256=sha256((s.LOCAL/'preparation-path-recovery.json').read_bytes()),
         recovery_CPU_seconds=process_time()-cpu, recovery_wall_seconds=perf_counter()-wall)
    save(s.LOCAL/'protocol.json', p); s.verified()
    print('N45 exact existing8sources24registered draws recovered; regeneration0/outcomes0')


if __name__ == '__main__': run()
