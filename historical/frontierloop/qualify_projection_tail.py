"""Additive verification of ALL G3 members above its original 16q check cap.

Do not rewrite G3 records or enlarge its original cap. A separately recorded
18q/120sec/1MiB qualification budget permits only the previously skipped tail.
No energy, CI, shots, statevector files, or resource-based member selection.
"""
import sqlite3
import json
from pathlib import Path
from time import process_time, perf_counter
from .paths import ROOT
from .io import load, save, sha256, space
from .projection_compile import compile_member
from .frozen_projection_screen import read_selected


def run():
    run_dir = ROOT/'runs/G3-FULLBANK-PROJECTION-v1'
    output = ROOT/'qualification/G3-projection-independent-tail-v1'
    if (output/'started.json').exists():
        raise RuntimeError('Tail qualification already started; no blind retry')
    terminal = load(run_dir/'terminal.json')
    db_path = run_dir/'compiled-members.sqlite'
    if sha256(db_path.read_bytes()) != terminal['database_SHA256']:
        raise RuntimeError('G3 terminal database changed')
    db = sqlite3.connect('file:'+str(db_path)+'?mode=ro', uri=True)
    try:
        pending = [json.loads(x[0]) for x in db.execute('SELECT record FROM members ORDER BY case_id,idx')
                   if json.loads(x[0]).get('independent_probability', {}).get('status') == 'SYMBOLIC_ONLY_ABOVE_FIXED_ACTIVE_CAP']
    finally:
        db.close()
    frozen = load(ROOT/'freeze-G3-FULLBANK-PROJECTION.json')
    plan = {'scope': 'all previously unverified-by-Qiskit G3 rows, no outcome/resource filtering',
        'targets': [{'case': x['case'], 'index': x['index'], 'member_id': x['member_id']} for x in pending],
        'max_active_qubits': 18, 'max_wall_seconds': 120, 'max_saved_bytes': 1048576,
        'source_terminal_SHA256': sha256((run_dir/'terminal.json').read_bytes()),
        'G3_database_SHA256': terminal['database_SHA256'],
        'implementation_SHA256': sha256(Path(__file__).read_bytes()),
        'new_shots': 0, 'new_energy_or_CI': 0}
    save(output/'started.json', plan)
    cpu, wall = process_time(), perf_counter()
    rows = []
    for old in pending:
        space()
        if old['projected_active_qubits'] > 18 or perf_counter()-wall > 120:
            raise RuntimeError('Preserve tail as not checked; fixed qualification bound exceeded')
        source = next(x for x in frozen['sources'] if x['case'] == old['case'])
        member = next(x for x in source['members'] if x['index'] == old['index'])
        if member['member_id'] != old['member_id']:
            raise RuntimeError('Tail source mismatch')
        blobs = read_selected(list(member['artifacts'].values()))
        meta = dict(load(source['meta_path']), case_id=source['case'])
        new, _, _ = compile_member(meta, member, blobs, max_active=18)
        if new['resources'] != old['resources'] or new['policy'] != old['policy']:
            raise RuntimeError('Tail recompilation changed frozen resource policy')
        rows.append({'case': new['case'], 'index': new['index'], 'member_id': new['member_id'],
            'independent_probability': new['independent_probability'], 'resources_and_policy_unchanged': True})
    save(output/'result.json', {'status': 'ALL_PREVIOUSLY_SKIPPED_TAIL_CHECKS_PASSED', 'rows': rows,
        'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall,
        'original_G3_records_unchanged': sha256(db_path.read_bytes()) == terminal['database_SHA256'],
        'new_shots': 0, 'new_energy_or_CI': 0, 'statevector_saved': False})
    print('TAIL_VERIFIED', len(rows), 'members; original G3 unchanged')


if __name__ == '__main__':
    run()
