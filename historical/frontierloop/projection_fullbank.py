"""All N5 members; compact storage and unchanged outcome-blind guarded policy."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
from time import process_time, perf_counter
import traceback
import xml.etree.ElementTree as ET
from .paths import ROOT, OLD
from .io import load, save, sha256, digest, space
from .frozen_projection_screen import read_selected
from .projection_compile import compile_member

RUN = ROOT/'runs/G3-FULLBANK-PROJECTION-v1'
PROTOCOL = ROOT/'protocol-G3-FULLBANK-PROJECTION.json'
FREEZE = ROOT/'freeze-G3-FULLBANK-PROJECTION.json'


def source_plans():
    prefix = load(ROOT/'freeze-G2-FROZEN-OCCUPATION.json')
    plans = []
    db = sqlite3.connect('file:'+str(OLD/'raw-archives-v1/path-index.sqlite')+'?mode=ro', uri=True)
    try:
        for source in prefix['sources']:
            if source['status'] != 'AVAILABLE':
                plans.append(source)
                continue
            path = Path(source['source_result'])
            if sha256(path.read_bytes()) != source['source_result_SHA256']:
                raise RuntimeError('G2/C5 source changed')
            result = load(path)
            members = []
            for i, row in enumerate(result['member_records']):
                mid = digest(row['spec'])
                if mid != row['member_id']:
                    raise ValueError('Frozen member identity mismatch')
                entries = {}
                for name in ['member.json.gz', 'circuit.qpy.gz']:
                    relative = 'native-SqDRIFT/'+source['case']+'/members/'+mid+'/'+name
                    p = OLD/relative
                    if p.exists():
                        blob = p.read_bytes()
                        entry = {'path': relative, 'bundle': None, 'sha256': sha256(blob), 'bytes': len(blob)}
                    else:
                        e = db.execute('SELECT bundle,sha256,bytes FROM members WHERE path=?', (relative,)).fetchone()
                        if e is None:
                            raise FileNotFoundError(relative)
                        entry = {'path': relative, 'bundle': e[0], 'sha256': e[1], 'bytes': e[2]}
                    entries[name] = entry
                members.append({'index': i, 'member_id': mid, 'spec': row['spec'],
                    'historical_member_shots': row['shots'], 'artifacts': entries})
            if len(members) != 783 or sum(x['historical_member_shots'] for x in members) != 100000:
                raise ValueError('Whole frozen bank/shot weights mismatch')
            counts = path.parent/'final-counts.npz'
            c5 = next(x for x in load(ROOT/'manifest-C5-CONTROLS.json')['cells']
                      if x['case'] == source['case'] and x['method'] == 'D_SQDRIFT_N5')
            if sha256(counts.read_bytes()) != c5['source_counts_SHA256']:
                raise RuntimeError('Whole-bank original final counts changed')
            plans.append({**source, 'members': members, 'source_counts_path': str(counts),
                'source_counts_SHA256': c5['source_counts_SHA256']})
    finally:
        db.close()
    return plans


def binding():
    xml = ROOT/'qualification-projection-fullbank-v1-tests.xml'
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    if sum(int(x.get('tests', '0')) for x in suites) != 12 or any(
            int(x.get(k, '0')) for x in suites for k in ['failures', 'errors', 'skipped']):
        raise RuntimeError('12 projection/policy qualification tests required')
    g2 = load(ROOT/'freeze-G2-FROZEN-OCCUPATION.json')
    if any(version(k) != v for k, v in g2['versions'].items()):
        raise RuntimeError('Pinned G2 dependency versions changed')
    for p in [ROOT/'frontierloop/frozen_projection.py', OLD/'paperstudy/lucj.py', OLD/'paperstudy/sqdrift_execution.py']:
        if sha256(p.read_bytes()) != g2['implementation_SHA256'][str(p)]:
            raise RuntimeError('G2 frozen kernel changed')
    return {'protocol_SHA256': sha256(PROTOCOL.read_bytes()), 'qualifier_SHA256': sha256(xml.read_bytes()),
        'G2_terminal_SHA256': sha256((ROOT/'runs/G2-FROZEN-OCCUPATION-v1/terminal.json').read_bytes()),
        'G2_freeze_SHA256': sha256((ROOT/'freeze-G2-FROZEN-OCCUPATION.json').read_bytes()),
        'R2_comparison_csv_SHA256': sha256((ROOT/'analysis/R2-common-CI-all.csv').read_bytes()),
        'implementation_SHA256': {str(p): sha256(p.read_bytes()) for p in [Path(__file__),
            ROOT/'frontierloop/projection_compile.py', ROOT/'test_projection_compile.py',
            ROOT/'frontierloop/frozen_projection_screen.py']},
        'dependencies': g2['versions'], 'sources': source_plans(), 'new_model_shots': 0}


def run():
    frozen = load(FREEZE)
    if binding() != frozen:
        raise RuntimeError('G3 pre-outcome freeze changed')
    if (RUN/'started.json').exists():
        raise RuntimeError('G3 already started; preserve partial and never blindly restart')
    spec = load(PROTOCOL); space()
    cpu, wall = process_time(), perf_counter()
    save(RUN/'started.json', {'freeze_SHA256': sha256(FREEZE.read_bytes()), 'new_model_shots': 0})
    db = sqlite3.connect(RUN/'compiled-members.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,member_id TEXT,record TEXT,certificate_gz BLOB,qpy_gz BLOB,PRIMARY KEY(case_id,idx))')
    db.commit()
    prefix = {(x['case'], x['index']): x for x in load(ROOT/'runs/G2-FROZEN-OCCUPATION-v1/terminal.json')['rows']
              if x['status'] == 'RESOURCE_PREFLIGHT_COMPLETE'}
    statuses = Counter(); case_reports = []
    try:
        for source in frozen['sources']:
            case = source['case']
            if source['status'] != 'AVAILABLE':
                case_reports.append({'case': case, 'status': source['status']})
                continue
            blobs = read_selected([e for m in source['members'] for e in m['artifacts'].values()])
            meta = dict(load(source['meta_path']), case_id=case)
            if sha256(Path(source['meta_path']).read_bytes()) != source['meta_SHA256'] or meta['problem_id'] != source['problem_id']:
                raise RuntimeError('Physical identity changed')
            case_status = Counter()
            for plan in source['members']:
                space()
                if process_time()-cpu > spec['max_CPU_seconds'] or perf_counter()-wall > spec['max_wall_seconds']:
                    raise TimeoutError('G3 frozen CPU/wall budget; preserve partial')
                if sum(p.stat().st_size for p in RUN.iterdir() if p.is_file()) > spec['max_new_directory_bytes']:
                    raise OSError('G3 compact output cap')
                try:
                    row, certificate, circuit = compile_member(meta, plan, blobs,
                        spec['actual_member_independent_check_max_active_qubits'])
                    previous = prefix.get((case, plan['index']))
                    if previous is not None:
                        for name, key in [('NATIVE', 'native_recount_resources'),
                                          ('RUSTIQ', 'same_layout_Rustiq_resources'), ('PROJECTED', 'projected_resources')]:
                            if row['resources'][name] != previous[key]:
                                raise RuntimeError('G2 prefix all-compiler resource regression')
                        row['G2_prefix_regression'] = 'PASSED'
                except Exception as exc:
                    row = {'status': 'FAILED_ENGINEERING_PRESERVED', 'case': case, 'index': plan['index'],
                        'member_id': plan['member_id'], 'error': repr(exc), 'traceback': traceback.format_exc(),
                        'new_model_shots': 0}
                    certificate, circuit = None, None
                db.execute('INSERT INTO members VALUES(?,?,?,?,?,?)', (case, plan['index'], plan['member_id'],
                    json.dumps(row, sort_keys=True, allow_nan=False), certificate, circuit)); db.commit()
                statuses[row['status']] += 1; case_status[row['status']] += 1
                if (plan['index']+1) % 64 == 0:
                    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                    save(RUN/'progress.json', {'case': case, 'next_index': plan['index']+1,
                        'statuses': dict(statuses), 'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall})
                    print(case, plan['index']+1, dict(statuses), flush=True)
            case_reports.append({'case': case, 'statuses': dict(case_status), 'members': len(source['members'])})
            print(case, 'TERMINAL', dict(case_status), flush=True)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise RuntimeError('Compact results database failed integrity check')
    finally:
        db.close()
    save(RUN/'terminal.json', {'status': 'WHOLE_FROZEN_BANK_RESOURCE_TERMINAL', 'cases': case_reports,
        'statuses': dict(statuses), 'CPU_seconds': process_time()-cpu, 'wall_seconds': perf_counter()-wall,
        'database_SHA256': sha256((RUN/'compiled-members.sqlite').read_bytes()),
        'freeze_SHA256': sha256(FREEZE.read_bytes()), 'new_model_shots': 0,
        'new_energy_evaluations': 0, 'new_CI_evaluations': 0, 'goal_complete': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); args = p.parse_args()
    if args.freeze:
        b = binding()
        if FREEZE.exists() and load(FREEZE) != b:
            raise RuntimeError('Never replace G3 freeze')
        save(FREEZE, b)
        print('G3_FROZEN', sum(len(x.get('members', [])) for x in b['sources']), 'members; no outcomes used')
    else:
        run()
