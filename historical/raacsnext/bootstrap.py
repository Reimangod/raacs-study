import sqlite3
from time import time
from .paths import ROOT
from .io import save, load, digest, sha256, space_check, verify_previous_completion


def initialize():
    protocol = load(ROOT/'protocol.json')
    if len(protocol['cases'])*(len(protocol['variants'])+len(protocol['controls']))*len(protocol['pilot_seeds']) != 168:
        raise ValueError('Pilot scope mismatch')
    space_check()
    previous = verify_previous_completion()
    state = ROOT/'campaign-start.json'
    if not state.exists():
        started = time()
        save(state, {'started_unix': started, 'pilot_deadline_unix': started+86400,
            'protocol_SHA256': sha256((ROOT/'protocol.json').read_bytes()),
            'previous_evidence': previous, 'actual_QPU_shots': 0,
            'no_new_human_approval_gate': True})
    elif load(state)['protocol_SHA256'] != sha256((ROOT/'protocol.json').read_bytes()):
        raise RuntimeError('Frozen protocol changed')
    with sqlite3.connect(ROOT/'queue.sqlite') as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS cells(
            id TEXT PRIMARY KEY,stage TEXT,case_id TEXT,method TEXT,budget INTEGER,
            seed INTEGER,status TEXT,reason TEXT,result_path TEXT);
            CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY,token TEXT,pid INTEGER,started REAL);
        ''')
        for case in protocol['cases']:
            for method in protocol['variants']+protocol['controls']:
                for seed in protocol['pilot_seeds']:
                    key = digest(['RAACS-practical-v2','DEVELOPMENT',case,method,100000,seed])
                    db.execute('INSERT OR IGNORE INTO cells VALUES(?,?,?,?,?,?,?,?,?)',
                        (key,'DEVELOPMENT',case,method,100000,seed,'NOT_STARTED',None,None))
        count = db.execute('SELECT COUNT(*) FROM cells').fetchone()[0]
    save(ROOT/'S0-setup.json', {'planned_pilot_cells': count,'protocol_SHA256':sha256((ROOT/'protocol.json').read_bytes()),
        'old_evidence':previous,'free_bytes':space_check(),'new_quantum_evaluations':0,
        'setup_is_not_completed_research':True})
    print({'planned_cells':count,'old_numeric_and_kernel_evidence_unchanged':True,
        'preexisting_documentary_changes':previous['documentary_changes_since_old_completion']},flush=True)


if __name__=='__main__':
    initialize()
