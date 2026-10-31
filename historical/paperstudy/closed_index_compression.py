"""One-off transparent compression of the offline archive path index.

The archive controller must be stopped; its OS lock is held throughout. This
never applies to the live budget/queue SQLite or to scientific result bytes.
The SQL WAL is checkpointed and closed before the ordinary, byte-identical
filesystem compression. Readers/writers holding the index refuse migration.
"""
import fcntl,json,sqlite3,subprocess
from time import time
from .io import ROOT,save_json
from .filesystem_compression import compress_one


def assert_no_index_users(path):
    result=subprocess.run(['rtk','proxy','lsof','-t',str(path)],text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    if result.returncode not in(0,1):raise RuntimeError('Cannot establish exclusive closed-index ownership')
    if result.stdout.strip():raise RuntimeError('Archive index is still open; do not migrate a live database')


def run(root=ROOT,record_name='closed-archive-index.json'):
    path=root/'raw-archives-v1/path-index.sqlite'
    destination=root/'filesystem-compression-v1'/record_name
    if destination.exists():raise FileExistsError('Storage attempt evidence must use a new record name')
    with(root/'raw-archive-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_no_index_users(path)
        db=sqlite3.connect(path,timeout=10)
        try:
            checkpoint=db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if checkpoint[0]:raise RuntimeError('Archive-index WAL still busy')
            if db.execute('PRAGMA journal_mode=DELETE').fetchone()[0]!='delete':
                raise RuntimeError('Cannot close archive-index WAL before migration')
            columns=db.execute('PRAGMA table_info(members)').fetchall()
            if [x[1]for x in columns]!=['path','bundle','sha256','bytes']:
                raise RuntimeError('Unrecognized archive-index schema')
        finally:db.close()
        assert_no_index_users(path)
        record=compress_one(path,root)
        # Exact byte equality already verifies every indexed row and schema.
        # A new read-only SQL query additionally checks the normal access path.
        db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        try:
            sample=db.execute('SELECT path,bundle,sha256,bytes FROM members LIMIT 1').fetchone()
            if sample is None:raise RuntimeError('Unexpected empty archive index')
        finally:db.close()
        record.update(observed_unix=time(),archive_controller_exclusively_offline=True,
                      live_queue_and_budget_untouched=True,scientific_artifact_bytes_unchanged=True,
                      ordinary_SQL_read_verified=True)
        save_json(destination,record)
        print(json.dumps(record,indent=2),flush=True)
        return record


if __name__=='__main__':run()
