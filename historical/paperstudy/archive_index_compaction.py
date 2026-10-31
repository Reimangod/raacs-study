"""Offline layout-only reduction of the rebuildable archive path index.

WITHOUT ROWID removes SQLite's second physical copy of every long path key.
All four public columns and every row are retained exactly. Raw archives,
manifests, scientific artifacts and their hashes are not changed.
"""
import fcntl,hashlib,json,os,resource,sqlite3,uuid
from time import perf_counter,time
from .io import ROOT,save_json
from .filesystem_compression import file_sha256
from .closed_index_compression import assert_no_index_users

SQL='''
CREATE TABLE archive_bundles(id INTEGER PRIMARY KEY,path TEXT NOT NULL UNIQUE);
CREATE TABLE archived_path_rows(path TEXT PRIMARY KEY,bundle_id INTEGER NOT NULL REFERENCES archive_bundles(id),checksum BLOB NOT NULL,nbytes INTEGER NOT NULL) WITHOUT ROWID;
CREATE VIEW members AS SELECT p.path,b.path AS bundle,
CASE WHEN typeof(p.checksum)='blob' THEN lower(hex(p.checksum)) ELSE p.checksum END AS sha256,
p.nbytes AS bytes FROM archived_path_rows p JOIN archive_bundles b ON b.id=p.bundle_id;
CREATE TRIGGER members_insert INSTEAD OF INSERT ON members BEGIN
INSERT INTO archive_bundles(path) SELECT NEW.bundle WHERE NOT EXISTS(SELECT 1 FROM archive_bundles WHERE path=NEW.bundle);
INSERT OR REPLACE INTO archived_path_rows VALUES(NEW.path,(SELECT id FROM archive_bundles WHERE path=NEW.bundle),NEW.sha256,NEW.bytes);
END;
'''


def rows_digest(db):
    value=hashlib.sha256();count=0
    for row in db.execute('SELECT path,bundle,sha256,bytes FROM members ORDER BY path'):
        value.update(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode()+b'\n');count+=1
    return count,value.hexdigest()


def remove_redundant_secondary(path,root):
    db=sqlite3.connect(path)
    try:
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='archived_bundle_lookup'").fetchone()
        if not exists:return {'status':'ALREADY_COMPACTED'}
        tick=perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF)
        before=path.stat();left=rows_digest(db)
        # WITHOUT ROWID secondary indexes repeat the full primary path. Public
        # path lookup keeps its primary index; infrequent bundle enumeration
        # instead scans the table. No logical row or constraint is removed.
        db.execute('DROP INDEX archived_bundle_lookup');db.commit();db.execute('VACUUM')
        right=rows_digest(db)
        if left!=right:raise RuntimeError('Index-layout row preservation failed')
        after=path.stat();now=resource.getrusage(resource.RUSAGE_SELF)
        record={'status':'EVERY_ROW_VERIFIED_REDUNDANT_SECONDARY_REMOVAL','rows':left[0],
            'ordered_all_rows_SHA256_before_and_after':left[1],
            'allocated_bytes_before':before.st_blocks*512,'allocated_bytes_after':after.st_blocks*512,
            'allocated_bytes_reduced':(before.st_blocks-after.st_blocks)*512,
            'public_primary_path_lookup_and_all_rows_retained':True,
            'scientific_artifacts_archives_manifests_and_hashes_unchanged':True,'new_model_shots':0,
            'wall_seconds':perf_counter()-tick,'CPU_seconds':now.ru_utime+now.ru_stime-usage.ru_utime-usage.ru_stime}
    finally:db.close()
    save_json(root/'filesystem-compression-v1/archive-index-redundant-secondary-removal.json',record)
    print(json.dumps(record,indent=2),flush=True);return record


def run(root=ROOT):
    path=root/'raw-archives-v1/path-index.sqlite'
    temporary=path.with_name('.compact-index-'+uuid.uuid4().hex+'.sqlite')
    tick=perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF)
    with(root/'raw-archive-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_no_index_users(path)
        db=sqlite3.connect(path,timeout=10)
        try:
            if db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]:raise RuntimeError('Archive WAL busy')
            if db.execute('PRAGMA journal_mode=DELETE').fetchone()[0]!='delete':raise RuntimeError('Archive WAL not closed')
            schema=db.execute("SELECT sql FROM sqlite_master WHERE name='members'").fetchone()[0]
        finally:db.close()
        if 'archived_path_rows'in schema:return remove_redundant_secondary(path,root)
        before=path.stat();source_hash=file_sha256(path)
        source=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
        target=sqlite3.connect(temporary)
        try:
            target.execute('PRAGMA cache_size=-4096');target.executescript(SQL)
            bundles={}
            cursor=source.execute('SELECT path,bundle,sha256,bytes FROM members ORDER BY path')
            while True:
                rows=cursor.fetchmany(2048)
                if not rows:break
                converted=[]
                for name,bundle,checksum,nbytes in rows:
                    if bundle not in bundles:
                        bundles[bundle]=len(bundles)+1
                        target.execute('INSERT INTO archive_bundles VALUES(?,?)',(bundles[bundle],bundle))
                    packed=bytes.fromhex(checksum)
                    if len(packed)!=32 or packed.hex()!=checksum:raise RuntimeError('Invalid original SHA256 representation')
                    converted.append((name,bundles[bundle],packed,nbytes))
                target.executemany('INSERT INTO archived_path_rows VALUES(?,?,?,?)',converted)
            target.commit()
            left=rows_digest(source);right=rows_digest(target)
            if left!=right:raise RuntimeError('Every-row archive-index equivalence failed')
            if target.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('New archive index failed SQL check')
        except BaseException:
            target.close();source.close()
            if temporary.exists():temporary.unlink()
            raise
        else:target.close();source.close()
        try:
            assert_no_index_users(path)
            if file_sha256(path)!=source_hash:raise RuntimeError('Archive index changed during offline conversion')
            after=temporary.stat();new_hash=file_sha256(temporary)
            if after.st_blocks>=before.st_blocks:raise RuntimeError('No storage benefit; original index retained')
            record={'status':'EVERY_ROW_VERIFIED_LAYOUT_MIGRATION','path':str(path.relative_to(root)),
                'original_file_SHA256':source_hash,'replacement_file_SHA256':new_hash,
                'rows':left[0],'ordered_all_rows_SHA256_before_and_after':left[1],
                'allocated_bytes_before':before.st_blocks*512,'allocated_bytes_after':after.st_blocks*512,
                'allocated_bytes_reduced':(before.st_blocks-after.st_blocks)*512,
                'public_columns_queries_and_all_logical_values_unchanged':True,
                'layout':'v2: intern repeated bundle strings and pack SHA256 bytes behind compatible members view',
                'scientific_artifacts_archives_manifests_and_hashes_unchanged':True,
                'new_model_shots':0,'observed_unix':time()}
            directory=root/'filesystem-compression-v1';directory.mkdir(exist_ok=True)
            save_json(directory/'archive-index-layout-v2-intent.json',record)
            with temporary.open('rb')as stream:os.fsync(stream.fileno())
            os.replace(temporary,path)
            if file_sha256(path)!=new_hash:raise RuntimeError('Replaced index failed literal-byte check')
            now=resource.getrusage(resource.RUSAGE_SELF)
            record.update(wall_seconds=perf_counter()-tick,CPU_seconds=now.ru_utime+now.ru_stime-usage.ru_utime-usage.ru_stime)
            save_json(directory/'archive-index-layout-v2-completion.json',record)
            print(json.dumps(record,indent=2),flush=True);return record
        finally:
            if temporary.exists():temporary.unlink()


if __name__=='__main__':run()
