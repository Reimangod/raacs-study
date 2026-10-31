"""Transparent APFS compression of closed result JSON; content/API unchanged.

This is a storage-layout migration, not numerical reserialization. macOS ditto
is qualified on literal bytes first; every replacement is hash-verified and
atomic. Live data, SQLite, source files, counts and environments are excluded.
"""
import argparse,fcntl,hashlib,json,os,resource,sqlite3,stat,subprocess,sys,uuid
from pathlib import Path
from time import sleep,perf_counter,time
from .io import ROOT,save_json,sha256

COMPRESSED=getattr(stat,'UF_COMPRESSED',0x20)


def file_sha256(path):
    """Bounded-memory literal-byte digest, including large closed artifacts."""
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()


def compress_one(path,root=ROOT):
    if sys.platform!='darwin':raise RuntimeError('Only qualified native macOS operation')
    path=Path(path)
    if path.is_symlink()or root.resolve()not in path.resolve().parents:raise ValueError('Own regular files only')
    before=path.stat()
    if before.st_flags&COMPRESSED:return {'status':'ALREADY_COMPRESSED','path':str(path.relative_to(root))}
    expected=file_sha256(path);temporary=path.with_name('.fscompress-'+uuid.uuid4().hex)
    tick=perf_counter();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        subprocess.run(['rtk','proxy','/usr/bin/ditto','--noclone','--hfsCompression',str(path),str(temporary)],
                       check=True,timeout=300,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if file_sha256(temporary)!=expected:raise RuntimeError('Filesystem compression byte verification failed')
        current=path.stat()
        if(current.st_ino,current.st_size,current.st_mtime_ns)!=(before.st_ino,before.st_size,before.st_mtime_ns)or file_sha256(path)!=expected:
            raise RuntimeError('Source changed during storage migration; refuse replacement')
        after=temporary.stat();gain=(before.st_blocks-after.st_blocks)*512
        record={'status':'COMPRESSED'if gain>0 and after.st_flags&COMPRESSED else'NO_ALLOCATION_BENEFIT_SOURCE_RETAINED',
            'path':str(path.relative_to(root)),'content_SHA256_before_and_after':expected,'logical_bytes_unchanged':before.st_size,
            'allocated_bytes_before':before.st_blocks*512,'allocated_bytes_after':after.st_blocks*512,
            'allocated_byte_reduction_not_guaranteed_global_free_space':max(0,gain),
            'file_name_format_and_normal_reads_unchanged':True,'new_model_shots':0,'numerical_values_changed':False}
        if record['status']=='COMPRESSED':
            with temporary.open('rb')as stream:os.fsync(stream.fileno())
            os.replace(temporary,path)
            if file_sha256(path)!=expected:raise RuntimeError('Post-replacement byte verification failed')
        new=resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(wall_seconds=perf_counter()-tick,
            storage_subprocess_CPU_seconds=new.ru_utime+new.ru_stime-usage.ru_utime-usage.ru_stime)
        return record
    finally:
        if temporary.exists():temporary.unlink()


def completed_sources():
    db=sqlite3.connect(ROOT/'run-queue.sqlite');paths=set()
    for table in('planned_cells','sensitivity_cells','basis_cells'):
        if db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(table,)).fetchone():
            paths.update(ROOT/p for(p,)in db.execute('SELECT result_path FROM '+table+' WHERE status="COMPLETED"AND result_path IS NOT NULL'))
    db.close()
    # TN results are immutable commits after independent final sampling.
    paths.update(ROOT.glob('common-runs/S_TN_INDEPENDENT_FINAL/*/*/*/result.json'))
    return sorted(paths)


def compression_records(directory):
    records=[json.loads(p.read_text())for p in Path(directory).glob('*.json')if p.name!='status.json']
    # The same storage directory also holds layout-only SQL migration evidence.
    # Those records are not literal-byte filesystem-compression records.
    return [r for r in records if 'allocated_byte_reduction_not_guaranteed_global_free_space'in r]


def sweep(maximum=None):
    directory=ROOT/'filesystem-compression-v1';directory.mkdir(exist_ok=True);changed=0
    for path in completed_sources():
        s=path.stat()
        if s.st_size<2**20 or s.st_flags&COMPRESSED:continue
        if json.loads(path.read_text()).get('status')!='COMPLETED':raise RuntimeError('Only closed result records')
        record=compress_one(path);save_json(directory/(sha256(str(path.relative_to(ROOT)).encode())+'.json'),record)
        if record['status']=='COMPRESSED':changed+=1;print('TRANSPARENT_COMPRESSION',record['path'],record['allocated_byte_reduction_not_guaranteed_global_free_space'],flush=True)
        if maximum is not None and changed>=maximum:break
    records=compression_records(directory)
    save_json(directory/'status.json',{'observed_unix':time(),'compressed_files':sum(r['status']=='COMPRESSED'for r in records),
        'verified_allocated_byte_reduction':sum(r['allocated_byte_reduction_not_guaranteed_global_free_space']for r in records),
        'same_file_bytes_paths_formats_and_scientific_hashes':True,'no_research_information_discarded':True})


def main(watch=False,maximum=None):
    with(ROOT/'filesystem-compression-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            sweep(maximum)
            if not watch:return
            db=sqlite3.connect(ROOT/'run-queue.sqlite')
            pending=db.execute('SELECT COUNT(*) FROM planned_cells WHERE status IN("NOT_STARTED","RUNNING","WAITING_REMOTE_EXECUTOR")').fetchone()[0];db.close()
            if not pending:return
            sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--maximum',type=int)
    a=p.parse_args();main(a.watch,a.maximum)
