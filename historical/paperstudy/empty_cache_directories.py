"""Prune only empty archived cache leaves of completely closed cases.

No unlink/rmtree is used. Every removed directory name is already recoverable
from a durable archived member path. Nonempty, live, unarchived and unrelated
directories remain untouched. This does not change any scientific file bytes.
"""
import argparse,errno,fcntl,gzip,json,os,resource,shutil,sqlite3,uuid
from time import perf_counter,time
from .io import ROOT,save_json
from .filesystem_compression import file_sha256

CASES={'NAPH20-P1':35,'N2-32-R1p2-P2':30,'N2-32-R2p4-P2':30}


def closed_case(root,case,kind):
    if case not in CASES:raise ValueError('Only registered, closed non-Fe cases may be cleaned')
    db=sqlite3.connect(root/'run-queue.sqlite')
    try:
        for table,expected in(('planned_cells',120),('sensitivity_cells',CASES[case])):
            rows=db.execute('SELECT status,COUNT(*) FROM '+table+' WHERE case_id=? GROUP BY status',(case,)).fetchall()
            if rows!=[('COMPLETED',expected)]:raise RuntimeError('Case still has unfinished scientific work')
    finally:db.close()
    if kind=='rustiq':
        paths=list((root/'rustiq-control-v1/common-readout-resource-joins'/case).glob('*.json'))
        expected={(b,s)for b in(10000,100000,1000000)for s in(101,211,307,401,503,601,701,809,907,1009)}
        records=[json.loads(p.read_text())for p in paths]
        if len(records)!=30 or {(r['budget'],r['seed'])for r in records}!=expected:
            raise RuntimeError('Compiler case still has unfinished work')
        if any(r['case_id']!=case or not r['source_counts_unchanged']or r['new_model_shots']!=0 for r in records):
            raise RuntimeError('Compiler source mismatch')


def clean(case,kind='native',maximum=None,root=ROOT):
    if kind not in('native','rustiq'):raise ValueError('Unknown generated cache kind')
    with(root/'storage-empty-directory-cleanup.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);closed_case(root,case,kind)
        directory=(root/'native-SqDRIFT'/case/'members'if kind=='native'else root/'rustiq-control-v1/members'/case)
        if directory.is_symlink()or root.resolve()not in directory.resolve().parents:raise ValueError('Own regular cache only')
        archive=sqlite3.connect('file:'+str(root/'raw-archives-v1/path-index.sqlite')+'?mode=ro',uri=True)
        output=root/'storage-metadata-v1';output.mkdir(exist_ok=True)
        identity=case+'-'+kind+'-'+uuid.uuid4().hex;listing=output/(identity+'.paths.jsonl.gz')
        tick=perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF);free=shutil.disk_usage(root).free
        removed=nonempty=unarchived=nontarget=0
        try:
            with gzip.open(listing,'wt',encoding='utf-8',compresslevel=6)as stream:
                for entry in os.scandir(directory):
                    if len(entry.name)!=64 or set(entry.name)-set('0123456789abcdef')or not entry.is_dir(follow_symlinks=False):
                        nontarget+=1;continue
                    relative=str(__import__('pathlib').Path(entry.path).relative_to(root));prefix=relative+'/'
                    cursor=archive.execute('SELECT path FROM members WHERE path>=? AND path<? ORDER BY path LIMIT 1',(prefix,prefix+'\uffff'))
                    row=cursor.fetchone();cursor.close()
                    if not row or not row[0].startswith(prefix):unarchived+=1;continue
                    try:os.rmdir(entry.path)
                    except OSError as error:
                        if error.errno in(errno.ENOTEMPTY,errno.EEXIST,errno.ENOENT):nonempty+=1;continue
                        raise
                    stream.write(json.dumps({'empty_directory_removed':relative,'preserved_archived_member_witness':row[0]})+'\n')
                    removed+=1
                    if maximum is not None and removed>=maximum:break
        finally:archive.close()
        now=resource.getrusage(resource.RUSAGE_SELF)
        record={'status':'EMPTY_ARCHIVED_CLOSED_CACHE_DIRECTORIES_PRUNED','case_id':case,'kind':kind,
            'empty_directories_removed':removed,'nonempty_or_concurrently_missing_kept':nonempty,
            'unarchived_names_kept':unarchived,'unrelated_entries_kept':nontarget,
            'files_deleted':0,'scientific_artifacts_counts_archives_and_hashes_changed':False,
            'removed_names_and_archive_witnesses':str(listing.relative_to(root)),'listing_SHA256':file_sha256(listing),
            'new_model_shots':0,'CPU_seconds':now.ru_utime+now.ru_stime-usage.ru_utime-usage.ru_stime,
            'wall_seconds':perf_counter()-tick,'free_bytes_before':free,'free_bytes_after':shutil.disk_usage(root).free,
            'global_free_space_delta_not_uniquely_attributed_to_this_operation':True,'observed_unix':time()}
        save_json(output/(identity+'.json'),record);print(json.dumps(record,indent=2),flush=True);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True,choices=CASES)
    parser.add_argument('--kind',choices=('native','rustiq'),default='native');parser.add_argument('--maximum',type=int)
    args=parser.parse_args();clean(args.case,args.kind,args.maximum)
