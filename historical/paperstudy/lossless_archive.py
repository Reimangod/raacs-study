"""Lossless storage compaction of CLOSED generated raw-artifact groups.

Every byte remains retrievable by original path and SHA256 in tar.gz+SQLite.
This is a representation migration, not scientific record/data removal.
No results, final counts, sources, live draws or historical studies are moved.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import os
import sqlite3
import tarfile
from time import sleep,time

from .io import ROOT,save_json,digest,sha256


def index(root=ROOT):
    directory=root/"raw-archives-v1";directory.mkdir(exist_ok=True)
    db=sqlite3.connect(directory/"path-index.sqlite",timeout=30)
    db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS members(path TEXT PRIMARY KEY,bundle TEXT NOT NULL,sha256 TEXT NOT NULL,bytes INTEGER NOT NULL)")
    return db


def read_artifact(relative,root=ROOT):
    path=root/relative
    if path.exists():return path.read_bytes()
    db=index(root);row=db.execute("SELECT bundle,sha256,bytes FROM members WHERE path=?",(str(relative),)).fetchone();db.close()
    if row is None:raise FileNotFoundError(relative)
    with tarfile.open(root/row[0],"r:gz") as archive:
        blob=archive.extractfile(str(relative)).read()
    if len(blob)!=row[2] or sha256(blob)!=row[1]:raise RuntimeError("Archived scientific artifact failed digest")
    return blob


def compact(paths,bundle_name,root=ROOT):
    directory=root/"raw-archives-v1";directory.mkdir(exist_ok=True)
    target=directory/(bundle_name+".tar.gz");manifest=directory/(bundle_name+".json")
    db=index(root)
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if not target.exists() or sha256(target.read_bytes())!=record["archive_sha256"]:raise RuntimeError("Archived bundle changed")
        rows=db.execute("SELECT path,sha256,bytes FROM members WHERE bundle=?",(str(target.relative_to(root)),)).fetchall()
    else:
        rows=[]
        for p in sorted(set(paths)):
            if root not in p.resolve().parents or p.is_symlink() or not p.is_file():raise ValueError("Unsafe archive path")
            rel=str(p.relative_to(root));blob=p.read_bytes();rows.append((rel,sha256(blob),len(blob)))
        if not rows:
            db.close();return None
        temporary=target.with_suffix(".pending")
        with tarfile.open(temporary,"w:gz",compresslevel=6) as archive:
            for relative,expected,size in rows:
                p=root/relative
                if sha256(p.read_bytes())!=expected:raise RuntimeError("Raw file changed during archive creation")
                archive.add(p,arcname=relative,recursive=False)
        with temporary.open("rb") as f:os.fsync(f.fileno())
        # Verify actual decoded bytes, not just member sizes or tar integrity.
        with tarfile.open(temporary,"r:gz") as archive:
            info={m.name:m for m in archive.getmembers()}
            if set(info)!={x[0] for x in rows}:raise RuntimeError("Archive membership mismatch")
            for relative,expected,size in rows:
                blob=archive.extractfile(relative).read()
                if len(blob)!=size or sha256(blob)!=expected:raise RuntimeError("Archive round-trip byte mismatch")
        temporary.replace(target)
        db.execute("BEGIN IMMEDIATE")
        for relative,expected,size in rows:
            previous=db.execute("SELECT sha256,bytes FROM members WHERE path=?",(relative,)).fetchone()
            if previous is not None and previous!=(expected,size):raise RuntimeError("Original path already binds different bytes")
            db.execute("INSERT OR REPLACE INTO members VALUES(?,?,?,?)",(relative,str(target.relative_to(root)),expected,size))
        db.commit()
        record={"status":"VERIFIED_LOSSLESS_REPRESENTATION_MIGRATION","archive":str(target.relative_to(root)),
                "archive_sha256":sha256(target.read_bytes()),"files":len(rows),"original_bytes":sum(x[2] for x in rows),
                "archive_bytes":target.stat().st_size,"all_original_paths_indexed":True,"all_original_bytes_SHA256_verified":True,
                "research_information_discarded":False,"live_or_historical_files_touched":False,"created_unix":time()}
        save_json(manifest,record)
    # Only after durable archive + index + full byte verification. A crash here
    # leaves harmless duplicate representations, not missing scientific data.
    for relative,expected,size in rows:
        p=root/relative
        if p.exists():
            if sha256(p.read_bytes())!=expected:raise RuntimeError("Refuse to replace changed live artifact")
            p.unlink()
    db.close()
    return record


def sweep():
    db=sqlite3.connect(ROOT/"run-queue.sqlite")
    groups=db.execute("SELECT case_id,budget,COUNT(*) FROM planned_cells WHERE method_id='D_SQDRIFT_N20' AND status='COMPLETED' GROUP BY case_id,budget HAVING COUNT(*)=10").fetchall()
    completed=[]
    for case,B,_ in groups:
        name=case+"-D-N20-B"+str(B)
        if (ROOT/"raw-archives-v1"/(name+".json")).exists():continue
        sources=db.execute("SELECT id,result_path FROM planned_cells WHERE case_id=? AND budget=? AND method_id='D_SQDRIFT_N20' AND status='COMPLETED'",(case,B)).fetchall()
        files=[]
        for key,relative in sources:
            if db.execute("SELECT COUNT(*) FROM active_claims WHERE cell=?",(key,)).fetchone()[0]:raise RuntimeError("Active cell may not be archived")
            result=json.loads((ROOT/relative).read_text())
            if result["budget_ledger"]["unknown_or_inflight_draws"] or result["budget_ledger"]["known_actual_shots"]!=B:raise RuntimeError("Unreconciled raw work")
            drawdir=(ROOT/relative).parent/"draws"
            files.extend(p for p in drawdir.iterdir() if p.is_file())
            for member in result["member_records"]:
                directory=ROOT/"native-SqDRIFT"/case/"members"/digest(member["spec"])
                if directory.exists():files.extend(p for p in directory.iterdir() if p.is_file())
        record=compact(files,name)
        if record:completed.append(record);print("LOSSLESS_ARCHIVE",name,record["files"],record["original_bytes"],record["archive_bytes"],flush=True)
    db.close();return completed


def closed_run_sweep():
    """Archive after each closed expensive run, not after a ten-run burst.

    The same lossless byte/index checks apply. This avoids several GiB of
    allocation overhead before a high-shot group of ten can be compacted.
    """
    db=sqlite3.connect(ROOT/"run-queue.sqlite");sources=[]
    sources += db.execute("SELECT id,result_path FROM planned_cells WHERE method_id='D_SQDRIFT_N20' AND budget=1000000 AND status='COMPLETED'").fetchall()
    if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='sensitivity_cells'").fetchone()[0]:
        sources += db.execute("SELECT id,result_path FROM sensitivity_cells WHERE method_id LIKE 'D_%' AND status='COMPLETED'").fetchall()
    if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='basis_cells'").fetchone()[0]:
        sources += db.execute("SELECT id,result_path FROM basis_cells WHERE method_id LIKE 'D_%' AND status='COMPLETED'").fetchall()
    for p in ROOT.glob("common-runs/P1_SETTINGS_BRIDGE_NOT_REPRODUCTION/*/*/*/result.json"):
        sources.append((digest(str(p.relative_to(ROOT))),str(p.relative_to(ROOT))))
    for key,relative in sources:
        name="closed-run-"+key
        if (ROOT/"raw-archives-v1"/(name+".json")).exists():continue
        if db.execute("SELECT COUNT(*) FROM active_claims WHERE cell=?",(key,)).fetchone()[0]:continue
        result=json.loads((ROOT/relative).read_text())
        if result["budget_ledger"]["unknown_or_inflight_draws"]:raise RuntimeError("Unreconciled completed draw group")
        files=[];drawdir=(ROOT/relative).parent/"draws"
        if drawdir.exists():files.extend(p for p in drawdir.iterdir()if p.is_file())
        for member in result["member_records"]:
            directory=ROOT/"native-SqDRIFT"/result["case_id"]/"members"/digest(member["spec"])
            if directory.exists():files.extend(p for p in directory.iterdir()if p.is_file())
        record=compact(files,name)
        if record:print("CLOSED_RUN_LOSSLESS",result["case_id"],result["budget"],result["seed"],record["files"],flush=True)
    db.close()


def closed_compiler_sweep():
    for joined in ROOT.glob("rustiq-control-v1/common-readout-resource-joins/*/*.json"):
        name="Rustiq-closed-"+digest(str(joined.relative_to(ROOT)))
        if(ROOT/"raw-archives-v1"/(name+".json")).exists():continue
        join=json.loads(joined.read_text());source=ROOT/join["source_result"]
        if sha256(source.read_bytes())!=join["source_result_sha256"]:raise RuntimeError("Compiler/source drift before archive")
        result=json.loads(source.read_text());files=[]
        for member in result["member_records"]:
            directory=ROOT/"rustiq-control-v1/members"/join["case_id"]/digest(member["spec"])
            if directory.exists():files.extend(p for p in directory.iterdir()if p.is_file())
        record=compact(files,name)
        if record:print("COMPILER_LOSSLESS",join["case_id"],join["budget"],join["seed"],record["files"],flush=True)


def main(watch=False):
    with (ROOT/"raw-archive-controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            closed_run_sweep()
            closed_compiler_sweep()
            sweep()
            if not watch:return
            db=sqlite3.connect(ROOT/"run-queue.sqlite")
            pending=db.execute("SELECT COUNT(*) FROM planned_cells WHERE method_id='D_SQDRIFT_N20' AND status IN ('NOT_STARTED','RUNNING','WAITING_REMOTE_EXECUTOR')").fetchone()[0]
            if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='sensitivity_cells'").fetchone()[0]:
                pending+=db.execute("SELECT COUNT(*) FROM sensitivity_cells WHERE method_id LIKE 'D_%' AND status IN ('NOT_STARTED','RUNNING')").fetchone()[0]
            if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='basis_cells'").fetchone()[0]:
                pending+=db.execute("SELECT COUNT(*) FROM basis_cells WHERE method_id LIKE 'D_%' AND status IN ('NOT_STARTED','RUNNING')").fetchone()[0]
            db.close()
            if not pending:return
            sleep(45)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");main(p.parse_args().watch)
