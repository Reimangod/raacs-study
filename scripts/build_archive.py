#!/usr/bin/env python3
"""Build a non-destructive, content-addressed research snapshot from saved files.

No research runner is imported. No molecular sampling/optimization is performed.
Only this new repository and its ignored artifacts directory are written.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"))
from raacs_study.evidence import canonical, normalize, read_csv, sha256_file, write_csv, write_json

ROOT_NAMES=(
    "raacs-frontier-research-loop-2026-10-25",
    "raacs-practical-frontier-v2-2026-10-25",
    "raacs-lucj-sqdrift-paper-aligned-plan-2026-10-25",
    "paper-aligned-confirmation-2026-10-28",
    "b4-small-molecule-comparison-2026-10-28-v1",
    "b4-lih15-from-hf-2026-10-28-v1",
    "raacs-comparison-expanded-2026-10-25",
)
SKIP_DIRS={".git","runtime",".runtime",".venv","venv","__pycache__","node_modules",
           ".pytest_cache",".mpl-cache",".uv-cache","site-packages"}
SKIP_NAMES={".DS_Store","B4_small_molecule_comparison.zip","gpu-inputs.tar.gz","gpu-results.tar.gz"}
SMALL_CASES=("LiH_R1.5","LiH_R3.0","BeH2_R1.3","BeH2_R2.5","H6_R1.4","H6_R2.2")
LARGE_CASES=("NAPH20-BLOCK-BOYS","N2-32-R1p2-P2","N2-32-R2p4-P2","FE2S2-40-P2")
TEXT={".json",".jsonl",".csv",".py",".md",".txt",".toml",".yaml",".yml",".sh",".sbatch",".log",".cfg",".ini"}
SECRETS={
    "private_key":re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "github_token":re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws_access_id":re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "openai_key":re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
}


def files_in(root,workspace,excluded):
    for base,dirs,names in os.walk(root,followlinks=False):
        dirs.sort()
        for name in dirs[:]:
            p=Path(base)/name
            if name in SKIP_DIRS or name.startswith(".venv") or p.is_symlink():
                excluded.append({"path":p.relative_to(workspace).as_posix(),"reason":"environment/cache/git or symlink directory"})
                dirs.remove(name)
        for name in sorted(names):
            p=Path(base)/name
            if p.is_symlink() or name in SKIP_NAMES or p.suffix in {".pyc",".pyo",".pem",".key"}or name.endswith(".sqlite-shm")or name==".env"or name.startswith(".env."):
                excluded.append({"path":p.relative_to(workspace).as_posix(),"reason":"cache/transport copy/credential-shaped file/symlink"})
                continue
            yield p


def zip_entry(archive,name,stream,size):
    info=zipfile.ZipInfo(name,date_time=(2026,10,31,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o100644<<16
    with archive.open(info,"w",force_zip64=size>=2**31)as dest:
        shutil.copyfileobj(stream,dest,1024*1024)


def archive_root(workspace,root,destination,excluded,secret_hits,selected_files=None):
    if destination.exists():raise FileExistsError("Refuse to overwrite an archive: "+str(destination))
    pending=destination.with_suffix(".zip.pending")
    if pending.exists():raise FileExistsError("Inspect existing partial capture: "+str(pending))
    inventory=[];blobs=set();source_bytes=0;unique_bytes=0;tick=time.monotonic()
    with zipfile.ZipFile(pending,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True)as archive:
        for i,p in enumerate(selected_files if selected_files is not None else files_in(root,workspace,excluded)):
            before=p.stat()
            digest=sha256_file(p)
            rel=p.relative_to(workspace).as_posix()
            if p.suffix.lower()in TEXT and before.st_size<=5*1024**2:
                raw=p.read_bytes()
                for kind,pattern in SECRETS.items():
                    if pattern.search(raw):secret_hits.append({"path":rel,"kind":kind,"sha256":digest})
            blob="blobs/"+digest[:2]+"/"+digest
            if digest not in blobs:
                with p.open("rb")as stream:zip_entry(archive,blob,stream,before.st_size)
                blobs.add(digest);unique_bytes+=before.st_size
            after=p.stat()
            if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError("Source changed during capture: "+rel)
            inventory.append({"source_relative_path":rel,"bytes":before.st_size,"sha256":digest,"blob":blob})
            source_bytes+=before.st_size
            if(i+1)%10000==0:print(json.dumps({"capture":root.name,"files":i+1,"raw_MiB":round(source_bytes/1024**2,1)}),flush=True)
        payload=b"".join(canonical(r)+b"\n"for r in inventory)
        zip_entry(archive,"inventory.jsonl",io.BytesIO(payload),len(payload))
        notes=canonical({"schema":"raacs-evidence-cas-1","logical_root":root.relative_to(workspace).as_posix(),
                         "original_files":len(inventory),"unique_blobs":len(blobs),"original_bytes":source_bytes,
                         "unique_original_bytes":unique_bytes,"lossless":True,"paths_are_workspace_relative":True})
        zip_entry(archive,"archive.json",io.BytesIO(notes),len(notes))
    pending.replace(destination)
    report={"name":destination.name,"sha256":sha256_file(destination),"bytes":destination.stat().st_size,
            "logical_root":root.relative_to(workspace).as_posix(),"original_files":len(inventory),
            "unique_blobs":len(blobs),"original_bytes":source_bytes,"unique_original_bytes":unique_bytes,
            "format":"raacs-evidence-cas-1","role":"historical evidence, not independently confirmed endpoints"}
    print(json.dumps({"captured":destination.name,"MiB":round(report["bytes"]/1024**2,2),"files":len(inventory),"seconds":round(time.monotonic()-tick,2)}),flush=True)
    return report,inventory


def copy_original(workspace,path,target,source_entries):
    before=path.stat();target.parent.mkdir(parents=True,exist_ok=True)
    payload=path.read_bytes();after=path.stat()
    if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError("Changing source")
    target.write_bytes(payload)
    source_entries.append({"source_relative_path":path.relative_to(workspace).as_posix(),
                           "destination":target.relative_to(REPO).as_posix(),"bytes":len(payload),
                           "sha256":hashlib.sha256(payload).hexdigest(),"kind":"byte-identical original"})


def collect_sources(workspace,source_entries):
    roots={
        "frontierloop":workspace/"outputs/raacs-frontier-research-loop-2026-10-25/frontierloop",
        "raacsnext":workspace/"outputs/raacs-practical-frontier-v2-2026-10-25/raacsnext",
        "paperstudy":workspace/"outputs/raacs-lucj-sqdrift-paper-aligned-plan-2026-10-25/execution-v1/paperstudy",
        "benchmark":workspace/"ceo-sampler-benchmark/benchmark",
    }
    # Byte-identical package snapshots, not an invented portable rewrite.
    for package,root in roots.items():
        for p in sorted(root.rglob("*.py")):
            if "__pycache__"in p.parts:continue
            copy_original(workspace,p,REPO/"historical"/package/p.relative_to(root),source_entries)
    vendor=workspace/"ceo-sampler-benchmark/vendor/ceo"
    for p in sorted(vendor.rglob("*")):
        if p.is_file()and "__pycache__"not in p.parts and(p.suffix==".py"or p.name in {"LICENSE","NOTICE","COPYING"}):
            copy_original(workspace,p,REPO/"historical/vendor/ceo"/p.relative_to(vendor),source_entries)
    for name in ("protocol-B4.json","freeze-B4.json","run_b4.py"):
        p=workspace/"outputs/raacs-frontier-research-loop-2026-10-25"/name
        if p.exists():copy_original(workspace,p,REPO/"configs/original-B4"/name,source_entries)
    for label,rel in (("small","b4-small-molecule-comparison-2026-10-28-v1/protocol.json"),
                      ("large","paper-aligned-confirmation-2026-10-28/protocol-v1.json"),
                      ("additional-naphthalene","paper-aligned-confirmation-2026-10-28/naphthalene-extension-v1/protocol.json")):
        p=workspace/"outputs"/rel
        copy_original(workspace,p,REPO/"configs"/(label+"-original-protocol.json"),source_entries)
    policy=[]
    frontier=workspace/"outputs/raacs-frontier-research-loop-2026-10-25"
    for p in sorted(frontier.glob("protocol*.json")):
        copy_original(workspace,p,REPO/"configs/exploratory"/p.name,source_entries)
        policy.append({"historical_configuration":p.name,"path":"configs/exploratory/"+p.name,
                       "sha256":sha256_file(p),"role":"exploratory; not fresh independent confirmation"})
    write_json(REPO/"configs/policy-registry.json",{
        "schema":"policy-registry-1","display_alias":{"B4":"RAACS"},
        "not_aliases":["ACTIVE","earlier RAACS variants"],
        "canonical_implementation":"historical/frontierloop/spin_beam.py:SpinBeamEngine",
        "inherited_search":"historical/frontierloop/beam.py:BeamEngine",
        "support_packages":["historical/frontierloop","historical/raacsnext","historical/paperstudy","historical/benchmark"],
        "entrypoint_warning":"Snapshots are for exact source lineage. Historical paths/runners are not the portable CLI.",
        "cohort_protocols":["configs/small-original-protocol.json","configs/large-original-protocol.json","configs/additional-naphthalene-original-protocol.json"],
        "exploratory_configurations":policy,
    })
    # Record static local imports; dynamic dependencies and external native builds
    # are not falsely certified by this static check.
    unresolved=[]
    local_names=set(roots)
    for folder in (REPO/"historical").iterdir():
        for file in folder.glob("*.py"):
            try:tree=ast.parse(file.read_text())
            except SyntaxError:continue
            for node in ast.walk(tree):
                names=[]
                if isinstance(node,ast.Import):names=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom)and node.level==0 and node.module:names=[node.module]
                for name in names:
                    if name.split(".")[0]in local_names:
                        path=REPO/"historical"/Path(*name.split("."))
                        if not path.with_suffix(".py").exists()and not(path/"__init__.py").exists():
                            unresolved.append({"source":file.relative_to(REPO).as_posix(),"import":name})
    write_json(REPO/"data/static-import-audit.json",{"unresolved_local_static_imports":unresolved,
        "scope":"top-level Python packages only; no assertion of full runtime closure or dynamic imports"})


def build_tables(workspace,source_entries):
    small=workspace/"outputs/b4-small-molecule-comparison-2026-10-28-v1"
    old_archive=workspace/"exports/B4_vs_LUCJ_SqDRIFT_experiment_data_2026-10-28.zip"
    tables=[("small-exact-singlet","data/original-tables/small-all_CI_shot_results.csv",(small/"tables/all_CI_shot_results.csv").read_bytes())]
    with zipfile.ZipFile(old_archive)as archive:
        for cohort,src,dest in (("large-85120","tables/existing_85120shots_all_endpoints.csv","large-all-endpoints.csv"),
                                ("naphthalene-additional","tables/new_naphthalene_all_endpoints.csv","naphthalene-all-endpoints.csv")):
            tables.append((cohort,"data/original-tables/"+dest,archive.read(src)))
    rows=[];original=[];lineage=[]
    for cohort,name,payload in tables:
        p=REPO/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
        parsed=read_csv(payload)
        lineage.append({"cohort":cohort,"path":name,"rows":len(parsed),"sha256":hashlib.sha256(payload).hexdigest(),
                        "origin":"saved original table"if cohort=="small-exact-singlet"else"preserved prior export view; numeric rows unchanged"})
        for i,row in enumerate(parsed,1):
            normalized,raw=normalize(row,cohort,name,i);rows.append(normalized);original.append(raw)
    write_csv(REPO/"data/catalog.csv",rows)
    (REPO/"data/original-rows.jsonl").write_bytes(b"".join(canonical(x)+b"\n"for x in original))
    write_json(REPO/"data/table-lineage.json",lineage)
    for p in sorted((small/"tables").glob("*.csv")):
        if p.name=="all_CI_shot_results.csv":continue
        copy_original(workspace,p,REPO/"data/original-tables"/("small-"+p.name),source_entries)
    missing=read_csv((small/"tables/missing_sources.csv").read_bytes())
    write_csv(REPO/"data/missing-sources.csv",[{"cohort":"small-exact-singlet",**r}for r in missing])
    roots=read_csv((small/"tables/unique_CI_roots.csv").read_bytes())
    chosen=[]
    for size in(1,4):
        candidates=[r for r in roots if r["case"]=="LiH_R1.5"and len(json.loads(r["selected_spin_strings"]))==size]
        chosen.append(candidates[0])
    write_json(REPO/"data/fixtures/lih15-saved-bases.json",chosen)
    for name in("hamiltonian.npz","reference.json","reference.npz"):
        # Full reference input is archived; no new FCI computation or use in policy.
        p=workspace/"ceo-sampler-benchmark/results/standard-v1/references/LiH_R1.5"/name
        copy_original(workspace,p,REPO/"data/fixtures"/name,source_entries)
    cases=[]
    for case in SMALL_CASES:
        base=workspace/"ceo-sampler-benchmark/results/standard-v1/references"/case
        p=base/"reference.json";ref=json.loads(p.read_text())
        cases.append({"case":case,"cohort":"small-exact-singlet","n_qubits":ref["n_qubits"],
                      "physical_metadata":ref,"source_relative_path":p.relative_to(workspace).as_posix(),"source_sha256":sha256_file(p),
                      "reference_scope":"saved FCI metadata; this packaging does not calculate new FCI"})
    physical=workspace/"outputs/raacs-lucj-sqdrift-paper-aligned-plan-2026-10-25/execution-v1/data/physical"
    for case in LARGE_CASES:
        p=physical/case/"problem.json";ref=json.loads(p.read_text())
        cases.append({"case":case,"cohort":"large-85120; additional naphthalene separate readout cohort when applicable",
                      "physical_metadata":ref,"source_relative_path":p.relative_to(workspace).as_posix(),"source_sha256":sha256_file(p),
                      "reference_scope":"registered reference; independent reference certification is not implied"})
    write_json(REPO/"data/cases.json",cases)
    population=[]
    src_rows=read_csv((small/"tables/source_and_resource_costs.csv").read_bytes())
    for r in src_rows:
        population.append({"cohort":"small-exact-singlet","case":r["case"],"historical_method_id":r["method"],
                           "replicate":r["replicate"],"source_status_original":r["status"],"readout_availability":"AVAILABLE",
                           "source_directory_original":r["source_directory"]})
    for r in missing:
        population.append({"cohort":"small-exact-singlet","case":r["case"],"historical_method_id":r["method"],
                           "replicate":r["replicate"],"source_status_original":r["status"],"readout_availability":"MISSING",
                           "source_directory_original":""})
    write_csv(REPO/"data/source-population.csv",population)
    write_json(REPO/"data/coverage.json",{
        "endpoint_rows":len(rows),"endpoint_rows_by_cohort":dict(Counter(r["cohort"]for r in rows)),
        "physical_conditions":len(cases),"small_planned_source_replicates":len(population),
        "small_available_source_replicates":len(src_rows),"small_missing_source_replicates":len(missing),
        "recorded_replicate_labels_by_cohort":{c:sorted({r["replicate"]for r in rows if r["cohort"]==c})for c in sorted({r["cohort"]for r in rows})},
        "units_warning":"Endpoint rows, seed labels, distinct structures, model samples and CI roots are different units.",
        "missingness_warning":"small missing source replicates are listed separately, not invented as zero-energy endpoints",
        "new_model_samples":0,"new_QPU_shots":0,"new_FCI_reference_computations":0,
    })


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--metadata-only",action="store_true");parser.add_argument("--archives-only",action="store_true")
    args=parser.parse_args();workspace=args.workspace.resolve()
    if REPO.is_relative_to(workspace/"outputs"):raise RuntimeError("New archive must not be a captured output subtree")
    if not args.archives_only:
        source_entries=[]
        collect_sources(workspace,source_entries)
        build_tables(workspace,source_entries)
        write_json(REPO/"data/copied-source-manifest.json",source_entries)
        print(json.dumps({"metadata_built":True,"copied_originals":len(source_entries)}),flush=True)
    if not args.metadata_only:
        out=REPO/"artifacts";out.mkdir(exist_ok=True)
        excluded=[];secret_hits=[];archives=[];all_inventory=[]
        for name in ROOT_NAMES:
            root=workspace/"outputs"/name
            report,inventory=archive_root(workspace,root,out/(name+".zip"),excluded,secret_hits)
            archives.append(report)
            all_inventory.extend({"archive":report["name"],**r}for r in inventory)
        # Physical reference files are outside the study roots; they are inputs,
        # not an invitation to publish all unrelated benchmark outcomes.
        refroot=workspace/"ceo-sampler-benchmark/results/standard-v1/references"
        selected=[]
        for case in SMALL_CASES:
            src=refroot/case
            for p in sorted(src.iterdir()):
                if p.is_file()and p.suffix in {".json",".npz",".csv",".parquet"}:
                    selected.append(p)
        report,inventory=archive_root(workspace,refroot,out/"small-reference-inputs.zip",excluded,secret_hits,selected)
        report["logical_root"]="ceo-sampler-benchmark/results/standard-v1/references (six selected cases)"
        archives.append(report)
        all_inventory.extend({"archive":report["name"],**r}for r in inventory)
        write_json(REPO/"data/archive-manifest.json",{
            "schema":"raacs-research-archives-1","publication_status":"PRIVATE_GITHUB_DELIVERY_NO_ZENODO_DOI",
            "original_workspace_prefix":str(workspace),
            "archives":archives,"original_file_count":sum(x["original_files"]for x in archives),
            "original_bytes":sum(x["original_bytes"]for x in archives),"archive_bytes":sum(x["bytes"]for x in archives),
            "source_files_modified":False,"new_molecular_sampling":False,
            "remote_download_command_template":"gh release download data-v0.1.0 --repo Reimangod/raacs-study --dir data/downloads",
            "zenodo_version_doi":None,
        })
        packed=b"".join(canonical(r)+b"\n"for r in all_inventory)
        # gzip mtime=0 makes the inventory reproducible.
        (REPO/"data/source-inventory.jsonl.gz").write_bytes(gzip.compress(packed,compresslevel=6,mtime=0))
        write_json(REPO/"data/capture-exclusions.json",excluded)
        write_json(REPO/"data/credential-pattern-scan.json",{
            "scope":"UTF-8/text-shaped files <=5MiB; source archive capture only; patterns, not a full security audit",
            "patterns":list(SECRETS),"hits":secret_hits,"values_never_recorded":True,
        })
        print(json.dumps({"archives_built":len(archives),"original_files":len(all_inventory),
                          "archive_MiB":round(sum(x["bytes"]for x in archives)/1024**2,2),"secret_pattern_hits":len(secret_hits)}),flush=True)


if __name__=="__main__":main()
