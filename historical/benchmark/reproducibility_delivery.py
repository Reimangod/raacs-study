"""Compact code snapshot and streaming-hashed index, not a raw-data duplicate.

No numerical experiment, sampling, optimizer or source-artifact mutation.
The local dataset tree remains necessary; the archive is not a complete data copy.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
from io import BytesIO,TextIOWrapper
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from time import perf_counter
from .common import ROOT,save_binary,save_json


def streaming_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def source_paths(root):
    paths=[]
    for name in ('pyproject.toml','uv.lock','AGENTS.md','README.md','STUDY_PROTOCOL.md','THIRD_PARTY_NOTICES.md'):
        paths.append(root/name)
    for name in ('benchmark','vendor','tests','gpu'):
        if (root/name).exists():
            paths.extend(p for p in (root/name).rglob('*') if p.is_file() and not p.is_symlink()
                and '__pycache__' not in p.parts and (p.suffix in ('.py','.sh','.toml') or p.name.startswith(('LICENSE','NOTICE'))))
    paths.extend(p for p in (root/'study-v1').glob('*.md') if p.name not in ('PROGRESS.md','REQUIREMENTS.md'))
    paths.extend(p for p in (root/'dashboard').iterdir() if p.name in (
        'app.js','stronger.js','overview.js','style.css','index.html','README.md') or p.suffix=='.py')
    return sorted(set(paths))


def make_archive(path, files, root):
    # Deterministic metadata; do not preserve user IDs or incidental mtimes.
    with path.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
        for p in files:
            content=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(root)))
            info.size=len(content);info.mode=0o644;info.mtime=0
            tar.addfile(info,BytesIO(content))


def verify_archive(path, manifest, destination):
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers()
        if {m.name for m in members}!=set(manifest):raise ValueError('Code snapshot member mismatch')
        for member in members:
            rel=Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or not member.isfile():raise ValueError('Unsafe archive member')
            raw=tar.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest()!=manifest[member.name]:raise ValueError('Archived code byte mismatch')
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)


def build(output=None):
    root=ROOT;out=Path(output or root/'results/reproducibility-delivery-v1').resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if any(streaming_sha(out/p)!=sha for p,sha in old['outputs'].items()):raise ValueError('Delivery artifact changed')
        return {'status':'VERIFIED_EXISTING_DELIVERY','whole_goal_complete':False}
    if shutil.disk_usage(out).free<2.75*2**30:raise RuntimeError('Registered free-disk floor')
    started=perf_counter();files=source_paths(root)
    code={str(p.relative_to(root)):streaming_sha(p) for p in files}
    save_json(out/'code-files.json',code)
    make_archive(out/'research-code.tar.gz',files,root)
    # Hash every actual regular result artifact, excluding this new package.
    # Directory symlinks are indexed as links, not followed/counted as new data.
    rows=0;total=0;links=0;groups={}
    with (out/'dataset-artifacts.csv.gz').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=1) as zipped:
        with TextIOWrapper(zipped,encoding='utf-8',newline='') as text:
            w=csv.writer(text);w.writerow(['path','kind','size_bytes','sha256','link_target','target_exists'])
            for p in sorted((root/'results').rglob('*')):
                if p==out or out in p.parents:continue
                if p.is_symlink():
                    target=os.readlink(p);w.writerow([str(p.relative_to(root)),'symlink',None,None,target,p.exists()]);links+=1
                elif p.is_file():
                    size=p.stat().st_size;sha=streaming_sha(p);w.writerow([str(p.relative_to(root)),'file',size,sha,'',''])
                    rows+=1;total+=size
                    top=p.relative_to(root/'results').parts[0]
                    g=groups.setdefault(top,{'files':0,'bytes':0});g['files']+=1;g['bytes']+=size
                if rows and rows%5000==0 and shutil.disk_usage(out).free<2.75*2**30:
                    raise RuntimeError('Disk floor during streaming index; partial index retained')
    commands=[]
    with tempfile.TemporaryDirectory(prefix='ceo-code-extraction-check-') as temp:
        extracted=Path(temp);verify_archive(out/'research-code.tar.gz',code,extracted)
        env=dict(os.environ,PYTHONPATH=str(extracted),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        command=[sys.executable,'-c',
            'import benchmark;from pathlib import Path;print(Path(benchmark.__file__).resolve());'
            'import benchmark.fe_s_feasibility,benchmark.resource_adaptive_sampler,benchmark.report,benchmark.plot']
        check=subprocess.run(command,cwd=extracted,env=env,capture_output=True,text=True,timeout=45)
        if check.returncode or str(extracted) not in check.stdout:raise RuntimeError('Extracted-code import check failed '+check.stderr)
        commands.append({'command':command[1:],'exit_code':check.returncode,'source':'EXTRACTED_ARCHIVE_NOT_EDITABLE_INSTALL'})
        for phase,mode in [('reanalysis','quick'),('baselines','standard'),('early_stopping','standard'),('new_sampler','quick'),('scaling','standard')]:
            command=[sys.executable,'-m','benchmark.run','--phase',phase,'--mode',mode,'--help']
            check=subprocess.run(command,cwd=extracted,env=env,capture_output=True,text=True,timeout=45)
            if check.returncode or 'usage:' not in check.stdout:raise RuntimeError('Extracted CLI help failed '+check.stderr)
            commands.append({'command':command[1:],'exit_code':check.returncode,'scope':'ARGUMENT_DISPATCH_ONLY_NO_EXPERIMENT'})
    save_json(out/'extracted-code-validation.json',{'status':'ARCHIVE_BYTES_IMPORTS_AND_FIVE_PHASE_DISPATCH_CHECKED',
        'archive_files':len(files),'commands':commands,'new_experiments':0,'new_eigensolver_calls':0,
        'fresh_dependency_install_performed':False,'dataset_was_not_copied_or_reexecuted':True,
        'scope':'Fresh code extraction using current installed interpreter/dependencies; not a clean-machine/environment reinstallation'})
    index={'dataset_root_relative':'results','regular_files':rows,'regular_bytes':total,'symlinks_not_new_data':links,
        'cohorts':groups,'raw_data_in_archive':False,'requires_existing_dataset_tree':True,
        'all_indexed_regular_files_streaming_SHA256_checked':True,
        'experimental_independence_not_inferred_from_file_count':True,
        'Fe_public_input_redistribution_license':'NOT_ESTABLISHED_DO_NOT_PUBLISH_INPUT',
        'raw_dataset_changed':False,'code_files':len(files),
        'runtime_versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','pyscf','qiskit','ffsim','quimb','qiskit-addon-sqd')},
        'uv_lock_sha256':streaming_sha(root/'uv.lock'),'wall_seconds':perf_counter()-started}
    save_json(out/'dataset-index.json',index)
    readme=f'''# Reproducible research delivery — local, not published

## Start here

- Main15-question/10-metric report and18-figure index: `study-v1/CURRENT_DELIVERY_V3.md`.
- Fe2/Fe4 availability correction and actual Fe2 supplement: `study-v1/CURRENT_DELIVERY_V4.md`.
- Original367-item requirement map: `results/full-objective-audit-2026-10-23-v3/`.
- Interactive original30-case/B1 viewer: `dashboard/CEO_LUCJ_Explorer.html`.

## Contents

`research-code.tar.gz`: {len(files)} source/config/license/protocol files, exact
bytes listed in `code-files.json`. No `.venv`, credentials, model cache, raw
results, downloaded Fe data or third-party remote repository is bundled.

`dataset-artifacts.csv.gz`: streaming SHA256/size index of {rows} regular
artifacts ({total} bytes) already present in the local results tree. {links}
symlinks are links, not independent data or duplicated disk bytes. The count of
files is NOT the count of experiments or independent statistical samples.
`dataset-index.json` gives per-directory inventory and current runtime versions.

**This is a compact code snapshot and dataset locator, not a self-contained
copy of all raw data.** Keep the existing `results/` tree with its original
relative layout. Do not delete it after producing this package. Partial,
failed, capped and historical versions remain indexed, not relabelled success.
Source author Fe integral redistribution permission is unestablished; do not
publish that raw input merely because this local index mentions it.

## Restore / run

Extract code into a separate empty project folder (do not overwrite the active
research). Retain/copy the existing raw dataset tree separately only if storage
permits; do not duplicate it automatically. Install the pinned base runtime:

```bash
uv sync --frozen
uv run python -m pytest benchmark/tests dashboard -q
uv run python -m benchmark.report --scalable-study --current-evidence --mode standard --output results/research-synthesis-standard-v3
uv run python -m benchmark.plot --all --scalable-study --current-evidence --mode standard --output results/current18figure-index-standard-v1
```

Existing content-addressed report/plot calls verify/reuse completed outputs;
do not rerun historical non-read-only renderers as verifiers. The official
SqDRIFT construction path uses its separately pinned Qiskit2.5 SDK overlay,
documented in `study-v1/SQDRIFT_RUNTIME_QUALIFICATION.md`; base environment is
Qiskit2.6. No new overlay install or external service access was performed here.
Historical A100 execution artifacts are data, not permission to launch jobs.

The archive was actually extracted into a fresh temporary folder. Every code
member byte was checked; imports were confirmed to come from the extraction,
and all five phase CLI dispatches exited0 with `--help`. This verifies code
packaging, not a clean-machine dependency installation or rerun of13GiB data.
`extracted-code-validation.json` records that exact limited scope.

No new numerical experiment, optimizer, molecular solve, QPU shot or GPU job was
run for packaging. No upload, GitHub change, publication or author message.
Universal RAACS/CEO advantage, quantum-sampling necessity, historical total
quantum-training cost advantage and large-system ground accuracy remain
unestablished; the research report preserves negative and conditional results.
This package alone is not a whole-goal completion declaration.
'''
    save_binary(out/'README.md',readme.encode())
    outputs={p.name:streaming_sha(p) for p in out.iterdir() if p.is_file()}
    record={'schema':'compact-research-reproducibility-delivery-v1','status':'CODE_ARCHIVED_DATA_INDEXED_AND_EXTRACTION_VERIFIED',
        'script_sha256':streaming_sha(__file__),'outputs':outputs,
        'regular_dataset_files':rows,'regular_dataset_bytes':total,'source_files':len(files),
        'new_experiments':0,'new_eigensolver_calls':0,'hardware_shots':0,'whole_goal_complete':False}
    save_json(out/'manifest.json',record)
    return {'status':record['status'],'source_files':len(files),'dataset_files':rows,'dataset_bytes':total,'whole_goal_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    print(json.dumps(build(p.parse_args().output)))
