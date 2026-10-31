"""Add-on archive with exact goal bytes and actual restored report execution.

Reuses the frozen base code archive and local raw datasets, without copying
13GiB or changing a bound numerical source. No clean-machine install claim.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from .common import ROOT,save_binary,save_json
from .reproducibility_delivery import streaming_sha,verify_archive
from .portable_report_entry import OBJECTIVE_SHA256

OWNER_GOAL=Path('/Users/rei/.codex/attachments/b0f3686d-83a1-477c-bc5a-3ec6a43e7830/goal-objective.md')
BASE=ROOT/'results/reproducibility-delivery-v1'


def build(output=None):
    root=Path(output or ROOT/'results/portable-delivery-addon-v1').resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'validation.json').exists():
        old=json.loads((root/'validation.json').read_text())
        if any(streaming_sha(ROOT/p)!=sha for p,sha in old['inputs'].items()):raise ValueError('Portable validation input drift')
        if any(streaming_sha(root/p)!=sha for p,sha in old['outputs'].items()):raise ValueError('Portable outputs drift')
        return {'status':'VERIFIED_EXISTING_PORTABLE_REPORT_PROOF','new_experiments':0}
    if streaming_sha(OWNER_GOAL)!=OBJECTIVE_SHA256:raise ValueError('Owner objective byte identity changed')
    inputs={str(p.relative_to(ROOT)):streaming_sha(p) for p in (
        BASE/'manifest.json',BASE/'research-code.tar.gz',BASE/'code-files.json',
        ROOT/'benchmark/portable_report_entry.py',ROOT/'benchmark/portable_delivery_validation.py')}
    # Generated data artifacts preserve the original absence of a final newline.
    # Markdown authoring tools normally append one; do not normalize goal bytes.
    save_binary(root/'goal-objective.md',OWNER_GOAL.read_bytes())
    addon_files={
        'study-v1/authoritative-goal-input.md':(root/'goal-objective.md').read_bytes(),
        'benchmark/portable_report_entry.py':(ROOT/'benchmark/portable_report_entry.py').read_bytes(),
        'benchmark/portable_delivery_validation.py':(ROOT/'benchmark/portable_delivery_validation.py').read_bytes()}
    with tarfile.open(root/'portable-report-addon.tar.gz','w:gz') as tar:
        for name,raw in addon_files.items():
            entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=0o644;entry.mtime=0
            tar.addfile(entry,BytesIO(raw))
    code=json.loads((BASE/'code-files.json').read_text())
    with tempfile.TemporaryDirectory(prefix='ceo-restored-report-') as temp:
        restored=Path(temp)
        verify_archive(BASE/'research-code.tar.gz',code,restored)
        # Only previously authored add-on bytes, each independently verified.
        import hashlib
        addon_hashes={p:hashlib.sha256(raw).hexdigest() for p,raw in addon_files.items()}
        verify_archive(root/'portable-report-addon.tar.gz',addon_hashes,restored)
        (restored/'results').symlink_to(ROOT/'results',target_is_directory=True)
        # Write only into a new delivery-owned derivative namespace. All actual
        # experiments are read-only inputs; never replace historical snapshots.
        destination=restored/'results'/root.relative_to(ROOT/'results')/'restored-report'
        command=[sys.executable,'-m','benchmark.portable_report_entry',
            '--objective','study-v1/authoritative-goal-input.md','--mode','standard',
            '--output',str(destination)]
        env=dict(os.environ,PYTHONPATH=str(restored),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        # Explicitly deny the old attachment, including stat/open, in this child.
        # This proves it did not accidentally fall back to a host-only goal file.
        launcher='''import os,runpy,sys
from pathlib import Path
base=Path.cwd()
import benchmark
assert Path(benchmark.__file__).resolve().is_relative_to(base)
old='/Users/rei/.codex/attachments/b0f3686d-83a1-477c-bc5a-3ec6a43e7830/goal-objective.md'
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and os.fsdecode(args[0])==old:
        raise RuntimeError('ORIGINAL_OWNER_ATTACHMENT_ACCESS_FORBIDDEN')
sys.addaudithook(guard)
sys.argv=['benchmark.portable_report_entry']+sys.argv[1:]
runpy.run_module('benchmark.portable_report_entry',run_name='__main__')
'''
        actual=[sys.executable,'-c',launcher]+command[3:]
        completed=subprocess.run(actual,cwd=restored,env=env,capture_output=True,text=True,timeout=50)
        save_binary(root/'restored-report.stdout.txt',completed.stdout.encode())
        save_binary(root/'restored-report.stderr.txt',completed.stderr.encode())
        if completed.returncode:raise RuntimeError('Actual restored report failed: '+completed.stderr)
        answer=json.loads(completed.stdout.strip())
        # Source keys are project-relative except original archival dependencies.
        # The new exact goal must be bound under the portable project location.
        snapshots=list((root/'restored-report/snapshots').iterdir())
        if len(snapshots)!=1:raise ValueError('Unexpected portable report namespace')
        snapshot=snapshots[0];proof=json.loads((snapshot/'source-evidence.json').read_text())
        if proof['source_sha256'].get('study-v1/authoritative-goal-input.md')!=OBJECTIVE_SHA256:
            raise ValueError('Report did not bind the exact portable goal input')
        if str(OWNER_GOAL) in proof['source_sha256']:raise ValueError('Report retained forbidden host goal input')
        historical=ROOT/'results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1'
        same=[]
        for name in ('questions.json','primary-metrics.json','HCI-all-planned-same-d.csv',
            'HCI-conditional-summary.csv','workflow-method-summary.csv','workflow-commonR-and-actuald.csv',
            'all-planned-workflows.csv','gate-noise-paired-molecule-contrasts.csv','direct-configuration-comparisons.csv'):
            if (snapshot/name).read_bytes()!=(historical/name).read_bytes():raise ValueError('Portable scientific table changed: '+name)
            same.append(name)
    note='''# Portable reporting add-on — exact goal input, no new experiment

Keep the original base archive and raw dataset tree. Extract
`research-code.tar.gz`, then this `portable-report-addon.tar.gz` into an empty
project directory. The add-on carries the exact original goal bytes at
`study-v1/authoritative-goal-input.md` and a small input-location adapter.
The frozen original report source is not edited. Use:

```bash
uv sync --frozen
uv run python -m benchmark.portable_report_entry --objective study-v1/authoritative-goal-input.md --mode standard --output results/portable-report
```

An actual fresh source extraction was tested using the current installed Python
dependencies and a symlink to the unchanged local data tree (no13GiB copy).
The original absolute goal attachment was forbidden by a Python file-open audit
hook; the report still regenerated and bound the portable input SHA256.
All nine scientific JSON/CSV tables are byte-identical to the existing current
synthesis. Provenance/snapshot identity differs because the input path changed;
the report is not relabelled as the old snapshot. Figures are existing sources,
not rerendered here. No solver, optimizer, new sampling acquisition or GPU job.

This is not a fresh dependency install or full experiment rerun. Dashboard
tests additionally require Node.js (`node` on PATH); `uv sync` does not install
Node. SqDRIFT reconstruction still uses the separately documented pinned SDK
overlay. Raw data, original code archives and prior failed evidence stay intact.
The newer Fe2 feasibility supplement remains separately linked by
`study-v1/CURRENT_DELIVERY_V4.md`; portable synthesis preserves older scientific
tables rather than silently updating their Fe status or overwriting them.
'''
    save_binary(root/'README.md',note.encode())
    outputs={str(p.relative_to(root)):streaming_sha(p) for p in root.rglob('*') if p.is_file()}
    validation={'schema':'portable-report-execution-proof-v1','status':'RESTORED_REPORT_EXECUTED_WITH_ORIGINAL_ATTACHMENT_FORBIDDEN',
        'inputs':inputs,'outputs':outputs,'objective_sha256':OBJECTIVE_SHA256,
        'restored_report_snapshot':str(snapshot.relative_to(ROOT)), 'unchanged_scientific_tables':same,
        'actual_child_exit_code':completed.returncode,'original_goal_file_open_denied':True,
        'fresh_dependency_install':False,'raw_data_copied':False,'new_experiments':0,'new_solver_calls':0,
        'new_FCI_solves':0,'hardware_shots':0,'whole_goal_complete':False}
    save_json(root/'validation.json',validation)
    return {'status':validation['status'],'identical_scientific_tables':len(same),'snapshot':str(snapshot.relative_to(ROOT))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    print(json.dumps(build(p.parse_args().output)))
