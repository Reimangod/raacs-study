from io import BytesIO
import json
from pathlib import Path
import shutil

import numpy as np
from paperstudy.io import atomic_bytes as _atomic, digest, sha256
from .paths import ROOT, OLD


def new_path(path):
    path = Path(path).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError('This study cannot write outside its new root')
    return path


def write_bytes(path, value):
    path = new_path(path)
    _atomic(path, value)


def save(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False)+'\n').encode())


def load(path):
    return json.loads(Path(path).read_text())


def array_bytes(**arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def space_check(required=0):
    free = shutil.disk_usage(ROOT).free
    if free < 3*2**30 + required:
        raise OSError('Disk safety boundary: no research/cache deletion; free='+str(free))
    return free


def verify_previous_completion():
    record = load(OLD/'FINAL_COMPLETION.json')
    if not record['whole_experiment_complete']:
        raise RuntimeError('Prior scope not closed')
    bad = []
    documentary = []
    for p,h in record['evidence_SHA256'].items():
        blob = (OLD/p).read_bytes()
        if sha256(blob) == h:
            continue
        # A later comparison-report link was added outside this new study.
        # Prove that this is the ONLY byte change; never rewrite old evidence.
        if p == 'analysis-v1/README_JA.md':
            lines = blob.splitlines(keepends=True)
            additions = [x for x in lines if x.startswith('- [拡張比較表：'.encode())]
            stripped = b''.join(x for x in lines if x not in additions)
            if len(additions) == 1 and sha256(stripped) == h:
                documentary.append({'path':p,'old_SHA256':h,'current_SHA256':sha256(blob),
                    'change':'One additive link to table-comparison-v2/REPORT_JA.md',
                    'old_content_reconstructed_SHA256':sha256(stripped),
                    'numerical_or_scientific_change':False,'old_file_modified_by_this_study':False})
                continue
        bad.append(p)
    if bad:
        raise RuntimeError('Prior frozen evidence changed: '+str(bad))
    for p,h in record['frozen_scientific_file_SHA256'].items():
        if sha256(Path(p).read_bytes()) != h:
            raise RuntimeError('Prior scientific kernel changed: '+p)
    return {'completion_SHA256': sha256((OLD/'FINAL_COMPLETION.json').read_bytes()),
        'bound_files_checked': len(record['evidence_SHA256']), 'scientific_mismatches': bad,
        'documentary_changes_since_old_completion':documentary}
