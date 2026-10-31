from io import BytesIO
import json
from pathlib import Path
import shutil
import numpy as np
from paperstudy.io import atomic_bytes,digest,sha256
from .paths import ROOT

def safe(path):
    path=Path(path).resolve()
    if ROOT not in path.parents:raise ValueError('New research root only')
    return path

def write(path,blob):atomic_bytes(safe(path),blob)
def save(path,value):write(path,(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def load(path):return json.loads(Path(path).read_text())
def arrays(**a):
    stream=BytesIO();np.savez_compressed(stream,**a);return stream.getvalue()
def space():
    free=shutil.disk_usage(ROOT).free
    if free<3*2**30+2**20:raise OSError('Free disk floor; no automatic deletion')
    return free
