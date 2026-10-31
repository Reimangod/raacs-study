"""N53 successor: transport exact probe bytes instead of cross-platform RNG replay."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import worker_base


def run(root):
    p=json.loads((root/'protocol.json').read_text())
    lookup={(c['D'],c['dense_seed']):c for c in p['cases']}
    def exact_probes(D,seed):
        c=lookup[(D,seed)];f=root/c['file']
        if hashlib.sha256(f.read_bytes()).hexdigest()!=c['SHA256']:
            raise ValueError('Explicit probe packet digest differs')
        with np.load(f,allow_pickle=False) as a:x=a['explicit_dense_probes']
        if x.shape!=(2,D) or x.dtype!=np.float64:
            raise ValueError('Explicit float64 probe shape differs')
        if hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()!=c['dense_input_SHA256']:
            raise ValueError('Exact CPU oracle input bytes differ')
        return x
    worker_base.dense_probes=exact_probes
    worker_base.run(root)


if __name__=='__main__':run(Path(sys.argv[1]))
