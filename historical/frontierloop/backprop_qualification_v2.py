"""Reuse the exact N31 qualification body with additive version-bound adapter.

V1's failure and all byte-pinned code are retained. No scientific fixture,
tolerance, finite-difference method, or cap is changed.
"""
from pathlib import Path
import inspect
import importlib.metadata
import ffsim.linalg.util as util
from . import backprop_qualification as prior
from . import ucj_backprop_overlay as old_adapter
from . import ucj_backprop_overlay_v2 as adapter
from .paths import ROOT
from .io import load,save,sha256,space

LOCAL=ROOT/'qualification/N31-analytical-UCJ-v2'
PRIOR=prior.LOCAL


def code_bindings():
    return {name:sha256(Path(file).read_bytes()) for name,file in dict(
        wrapper=__file__,qualification_body=prior.__file__,adapter_v1=old_adapter.__file__,adapter_v2=adapter.__file__).items()}


def prepare():
    space();adapter.verified_sources()
    if LOCAL.exists():raise RuntimeError('Existing successor: inspect, never overwrite')
    p=prior.verified();failure=load(PRIOR/'failed.json')
    if 'unexpected keyword argument' not in failure['error'] or failure['rows']:
        raise RuntimeError('Expected preserved API-only pre-result failure')
    p=dict(p,version='N31_PINNED_LICENSED_UCJ_BACKPROP_QUALIFICATION_V2',code_bindings=code_bindings(),
        predecessor_failure_SHA256=sha256((PRIOR/'failed.json').read_bytes()),
        predecessor_protocol_SHA256=sha256((PRIOR/'protocol.json').read_bytes()),
        API_change='private validator adds norb range check; scientific bodies/fixtures/caps unchanged')
    LOCAL.mkdir(parents=True);save(LOCAL/'protocol.json',p)
    print('N31_V2_FROZEN_ADDITIVE_API_BINDING_ONLY',flush=True)


def verified():
    p=load(LOCAL/'protocol.json');prior.verified()
    if p['code_bindings']!=code_bindings():raise RuntimeError('Successor source drift')
    if sha256((PRIOR/'failed.json').read_bytes())!=p['predecessor_failure_SHA256']:raise RuntimeError('Predecessor evidence drift')
    if sha256((PRIOR/'protocol.json').read_bytes())!=p['predecessor_protocol_SHA256']:raise RuntimeError('Predecessor protocol drift')
    for name,digest in p['local_helper_SHA256'].items():
        if sha256(inspect.getsource(getattr(util,name)).encode())!=digest:raise RuntimeError('Installed helper drift')
    return p


def run():
    verified()
    # Rebind only this process's qualification module, not library installation.
    prior.LOCAL=LOCAL
    prior.overlay=adapter
    prior.verified=verified_runtime
    prior.run()


def verified_runtime():
    p=load(LOCAL/'protocol.json');adapter.verified_sources()
    if p['code_bindings']!=code_bindings():raise RuntimeError('Qualification source drift')
    if {name:importlib.metadata.version(name) for name in p['packages']}!=p['packages']:raise RuntimeError('Runtime drift')
    for name,digest in p['local_helper_SHA256'].items():
        if sha256(inspect.getsource(getattr(util,name)).encode())!=digest:raise RuntimeError('Installed helper drift')
    return p


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
