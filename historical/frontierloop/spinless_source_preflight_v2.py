"""N34 additive exact spinless gate-definition expansion, not a new ansatz."""
from pathlib import Path
from qiskit import QuantumCircuit
from . import spinless_source_preflight as prior
from .paths import ROOT
from .io import load,save,sha256,space

LOCAL=ROOT/'runs/N34-SPINLESS-L1-PREFLIGHT-v2'
PRIOR=prior.LOCAL
METHOD=prior.METHOD
_BASE_VERIFY=prior.verified


def primitive(q,k):
    # The frozen N16 plan recognizes spin-balanced names, not the SDK's spinless
    # wrapper names. Expand ONLY the actual stored exact definitions to the SAME
    # supported XX+YY/p/cp/X primitives; do not approximate a fermionic rotation.
    names=('slater_spinless_jw','diag_coulomb_spinless_jw','orb_rot_spinless_jw')
    expanded=q.copy()
    for _ in range(3):
        if not any(item.operation.name in names for item in expanded):break
        expanded=expanded.decompose(gates_to_decompose=list(names))
    if any(item.operation.name in names for item in expanded):raise RuntimeError('Unresolved actual spinless gate definitions')
    n=q.num_qubits;pad=QuantumCircuit(2*n);pad.compose(expanded,qubits=list(range(n)),inplace=True)
    module=prior.legacy.qualified_module();cfg,a=module.plan(pad,n,(k,0));v=module.numpy_state(cfg,a)
    return v,cfg,a


def prepare():
    p=_BASE_VERIFY();space();failure=load(PRIOR/'failed.json')
    if failure['fixtures'] or failure['sources'] or 'Initial particle sector differs' not in failure['error']:raise RuntimeError('Expected preserved pre-result wrapper-name failure')
    if LOCAL.exists():raise RuntimeError('Existing additive N34 successor; do not overwrite')
    p=dict(p,version='N34_SPINLESS_L1_EXACT_DEFINITION_EXPANSION_SUCCESSOR_V2',
        predecessor_failure_SHA256=sha256((PRIOR/'failed.json').read_bytes()),
        predecessor_protocol_SHA256=sha256((PRIOR/'protocol.json').read_bytes()),
        successor_code_SHA256=sha256(Path(__file__).read_bytes()),
        only_change='EXPAND_EXACT_SPINLESS_WRAPPER_DEFINITIONS_BEFORE_UNCHANGED_QUALIFIED_CPU_PRIMITIVE_PLAN')
    LOCAL.mkdir(parents=True);save(LOCAL/'protocol.json',p)
    print('N34_V2_FROZEN_EXACT_WRAPPER_EXPANSION_NO_SCIENCE_CHANGE',flush=True)


def verified():
    old=prior.LOCAL;prior.LOCAL=LOCAL
    try:p=_BASE_VERIFY()
    finally:prior.LOCAL=old
    if p['successor_code_SHA256']!=sha256(Path(__file__).read_bytes()):raise RuntimeError('Successor adapter changed')
    if p['predecessor_failure_SHA256']!=sha256((PRIOR/'failed.json').read_bytes()) or p['predecessor_protocol_SHA256']!=sha256((PRIOR/'protocol.json').read_bytes()):raise RuntimeError('Old failure/protocol changed')
    return p


def run():
    verified();prior.LOCAL=LOCAL;prior.primitive=primitive;prior.verified=verified
    prior.run()


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
