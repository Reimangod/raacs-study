# Private import adapter for Apache-2.0 ffsim source, IBM copyright retained
# in reference/N31-ffsim-e28eace. Adapter written for this study, not upstream.
"""Load byte-pinned upstream functions without upgrading installed ffsim.

Only import bindings are adapted. Energy and determinant function bodies are
the upstream AST nodes, unmodified. Unsupported variant parameter builders
raise, rather than pretending to implement those variants.
"""
import ast
import functools
from pathlib import Path
import types
import ffsim
import numpy as np
from .paths import ROOT
from .io import load,sha256

VENDOR=ROOT/'reference/N31-ffsim-e28eace'
COMMIT='e28eace7db0e2a7964adfbf18d5a5f916c49d783'


def verified_sources():
    m=load(VENDOR/'manifest.json')
    if m['commit']!=COMMIT or m['license']!='Apache-2.0':raise RuntimeError('Pinned source/permission context changed')
    total=0
    for name,row in m['files'].items():
        file=VENDOR/name
        if file.parent!=VENDOR or sha256(file.read_bytes())!=row['SHA256'] or file.stat().st_size!=row['bytes']:raise RuntimeError('Upstream source bytes changed')
        total+=row['bytes']
    if total>2*2**20:raise RuntimeError('Small-source capture cap')
    return m


def function_nodes(filename,names):
    tree=ast.parse((VENDOR/filename).read_text())
    selected={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    if set(selected)!=set(names):raise RuntimeError('Expected explicit upstream helper set')
    return [selected[n] for n in names]


def unsupported(*args,**kwargs):raise NotImplementedError('Only balanced one-layer UCJ is qualified by this adapter')


@functools.cache
def module():
    verified_sources()
    import jax
    import jax.numpy as jnp
    import ffsim.linalg.util as u
    from ffsim.variational.util import validate_interaction_pairs
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_enable_compilation_cache',False)
    mod=types.ModuleType('n31_pinned_ffsim_ucj_energy')
    env=mod.__dict__
    env.update(jax=jax,jnp=jnp,np=np,
        MolecularHamiltonian=ffsim.MolecularHamiltonian,MolecularHamiltonianSpinless=ffsim.MolecularHamiltonianSpinless,
        UCJOpSpinBalanced=ffsim.UCJOpSpinBalanced,UCJOpSpinUnbalanced=ffsim.UCJOpSpinUnbalanced,UCJOpSpinless=ffsim.UCJOpSpinless,
        rotate_one_body_tensor=u.rotate_one_body_tensor,rotate_two_body_tensor=u.rotate_two_body_tensor,
        real_symmetrics_from_parameters_jax=u.real_symmetrics_from_parameters_jax,
        unitary_from_parameters_jax=u.unitary_from_parameters_jax,
        validate_interaction_pairs=validate_interaction_pairs,
        ucj_op_spin_unbalanced_from_parameters_jax=unsupported,ucj_op_spinless_from_parameters_jax=unsupported)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    nodes=function_nodes('linalg_util.py',['all_indices','upper_triangular_indices','real_symmetric_from_parameters_jax'])
    nodes+=function_nodes('ucj_spin_balanced.py',['ucj_op_spin_balanced_from_parameters_jax'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),str(VENDOR/'pinned_helpers'),'exec'),env)
    tree=ast.parse((VENDOR/'ucj_energy.py').read_text())
    skipped=[];body=[]
    allowed={'ffsim.hamiltonians.molecular_hamiltonian','ffsim.linalg.util','ffsim.variational.ucj_spin_balanced',
             'ffsim.variational.ucj_spin_unbalanced','ffsim.variational.ucj_spinless','ffsim.variational.util'}
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module in allowed:
            skipped.append(node.module);continue
        if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('ffsim'):
            raise RuntimeError('Unexpected upstream import; audit instead of broad replacement')
        body.append(node)
    if set(skipped)!=allowed:raise RuntimeError('Expected exact API-import adaptation set')
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(VENDOR/'ucj_energy.py'),'exec'),env)
    mod.adapter_imports=skipped
    return mod


def energy(op,h,g,core,nelec,chunk_size=128):
    if op.n_reps!=1 or tuple(nelec)[0]!=tuple(nelec)[1] or op.norb>20:
        raise ValueError('Registered closed-shell balanced one-layer <=20orbital scope')
    value=module().ucj_energy_spin_balanced(op,ffsim.MolecularHamiltonian(h,g,constant=core),tuple(nelec),chunk_size=chunk_size)
    if not np.isfinite(value):raise FloatingPointError('Nonfinite analytical energy; not a valid result')
    return float(value)


def gradient_function(h,g,core,nelec,pairs,final=True,chunk_size=128):
    if nelec[0]!=nelec[1] or len(h)>20:raise ValueError('Registered closed-shell <=20orbital scope')
    fun=module().ucj_energy_and_grad_func_spin_balanced(ffsim.MolecularHamiltonian(h,g,constant=core),tuple(nelec),
        interaction_pairs=pairs,with_final_orbital_rotation=final,chunk_size=chunk_size)
    def checked(x):
        e,v=fun(x)
        if not np.isfinite(e) or not np.isfinite(v).all():raise FloatingPointError('Nonfinite analytical energy/gradient')
        return float(e),np.asarray(v)
    return checked
