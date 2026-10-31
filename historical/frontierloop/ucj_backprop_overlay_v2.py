"""N31 additive API compatibility; no installed ffsim module is patched.

The upstream energy/determinant bodies remain unchanged. The only successor
binding supplies the norb validation keyword absent in installed ffsim 0.0.84.
"""
import functools
import numpy as np
import ffsim
from ffsim.variational.util import validate_interaction_pairs as installed_validate
from . import ucj_backprop_overlay as original

VENDOR=original.VENDOR
COMMIT=original.COMMIT
verified_sources=original.verified_sources


def validate_interaction_pairs(interaction_pairs,ordered,*,norb):
    installed_validate(interaction_pairs,ordered=ordered)
    if interaction_pairs is not None:
        for pair in interaction_pairs:
            if len(pair)!=2 or any(not isinstance(i,(int,np.integer)) or not 0<=i<norb for i in pair):
                raise ValueError('Interaction indices must be integers in the registered orbital range')


@functools.cache
def module():
    # __wrapped__ constructs a NEW private module, not original's cached one.
    mod=original.module.__wrapped__()
    mod.validate_interaction_pairs=validate_interaction_pairs
    return mod


def energy(op,h,g,core,nelec,chunk_size=128):
    if op.n_reps!=1 or tuple(nelec)[0]!=tuple(nelec)[1] or op.norb>20:
        raise ValueError('Registered closed-shell balanced one-layer <=20orbital scope')
    value=module().ucj_energy_spin_balanced(op,ffsim.MolecularHamiltonian(h,g,constant=core),tuple(nelec),chunk_size=chunk_size)
    if not np.isfinite(value):raise FloatingPointError('Nonfinite analytical energy')
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
