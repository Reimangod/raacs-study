"""Qualified energy-only particle-hole representation of the same S=0 CI space.

Select electron configurations first, then complement exactly those strings.
Do not resample, insert a different HF, or treat hole-RDMs as electron-RDMs.
This module is not wired into frozen B4 or C5.
"""
import numpy as np
import ffsim
from .energy_cache import hole_integrals
from .singlet import basis,diagonalize


def energy(h,g,core,nelec,strings,seconds=30):
    if np.iscomplexobj(h) or np.iscomplexobj(g):
        raise ValueError('Only the qualified real-integral transformation is supported')
    n=len(h)
    hh,gg,cc=hole_integrals(h,g,core)
    hole_strings=np.sort(((1<<n)-1)^np.asarray(strings,dtype=np.int64))
    holes=(n-nelec[0],n-nelec[1])
    q,info=basis(hole_strings,n,holes)
    result,_=diagonalize(ffsim.MolecularHamiltonian(hh,gg,constant=cc),holes,hole_strings,q,
        {'per_CI_wall_cap_seconds':seconds,'eigenvector_residual_tolerance':1e-5})
    return dict(result,spin_basis=info,representation='EXACT_PARTICLE_HOLE_ENERGY_ONLY',
        electron_nelec=list(nelec),hole_nelec=list(holes),new_model_shots=0,
        original_electron_selected_dimension=len(strings)**2,
        RDM_is_in_hole_representation_not_electron_RDM=True)
