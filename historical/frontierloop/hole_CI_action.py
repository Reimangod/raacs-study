"""Exact electron-coordinate H-vector action through existing hole integrals.

The selected determinant basis and physical H are unchanged. This is a
classical backend representation change, not a new ansatz or an approximation.
"""
import numpy as np
from .energy_cache import hole_integrals
from .native_CI_kernel_feasibility import action_inputs as electron_inputs


def action_inputs(h,g,n,ne,strings):
    es=np.array(sorted(strings),dtype=np.int64)
    if ne[0]!=ne[1] or ne[0]<=n/2 or len(set(map(int,es)))!=len(es): raise ValueError('Balanced high-filling unique basis required')
    if n-ne[0]<2: raise ValueError('This Fe timing adapter requires at least two holes per spin')
    if any(int(s).bit_count()!=ne[0] for s in es): raise ValueError('Wrong electron sector')
    hs=np.sort(((1<<n)-1)^es)
    lookup={int(s):i for i,s in enumerate(es)}
    order=np.array([lookup[((1<<n)-1)^int(s)] for s in hs])
    signs=np.array([(-1.)**sum(k for k in range(n) if int(es[i])&(1<<k)) for i in order])
    phase=signs[:,None]*signs[None,:]
    hh,gg,shift=hole_integrals(h,g,0.)
    base,info=electron_inputs(hh,gg,n,(n-ne[0],n-ne[1]),hs)
    def hop(v):
        x=np.asarray(v).reshape(len(es),len(es))
        hx=x[np.ix_(order,order)]*phase
        hy=base(hx.ravel()).reshape(hx.shape)+shift*hx
        y=np.empty_like(x);y[np.ix_(order,order)]=hy*phase
        return y.ravel()
    return hop,dict(info,particle_hole_used=True,
                    electron_nelec=list(ne),solver_hole_nelec=[n-ne[0],n-ne[1]],
                    full_reference_constant_shift_Ha=shift,
                    vector_order='ORIGINAL_SORTED_ELECTRON_CARTESIAN_WITH_PHASE_AND_COMPLEMENT_INTERNAL_ONLY')
