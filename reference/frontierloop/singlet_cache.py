"""Exact S=0 scoring cache for a future sampler policy, no orbital optimization.

Each instance is scoped to fixed physical integrals. Only identical selected
determinant bases share an energy. Histograms still select their own basis.
"""
import numpy as np
import ffsim
from paperstudy.ci import common_strings,interleaved_to_blocked
from .singlet import basis,diagonalize


class SingletEnergyCache:
    def __init__(self,meta,h,g):
        self.meta=meta
        self.ham=ffsim.MolecularHamiltonian(h,g,constant=meta['core_energy'])
        self.spec={'per_CI_wall_cap_seconds':30,'eigenvector_residual_tolerance':1e-5}
        self.cache={}
        self.calls=self.hits=self.matvecs=0

    def energy(self,counts,r):
        m=self.meta
        keys=sorted(counts)
        ids=np.array([interleaved_to_blocked(x,m['norb']) for x in keys],np.uint64)
        values=np.array([counts[x] for x in keys],np.uint64)
        strings,_=common_strings(ids,values,m['norb'],(m['nalpha'],m['nbeta']),r)
        key=tuple(map(int,strings))
        if key in self.cache:
            self.hits+=1
            return self.cache[key]['energy_Ha']
        q,info=basis(strings,m['norb'],(m['nalpha'],m['nbeta']))
        result,_=diagonalize(self.ham,(m['nalpha'],m['nbeta']),strings,q,self.spec)
        result.update(spin_basis=info)
        self.cache[key]=result
        self.calls+=1
        self.matvecs+=result['matvec_calls']
        return result['energy_Ha']
