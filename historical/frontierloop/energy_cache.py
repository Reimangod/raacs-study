"""Exact projected-Hamiltonian energy caching and particle-hole contraction.

Cache the selected determinant BASIS, not the raw shot histogram. Histograms
still determine that basis independently; no count-dependent labels are cached.
For high filling, complement the already-selected strings (never inject a new
hole-HF string) and use the exactly transformed Hamiltonian.
"""
from time import perf_counter
import numpy as np
from paperstudy.ci import common_strings,interleaved_to_blocked,CountedSelectedCI


def hole_integrals(h,g,core):
    hh=-h-2*np.einsum('pqrr->pq',g)+np.einsum('prrq->pq',g)
    full=float(core+2*np.trace(h)+2*np.einsum('pprr->',g)-np.einsum('prrp->',g))
    return hh,g,full


def basis_energy(h,g,core,nelec,strings,seconds=30,holes=False):
    n=len(h);s=np.array(strings,np.int64);ne=tuple(nelec)
    if holes:
        h,g,core=hole_integrals(h,g,core);s=np.sort(((1<<n)-1)^s);ne=(n-ne[0],n-ne[1])
    if ne==(0,0):return {'energy_Ha':float(core),'converged':True,'residual_norm':0.,'matvec_calls':0,'holes':holes}
    solver=CountedSelectedCI(perf_counter()+seconds);solver.verbose=0;solver.max_memory=1024
    energy,vector=solver.kernel_fixed_space(h,g,n,ne,(s,s),tol=1e-10,max_cycle=200,max_space=20,
        davidson_only=True,ecore=float(core),verbose=0)
    residual=solver.contract_2e(solver.absorb_h1e(h,g,n,ne,.5),vector,n,ne)-(float(energy)-core)*vector
    norm=float(np.linalg.norm(residual))
    if not solver.converged or not np.isfinite(energy)or norm>1e-5:
        raise RuntimeError('Projected CI energy not numerically qualified')
    return {'energy_Ha':float(energy),'converged':bool(solver.converged),'residual_norm':norm,
        'matvec_calls':solver.matvec_calls,'holes':holes,'selected_physical_dimension':len(s)**2}


class EnergyCache:
    def __init__(self,meta,h,g):
        self.meta=meta;self.h,self.g=h,g;self.cache={};self.calls=self.hits=self.matvecs=0
        self.holes=meta['nalpha']>meta['norb']/2 and meta['nbeta']>meta['norb']/2
    def energy(self,counts,r):
        m=self.meta;ids=np.array([interleaved_to_blocked(x,m['norb'])for x in sorted(counts)],np.uint64)
        values=np.array([counts[x]for x in sorted(counts)],np.uint64)
        strings,_=common_strings(ids,values,m['norb'],(m['nalpha'],m['nbeta']),r)
        key=tuple(map(int,strings))
        if key in self.cache:self.hits+=1;return self.cache[key]['energy_Ha']
        result=basis_energy(self.h,self.g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,holes=self.holes)
        self.cache[key]=result;self.calls+=1;self.matvecs+=result['matvec_calls']
        return result['energy_Ha']
