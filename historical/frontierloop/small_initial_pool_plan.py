"""Uniform source-only initial-pool selection; no past energies or winners."""
import math
import numpy as np


def choose(ids,values,n,nelec,r=8,coefficient_source=False):
    ids=np.asarray(ids);values=np.asarray(values)
    if ids.ndim!=1 or values.ndim!=1 or len(ids)!=len(values) or not len(ids):raise ValueError('Nonempty equal-length explicit source required')
    if not np.issubdtype(ids.dtype,np.integer) or ids.dtype==np.bool_ or len(set(map(int,ids)))!=len(ids):raise ValueError('Unique integer physical IDs required')
    if nelec[0]!=nelec[1] or not 1<=r<=128 or isinstance(r,bool):raise ValueError('Balanced bounded registered pool only')
    if not np.isfinite(values).all():raise ValueError('Nonfinite source')
    if coefficient_source:
        scores=np.asarray(abs(values)**2,dtype=float)
        if abs(math.fsum(map(float,scores))-1)>1e-8:raise ValueError('Coefficient source not normalized')
    else:
        if not np.issubdtype(values.dtype,np.integer) or values.dtype==np.bool_ or np.any(values<=0):raise ValueError('Positive integer counts required')
        scores=values
    weights={};mask=(1<<n)-1;hf=(1<<nelec[0])-1
    for x,w in zip(ids,scores):
        if int(x)<0 or int(x)>=1<<(2*n):raise ValueError('Physical ID outside register')
        a,b=int(x)&mask,int(x)>>n
        if a.bit_count()!=nelec[0] or b.bit_count()!=nelec[1]:raise ValueError('Source particle-sector mismatch')
        if w:
            value=float(w) if coefficient_source else int(w)
            weights[a]=weights.get(a,0)+value;weights[b]=weights.get(b,0)+value
    return sorted([hf]+[s for s in sorted(weights,key=lambda s:(-weights[s],s)) if s!=hf][:r-1])


def initial_strings(source,n,nelec,r=8):
    if source.get('kind')=='CLASSICAL_HF_ONLY':return [(1<<nelec[0])-1]
    with np.load(source['payload'],allow_pickle=False) as a:
        coeff=source['kind']=='CLASSICAL_COEFFICIENT_MARGINAL'
        if source['kind'] not in ['CLASSICAL_COEFFICIENT_MARGINAL','CLASSICAL_FULL_COUNTS','QUANTUM_FULL_COUNTS']:
            raise ValueError('Unknown source kind, no implicit substitute')
        return choose(a['blocked_ids'],a['coefficients'] if coeff else a['counts'],n,nelec,r=r,coefficient_source=coeff)
