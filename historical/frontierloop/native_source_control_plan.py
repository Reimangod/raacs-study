"""Source-only N6 initial spaces; no evaluation energy or FCI is accepted."""
import math
import numpy as np
from .native_hci_plan import frequency_strings


METHODS=('DETERMINISTIC_CISD_MARGINAL','C5_E_LUCJ_TN_COMMON_L1')


def coefficient_strings(ids,values,n,nelec,r=128):
    ids=np.asarray(ids);values=np.asarray(values)
    if ids.ndim!=1 or values.ndim!=1 or len(ids)!=len(values) or not np.issubdtype(ids.dtype,np.integer) or ids.dtype==np.bool_:
        raise ValueError('Explicit equal-length integer IDs and coefficients required')
    if nelec[0]!=nelec[1] or r!=128 or len(set(map(int,ids)))!=len(ids):raise ValueError('Registered balanced unique coefficient source required')
    if not np.isfinite(values).all():raise ValueError('Nonfinite source coefficients')
    weights={};mask=(1<<n)-1;norm=math.fsum(float(abs(v)**2) for v in values)
    if abs(norm-1)>1e-8:raise ValueError('Coefficient source not normalized')
    for x,v in zip(ids,values):
        if int(x)<0 or int(x)>=1<<(2*n):raise ValueError('Coefficient ID outside physical register')
        a,b=int(x)&mask,int(x)>>n
        if a.bit_count()!=nelec[0] or b.bit_count()!=nelec[1]:raise ValueError('Coefficient source outside particle sector')
        w=float(abs(v)**2)
        if w:
            weights[a]=weights.get(a,0.)+w;weights[b]=weights.get(b,0.)+w
    hf=(1<<nelec[0])-1
    return sorted([hf]+[x for x in sorted(weights,key=lambda x:(-weights[x],x)) if x!=hf][:r-1])


def initial_strings(source,n,nelec):
    with np.load(source['payload'],allow_pickle=False) as a:
        if source['kind']=='CLASSICAL_COEFFICIENT_MARGINAL':return coefficient_strings(a['blocked_ids'],a['coefficients'],n,nelec)
        if source['kind']=='QUANTUM_FULL_COUNTS':return frequency_strings(a['blocked_ids'],a['counts'],n,(1<<nelec[0])-1)
    raise ValueError('No substitute or unknown source kind')
