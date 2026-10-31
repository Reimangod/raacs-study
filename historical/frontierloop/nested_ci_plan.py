"""N20 source-frequency ranking: no energy or evaluation reference as input."""
from numbers import Integral
import numpy as np


def frequency_basis(ids,counts,n,nelec,r):
    if nelec[0]!=nelec[1] or r not in (128,256,512,1024):
        raise ValueError('Registered balanced R128/256/512/1024 only')
    if len(ids)!=len(counts) or not len(ids) or len(set(map(int,ids)))!=len(ids):
        raise ValueError('Unique positive full-register counts required')
    scores={}; mask=(1<<n)-1
    for x,c in zip(ids,counts):
        if not isinstance(x,Integral) or not isinstance(c,Integral) or isinstance(x,(bool,np.bool_)) or isinstance(c,(bool,np.bool_)) or c<=0 or not 0<=x<1<<(2*n):
            raise ValueError('Integer IDs and counts inside physical register required')
        a,b=int(x)&mask,int(x)>>n
        if (a.bit_count(),b.bit_count())!=tuple(nelec): raise ValueError('Source particle sector differs')
        for s in (a,b): scores[s]=scores.get(s,0)+int(c)
    hf=(1<<nelec[0])-1
    rank=sorted((s for s in scores if s!=hf),key=lambda s:(-scores[s],s))
    return sorted([hf,*rank[:r-1]])
