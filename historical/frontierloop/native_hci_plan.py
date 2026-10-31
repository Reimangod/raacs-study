"""Common source-independent HCI proposal rules, not new quantum sampling."""
import math
from numbers import Integral


def frequency_strings(ids,counts,n,hf,r=128):
    scores={};mask=(1<<n)-1
    if len(ids)!=len(counts) or not 1<=r<=128:raise ValueError('Explicit equal-length counts and bounded R required')
    for x,c in zip(ids,counts):
        if not isinstance(x,Integral) or not isinstance(c,Integral) or isinstance(x,bool) or isinstance(c,bool) or c<=0 or x<0 or x>=1<<(2*n):
            raise ValueError('Do not coerce unknown/noninteger source IDs or counts')
        if (int(x)&mask).bit_count()!=int(hf).bit_count() or (int(x)>>n).bit_count()!=int(hf).bit_count():
            raise ValueError('Frequency source outside registered balanced particle sector')
        for s in [int(x)&mask,int(x)>>n]:scores[s]=scores.get(s,0)+int(c)
    return sorted([int(hf)]+sorted([s for s in scores if s!=hf],key=lambda s:(-scores[s],s))[:r-1])


def marginal_strings(pairs,n,nelec,hf,r=128):
    if not pairs or not all(math.isfinite(v) for _,_,v in pairs) or len({(a,b) for a,b,_ in pairs})!=len(pairs):
        raise ValueError('Unique finite native determinant coefficients required')
    scores={};norm=math.fsum(float(v)**2 for _,_,v in pairs)
    if abs(norm-1)>1e-8:raise ValueError('Native CI coefficients incomplete or not normalized')
    for a,b,v in pairs:
        if a<0 or b<0 or a>=1<<n or b>=1<<n or a.bit_count()!=nelec[0] or b.bit_count()!=nelec[1]:
            raise ValueError('Native proposed determinant outside physical particle sector')
        scores[a]=scores.get(a,0.)+.5*v*v;scores[b]=scores.get(b,0.)+.5*v*v
    return sorted([int(hf)]+sorted([s for s in scores if s!=hf],key=lambda s:(-scores[s],s))[:r-1]),norm
