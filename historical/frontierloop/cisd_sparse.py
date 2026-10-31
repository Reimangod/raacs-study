"""Sparse determinant form of restricted CISD, without a full Fock vector.

Mapping follows installed PySCF ci.cisd.to_fcivec, but creates only its known
nonzero determinant pairs. This is classical benchmark data, not a quantum
state prepared for free and not input to the RAACS sampler policy.
"""
import numpy as np
from pyscf.ci import cisd
from pyscf.fci import cistring


def coefficients(vector,norb,nocc):
    c0,c1,c2=cisd.cisdvec_to_amplitudes(vector,norb,nocc,copy=False)
    nvir=norb-nocc
    a1,s1=cisd.tn_addrs_signs(norb,nocc,1)
    strings1=cistring.addrs2str(norb,nocc,a1)
    hf=(1<<nocc)-1
    ids=[hf|(hf<<norb)];values=[c0]
    singles=c1.ravel()*s1
    for s,v in zip(strings1,singles):
        ids.extend([hf|(int(s)<<norb),int(s)|(hf<<norb)])
        values.extend([v,v])
    opposite=c2.transpose(0,2,1,3).reshape(nocc*nvir,-1)
    opposite=np.einsum('i,j,ij->ij',s1,s1,opposite)
    for i,a in enumerate(strings1):
        for j,b in enumerate(strings1):
            ids.append(int(a)|(int(b)<<norb));values.append(opposite[i,j])
    if nocc>1 and nvir>1:
        same=c2-c2.transpose(1,0,2,3)
        oo=np.tril_indices(nocc,-1);vv=np.tril_indices(nvir,-1)
        same=same[oo][:,vv[0],vv[1]]
        a2,s2=cisd.tn_addrs_signs(norb,nocc,2)
        strings2=cistring.addrs2str(norb,nocc,a2)
        for s,v in zip(strings2,same.ravel()*s2):
            ids.extend([hf|(int(s)<<norb),int(s)|(hf<<norb)])
            values.extend([v,v])
    ids=np.asarray(ids,dtype=np.uint64);values=np.asarray(values)
    if len(set(map(int,ids)))!=len(ids):
        raise RuntimeError('CISD sparse mapping produced duplicate determinants')
    order=np.argsort(ids)
    return ids[order],values[order]


def marginal_strings(ids,values,norb,nocc,r):
    weights={};mask=(1<<norb)-1
    for x,v in zip(ids,values):
        a,b=int(x)&mask,int(x)>>norb
        w=float(abs(v)**2)
        if w==0:
            continue
        weights[a]=weights.get(a,0.)+w
        weights[b]=weights.get(b,0.)+w
    hf=(1<<nocc)-1
    selected=[hf]+[s for s in sorted(weights,key=lambda s:(-weights[s],s)) if s!=hf][:r-1]
    return np.array(sorted(selected),np.int64)
