"""N52 mathematical pilot: projected Cartesian CI as one-spin blocks and
factorized opposite-spin Coulomb actions. No truncation or new quantum ansatz.
Small CPU fixtures only; no claim of a production CUDA implementation.
"""
import numpy as np
from .joint_ci_sparse import matrices


def one_body_generators(n,k,strings):
    s=list(map(int,strings));lookup={v:i for i,v in enumerate(s)}
    if len(lookup)!=len(s) or any(v<0 or v>=1<<n or v.bit_count()!=k for v in s):raise ValueError('Unique fixed one-spin sector required')
    E=np.zeros((n,n,len(s),len(s)))
    for j,word in enumerate(s):
        for q in range(n):
            if not word&(1<<q):continue
            sign_q=(-1)**((word&((1<<q)-1)).bit_count());removed=word^(1<<q)
            for p in range(n):
                if removed&(1<<p):continue
                target=removed|(1<<p)
                if target in lookup:E[p,q,lookup[target],j]=sign_q*((-1)**((removed&((1<<p)-1)).bit_count()))
    return E


def factor_action(h,g,nelec,strings_a,strings_b,library):
    n=len(h);ka,kb=nelec
    EA=one_body_generators(n,ka,strings_a);EB=one_body_generators(n,kb,strings_b)
    # Do NOT form same-spin H from products of projected E matrices: that would
    # wrongly discard paths through intermediate determinants outside the basis.
    HA,_,_=matrices(h,g,(ka,0),[(int(a),0) for a in strings_a],library)
    HB,_,_=matrices(h,g,(0,kb),[(0,int(b)) for b in strings_b],library)
    vals,vec=np.linalg.eigh(g.reshape(n*n,n*n))
    L=(vec*np.sqrt(abs(vals))[None,:]).T.reshape(n*n,n,n)
    FA=np.einsum('lpq,pqij->lij',L,EA);FB=np.einsum('lpq,pqij->lij',L,EB)
    signs=np.sign(vals)
    reconstructed=np.einsum('l,lpq,lrs->pqrs',signs,L,L)
    error=float(np.max(abs(reconstructed-g)))
    def hop(v):
        C=np.asarray(v).reshape(len(strings_a),len(strings_b))
        out=HA@C+(HB@C.T).T
        out+=np.einsum('l,lij,jk,lmk->im',signs,FA,C,FB,optimize=True)
        return out.ravel()
    def spin(v):
        C=np.asarray(v).reshape(len(strings_a),len(strings_b))
        out=(((ka-kb)/2)**2+(ka+kb)/2)*C
        out=out-np.einsum('pqij,jk,qpmk->im',EA,C,EB,optimize=True)
        return out.ravel()
    return hop,spin,dict(ERI_reconstruction_error=error,retained_factor_count=n*n,
         eigenvalue_cutoff=None,no_factor_truncation=True,
         one_spin_dimensions=[len(strings_a),len(strings_b)])
