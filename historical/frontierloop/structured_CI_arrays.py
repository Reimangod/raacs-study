"""Sparse one-body transitions for exact, balanced Cartesian CI actions.

Only spin-block matrices are stored, never the D-by-D Cartesian Hamiltonian.
No quantum circuit, selected determinant basis, or ERI factor is pruned.
"""
import numpy as np
from .joint_ci_sparse import matrices


def transitions(n,k,strings):
    s=list(map(int,strings));lookup={x:i for i,x in enumerate(s)}
    if len(lookup)!=len(s) or not s or any(x<0 or x>=1<<n or x.bit_count()!=k for x in s):raise ValueError('Unique fixed spin sector required')
    pq=[];rows=[];cols=[];sign=[];occ=np.zeros((len(s),n))
    for j,word in enumerate(s):
        for q in range(n):
            if not word&(1<<q):continue
            occ[j,q]=1
            sign_q=(-1)**((word&((1<<q)-1)).bit_count());removed=word^(1<<q)
            for p in range(n):
                if removed&(1<<p):continue
                target=removed|(1<<p)
                if target in lookup:
                    pq.append(p*n+q);rows.append(lookup[target]);cols.append(j)
                    sign.append(sign_q*((-1)**((removed&((1<<p)-1)).bit_count())))
    return dict(pq=np.asarray(pq,np.int32),rows=np.asarray(rows,np.int32),
         cols=np.asarray(cols,np.int32),signs=np.asarray(sign,np.float64),occupancies=occ)


def make_arrays(h,g,k,strings,library):
    n=len(h);s=np.asarray(strings,np.int64)
    if np.iscomplexobj(h) or np.iscomplexobj(g):raise ValueError('Qualified real-integral scope only')
    tr=transitions(n,k,s)
    HA,_,info=matrices(h,g,(k,0),[(int(a),0) for a in s],library,seconds=30)
    values,vectors=np.linalg.eigh(g.reshape(n*n,n*n))
    L=(vectors*np.sqrt(abs(values))[None,:]).T
    error=float(np.max(abs((L.T*np.sign(values))@L-g.reshape(n*n,n*n))))
    if error>1e-10:raise ValueError('Untruncated signed ERI factorization differs')
    arrays=dict(n=np.int64(n),k=np.int64(k),strings=s,one_spin_H=HA.toarray(),
                factors=L,factor_signs=np.sign(values),**tr)
    return arrays,dict(ERI_reconstruction_error=error,retained_factor_count=n*n,
           eigenvalue_cutoff=None,one_spin_matrix=info,
           dense_projected_H_built=False,sparse_transition_count=len(tr['pq']))
