"""Exact float64 Cartesian H/S2 action, NumPy test or CuPy device backend.

The caller controls device selection, memory limits, contexts, and timing.
No CI eigensolver or chemical-accuracy claim is made by this module.
"""
import numpy as np


def dense_probes(D,seed):
    # No reduction-based normalization: CPU-generated input bytes are identical
    # across BLAS implementations; use an elementwise, explicit sqrt(D) scale.
    return np.random.default_rng(seed).normal(size=(2,D))/np.sqrt(float(D))


class StructuredAction:
    def __init__(self,xp,data,tile=8):
        self.xp=xp;self.n=int(data['n']);self.k=int(data['k']);self.R=len(data['strings']);self.tile=tile
        if tile<1 or data['one_spin_H'].shape!=(self.R,self.R):raise ValueError('Shape or tile mismatch')
        self.H=xp.asarray(data['one_spin_H'],dtype=xp.float64)
        L=xp.asarray(data['factors'],dtype=xp.float64)
        self.sign=xp.asarray(data['factor_signs'],dtype=xp.float64)
        self.F=xp.zeros((len(L),self.R,self.R),dtype=xp.float64)
        rows=xp.asarray(data['rows']);cols=xp.asarray(data['cols']);pq=xp.asarray(data['pq'])
        signs=xp.asarray(data['signs'],dtype=xp.float64)
        for start in range(0,len(L),tile):
            li=xp.arange(start,min(start+tile,len(L)))
            xp.add.at(self.F,(li[:,None],rows[None,:],cols[None,:]),L[li[:,None],pq[None,:]]*signs[None,:])
        occ=xp.asarray(data['occupancies'],dtype=xp.float64)
        self.spin_diag=self.k-occ@occ.T
        self.off=[]
        for p in range(self.n):
            for q in range(self.n):
                if p==q:continue
                idx=np.flatnonzero(data['pq']==p*self.n+q)
                if len(idx):self.off.append((rows[idx],cols[idx],signs[idx]))

    def physical_H(self,v):
        xp=self.xp;C=v.reshape(self.R,self.R)
        out=self.H@C+C@self.H.T
        for start in range(0,len(self.F),self.tile):
            F=self.F[start:start+self.tile]
            work=(F@C)@xp.swapaxes(F,1,2)
            out+=xp.sum(work*self.sign[start:start+len(F),None,None],axis=0)
        return out.ravel()

    def spin_S2(self,v):
        xp=self.xp;C=v.reshape(self.R,self.R)
        out=self.spin_diag*C
        # E_qp = E_pq^T. Use its correct row/column reversal, not a guessed
        # Hermitian product; each factor acts in a different spin sector.
        for rows,cols,signs in self.off:
            values=signs[:,None]*signs[None,:]*C[cols[:,None],rows[None,:]]
            xp.add.at(out,(rows[:,None],cols[None,:]),-values)
        return out.ravel()

    def penalized(self,v):return self.physical_H(v)+.1*self.spin_S2(v)
