"""Real-integral particle-hole RDM conversion for a future faster CI backend.

Both tensors use PySCF/ffsim spin-summed RDM conventions. This is classical
representation conversion, not a new measurement and not a state preparation.
Not used by frozen C5 or B4.
"""
import numpy as np


def electron_rdms(hole_dm1,hole_dm2):
    if np.iscomplexobj(hole_dm1) or np.iscomplexobj(hole_dm2):
        raise ValueError('Only independently qualified real RDMs are supported')
    n=len(hole_dm1);eye=np.eye(n);d=hole_dm1
    one=2*eye-d.T
    two=(hole_dm2.transpose(1,0,3,2)
        +4*np.einsum('pq,rs->pqrs',eye,eye)-2*np.einsum('ps,rq->pqrs',eye,eye)
        -2*np.einsum('rs,pq->pqrs',eye,d)+np.einsum('ps,rq->pqrs',eye,d)
        +np.einsum('rq,ps->pqrs',eye,d)-2*np.einsum('pq,rs->pqrs',eye,d))
    return one,two
