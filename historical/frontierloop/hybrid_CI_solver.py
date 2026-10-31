"""The N51 PySCF Davidson iteration with an externally supplied exact action."""
from time import process_time,perf_counter
import numpy as np
from pyscf import lib


def exact_diagonal(data):
    n=int(data['n']);occ=np.asarray(data['occupancies'],float)
    L=np.asarray(data['factors']);fd=L[:,np.arange(n)*(n+1)]@occ.T
    one=np.diag(data['one_spin_H'])
    physical=one[:,None]+one[None,:]+(fd.T*np.asarray(data['factor_signs']))@fd
    return (physical+.1*(int(data['k'])-occ@occ.T)).ravel()


def solve(action,diagonal,initial,*,max_actions=300,wall_seconds=180,CPU_seconds=600,progress=None):
    """No reference energy or saved final root is accepted as an input."""
    cpu,wall=process_time(),perf_counter();calls=0
    diag=np.asarray(diagonal,float);v=np.asarray(initial,float).copy()
    if diag.ndim!=1 or v.shape!=diag.shape or not np.all(np.isfinite(v)) or not np.all(np.isfinite(diag)):
        raise ValueError('Finite matching real vectors required')
    norm=float(np.linalg.norm(v))
    if norm==0:raise ValueError('Initial vector cannot be zero')
    v/=norm
    def hop(x):
        nonlocal calls
        if calls>=max_actions or perf_counter()-wall>wall_seconds or process_time()-cpu>CPU_seconds:
            raise TimeoutError('N54 registered H-action/CPU/wall cap')
        calls+=1;y=np.asarray(action(np.asarray(x,float)),float)
        if y.shape!=v.shape or not np.all(np.isfinite(y)):raise ValueError('Invalid action output')
        if progress is not None and calls%10==0:
            progress(dict(matvec_calls=calls,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall))
        return y
    probe=np.zeros_like(v);probe[int(np.argmin(diag))]=1
    diagonal_error=abs(float(probe@hop(probe))-float(np.min(diag)))
    if diagonal_error>1e-8:raise ValueError('Exact projected diagonal differs')
    if len(v)<=2:
        values,vectors=np.linalg.eigh(np.column_stack([hop(np.eye(len(v))[:,i]) for i in range(len(v))]))
        i=int(np.argmin(values));energy=float(values[i]);vector=vectors[:,i];flag=True
    else:
        flags,values,states=lib.davidson1(lambda xs:[hop(x) for x in xs],[v],
            lambda x,e,*args:x/(diag-e+1e-4),tol=1e-12,tol_residual=1e-10,
            max_cycle=500,max_space=24,nroots=1,max_memory=1536,lindep=1e-20,verbose=0)
        i=int(np.argmin(values));energy=float(values[i]);vector=np.asarray(states[i]);flag=bool(flags[i])
    residual=float(np.linalg.norm(hop(vector)-energy*vector))
    if residual>1e-8 or abs(float(vector@vector)-1)>1e-10:
        raise ValueError('Returned root failed unchanged explicit residual/normalization')
    return dict(objective_without_core=energy,residual_norm=residual,matvec_calls=calls,
        eigensolver_internal_converged=flag,preconditioner_diagonal_probe_error_Ha=diagonal_error,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        eigensolver='PYSCF_DAVIDSON_DIAGONAL_PRECONDITIONED_CPU_DRIVER_EXTERNAL_ACTION',
        requested_eigenpairs=1,not_global_optimality_certificate=True),vector
