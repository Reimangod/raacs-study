"""Additive matrix-free lowest-root diagnostic; not a silent C3 replacement."""
from time import perf_counter
import ffsim
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
from pyscf.fci import selected_ci
from paperstudy.ci import CountedSelectedCI


def solve(ham,nelec,strings,penalty=.1,seconds=60,seed=20261025,known_vector=None):
    """Use H + penalty*S²; generic start avoids a single determinant's symmetry.

    k=2 checks the first gap but is not a mathematical global root certificate.
    The returned state's physical energy, penalty, residual and Rayleigh checks
    are independent of PySCF's Davidson root selection. Costs count every hop.
    """
    strings=np.asarray(strings,np.int64);shape=(len(strings),)*2;d=len(strings)**2;n=ham.norb
    solver=CountedSelectedCI(perf_counter()+seconds);solver.verbose=0
    h2=solver.absorb_h1e(ham.one_body_tensor,ham.two_body_tensor,n,nelec,.5)
    links=selected_ci._all_linkstr_index((strings,strings),n,nelec)
    def sci(x):
        # ARPACK eigenvector columns may be strided. PySCF native CI/RDM
        # contractions require a C-contiguous coefficient buffer.
        return selected_ci._as_SCIvector(np.array(x,dtype=np.float64,order='C',copy=True).reshape(shape),(strings,strings))
    def hop(x):
        v=sci(x)
        return (solver.contract_2e(h2,v,n,nelec,link_index=links)+penalty*solver.contract_ss(v,n,nelec)).ravel()
    rng=np.random.default_rng(seed)
    symmetry_errors=[];expectation_errors=[]
    for _ in range(3):
        a,b=rng.normal(size=(2,d));a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        symmetry_errors.append(float(abs(np.vdot(a,hop(b))-np.vdot(hop(a),b))))
        v=sci(a);one,two=solver.make_rdm12(v,n,nelec)
        rdm=ffsim.ReducedDensityMatrix(one,two)
        physical=float(rdm.expectation(ham).real)
        s2=float(solver.spin_square(v,n,nelec)[0])
        expectation_errors.append(float(abs(np.vdot(a,hop(a))+ham.constant-physical-penalty*s2)))
    if max(symmetry_errors)>1e-10 or max(expectation_errors)>1e-8:
        raise RuntimeError('Independent matrix-free penalty action/RDM mismatch')
    v0=rng.normal(size=d);v0/=np.linalg.norm(v0)
    if known_vector is not None:
        z=np.asarray(known_vector).ravel();v0=z/np.linalg.norm(z)+.01*v0;v0/=np.linalg.norm(v0)
    if d<=2:
        matrix=np.column_stack([hop(np.eye(d)[:,i]) for i in range(d)])
        values,vectors=np.linalg.eigh(matrix)
    else:
        values,vectors=eigsh(LinearOperator((d,d),matvec=hop,dtype=np.float64),
            k=2,which='SA',v0=v0,tol=1e-12,maxiter=1000)
    index=np.argsort(values);values=values[index];vector=vectors[:,index[0]]
    residual=float(np.linalg.norm(hop(vector)-values[0]*vector))
    v=sci(vector);one,two=solver.make_rdm12(v,n,nelec);rdm=ffsim.ReducedDensityMatrix(one,two)
    physical=float(rdm.expectation(ham).real);s2=float(solver.spin_square(v,n,nelec)[0])
    objective=float(values[0]+ham.constant)
    if residual>1e-8 or abs(objective-physical-penalty*s2)>1e-8:
        raise RuntimeError(f'Lowest-root diagnostic residual/physical consistency failed: residual={residual}, consistency={objective-physical-penalty*s2}')
    row={'physical_energy_Ha':physical,'penalized_energy_Ha':objective,'spin_square':s2,
        'lowest_two_eigenvalues_Ha':(values[:2]+ham.constant).tolist(),'residual_norm':residual,
        'random_bilinear_hermiticity_errors':symmetry_errors,'independent_RDM_expectation_errors':expectation_errors,
        'matvec_calls':solver.matvec_calls,'seed':seed,'solver':'SciPy eigsh which=SA k=2 matrix-free physical H plus projected S2',
        'not_global_optimality_or_exact_singlet_certificate':True}
    return row,vector.reshape(shape),rdm
