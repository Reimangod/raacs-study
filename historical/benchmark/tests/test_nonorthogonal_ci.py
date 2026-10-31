from itertools import combinations
import numpy as np
import pytest
from scipy.linalg import expm
from benchmark.common import ROOT
from benchmark.chemistry import load_reference
from benchmark.integral_hamiltonian import IntegralHamiltonian
from benchmark.nonorthogonal_ci import NonorthogonalHamiltonian,occupied_columns,generalized_ground


def exterior(columns):
    """Independent small-register audit only; never a production solver."""
    n,ne = columns.shape
    vector = np.zeros(1<<n,complex)
    for occ in combinations(range(n),ne):
        idx = sum(1<<(n-1-q) for q in occ)
        vector[idx] = np.linalg.det(columns[list(occ)])
    return vector


def test_zero_overlap_single_double_and_disconnected_pairs():
    r = load_reference(ROOT/'results/standard-v1/references/H4_R1.1')
    m=r['meta']; h=IntegralHamiltonian.from_saved_integrals(r['h1'],r['eri'],m['hf_id'],m['rhf_energy'])
    noci=NonorthogonalHamiltonian(h.h1,h.eri,h.core)
    ids=[int(x) for x in r['ids']]
    nonzero_at_zero=0
    for idx in ids:
        a=occupied_columns(np.eye(h.norb),m['hf_id']);b=occupied_columns(np.eye(h.norb),idx)
        hij,sij=noci.pair(a,b)
        assert abs(hij-r['h'][m['hf_id'],idx])<1e-10
        assert abs(sij-(idx==m['hf_id']))<1e-10
        nonzero_at_zero+=abs(sij)<1e-12 and abs(hij)>1e-5
    assert nonzero_at_zero>1


def test_rotated_real_complex_and_near_orthogonal_against_exterior_projection():
    r=load_reference(ROOT/'results/standard-v1/references/H4_R1.1');m=r['meta']
    h=IntegralHamiltonian.from_saved_integrals(r['h1'],r['eri'],m['hf_id'],m['rhf_energy'])
    noci=NonorthogonalHamiltonian(h.h1,h.eri,h.core);rng=np.random.default_rng(771)
    columns=[]
    for complex_basis in (False,True):
        for _ in range(5):
            generator=rng.normal(size=(4,4))+ (1j*rng.normal(size=(4,4)) if complex_basis else 0)
            u=expm(.4*(generator-generator.conj().T))
            columns.append(occupied_columns(u,int(rng.choice(r['ids']))))
    columns += [occupied_columns(np.eye(4),m['hf_id'])]*2
    vectors=np.column_stack([exterior(x) for x in columns]);hh,ss=noci.matrices(columns)
    assert np.max(abs(ss-vectors.conj().T@vectors))<1e-10
    assert np.max(abs(hh-vectors.conj().T@(r['h']@vectors)))<1e-10
    result,c=generalized_ground(hh,ss)
    assert result['rank']<len(columns)
    assert np.isclose(np.linalg.norm(vectors@c),1,atol=1e-9)
    assert np.isclose(np.vdot(vectors@c,r['h']@(vectors@c)).real,result['energy_Ha'],atol=1e-9)


def test_phase_and_occupied_column_order_are_not_lost():
    h=NonorthogonalHamiltonian(np.diag([-2.,-1.]),np.zeros((2,)*4),0.)
    a=occupied_columns(np.eye(2),12);b=a[:,::-1]*1j
    hij,sij=h.pair(a,b)
    assert abs(sij-1)<1e-12 # column swap -1, two factors i -> -1
    assert abs(hij+4)<1e-12


def test_40qubit_named_slater_solver_has_no_hilbert_vector():
    n=20;h=NonorthogonalHamiltonian(np.diag(np.linspace(-2,1,n)),np.zeros((n,)*4),.3)
    hf=sum(1<<(2*n-1-q) for q in range(20));u=np.eye(n)
    a=occupied_columns(u,hf);b=occupied_columns(u,hf^(1<<39)^(1<<19))
    hh,ss=h.matrices([a,b]);out,_=generalized_ground(hh,ss)
    assert out['rank']==2 and hh.shape==(2,2)
    assert h.factors.nbytes<2*2**20


def test_rank_deficiency_and_bad_overlap_not_hidden():
    out,_=generalized_ground(np.full((2,2),-1.),np.ones((2,2)))
    assert out['rank']==1 and abs(out['energy_Ha']+1)<1e-10
    with pytest.raises(ValueError):generalized_ground(np.eye(2),np.diag([1,-.1]))
    with pytest.raises(ValueError):occupied_columns(np.ones((2,2)),12)
