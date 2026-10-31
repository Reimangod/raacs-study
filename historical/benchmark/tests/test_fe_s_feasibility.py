import json
import numpy as np
import pytest
from benchmark.common import ROOT
from benchmark.fe_s_feasibility import (header,verify_download,occupation_reference,
    independent_diagonal,explicit_matrix,apply_fermions)
from benchmark.integral_hamiltonian import IntegralHamiltonian
from benchmark.local_sampler_primitives import valid_sector


def test_headers_and_representation_boundaries():
    h=header(b' &FCI NORB=20,NELEC=30,MS2=0,\n &END')
    assert (h['n_qubits'],h['nalpha'],h['nbeta'])==(40,15,15)
    assert header(b' &FCI NORB=36,NELEC=54,MS2=0,')['n_qubits']==72
    for raw in (b'NORB=20,NELEC=31,MS2=0',b'NORB=20,NELEC=50,MS2=0',b'NORB=20'):
        with pytest.raises(ValueError):header(raw)
    with pytest.raises(ValueError):verify_download(b'not pinned physical data')


def test_occupation_reference_is_declared_not_rhf():
    x=occupation_reference(20,15,15)
    assert valid_sector(x,40,15,15)
    assert x.bit_count()==30
    assert apply_fermions(x,[(True,0)],40) is None
    assert apply_fermions(x,[(False,39)],40) is None
    assert 'NOT_VERIFIED_RHF' in (ROOT/'benchmark/fe_s_feasibility.py').read_text()


@pytest.mark.parametrize('case',['H2_R0.7','H4_R1.1','LiH_R1.5','BeH2_R1.3','H6_R1.1','N2_R1.1'])
def test_explicit_fermion_oracle_against_qualified_small_physical_inputs(case):
    folder=ROOT/'results/standard-v1/references'/case
    meta=json.loads((folder/'reference.json').read_text())
    with np.load(folder/'reference.npz') as z:h1=z['h1'];eri=z['eri']
    backend=IntegralHamiltonian.from_saved_integrals(h1,eri,meta['hf_id'],meta['rhf_energy'])
    na=meta['nalpha'];nb=meta['nbeta'];n=len(h1)
    assert abs(independent_diagonal(h1,eri,backend.core,na,nb)-meta['rhf_energy'])<1e-8
    hf=meta['hf_id'];ids=[hf]
    for src,dst in ((0,2*na),(1,2*nb+1)):
        if dst<2*n:ids.append(hf^(1<<(2*n-1-src))^(1<<(2*n-1-dst)))
    if na<n and nb<n:
        ids.append(hf^(1<<(2*n-1))^(1<<(2*n-2))^(1<<(2*n-1-2*na))^(1<<(2*n-2-2*nb)))
    assert np.allclose(explicit_matrix(ids,backend),backend.matrix(ids,ids),atol=1e-9,rtol=0)
