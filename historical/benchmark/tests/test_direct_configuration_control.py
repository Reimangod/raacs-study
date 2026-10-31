"""Small implementation checks; no production exploratory fork outcomes."""
import json
import numpy as np
import pytest
from benchmark.direct_configuration_control import ranked_direct_targets,solve,immutable,prepare,ROOT


def data():
    proposals=[{'proposal_target_ids':[9,3,8]},{'proposal_target_ids':[3,5]}]
    external={'ids':[3,5,8,9,99],'en_energy_importance':[.5,.5,0.,.1,100.]}
    return proposals,external


def test_all_eligible_bank_targets_not_global_or_two_only():
    p,e=data()
    assert ranked_direct_targets(p,e,[1],64)==[3,5,9]
    assert ranked_direct_targets(p,e,[1],3)==[3,5]
    assert 99 not in ranked_direct_targets(p,e,[1],64)


def test_existing_ids_and_zero_importance_not_added():
    p,e=data();assert ranked_direct_targets(p,e,[1,3],64)==[5,9]


@pytest.mark.parametrize('kind',['nan','negative','absent','duplicate','bool_cap'])
def test_invalid_candidate_evidence_is_not_silently_accepted(kind):
    p,e=data();cap=64
    if kind=='nan':e['en_energy_importance'][0]=float('nan')
    elif kind=='negative':e['en_energy_importance'][0]=-1.
    elif kind=='absent':p.append({'proposal_target_ids':[123]})
    elif kind=='duplicate':e['ids'][-1]=3
    elif kind=='bool_cap':cap=True
    with pytest.raises(ValueError):ranked_direct_targets(p,e,[1],cap)


def test_named_projected_solver_and_variational_nesting():
    class Tiny:
        def __init__(self):self.requests=0
        def matrix(self,l,r):
            self.requests+=len(l)*len(r)
            h=np.array([[0.,-.2,0.],[-.2,.3,-.1],[0.,-.1,.8]])
            return h[np.ix_(l,r)]
    b=Tiny();a=solve(b,[0]);c=solve(b,[0,1]);d=solve(b,[0,1,2])
    assert d['energy_Ha']<=c['energy_Ha']<=a['energy_Ha']
    assert d['internal_Ritz_residual']<1e-12 and b.requests==1+4+9


def test_oversized_space_rejected_before_backend_work():
    class Forbidden:
        def matrix(self,*args):raise AssertionError('Should not request matrix')
    with pytest.raises(ValueError):solve(Forbidden(),list(range(65)))


def test_binding_is_additive_not_overwritten(tmp_path):
    p=tmp_path/'binding.json';immutable(p,{'x':1});before=p.read_bytes()
    immutable(p,{'x':1})
    with pytest.raises(ValueError):immutable(p,{'x':2})
    assert p.read_bytes()==before


def test_immutable_source_destination_rejected_before_writes():
    with pytest.raises(ValueError,match='immutable'):prepare(ROOT/'results/standard-v1/forbidden-control')
    assert not (ROOT/'results/standard-v1/forbidden-control').exists()
