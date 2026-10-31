"""Regression for actual metadata-ID / integral-ID namespace mismatch."""
from copy import deepcopy
import hashlib
import json
import pytest
from benchmark.hci_paired_review_v2 import integral_identity


def fixture():
    binding=dict(h1='h1-bytes',eri='eri-bytes',core=1.,n_qubits=8,nalpha=2,nbeta=2,hf_id=240)
    pid=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    problem=dict(problem_id=pid,physical_integral_binding=binding,n_qubits=8,nalpha=2,nbeta=2,hf_id=240,HF_energy_Ha=-3.)
    meta=dict(problem_id='a'*64,n_qubits=8,nalpha=2,nbeta=2,hf_id=240,hf_energy_Ha=-3.,core_constant_Ha=1.)
    es=dict(meta_sha256='meta',integrals_sha256='archive')
    hs=dict(metadata_sha256='meta',archive_sha256='archive',problem=problem)
    return meta,es,hs,deepcopy(problem),pid,'meta','archive'


def test_namespaces_may_differ_but_physics_must_agree():
    args=fixture(); result=integral_identity(*args)
    assert result['identifiers_use_different_schemas'] is True
    assert result['metadata_builder_problem_id'] != result['integral_problem_id']
    assert result['integral_problem_id'] == args[4]


@pytest.mark.parametrize('defect',['meta_hash','archive_hash','HCI_id','sampler_id','sampler_core',
    'sampler_h1','sampler_sector','metadata_sector','metadata_hf','sampler_hf','metadata_core','missing_metadata_id'])
def test_a_real_physical_mismatch_is_not_excused_by_namespaces(defect):
    meta,es,hs,sampler,pid,msha,asha=fixture()
    if defect=='meta_hash':hs['metadata_sha256']='different'
    elif defect=='archive_hash':es['integrals_sha256']='different'
    elif defect=='HCI_id':pid='wrong'
    elif defect=='sampler_id':sampler['problem_id']='wrong'
    elif defect=='sampler_core':sampler['physical_integral_binding']['core']=2.
    elif defect=='sampler_h1':sampler['physical_integral_binding']['h1']='wrong'
    elif defect=='sampler_sector':sampler['nalpha']=1
    elif defect=='metadata_sector':meta['nbeta']=1
    elif defect=='metadata_hf':meta['hf_energy_Ha']=-2.
    elif defect=='sampler_hf':sampler['HF_energy_Ha']=float('nan')
    elif defect=='metadata_core':meta['core_constant_Ha']=2.
    elif defect=='missing_metadata_id':meta['problem_id']=''
    with pytest.raises(ValueError):integral_identity(meta,es,hs,sampler,pid,msha,asha)


def test_both_sides_sharing_a_bad_integral_digest_is_rejected():
    meta,es,hs,sampler,pid,msha,asha=fixture()
    hs['problem']['physical_integral_binding']['eri']='changed'
    sampler['physical_integral_binding']['eri']='changed'
    with pytest.raises(ValueError,match='content-addressed'):integral_identity(meta,es,hs,sampler,pid,msha,asha)
