import json
import sqlite3
from dataclasses import asdict
import pytest
from benchmark.common import ROOT,file_digest
from benchmark.molecular_strong_controls import configuration,queue,cell,readouts
from benchmark.resource_adaptive_sampler import integral_problem
from benchmark.circuit_sampling_runtime import SamplingSpec


def small_config():
    cfg=configuration('quick')
    cfg.update(total_model_shots=64,pilot_shots=4,free_floor_bytes=0)
    return cfg


def test_registered_cases_and_complete_methods():
    q=queue('quick');s=queue('standard')
    assert len(q)==5 and len(s)==26
    assert len({x['id'] for x in s})==26
    cfg=configuration('quick')
    assert len(cfg['methods'])==9 and cfg['total_model_shots']==2000
    assert cfg['pilot_shots']*cfg['QSCI_LUCJ_points']==256
    assert cfg['FCI_used'] is False and cfg['full_Fock_vector_allowed'] is False


def test_real_small_native_LUCJ_counts_budget_readouts_resume(tmp_path):
    p=ROOT/'results/standard-v1/references/H4_R1.1';cfg=small_config()
    for method in ('HF','LUCJ-MP2-2-fixed','LUCJ-CCSD-2-QSCI4','classical-SCI64'):
        folder=tmp_path/method
        r=cell(p,folder,method,cfg,tmp_path,tmp_path/'amplitudes')
        assert r['status']=='COMPLETED',r
        sha=file_digest(folder/'result.json')
        assert cell(p,folder,method,cfg,tmp_path,tmp_path/'amplitudes')==r
        assert file_digest(folder/'result.json')==sha
        if method=='classical-SCI64':continue
        assert r['new_primary_model_shots']==64
        db=sqlite3.connect(folder/'events.sqlite')
        assert db.execute('SELECT sum(requested) FROM draws').fetchone()[0]==64
        if method.endswith('QSCI4'):
            assert r['pilot_model_shots']==16 and r['independent_final_model_shots']==48
            assert 1<=r['n_optimized_variables_this_run']<=3
            assert db.execute('SELECT count(*) FROM draws').fetchone()[0]==5
        db.close()
        values=json.loads((folder/'readouts.json').read_text())
        assert [v['R'] for v in values]==[8,32,64]
        assert all(v['internal_residual']<1e-8 for v in values)
        assert all(values[i+1]['energy_Ha']<=values[i]['energy_Ha']+1e-10 for i in (0,1))


def test_capacity_boundary_is_pending_not_permanent_failure(tmp_path):
    cfg=small_config();cfg['free_floor_bytes']=2**60
    folder=tmp_path/'pending'
    r=cell(ROOT/'results/standard-v1/references/H2_R0.7',folder,'HF',cfg,tmp_path,tmp_path/'amplitudes')
    assert r['status']=='PENDING_CAPACITY_BOUNDARY' and r['terminal'] is False
    assert not (folder/'result.json').exists()
    db=sqlite3.connect(folder/'events.sqlite')
    assert db.execute('SELECT count(*) FROM draws').fetchone()[0]==0
    db.close()


def test_partial_pilot_never_tops_up_or_labels_independent_final(monkeypatch,tmp_path):
    from benchmark import molecular_strong_controls as m
    original=m.CircuitCountSampler.draw
    def partial(self,preparation,shots,seed,*args):
        ids,counts,meta=original(self,preparation,1,seed,*args)
        meta['requested_model_shots']=shots
        meta['sample_metadata']['status']='RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED'
        return ids,counts,meta
    monkeypatch.setattr(m.CircuitCountSampler,'draw',partial)
    folder=tmp_path/'partial'
    r=cell(ROOT/'results/standard-v1/references/H2_R0.7',folder,'LUCJ-CCSD-2-QSCI4',small_config(),tmp_path,tmp_path/'amplitudes')
    assert r['status']=='RESOURCE_CAPPED_PARTIAL_COUNTS_PRESERVED'
    assert r['new_primary_model_shots']==1
    assert r.get('final_energy_Ha') is None
    db=sqlite3.connect(folder/'events.sqlite')
    assert db.execute('SELECT role,requested FROM draws').fetchall()==[('QSCI-pilot',4)]
    db.close()
