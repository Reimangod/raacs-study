"""Bounded accounting/identity tests; no molecular solver, fitting or sampling."""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from benchmark import workflow_cost_study as w
from benchmark.workflow_cost_analysis import named_selection, frontier


def test_physical_export_excludes_oracle_arrays_and_preserves_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'ROOT', tmp_path)
    source = tmp_path / 'results/standard-v1/references/test'
    source.mkdir(parents=True)
    metadata = {'n_qubits': 4, 'nalpha': 1, 'nbeta': 1, 'hf_id': 12, 'rhf_energy': -1.,
                'fci_energy': -2., 'hf_weight_fci': .5}
    (source / 'reference.json').write_text(json.dumps(metadata))
    h1, eri = np.eye(2), np.zeros((2,2,2,2))
    np.savez(source / 'reference.npz', h1=h1, eri=eri, fci_state=np.ones(16), probabilities=np.ones(16)/16)
    before = w.sha(source / 'reference.npz')
    destination = w.stage_input('test', tmp_path / 'out')
    out = json.loads((destination / 'problem.json').read_text())
    assert set(out) == {'n_qubits','nalpha','nbeta','hf_id','hf_energy_Ha','schema'}
    with np.load(destination / 'integrals.npz') as z:
        assert set(z.files) == {'h1', 'eri'}
        assert np.array_equal(z['h1'], h1)
    assert w.sha(source / 'reference.npz') == before
    assert w.stage_input('test', tmp_path / 'out') == destination
    changed = dict(metadata, rhf_energy=-3.)
    (source / 'reference.json').write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='Immutable'):
        w.stage_input('test', tmp_path / 'out')


def test_task_identity_includes_budget_seed_method_and_cold_dependencies():
    cfg = {'seed': 101, 'total_model_shots': 2000, 'private_amplitudes': True}
    item = {'case': 'H2', 'method': 'RAACS-full-cost'}
    original = w.task_id(item, cfg)
    assert original != w.task_id(item, dict(cfg, seed=211))
    assert original != w.task_id(item, dict(cfg, total_model_shots=1000))
    assert original != w.task_id(dict(item, method='LUCJ-CCSD-2-fixed'), cfg)
    assert original == w.task_id(dict(reversed(list(item.items()))), cfg)


def test_all_paid_pilots_are_counted_and_started_draw_is_unknown(tmp_path):
    from benchmark.local_sampler_primitives import pack_counts64
    db = sqlite3.connect(tmp_path / 'events.sqlite')
    db.executescript('CREATE TABLE circuits(id TEXT,payload TEXT);'
                    'CREATE TABLE draws(id TEXT,circuit TEXT,role TEXT,requested INTEGER,counts BLOB,metadata TEXT);')
    for cid, gates, depth in [('pilot', 100, 50), ('final', 10, 5)]:
        db.execute('INSERT INTO circuits VALUES(?,?)', (cid, json.dumps({'resources':{'cnot_count':gates,'cnot_depth':depth}})))
    for did, cid, role, n in [('rejected', 'pilot', 'QSCI-pilot', 64), ('finaldraw', 'final', 'independent-final', 1936)]:
        db.execute('INSERT INTO draws VALUES(?,?,?,?,?,?)', (did,cid,role,n,
            pack_counts64(np.array([12],dtype=np.uint64),np.array([n])),json.dumps({'actual_raw_model_shots':n})))
    db.commit()
    result = {'status':'COMPLETED','final_draw_id':'finaldraw'}
    observed = w.events(tmp_path, 'LUCJ-CCSD-2-QSCI4', result)
    assert observed['known_model_shots'] == 2000
    assert observed['gate_shot_proxy'] == 64*100+1936*10
    assert observed['depth_shot_proxy'] == 64*50+1936*5
    assert observed['final_counts'] == {12:1936}
    assert not observed['unknown_inflight']
    db.execute('INSERT INTO draws VALUES(?,?,?,?,NULL,NULL)', ('inflight','pilot','QSCI-pilot',10))
    db.commit(); db.close()
    assert w.events(tmp_path, 'LUCJ-CCSD-2-QSCI4', {'status':'FAILED'})['unknown_inflight']
    missing = w.events(tmp_path / 'absent', 'RAACS-full-cost', {'status':'FAILED'})
    assert missing['unknown_inflight'] and missing['known_model_shots'] == 0


def test_matched_dimension_uses_frequency_order_not_numeric_sorted_prefix():
    info = {'hf_id':12, 'n_qubits':4, 'nalpha':1, 'nbeta':1}
    counts = {12:1, 3:20, 6:10, 9:30}
    assert named_selection(counts, info, 2) == [9,12]
    assert named_selection(counts, info, 3) == [3,9,12]


def test_observed_frontier_keeps_problem_groups_and_no_failure_as_win():
    def row(case, method, energy, wall, gates, status='COMPLETED'):
        return {'case':case,'method':method,'status':status,'unknown_inflight':False,
                'readouts':[{'energy_Ha':energy}],'total_workflow_wall_seconds':wall,
                'all_draw_CNOT_times_shots_proxy':gates}
    result = frontier([row('a','A',-2,1,1),row('a','B',-1,2,2),row('b','B',-3,3,3),
                       row('a','failed',-100,0,0,'FAILED')])
    assert len(result) == 3
    assert {r['method']:r['observed_nondominated'] for r in result if r['case']=='a'} == {'A':True,'B':False}
    assert next(r for r in result if r['case']=='b')['observed_nondominated']


def test_dependency_pid_reuse_is_not_treated_as_finished(monkeypatch):
    from benchmark import workflow_cost_sequence as sequence
    monkeypatch.setattr(sequence,'process_command',lambda pid:'python -m unrelated')
    with pytest.raises(RuntimeError,match='Foreign/reused'):
        sequence.dependency_snapshot([(123,'benchmark.gate_noise_sequence')])
    monkeypatch.setattr(sequence,'process_command',lambda pid:None)
    assert sequence.dependency_snapshot([(123,'benchmark.gate_noise_sequence')])[0]['OS_live'] is False


def test_receipt_or_manifest_drift_is_not_overwritten(tmp_path):
    path = tmp_path/'frozen.json'
    w.frozen(path, {'cost':None})
    before = path.read_bytes()
    w.frozen(path, {'cost':None})
    with pytest.raises(RuntimeError,match='Immutable'):
        w.frozen(path, {'cost':0})
    assert path.read_bytes() == before


def test_external_timer_receipt_resume_does_not_repeat_execution(tmp_path, monkeypatch):
    root = tmp_path/'results/total-workflow-quick-v1'
    root.parent.mkdir()
    monkeypatch.setattr(w, 'ROOT', tmp_path)
    cfg = {'queue':[{'case':'example','method':'LUCJ-CCSD-2-fixed'}],
           'free_floor_bytes':0,'output_cap_bytes':2**30,'timing_scope':'test-only-parent-observed-boundary'}
    monkeypatch.setattr(w,'prepare',lambda root,mode:cfg)
    calls=[]
    class Finished:
        pid=12345
        returncode=0
        def poll(self): return 0
    def spawn(*args,**kwargs):
        calls.append(args)
        folder=root/'cases/example/LUCJ-CCSD-2-fixed'
        w.save(folder/'result.json',{'status':'COMPLETED'})
        w.save(folder/'workflow-summary.json',{'unknown_inflight':False,'known_model_shots':2000})
        return Finished()
    monkeypatch.setattr(w.subprocess,'Popen',spawn)
    w.run(root,'quick')
    receipt=json.loads((root/'cases/example/LUCJ-CCSD-2-fixed/process-receipt.json').read_text())
    assert receipt['total_workflow_wall_seconds']>0 and receipt['total_model_shots']==2000
    assert receipt['all_model_work_known'] and len(calls)==1
    w.run(root,'quick')
    assert len(calls)==1
    (root/'cases/example/LUCJ-CCSD-2-fixed/result.json').write_text('{}')
    with pytest.raises(AssertionError):
        w.run(root,'quick')
    assert len(calls)==1


def test_independent_circuit_counter_reconstructs_raw_qpy_without_sampling(tmp_path):
    import zlib
    from qiskit import QuantumCircuit
    from benchmark.ansatz import circuit_bytes,resources
    from benchmark.local_sampler_primitives import pack_counts64
    from benchmark.workflow_cost_analysis import independent_evidence
    qc=QuantumCircuit(4);qc.x(2);qc.x(3);qc.cx(0,1);qc.cx(1,0)
    raw=zlib.compress(circuit_bytes(qc))
    resource=resources(qc,0,0)
    db=sqlite3.connect(tmp_path/'events.sqlite')
    db.executescript('CREATE TABLE circuits(id TEXT,sampling_qpy BLOB,payload TEXT);'
                    'CREATE TABLE draws(id TEXT,circuit TEXT,requested INTEGER,counts BLOB,metadata TEXT);')
    db.execute('INSERT INTO circuits VALUES(?,?,?)',('c',raw,json.dumps({'resources':resource})))
    metadata={'actual_raw_model_shots':20,'hardware_shots':0,'FCI_used':False,
              'layout_evidence':{'logical_to_simulator_qubit':[0,1,2,3],
                                 'backend_ids':[12],'backend_counts':[20]}}
    db.execute('INSERT INTO draws VALUES(?,?,?,?,?)',('d','c',20,
        pack_counts64(np.array([12],dtype=np.uint64),np.array([20])),json.dumps(metadata)))
    db.commit();db.close()
    result=independent_evidence(tmp_path,'LUCJ-CCSD-2-fixed',{'final_draw_id':'d'})
    assert result['shots']==20 and result['gate']==40
    assert result['final']=={12:20}
    assert result['resources']==resource
