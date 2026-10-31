import sqlite3
from qiskit import QuantumCircuit
from benchmark.common import ROOT,file_digest
from benchmark.multibasis_standard import settings,import_quick,unknown_parameter_resources
from benchmark.multibasis_study import unpack


def test_explicit_unknown_parameter_counts_not_false_zero():
    qc=QuantumCircuit(2);qc.cx(0,1)
    out=unknown_parameter_resources(qc,0,0)
    assert out['cnot_count']==1 and out['parameter_count'] is None and out['selected_operator_count'] is None


def test_quick_raw_reuse_preserves_counts_and_does_not_import_outcomes(tmp_path):
    cfg=settings();quick=ROOT/'results/multibasis-NOCI-quick-v1';before=file_digest(quick/'events.sqlite')
    import_quick(tmp_path,cfg);import_quick(tmp_path,cfg)
    db=sqlite3.connect(tmp_path/'events.sqlite');old=sqlite3.connect(f"file:{quick/'events.sqlite'}?mode=ro",uri=True)
    assert db.execute('SELECT count(*) FROM tasks').fetchone()==(0,)
    assert db.execute('SELECT count(*) FROM readouts').fetchone()==(0,)
    rows=db.execute('SELECT id,payload FROM draws').fetchall();assert len(rows)==152
    for did,blob in rows:
        current=unpack(blob);original=unpack(old.execute('SELECT payload FROM draws WHERE id=?',(did,)).fetchone()[0])
        assert (current['ids'],current['counts'],current['seed'])==(original['ids'],original['counts'],original['seed'])
        assert current['metadata']['new_quantum_acquisition_in_this_run'] is False
    assert before==file_digest(quick/'events.sqlite')
    db.close();old.close()
