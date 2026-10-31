import sqlite3
import pytest
from benchmark.online_parameter_controls import SCHEMA
from benchmark.online_parameter_sequence import prefix_snapshot,verify_prefix,complete


def test_committed_prefix_is_unchanged_when_new_tasks_append():
    db=sqlite3.connect(":memory:");db.executescript(SCHEMA)
    db.execute("INSERT INTO cells VALUES('old','{}','COMPLETED','{}')")
    snapshot=prefix_snapshot(db)
    db.execute("INSERT INTO cells VALUES('new','{}','COMPLETED','{}')")
    verify_prefix(db,snapshot)
    db.execute("UPDATE cells SET result='changed' WHERE id='old'")
    with pytest.raises(AssertionError):verify_prefix(db,snapshot)


def test_wrong_live_owner_never_dispatches_or_touches_database(monkeypatch,tmp_path):
    monkeypatch.setattr("benchmark.online_parameter_sequence.live_command",lambda pid:"some-unrelated-user-job")
    with pytest.raises(RuntimeError):complete(tmp_path,1234)
    assert not list(tmp_path.iterdir())
