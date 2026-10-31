"""CPU-owned durable task claims: an uncertain submission never becomes a free task.

This local SQLite authority is owned by the Mac dispatcher. Remote workers cannot
claim a second copy. A crashed/unknown task requires explicit evidence reconciliation.
"""
import json
import sqlite3
from pathlib import Path
from .common import digest


class Registry:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, spec TEXT NOT NULL, owner TEXT NOT NULL, state TEXT NOT NULL, evidence TEXT)")
        self.db.commit()

    def claim(self,spec,owner):
        identity=digest(spec)
        with self.db:
            try:self.db.execute("INSERT INTO tasks VALUES (?,?,?,?,?)",(identity,json.dumps(spec,sort_keys=True),owner,"CLAIMED",None))
            except sqlite3.IntegrityError:raise RuntimeError("Task already claimed; reconcile rather than duplicate: "+identity)
        return identity

    def update(self,identity,owner,state,evidence):
        if state not in ("SUBMITTED","RUNNING","COMPLETED","FAILED_PRESERVED","UNKNOWN_RECONCILE"):
            raise ValueError(state)
        with self.db:
            cursor=self.db.execute("UPDATE tasks SET state=?,evidence=? WHERE task_id=? AND owner=?",
                                   (state,json.dumps(evidence,sort_keys=True),identity,owner))
            if cursor.rowcount!=1:raise RuntimeError("Task owner mismatch")

    def rows(self):
        return self.db.execute("SELECT task_id,owner,state,evidence FROM tasks ORDER BY task_id").fetchall()

    def get(self,spec):
        return self.db.execute("SELECT task_id,owner,state,evidence FROM tasks WHERE task_id=?",(digest(spec),)).fetchone()
