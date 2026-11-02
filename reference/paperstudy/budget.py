"""Atomic shared shot reservations. Unknown outcomes are not silently refunded."""
import json
from pathlib import Path
import sqlite3


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS budgets(run TEXT PRIMARY KEY, cap INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS draws(
                    run TEXT, event TEXT, request TEXT NOT NULL, reserved INTEGER NOT NULL,
                    actual INTEGER, state TEXT NOT NULL, artifact_digest TEXT,
                    PRIMARY KEY(run,event));
                CREATE TABLE IF NOT EXISTS readouts(
                    run TEXT, id TEXT, draw TEXT NOT NULL, spec TEXT NOT NULL,
                    PRIMARY KEY(run,id));
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def register(self, run, cap):
        if not isinstance(cap, int) or cap < 1:
            raise ValueError("Positive registered shot cap")
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO budgets VALUES(?,?)", (run, cap))
            if db.execute("SELECT cap FROM budgets WHERE run=?", (run,)).fetchone()[0] != cap:
                raise ValueError("Frozen run budget cannot change")

    def reserve(self, run, event, request, shots):
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
            raise ValueError("Positive integer model shots")
        request = json.dumps(request, sort_keys=True, separators=(",", ":"))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT request,reserved,state FROM draws WHERE run=? AND event=?", (run, event)).fetchone()
            if old:
                if old[:2] != (request, shots):
                    raise ValueError("Event reuse with a different request/circuit/shot count")
                return old[2]
            row = db.execute("SELECT cap FROM budgets WHERE run=?", (run,)).fetchone()
            if row is None:
                raise ValueError("Run must be registered")
            charged = db.execute("SELECT COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE actual END),0) FROM draws WHERE run=?", (run,)).fetchone()[0]
            if charged+shots > row[0]:
                raise BudgetExceeded("Atomic shared budget would be exceeded")
            db.execute("INSERT INTO draws VALUES(?,?,?,?,?,?,?)", (run, event, request, shots, None, "RESERVED", None))
            return "NEW_RESERVATION"

    def complete(self, run, event, actual, artifact_digest):
        if not isinstance(actual, int) or isinstance(actual, bool) or actual < 0:
            raise ValueError("Known nonnegative actual shots")
        if len(artifact_digest) != 64 or any(c not in "0123456789abcdef" for c in artifact_digest):
            raise ValueError("Counts/evidence SHA256 required")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT reserved,actual,state,artifact_digest FROM draws WHERE run=? AND event=?", (run, event)).fetchone()
            if old is None or actual > old[0]:
                raise ValueError("Missing/insufficient reservation")
            if old[2] == "COMPLETED":
                if old[1] != actual or old[3] != artifact_digest:
                    raise ValueError("Immutable completed draw changed")
                return
            db.execute("UPDATE draws SET actual=?,state='COMPLETED',artifact_digest=? WHERE run=? AND event=?", (actual, artifact_digest, run, event))

    def record_readout(self, run, identity, draw, spec):
        spec = json.dumps(spec, sort_keys=True)
        with self.connect() as db:
            if db.execute("SELECT state FROM draws WHERE run=? AND event=?", (run, draw)).fetchone() != ("COMPLETED",):
                raise ValueError("CI readout must reference an actual completed draw")
            old = db.execute("SELECT draw,spec FROM readouts WHERE run=? AND id=?", (run, identity)).fetchone()
            if old and old != (draw, spec):
                raise ValueError("Readout identity changed")
            db.execute("INSERT OR IGNORE INTO readouts VALUES(?,?,?,?)", (run, identity, draw, spec))

    def snapshot(self, run):
        with self.connect() as db:
            cap = db.execute("SELECT cap FROM budgets WHERE run=?", (run,)).fetchone()[0]
            known, upper, unknown = db.execute("SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE actual END),0),SUM(actual IS NULL) FROM draws WHERE run=?", (run,)).fetchone()
            return {"cap": cap, "known_actual_shots": known, "charged_upper_bound": upper,
                    "unknown_or_inflight_draws": unknown or 0, "remaining_safe_budget": cap-upper}
