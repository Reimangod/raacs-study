"""Durable observer around the qualified official SQD workflow, not a new solver.

All SDK arithmetic/settings remain in the frozen v1 workflow. Each API boundary
is committed before the next operation, including if that operation raises.
One process owns this instrumented SDK module; concurrent threads are forbidden.
"""
from __future__ import annotations
import json
import sqlite3
import zlib
from unittest.mock import patch
import numpy as np
import qiskit_addon_sqd.fermion as fermion
from .common import digest
from .fermionic_sqd_workflow import full_fermionic_sqd, cartesian_ids
from .sqd import blocked_to_ids


def encode(value):
    return zlib.compress(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), 6)


def decode(value):
    return json.loads(zlib.decompress(value))


class DurableTrace:
    def __init__(self, db, task):
        self.db, self.task = db, task
        db.execute("CREATE TABLE IF NOT EXISTS trace(task TEXT,seq INTEGER,kind TEXT,payload BLOB,previous TEXT,sha TEXT,PRIMARY KEY(task,seq))")
        db.commit()
        if db.execute("SELECT 1 FROM trace WHERE task=?", (task,)).fetchone():
            raise RuntimeError("Existing partial trace must not be silently replayed")
        self.seq, self.previous = 0, None

    def emit(self, kind, payload):
        record = {"task": self.task, "seq": self.seq, "kind": kind, "payload": payload, "previous": self.previous}
        sha = digest(record)
        self.db.execute("INSERT INTO trace VALUES(?,?,?,?,?,?)", (self.task, self.seq, kind, encode(payload), self.previous, sha))
        self.db.commit()
        self.seq += 1
        self.previous = sha


def read_trace(db, task):
    previous = None
    records = []
    for seq, kind, raw, parent, sha in db.execute("SELECT seq,kind,payload,previous,sha FROM trace WHERE task=? ORDER BY seq", (task,)):
        payload = decode(raw)
        record = {"task": task, "seq": seq, "kind": kind, "payload": payload, "previous": parent}
        if seq != len(records) or parent != previous or digest(record) != sha:
            raise RuntimeError("SQD trace digest/sequence mismatch")
        records.append({**record, "sha": sha}); previous = sha
    return records


def full_sqd_durable(h1, eri, core, hf, na, nb, ids, counts, spec, trace):
    """Observe the real official calls; no extra optimization or sample acquisition."""
    raw = set(map(int, ids)); norb = len(h1)
    original_solve = fermion.solve_sci_batch
    original_recover = fermion.recover_configurations
    original_full = fermion.diagonalize_fermionic_hamiltonian
    trace.emit("START", {"raw_ids": list(map(int, ids)), "raw_counts": list(map(int, counts)),
                         "raw_shots": int(sum(counts)), "hardware_shots": 0, "FCI_used": False})

    def solve(*args, **kwargs):
        strings = args[0]
        trace.emit("SOLVE_REQUEST", {"spaces": [{"alpha": list(map(int, a)), "beta": list(map(int, b))} for a, b in strings]})
        results = original_solve(*args, **kwargs)
        for result in results:
            state = result.sci_state
            if state is None: raise RuntimeError("Official selected-space state unavailable")
            ids_and_weights = []
            for i, a in enumerate(state.ci_strs_a):
                for j, b in enumerate(state.ci_strs_b):
                    x = cartesian_ids([a], [b], norb)[0]
                    ids_and_weights.append((x, float(abs(state.amplitudes[i, j])**2)))
            ids_and_weights.sort()
            trace.emit("SOLVE_RESULT", {"energy_Ha": float(result.energy)+float(core),
                "selected_ids": [x for x, w in ids_and_weights], "coefficient_weights": [w for x, w in ids_and_weights],
                "alpha": list(map(int, state.ci_strs_a)), "beta": list(map(int, state.ci_strs_b)),
                "new_cartesian_ids_not_raw": sorted(set(x for x, w in ids_and_weights)-raw),
                "spin_square": float(state.spin_square()), "new_quantum_shots": 0})
        return results

    def recover(*args, **kwargs):
        trace.emit("RECOVERY_REQUEST", {"new_quantum_shots": 0})
        result = original_recover(*args, **kwargs)
        xs = list(map(int, blocked_to_ids(result[0])))
        trace.emit("RECOVERY_RESULT", {"ids": xs, "classical_weights": np.asarray(result[1], float).tolist(),
            "new_recovered_ids_not_raw": sorted(set(xs)-raw), "new_quantum_shots": 0})
        return result

    def full(*args, **kwargs):
        callback = kwargs["callback"]
        def durable_callback(results):
            trace.emit("ITERATION", {"batch_energies_Ha": [float(r.energy)+float(core) for r in results],
                "occupancies": [[list(map(float, r.orbital_occupancies[0])), list(map(float, r.orbital_occupancies[1]))] for r in results]})
            callback(results)
        kwargs["callback"] = durable_callback
        return original_full(*args, **kwargs)

    try:
        with patch.object(fermion, "solve_sci_batch", solve), patch.object(fermion, "recover_configurations", recover), patch.object(fermion, "diagonalize_fermionic_hamiltonian", full):
            out = full_fermionic_sqd(h1, eri, core, hf, na, nb, ids, counts, spec)
        trace.emit("COMPLETED", {"energy_Ha": out["energy_Ha"], "dimension": out["dimension"],
            "selected_ids": out["selected_interleaved_ids"], "new_quantum_shots": 0})
        return out
    except Exception as exc:
        trace.emit("FAILED_PRESERVED", {"type": type(exc).__name__, "error": str(exc), "new_quantum_shots": 0})
        raise
