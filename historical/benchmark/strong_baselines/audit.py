"""Independent final-state, ledger and readout checks of B1 cells."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from ..common import ROOT, file_digest, save_json
from ..chemistry import load_reference
from ..study.core import unpack_pairs


def audit(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["source"]); failures = []; cells = []
    for name, sha in manifest["implementation"].items():
        if file_digest(ROOT / name) != sha:
            failures.append("implementation drift: " + name)
    for case in manifest["case_ids"]:
        ref = load_reference(source / "references" / case)
        for spec in manifest["methods"]:
            d = root / "cells" / case / spec["name"]
            row = json.loads((d / "result.json").read_text())
            if row["status"] == "FAILED_PRESERVED":
                expected = spec["init"] == "ccsd" and not (source / "references" / case / "ccsd.json").exists()
                if not expected:
                    failures.append(f"unexpected failure {case}/{spec['name']}: {row.get('error')}")
                cells.append({"case_id": case, "method": spec["name"], "status": "PRESERVED_SOURCE_CCSD_MISSING" if expected else "UNEXPECTED_FAILURE"})
                continue
            if file_digest(d / "final.npz") != row["final_sha256"]:
                failures.append(f"final SHA {case}/{spec['name']}")
            with np.load(d / "final.npz") as z:
                state, p = z["state"], z["probability"]
            energy = float(np.vdot(state, ref["h"] @ state).real)
            assert np.isclose(np.linalg.norm(state), 1., atol=1e-9)
            assert abs(energy-row["vqe_energy"]) < 1e-9
            assert np.max(abs(p-abs(state)**2)) < 1e-12
            assert abs(p[ref["ids"]].sum()-1.) < 1e-9
            with (d / "preparation.qpy").open("rb") as stream:
                circuit = qpy.load(stream)[0]
            prepared = Statevector.from_instruction(circuit).data
            overlap = np.vdot(prepared, state)
            assert abs(abs(overlap)-1.) < 1e-8
            with sqlite3.connect(d / "objective.sqlite") as db:
                n, shots = db.execute("SELECT count(*),coalesce(sum(shots),0) FROM calls").fetchone()
                assert n == row["objective_evaluations"] <= manifest["budget"]["max_evals"]
                assert int(shots) == row["simulated_construction_shots"]
                for raw, budget in db.execute("SELECT counts,shots FROM calls WHERE shots>0"):
                    _, counts = unpack_pairs(raw)
                    assert counts.sum() == budget
            cells.append({"case_id": case, "method": spec["name"], "status": "PASS", "objective_evaluations": n})
        path = root / "readouts" / case / "readouts.sqlite"
        with sqlite3.connect(path) as db:
            for M, raw in db.execute("SELECT shots,counts FROM draws"):
                _, ns = unpack_pairs(raw)
                assert ns.sum() == M
            ns = db.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
            actual = db.execute("SELECT count(*) FROM outcomes").fetchone()[0]
            assert actual == ns*len(manifest["budget"]["M_grid"])*manifest["budget"]["evaluation_repeats"]*len(manifest["budget"]["R_grid"])
            assert db.execute("SELECT count(*) FROM pair_readouts").fetchone()[0] == 2*actual
            # Every recorded selected subspace must consist of valid determinants.
            import zlib
            valid = set(map(int, ref["ids"]))
            for raw in db.execute("SELECT ids FROM subspaces"):
                ids = np.frombuffer(zlib.decompress(raw[0]), dtype="<i4")
                assert len(ids) == len(set(ids)) and set(map(int, ids)) <= valid
    result = {"status": "FAILED" if failures else "QUICK_B1_VALIDATED" if manifest["mode"] == "quick" else "B1_VALIDATED_WITH_LIMITS",
              "cells": cells, "failures": failures, "hardware_shots": 0,
              "full_milestone2_complete": False, "whole_study_complete": False}
    save_json(root / "validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    if audit(p.parse_args().output)["failures"]:
        raise SystemExit(1)
