"""Independent artifact/count/dimension audit, with explicit fresh-QPY scope."""
from __future__ import annotations
import argparse
import gzip
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, rng_for, save_json
from .chemistry import load_reference
from .study.core import unpack_pairs, ranks
from .sqdrift_ensemble import member_specs
from .sqdrift_ensemble_evaluate import allocate, draw_members, load_members
from .time_evolution_primitives import apply_sequence


def audit(root, construction_only=False):
    root = Path(root); mf = json.loads((root / "manifest.json").read_text()); source = Path(mf["source"])
    problems = []; member_count = fresh = draws = components = controls = shots_total = ritz = 0
    with threadpool_limits(limits=1):
        for name, sha in mf["implementation"].items():
            if file_digest(ROOT / name) != sha:
                problems.append("bound code/input changed: "+name)
        for case in mf["case_ids"]:
            ref = load_reference(source / "references" / case); all_members = {}
            for spec in mf["methods"]:
                result = json.loads((root / "cells" / case / spec["name"] / "result.json").read_text())
                expected = member_specs(ref["meta"]["request_digest"], spec)
                assert result["member_ids"] == [m["member_id"] for m in expected]
                if result["status"] != "FIXED_ACTUAL_SQDRIFT_ENSEMBLE":
                    problems.append(case+"/"+spec["name"]+": failed members preserved")
                    continue
                for m in expected:
                    directory = root / "native" / case / "members" / m["member_id"]
                    rec = json.loads((directory / "result.json").read_text()); member_count += 1
                    assert rec["spec"] == m and rec["hardware_shots"] == 0 and not rec["FCI_used_for_selection"]
                    assert rec["group_first_moment_error"] < 1e-10 and rec["unit_group_norm_upper_bound"] <= 1+1e-9
                    assert abs(rec["valid_N_Sz_probability"]-1) < 1e-8
                    assert rec["state_parity_error"] < 1e-8
                    assert file_digest(directory / "probability.npz") == rec["probability_sha256"]
                    for filename, key in (("sdk-result.json.gz", "sdk_result_gzip_sha256"), ("sdk-circuit.qpy.gz", "sdk_circuit_gzip_sha256")):
                        assert file_digest(directory / filename) == rec["SDK_export_event"][key]
                    all_members[m["member_id"]] = directory
            ids = sorted(all_members)
            rng, _ = rng_for("SqDRIFT-independent-QPY-audit-v1", case)
            selected = ids if mf["mode"] == "quick" else sorted({ids[0], ids[-1], ids[int(rng.integers(len(ids)))]})
            for mid in selected:
                d = all_members[mid]; sdk = json.loads(gzip.decompress((d / "sdk-result.json.gz").read_bytes()))
                n = ref["meta"]["n_qubits"]; initial = np.zeros(1 << n, complex); initial[ref["meta"]["hf_id"]] = 1
                expected = apply_sequence(initial, sdk["sequence_for_independent_source_basis_parity"])
                qc = qpy.load(BytesIO(gzip.decompress((d / "preparation.qpy.gz").read_bytes())))[0]
                actual = Statevector.from_instruction(qc).data
                assert abs(abs(np.vdot(expected, actual))-1) < 1e-8
                qc = qpy.load(BytesIO(gzip.decompress((d / "sampling.qpy.gz").read_bytes())))[0]
                with np.load(d / "probability.npz") as z:
                    assert np.allclose(abs(Statevector.from_instruction(qc).data)**2, z["probability"], atol=1e-9)
                fresh += 1
            if construction_only:
                continue
            directory = root / "readouts" / case
            db = sqlite3.connect(directory / "readouts.sqlite")
            cp = {i: (sha, method, json.loads(meta)) for i, sha, method, meta in db.execute("SELECT id,sha,method,metadata FROM checkpoints")}
            for draw, c, M, rep, raw in db.execute("SELECT id,cp,shots,rep,counts FROM draws"):
                raw_ids, ns = unpack_pairs(raw); assert int(ns.sum()) == M; draws += 1
                role = cp[c][2]["role"]
                if role == "NEW_ENSEMBLE":
                    result = json.loads((root / "cells" / case / cp[c][1] / "result.json").read_text())
                    allocation = allocate(M, len(result["member_ids"]))
                    mrows = {mid: (m, seed, counts) for mid, m, seed, counts in db.execute("SELECT member_id,allocated_shots,seed,counts FROM member_draws WHERE draw=?", (draw,))}
                    assert set(mrows) == set(result["member_ids"])
                    aggregate = {}
                    for mid, m in zip(result["member_ids"], allocation):
                        allocated, seed, blob = mrows[mid]; ids2, counts = unpack_pairs(blob)
                        assert allocated == m == int(counts.sum())
                        _, expected_seed = rng_for("native-SqDRIFT-member-readout-v1", [case, result["task_id"], M, rep], mid)
                        assert seed == str(expected_seed)
                        components += 1; shots_total += m
                        for x, count in zip(ids2, counts):
                            aggregate[int(x)] = aggregate.get(int(x), 0)+int(count)
                    assert sorted(aggregate.items()) == sorted(zip(map(int, raw_ids), map(int, ns)))
                    if M == mf["budget"]["M_grid"][0] and rep == 0:
                        members, probabilities = load_members(root, case, result)
                        ii, nn, _ = draw_members(probabilities, allocation, [case, result["task_id"], M, rep], result["member_ids"])
                        assert np.array_equal(ii, raw_ids) and np.array_equal(nn, ns)
                        expected_cnot = sum(m["resources"]["cnot_count"]*a for m, a in zip(members, allocation))
                        assert db.execute("SELECT cnot_shots FROM ensemble_resources WHERE draw=?", (draw,)).fetchone()[0] == expected_cnot
                else:
                    row = db.execute("SELECT source_database,source_draw_id FROM reused_controls WHERE draw=?", (draw,)).fetchone()
                    assert row
                    src = sqlite3.connect(f"file:{row[0]}?mode=ro", uri=True)
                    original = src.execute("SELECT counts FROM draws WHERE id=?", (row[1],)).fetchone()[0]; src.close()
                    assert original == raw; controls += 1
            rows = list(db.execute("SELECT ids,energy,dimension FROM subspaces ORDER BY sha"))
            pick = np.linspace(0, len(rows)-1, min(5, len(rows)), dtype=int)
            for i in pick:
                blob, recorded, dim = rows[i]
                # Subspaces use a different packed ID format; see core.py.
                ids = np.frombuffer(__import__("zlib").decompress(blob), dtype="<i4")
                sub = ref["h"][ids][:, ids].toarray()
                energy = float(eigh(sub, subset_by_index=[0, 0], check_finite=False)[0][0])
                assert len(ids) == dim and abs(energy-recorded) < 1e-8; ritz += 1
            db.close()
    result = {"status": ("QUICK_SQDRIFT_ENSEMBLE_VALIDATED" if mf["mode"] == "quick" else "STANDARD_SQDRIFT_ENSEMBLE_AVAILABLE_VALIDATED") if not problems else "FAILED_PRESERVED",
              "issues": problems, "member_circuits_audited": member_count, "fresh_QPY_parity_members": fresh,
              "aggregate_draws": draws, "component_draws": components, "new_statistical_model_shots": shots_total,
              "reused_control_draws": controls, "fresh_independent_Ritz_checks": ritz, "construction_only": construction_only,
              "audit_scope": "all count totals/seed identities/member aggregation/control reuse; fresh RNG replay first M/rep0 per ensemble; QPY every quick member or3/case standard; Ritz5/case",
              "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / ("construction-validation.json" if construction_only else "validation.json"), result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--construction-only", action="store_true")
    args = p.parse_args(); result = audit(args.output, args.construction_only); print(json.dumps(result))
    if result["issues"]:
        raise SystemExit(1)
