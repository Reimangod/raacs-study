"""Actual molecular20–44q integral-only/MPS cohort; no FCI/full Fock vectors.

This version reuses, never edits, the frozen RAACS-v1 numerical engine.
Approximation checks compare identical circuits, not independently grown paths.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
from unittest.mock import patch
import zlib
import numpy as np
from qiskit import QuantumCircuit, qpy
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .molecular_problem_no_fci import MolecularSpec, build_problem
from .resource_adaptive_sampler import RAACSSpec, integral_problem, run_sampler
from .circuit_sampling_runtime import CircuitCountSampler, SamplingSpec
from .local_sampler_primitives import observed_ci, pack_counts64, valid_sector
from .mps_sampling import construct_mps, sample_mps
from .ansatz import circuit_bytes, measurement_equivalent_circuit, resources
from .study.core import rss_peak


def problem_queue(mode):
    if mode == "quick":
        specs = [MolecularSpec(m, 1.25) for m in ("H4", "H8", "H10")]
    elif mode == "standard":
        specs = [MolecularSpec(m, r) for m in ("H8", "H10", "H12", "H16", "H20", "H22") for r in (1.25, 1.8)]
        specs += [MolecularSpec("H2O", .96), MolecularSpec("C2", 1.25)]
        specs += [MolecularSpec("N2", 1.25, 2, n, "cc-pvdz") for n in (12, 16)]
    else:
        raise ValueError("Quick or standard")
    return [{"id": f"{s.molecule}_R{s.distance_A}_{digest(asdict(s))[:10]}", "spec": asdict(s)} for s in specs]


@contextmanager
def no_full_state_guards():
    """Prevent accidental small-backend/FCI fallbacks on the practical route."""
    import pyscf.fci.direct_spin1
    import pyscf.fci.direct_spin0
    import quimb.tensor as qtn
    from qiskit.quantum_info import Statevector
    def forbidden(*args, **kwargs):
        raise RuntimeError("SCALING_NO_FCI_NO_FULL_FOCK_STATE_BOUNDARY")
    with ExitStack() as stack:
        for obj, name in ((pyscf.fci.direct_spin1, "kernel"), (pyscf.fci.direct_spin0, "kernel"),
                          (Statevector, "from_instruction"), (qtn.MatrixProductState, "to_dense")):
            stack.enter_context(patch.object(obj, name, forbidden))
        yield


def configuration(mode):
    files = ("molecular_mps_scaling.py", "molecular_problem_no_fci.py", "resource_adaptive_sampler.py",
             "circuit_sampling_runtime.py", "local_sampler_primitives.py", "integral_hamiltonian.py",
             "integral_connections.py", "mps_sampling.py", "ansatz.py", "common.py", "study/core.py")
    return {"schema": "molecular-integral-MPS-scaling-v1", "mode": mode, "queue": problem_queue(mode),
            "scores": ["ci_gradient_local", "residual_cost", "full_cost"], "shots_per_cell": 2000,
            "bond_checks": [16, 32, 64, 128], "diagnostic_shots_per_bond": 384,
            "free_floor_bytes": 3*2**30, "output_boundary_bytes": 512*2**20,
            "implementation": {f"benchmark/{p}": file_digest(ROOT/"benchmark"/p) for p in files},
            "spec_sha256": file_digest(ROOT/"study-v1/MOLECULAR_MPS_SCALING.md"),
            "hardware_shots": 0, "FCI_allowed": False, "full_Fock_vector_allowed": False,
            "native_2Q_resources": None, "ground_state_reference": None}


def write_counts(path, ids, counts):
    path.write_bytes(pack_counts64(ids, counts))
    return {"path": path.name, "sha256": file_digest(path), "shots": int(counts.sum())}


def hf_control(problem, folder, M):
    folder.mkdir(parents=True, exist_ok=True)
    backend, info = integral_problem(problem); path = folder/"result.json"
    binding = {"problem": info, "shots": M, "method": "physical-HF-MPS", "version": 1}
    if path.exists():
        old = json.loads(path.read_text())
        if old["binding"] != binding or file_digest(folder/"counts.u64z") != old["counts"]["sha256"]:
            raise RuntimeError("HF control identity drift")
        return old
    qc = QuantumCircuit(info["n_qubits"], name="scaling-hf")
    for q in range(qc.num_qubits):
        if info["hf_id"] >> q & 1: qc.x(q)
    seed = rng_for("scaling-HF-v1", binding)[1]
    ids, counts, meta = CircuitCountSampler(SamplingSpec()).draw(qc, M, seed)
    ci = observed_ci(backend, dict(zip(map(int, ids), map(int, counts))), info["hf_id"], info["nalpha"], info["nbeta"], 64)
    (folder/"preparation.qpy").write_bytes(circuit_bytes(qc))
    result = {"status": meta["sample_metadata"]["status"], "binding": binding, "metadata": meta,
              "counts": write_counts(folder/"counts.u64z", ids, counts), "energy_Ha": ci["energy_Ha"],
              "actual_dimension": ci["actual_dimension"], "FCI_used": False, "hardware_shots": 0}
    save_json(path, result); return result


def same_circuit_bonds(problem, cell, folder, bonds, shots):
    folder.mkdir(parents=True, exist_ok=True); path = folder/"result.json"
    source = json.loads((cell/"result.json").read_text()); sha = source["event_database_sha256"]
    if file_digest(cell/"events.sqlite") != sha: raise RuntimeError("Source adaptive DB changed")
    allocations = [a for a in source["final_circuit_allocations"] if a["shots"] > 0]
    if len(allocations) != 1: raise ValueError("Registered scaling uses last-only physical circuit")
    cid = allocations[0]["circuit_id"]
    binding = {"source_database_sha256": sha, "preparation_id": cid, "bonds": bonds, "shots_per_bond": shots}
    if path.exists():
        result = json.loads(path.read_text())
        if result["binding"] != binding: raise RuntimeError("Bond diagnostic drift")
        for row in result["rows"]:
            if "counts" in row and file_digest(folder/row["counts"]["path"]) != row["counts"]["sha256"]:
                raise RuntimeError("Bond diagnostic counts changed")
        return result
    db = sqlite3.connect(f"file:{cell/'events.sqlite'}?mode=ro", uri=True)
    raw, actions = db.execute("SELECT preparation_qpy,actions FROM circuits WHERE id=?", (cid,)).fetchone(); db.close()
    prep_raw = zlib.decompress(raw)
    import hashlib
    if hashlib.sha256(prep_raw).hexdigest() != cid: raise RuntimeError("Physical preparation identity mismatch")
    prep = qpy.load(BytesIO(prep_raw))[0]; sampling, _ = measurement_equivalent_circuit(prep)
    (folder/"same-preparation.qpy").write_bytes(prep_raw)
    backend, info = integral_problem(problem); rows = []; states = []
    for bond in bonds:
        seed = rng_for("same-physical-circuit-bond-v1", binding, bond)[1]; start = perf_counter()
        try:
            state, build = construct_mps(sampling, max_bond=bond)
            ids, counts, sample = sample_mps(state, shots, seed)
            ci = observed_ci(backend, dict(zip(map(int, ids), map(int, counts))), info["hf_id"], info["nalpha"], info["nbeta"], 64)
            psi = state.psi.copy(); psi.normalize(); states.append((bond, psi))
            row = {"bond": bond, "status": sample["status"], "seed": str(seed), "build": build, "sample": sample,
                   "counts": write_counts(folder/f"bond-{bond}.u64z", ids, counts), "energy_Ha": ci["energy_Ha"],
                   "actual_dimension": ci["actual_dimension"], "valid_sector_fraction": sum(int(c) for x, c in zip(ids, counts)
                       if valid_sector(int(x), info["n_qubits"], info["nalpha"], info["nbeta"]))/max(1, int(counts.sum()))}
        except Exception as exc:
            row = {"bond": bond, "status": "ENGINEERING_OR_RESOURCE_FAILURE_PRESERVED", "error": repr(exc)}
        row["wall_seconds"] = perf_counter()-start; row["process_RSS_hwm_bytes"] = rss_peak(); rows.append(row)
        save_json(folder/"progress.json", {"binding": binding, "rows": rows})
    if states:
        refbond, ref = states[-1]
        for row in rows:
            matched = next((s for b, s in states if b == row["bond"]), None)
            if matched is not None:
                fidelity = float(abs(matched.H @ ref)**2)
                row["normalized_MPS_overlap_squared_to_highest_completed_bond"] = fidelity
                row["approximation_reference_bond"] = refbond
                if not -1e-8 <= fidelity <= 1+1e-8: raise RuntimeError("Invalid normalized MPS overlap")
    result = {"binding": binding, "rows": rows, "identical_physical_circuit": True,
              "not_exact_error_certificate": True, "used_for_policy_selection": False, "hardware_shots": 0}
    save_json(path, result); return result


def run(root, mode):
    import fcntl
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1), no_full_state_guards():
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = configuration(mode); path = root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != cfg: raise RuntimeError("Scaling identity drift; successor required")
        if mode == "standard":
            validation = ROOT/"results/molecular-MPS-scaling-quick-v1/validation.json"
            if not validation.exists() or json.loads(validation.read_text())["status"] != "SCALING_COUNTS_AND_BINDINGS_VALIDATED":
                raise RuntimeError("Actual quick scaling audit required before standard")
        if not path.exists() and shutil.disk_usage(root).free < cfg["free_floor_bytes"]+cfg["output_boundary_bytes"]:
            raise RuntimeError("Insufficient registered scaling capacity; no deletions")
        save_json(path, cfg); rows = []; started = perf_counter()
        for item in cfg["queue"]:
            size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if size > cfg["output_boundary_bytes"] or shutil.disk_usage(root).free < cfg["free_floor_bytes"]:
                raise RuntimeError("Scaling storage boundary; preserve partial results")
            case = root/"cases"/item["id"]; problem = case/"physical-problem"
            try:
                physical = build_problem(MolecularSpec(**item["spec"]), problem, small_full_H=False)
                if physical["meta"]["FCI_computed"] or physical["meta"]["full_Fock_H_constructed"]: raise RuntimeError("Forbidden physical source")
            except Exception as exc:
                save_json(case/"physical-failure.json", {"error": repr(exc), "spec": item["spec"]})
                rows.extend({"case": item["id"], "method": method, "status": "PHYSICAL_INPUT_FAILURE_PRESERVED"}
                            for method in ["HF"]+cfg["scores"])
                continue
            hf = hf_control(problem, case/"HF", cfg["shots_per_cell"])
            rows.append({"case": item["id"], "method": "HF", "status": hf["status"]})
            for score in cfg["scores"]:
                cell = case/score; failure = cell/"failure.json"
                if failure.exists():
                    rows.append({"case": item["id"], "method": score, "status": "ENGINEERING_FAILURE_PRESERVED"}); continue
                try:
                    spec = RAACSSpec(score=score, total_model_shots=cfg["shots_per_cell"], max_steps=4,
                        parent_connection_cap=64, target_cap=64, candidate_bank_cap=16, candidates_measured_per_step=2)
                    result = run_sampler(problem, cell, spec, SamplingSpec(backend="mps"))
                    same_circuit_bonds(problem, cell, case/"bond-diagnostics"/score, cfg["bond_checks"], cfg["diagnostic_shots_per_bond"])
                    rows.append({"case": item["id"], "method": score, "status": result["status"]})
                except Exception as exc:
                    save_json(failure, {"error": repr(exc), "partial_events_preserved": True})
                    rows.append({"case": item["id"], "method": score, "status": "ENGINEERING_FAILURE_PRESERVED"})
                save_json(root/"execution-status.json", {"status": "RUNNING", "terminal": len(rows), "planned": len(cfg["queue"])*4, "rows": rows})
        out = {"status": "SCALING_EXECUTED_PENDING_AUDIT_REPORT", "terminal": len(rows), "planned": len(cfg["queue"])*4,
               "rows": rows, "actual_wall_seconds": perf_counter()-started, "hardware_shots": 0,
               "FCI_used": False, "full_Fock_vector_used": False, "whole_study_complete": False}
        save_json(root/"execution-status.json", out); return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--output"); a = p.parse_args()
    print(json.dumps(run(a.output or ROOT/f"results/molecular-MPS-scaling-{a.mode}-v1", a.mode)))


if __name__ == "__main__": main()
