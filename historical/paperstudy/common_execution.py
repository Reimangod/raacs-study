"""Common-budget molecular execution, with paid/replayable primitive draws.

Native RAACS selection is reused unchanged. A resumed attempt reconstructs its
deterministic decisions from committed counts; quantum-model draws are not
repeated. Extra classical reconstruction time is retained as attempt work.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import sqlite3
from time import perf_counter, process_time, time
from unittest.mock import patch

import ffsim
import numpy as np
from qiskit import qpy

from benchmark import resource_adaptive_sampler as native
from benchmark.circuit_sampling_runtime import SamplingSpec
from benchmark.local_sampler_primitives import LocalQE, unpack_counts64

from .budget import BudgetLedger
from .ci import solve, interleaved_to_blocked
from .input_adapter import array_bytes
from .io import ROOT, PLAN_ROOT, atomic_bytes, assert_safe_physical_input, digest, save_json, sha256
from .lucj import amplitudes_from_saved, circuit_views, interaction_pairs, make_circuit, naive
from .sparse_raacs import ExactSparseSampler, traced_append


class UnknownDrawOutcome(RuntimeError):
    pass


class JournalSampler:
    """Persist result before ledger completion; unknown draws are never rerun."""
    def __init__(self, provider, directory, run_id, cap):
        self.provider = provider
        self.directory = Path(directory)
        self.run_id = run_id
        self.ledger = BudgetLedger(ROOT/"common-budget.sqlite")
        self.ledger.register(run_id, int(cap))
        self.replayed_draws = 0
        self.new_draws = 0

    @property
    def MPS_builds(self):
        return getattr(self.provider, "MPS_builds", 0)

    @property
    def exact_small_builds(self):
        return getattr(self.provider, "exact_small_builds", 0)

    def draw(self, preparation, shots, seed, n_parameters=0, n_operators=0):
        trace = (preparation.metadata or {}).get("paper_exact_sparse_actions", [] if n_operators == 0 else None)
        if trace is None:
            raise RuntimeError("RAACS physical trace is required for journal binding")
        request = {"n_qubits": preparation.num_qubits, "actions": trace,
                   "shots": int(shots), "seed": str(seed), "n_parameters": n_parameters,
                   "n_operators": n_operators, "hf_id": self.provider.hf_id}
        event = digest(request)
        directory = self.directory/event
        evidence = directory/"draw.json"
        state = self.ledger.reserve(self.run_id, event, request, int(shots))
        if evidence.exists():
            record = json.loads(evidence.read_text())
            if record["request"] != request or sha256((directory/"counts.npz").read_bytes()) != record["counts_sha256"]:
                raise RuntimeError("Committed count artifact/request mismatch")
            self.ledger.complete(self.run_id, event, record["actual_shots"], record["counts_sha256"])
            with np.load(directory/"counts.npz", allow_pickle=False) as a:
                ids, counts = a["ids"].copy(), a["counts"].copy()
            self.replayed_draws += 1
            return ids, counts, record["metadata"]
        if state != "NEW_RESERVATION":
            raise UnknownDrawOutcome("Reserved draw lacks committed counts; preserve reservation, do not duplicate work")
        tick = perf_counter()
        ids, counts, metadata = self.provider.draw(preparation, shots, seed, n_parameters, n_operators)
        payload = array_bytes({"ids": ids, "counts": counts})
        atomic_bytes(directory/"counts.npz", payload)
        record = {"event_id": event, "request": request, "counts_sha256": sha256(payload),
                  "actual_shots": int(counts.sum()), "metadata": metadata,
                  "new_primitive_wall_seconds": perf_counter()-tick}
        save_json(evidence, record)
        self.ledger.complete(self.run_id, event, record["actual_shots"], record["counts_sha256"])
        self.new_draws += 1
        return ids, counts, metadata


def load_physical(case):
    path = ROOT/"data/physical"/case
    metadata = json.loads((path/"problem.json").read_text())
    assert_safe_physical_input(metadata)
    payload = (path/"integrals.npz").read_bytes()
    with np.load(BytesIO(payload), allow_pickle=False) as a:
        h1, eri = a["h1"].copy(), a["eri"].copy()
    if {k: sha256(v.astype("<f8").tobytes()) for k, v in (("h1", h1), ("eri", eri))} != metadata["physical_arrays_sha256"]:
        raise RuntimeError("Physical integral digest changed")
    return metadata, h1, eri


def run_identity(case, method, budget, seed, lane):
    return digest(["paper-common-execution-v1", case, method, int(budget), int(seed), lane])


def load_operator(case, compressed, n_reps=1):
    meta, _, _ = load_physical(case)
    if compressed:
        path = ROOT/"qualification"/(case+"-author-compressed-fresh")
        record = json.loads((path/"result.json").read_text())
        payload = (path/"operator.npz").read_bytes()
        if record["status"] != "AUTHOR_INITIALIZER_RETURNED" or sha256(payload) != record["operator_sha256"]:
            raise RuntimeError("Compressed initializer binding invalid")
        if n_reps != 1:
            raise NotImplementedError("Extra compressed repetitions need their actual author initializer")
        with np.load(BytesIO(payload), allow_pickle=False) as a:
            op = ffsim.UCJOpSpinBalanced(a["diag_coulomb_mats"], a["orbital_rotations"], a["final_orbital_rotation"])
        cost = {"initialization_kind": record["initialization"],
                "amplitude_creation_CPU_seconds": record["original_amplitude_creation_CPU_seconds"],
                "compressed_initialization_CPU_seconds": record["CPU_process_seconds"],
                "compressed_initialization_wall_seconds": record["total_wall_seconds"],
                "all_initializer_stages_converged": record["all_stages_optimizer_success"],
                "source_result": str(path.relative_to(ROOT)/"result.json"), "operator_sha256": sha256(payload)}
    else:
        path = ROOT/"data/initialization-fresh"/case
        record = json.loads((path/"result.json").read_text())
        payload = (path/"amplitudes.npz").read_bytes()
        if record["status"] != "FRESH_INITIALIZATION_COMPLETED" or sha256(payload) != record["amplitudes_sha256"]:
            raise RuntimeError("Fresh amplitude binding invalid")
        tick, cpu = perf_counter(), process_time()
        t1, t2, _ = amplitudes_from_saved(path/"amplitudes.npz", meta["norb"], meta["nalpha"])
        op = naive(t1, t2, meta["norb"], n_reps)
        cost = {"initialization_kind": "FRESH_CONVERGED_"+record["selected_method"],
                "amplitude_creation_CPU_seconds": record["CPU_process_seconds"],
                "naive_factorization_CPU_seconds": process_time()-cpu,
                "naive_factorization_wall_seconds": perf_counter()-tick,
                "source_result": str(path.relative_to(ROOT)/"result.json")}
    return op, cost


def prepare_lucj(case, method, n_reps=1):
    """Fixed physical state once per configuration, not one state per readout."""
    path = ROOT/"prepared"/case/(method+"-L"+str(n_reps))
    path.mkdir(parents=True, exist_ok=True)
    meta, _, _ = load_physical(case)
    op, cost = load_operator(case, "COMPRESSED" in method, n_reps)
    filename = path/"construction.json"
    if filename.exists():
        old = json.loads(filename.read_text())
        if old["problem_id"] != meta["problem_id"]:
            raise RuntimeError("Prepared case binding changed")
        return op, old
    started, cpu = perf_counter(), process_time()
    prep, sampling, views = circuit_views(make_circuit(op, meta["norb"], (meta["nalpha"], meta["nbeta"])))
    for name, circuit in (("preparation", prep), ("sampling", sampling)):
        out = BytesIO()
        qpy.dump(circuit, out)
        atomic_bytes(path/(name+".qpy"), out.getvalue())
    operator_payload = array_bytes({"diag_coulomb_mats": op.diag_coulomb_mats,
                                   "orbital_rotations": op.orbital_rotations,
                                   "final_orbital_rotation": op.final_orbital_rotation})
    atomic_bytes(path/"operator.npz", operator_payload)
    report = {"case_id": case, "problem_id": meta["problem_id"], "method_id": method,
              "n_reps": n_reps, "operator_sha256": sha256(operator_payload), "resources": views,
              "coefficient_slots": len(op.to_parameters(interaction_pairs=interaction_pairs(meta["norb"]))),
              "nonredundant_independent_parameters": None,
              "parameter_unknown_reason": "ffsim coordinate length is not a rank certificate",
              "VQE_optimized_variables": 0, "initialization_cost": cost,
              "compilation_CPU_seconds": process_time()-cpu, "compilation_wall_seconds": perf_counter()-started,
              "fixed_coefficients_do_not_mean_zero_ansatz_parameters": True}
    save_json(filename, report)
    return op, report


def final_readouts(directory, meta, h1, eri, blocked_ids, counts):
    result = []
    for r in (8, 16, 32):
        filename = directory/("CI-r"+str(r)+".json")
        if filename.exists():
            item = json.loads(filename.read_text())
        else:
            item = solve(h1, eri, meta["core_energy"], (meta["nalpha"], meta["nbeta"]),
                         blocked_ids, counts, r)
            item["problem_id"] = meta["problem_id"]
            item["final_counts_sha256"] = sha256((directory/"final-counts.npz").read_bytes())
            save_json(filename, item)
        result.append(item)
    return result


def run_cell(case, method, budget, seed, lane="S3_ENGINEERING_PILOT", vector=None, construction=None):
    if method not in ("A_RAACS_CURRENT", "B_LUCJ_NAIVE_L1", "C_LUCJ_COMPRESSED_L1"):
        raise NotImplementedError("Unqualified method adapter not substituted")
    run_id = run_identity(case, method, budget, seed, lane)
    directory = ROOT/"common-runs"/lane/case/method/(str(budget)+"-"+str(seed))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = directory/"result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        meta, h1, eri = load_physical(case)
        binding = {"run_id": run_id, "case_id": case, "problem_id": meta["problem_id"],
                   "method_id": method, "budget": int(budget), "seed": int(seed), "lane": lane,
                   "scientific_plan_sha256": sha256((PLAN_ROOT/"plan.json").read_bytes()),
                   "runtime": {p: version(p) for p in ("numpy", "ffsim", "pyscf", "qiskit")},
                   "sampler_code_sha256": sha256(Path(__file__).read_bytes())}
        frozen = directory/"request.json"
        if frozen.exists() and json.loads(frozen.read_text()) != binding:
            raise RuntimeError("Actual run code/config drift; use versioned successor")
        save_json(frozen, binding)
        ledger = BudgetLedger(ROOT/"common-budget.sqlite")
        ledger.register(run_id, int(budget))
        start, cpu = perf_counter(), process_time()
        attempt_number = len(list((directory/"attempts").glob("*/started.json")))+1
        attempt_path = directory/"attempts"/str(attempt_number)
        save_json(attempt_path/"started.json", {"pid": os.getpid(), "start_unix": time(), "binding": binding})
        counts_path = directory/"final-counts.npz"
        try:
            if method == "A_RAACS_CURRENT":
                provider = ExactSparseSampler(SamplingSpec(backend="exact_sparse_QE", cache_entries=2), meta["hf_id"])
                journal = JournalSampler(provider, directory/"draws", run_id, budget)
                spec = native.RAACSSpec(total_model_shots=budget, max_steps=32,
                                       candidates_measured_per_step=4, CI_cap=64, seed=seed)
                original = LocalQE.append
                with patch.object(native, "CircuitCountSampler", lambda config: journal), patch.object(
                        LocalQE, "append", lambda self, parent, theta: traced_append(original, self, parent, theta)):
                    result = native.run_sampler(ROOT/"data/physical"/case, attempt_path/"native", spec,
                                  SamplingSpec(backend="exact_sparse_QE", cache_entries=2))
                native_db = sqlite3.connect(attempt_path/"native/events.sqlite")
                final = native_db.execute("SELECT circuit,counts64,metadata FROM draws WHERE role='independent_final'").fetchall()
                if len(final) != 1:
                    raise RuntimeError("Registered last_only expects one final circuit")
                ids, counts = unpack_counts64(final[0][1])
                blocked_ids = np.array([interleaved_to_blocked(int(x), meta["norb"]) for x in ids], np.uint64)
                metadata = json.loads(final[0][2])
                preparation_resources = metadata["preparation_resources"]
                sampling_resources = metadata["sampling_resources"]
                pilot_shots = result["pilot_raw_model_shots"]
                draw_rows = native_db.execute("SELECT role,raw_shots,metadata FROM draws").fetchall()
                cnot_apps = sum(int(row[1])*json.loads(row[2])["sampling_resources"]["cnot_count"] for row in draw_rows)
                detail = {"native_result": result, "quantum_draws_replayed_this_attempt": journal.replayed_draws,
                          "new_quantum_draws_this_attempt": journal.new_draws,
                          "exact_sparse_state_builds_this_attempt": provider.exact_sparse_builds,
                          "classical_reconstruction_on_resume_is_additional_work": True}
                native_db.close()
                optimized_variables = 0
                slots = len(result["accepted_operators"])
            else:
                if construction is None:
                    op, construction = prepare_lucj(case, method)
                else:
                    op, _ = load_operator(case, "COMPRESSED" in method)
                simulation = 0.
                if vector is None:
                    tick = perf_counter()
                    vector = ffsim.apply_unitary(ffsim.hartree_fock_state(meta["norb"], (meta["nalpha"], meta["nbeta"])),
                                      op, meta["norb"], (meta["nalpha"], meta["nbeta"]), copy=False)
                    simulation = perf_counter()-tick
                event = "independent-final"
                request = {"operator_sha256": construction["operator_sha256"], "problem_id": meta["problem_id"],
                           "seed": seed, "shots": budget, "role": event}
                state = ledger.reserve(run_id, event, request, int(budget))
                if counts_path.exists():
                    with np.load(counts_path, allow_pickle=False) as a:
                        blocked_ids, counts = a["blocked_ids"].copy(), a["counts"].copy()
                else:
                    if state != "NEW_RESERVATION":
                        raise UnknownDrawOutcome("LUCJ reserved draw without committed counts; no automatic rerun")
                    values = ffsim.sample_state_vector(vector, norb=meta["norb"], nelec=(meta["nalpha"], meta["nbeta"]),
                                shots=budget, bitstring_type=ffsim.BitstringType.INT, seed=seed)
                    blocked_ids, counts = np.unique(values, return_counts=True)
                    atomic_bytes(counts_path, array_bytes({"blocked_ids": blocked_ids.astype(np.uint64),
                                                          "counts": counts.astype(np.uint64)}))
                ledger.complete(run_id, event, int(counts.sum()), sha256(counts_path.read_bytes()))
                preparation_resources = construction["resources"]["preparation"]
                sampling_resources = construction["resources"]["Z_sampling_equivalent"]
                pilot_shots = 0
                cnot_apps = int(budget)*sampling_resources["cnot_count"]
                slots = construction["coefficient_slots"]
                optimized_variables = 0
                detail = {"construction": construction, "exact_sector_simulation_wall_seconds_this_attempt": simulation,
                          "fixed_state_reuse_across_independent_readout_seeds": vector is not None,
                          "statevector_saved": False, "vector_norm_squared": float(np.vdot(vector, vector).real)}
            if not counts_path.exists():
                atomic_bytes(counts_path, array_bytes({"blocked_ids": blocked_ids.astype(np.uint64), "counts": counts.astype(np.uint64)}))
            readouts = final_readouts(directory, meta, h1, eri, blocked_ids, counts)
            snapshot = ledger.snapshot(run_id)
            if snapshot["known_actual_shots"] != budget or snapshot["unknown_or_inflight_draws"]:
                raise RuntimeError("Primitive accounting does not reconcile to full registered budget")
            report = {**binding, "status": "COMPLETED", "pilot_model_shots": pilot_shots,
                      "independent_final_model_shots": int(counts.sum()), "actual_QPU_shots": 0,
                      "budget_ledger": snapshot, "preparation_resources": preparation_resources,
                      "sampling_resources": sampling_resources, "total_model_cnot_applications": int(cnot_apps),
                      "coefficient_slots": slots, "nonredundant_independent_parameters": None,
                      "VQE_optimized_variables": optimized_variables, "final_CI": readouts, "detail": detail,
                      "this_attempt_wall_seconds": perf_counter()-start, "this_attempt_CPU_seconds": process_time()-cpu,
                      "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "all_attempt_work_retained": True, "FCI_used_for_construction": False,
                      "is_primary_run": lane == "C_PRIMARY", "whole_study_complete": False}
            save_json(attempt_path/"completed.json", {"status": "COMPLETED", "wall_seconds": perf_counter()-start,
                       "CPU_seconds": process_time()-cpu})
            save_json(result_path, report)
            print(case, method, budget, seed, "COMPLETED", [r["energy_Ha"] for r in readouts], flush=True)
            return report
        except Exception as exc:
            save_json(attempt_path/"failed.json", {"error": repr(exc), "wall_seconds": perf_counter()-start,
                      "CPU_seconds": process_time()-cpu, "budget_ledger": ledger.snapshot(run_id)})
            raise


def pilot(method):
    case = "N2-32-R1p2-P2"
    if method not in ("A_RAACS_CURRENT", "B_LUCJ_NAIVE_L1", "C_LUCJ_COMPRESSED_L1"):
        raise NotImplementedError("No proxy for unqualified SqDRIFT")
    vector = construction = None
    if method != "A_RAACS_CURRENT":
        meta, _, _ = load_physical(case)
        op, construction = prepare_lucj(case, method)
        tick, cpu = perf_counter(), process_time()
        vector = ffsim.apply_unitary(ffsim.hartree_fock_state(meta["norb"], (meta["nalpha"], meta["nbeta"])),
                                    op, meta["norb"], (meta["nalpha"], meta["nbeta"]), copy=False)
        save_json(ROOT/"pilot-preparation"/(method+".json"), {"case_id": case, "method": method,
                  "simulation_CPU_seconds": process_time()-cpu, "simulation_wall_seconds": perf_counter()-tick,
                  "shared_simulation_once_not_free_algorithm_initialization": True,
                  "statevector_saved": False, "vector_size": int(vector.size)})
    for seed in (11003, 11027, 11047):
        run_cell(case, method, 10000, seed, vector=vector, construction=construction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", required=True, choices=["A_RAACS_CURRENT", "B_LUCJ_NAIVE_L1", "C_LUCJ_COMPRESSED_L1"])
    arguments = parser.parse_args()
    pilot(arguments.pilot)
