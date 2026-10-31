"""Native grouped random circuits, exact sparse simulation, common readout.

All unitary groups are applied, including diagonal/inert ones. No favorable
member selection, classical-CI state preparation or full-Fock large vector.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from qiskit import qpy
from qiskit.quantum_info import Statevector

from .budget import BudgetLedger
from .ci import interleaved_to_blocked
from .common_execution import final_readouts, load_physical, run_identity
from .input_adapter import array_bytes
from .io import ROOT, PLAN_ROOT, atomic_bytes, digest, save_json, sha256
from .lucj import circuit_views

PROJECT = ROOT.parents[2]/"ceo-sampler-benchmark"


def source_request(case):
    meta, h1, eri = load_physical(case)
    one, two = spinorb_from_spatial(h1, eri.transpose(0, 2, 3, 1))
    fermion = of.get_fermion_operator(of.InteractionOperator(0., one, .5*two))
    terms = [{"term": [[int(q), int(creation)] for q, creation in term],
              "coefficient": [float(c.real), float(c.imag)]} for term, c in sorted(fermion.terms.items())
             if term and abs(c) > 1e-12]
    return {"problem_id": meta["problem_id"], "n_qubits": meta["n_qubits"], "hf_id": meta["hf_id"],
            "fermion_terms": terms, "identity_scalar_omitted": meta["core_energy"],
            "FCI_or_reference_labels_used": False, "coefficient_zero_threshold": 1e-12}


class Session:
    def __init__(self, base, directory):
        self.directory = Path(directory)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT/".runtime-overlays/sqdrift-qiskit25-v1"), str(ROOT)])
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")})
        self.log = (self.directory/"worker-stderr.log").open("a")
        self.process = subprocess.Popen(["rtk", "proxy", sys.executable, "-u", "-m", "paperstudy.sqdrift_native_worker",
                    "--base", str(base), "--directory", str(self.directory)], cwd=PROJECT, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        if not self.read(180).get("ready"):
            raise RuntimeError("SqDRIFT native session not ready")

    def read(self, seconds):
        if not select.select([self.process.stdout], [], [], seconds)[0]:
            raise TimeoutError("Native exporter observation timeout; preserve process identity")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native exporter exited {self.process.poll()}; see preserved stderr")
        return json.loads(line)

    def export(self, spec, directory):
        self.process.stdin.write(json.dumps({"spec": spec, "relative_directory": str(directory.relative_to(self.directory))})+"\n")
        self.process.stdin.flush()
        reply = self.read(180)
        if reply["status"] != "NATIVE_GROUPED_MEMBER_EXPORTED":
            raise RuntimeError(reply)
        return reply

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"stop":true}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        self.log.close()


def sparse_group(state, terms, n, nelec, max_support=100000):
    for label, angle in terms:
        output = {x: np.cos(angle)*a for x, a in state.items()}
        flip = sum(1 << (n-1-q) for q, p in enumerate(label) if p in "XY")
        for x, a in state.items():
            phase = 1.+0j
            for q, p in enumerate(label):
                bit = (int(x) >> (n-1-q)) & 1
                if p == "Y":
                    phase *= -1j if bit else 1j
                elif p == "Z" and bit:
                    phase *= -1
            y = x ^ flip
            output[y] = output.get(y, 0j)-1j*np.sin(angle)*phase*a
        state = {x: a for x, a in output.items() if a != 0}
        if len(state) > 4*max_support:
            raise MemoryError("Exact sparse temporary support cap; no amplitude truncation")
    # Every verified complete fermionic group preserves Nalpha/Nbeta exactly.
    # Remove only floating-point leakage outside that proven invariant sector,
    # not small amplitudes within the physical state.
    kept = {}
    leakage = 0.
    for x, a in state.items():
        na = sum((x >> (n-1-q)) & 1 for q in range(0, n, 2))
        nb = sum((x >> (n-1-q)) & 1 for q in range(1, n, 2))
        if (na, nb) == tuple(nelec):
            kept[x] = a
        else:
            leakage += abs(a)**2
    if leakage > 1e-22:
        raise RuntimeError("Sampled unit group does not preserve the requested sector")
    if len(kept) > max_support:
        raise MemoryError("Exact sparse physical support cap; no substitute approximation")
    return kept, float(leakage)


def member_state(meta, member):
    state = {int(meta["hf_id"]): 1.+0j}
    leakage = 0.
    for group in member["source_mode_group_Pauli_sequence"]:
        state, error = sparse_group(state, group, meta["n_qubits"], (meta["nalpha"], meta["nbeta"]))
        leakage += error
    norm = float(sum(abs(a)**2 for a in state.values()))
    if abs(norm-1) > 1e-8:
        raise RuntimeError("Exact physical SqDRIFT state lost norm")
    return state, {"norm_squared": norm, "physical_support": len(state), "roundoff_outside_sector_weight": leakage,
                   "in_sector_amplitudes_truncated": False, "full_Fock_vector_used": False}


def prepare_member(case, spec, session, qualify=False):
    meta, _, _ = load_physical(case)
    directory = ROOT/"native-SqDRIFT"/case/"members"/digest(spec)
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory/"qualified-member.json"
    if filename.exists():
        return directory, json.loads(filename.read_text())
    if not (directory/"member.json.gz").exists():
        session.export(spec, directory)
    member = json.loads(gzip.decompress((directory/"member.json.gz").read_bytes()))
    if member["spec"] != spec or sha256((directory/"circuit.qpy.gz").read_bytes()) != member["circuit_file_sha256"]:
        raise RuntimeError("Native member artifact binding mismatch")
    qc = qpy.load(BytesIO(gzip.decompress((directory/"circuit.qpy.gz").read_bytes())))[0]
    prep, sampling, resource_views = circuit_views(qc)
    # The small independent test simulates the optimized native circuit, not
    # just another calculation using the same Pauli sequence.
    parity = None
    if qualify:
        if meta["n_qubits"] > 12:
            raise ValueError("Independent full-Fock verifier is small-fixture only")
        state, numerical = member_state(meta, member)
        expected = np.zeros(1 << meta["n_qubits"], float)
        for x, a in state.items():
            remapped = sum(((x >> (meta["n_qubits"]-1-q)) & 1) << member["permutation"][q]
                           for q in range(meta["n_qubits"]))
            expected[remapped] = abs(a)**2
        actual = abs(Statevector.from_instruction(prep).data)**2
        after_sampling = abs(Statevector.from_instruction(sampling).data)**2
        parity = {"preparation_probability_L1": float(np.sum(abs(expected-actual))),
                  "sampling_probability_L1": float(np.sum(abs(expected-after_sampling))), **numerical}
        if max(parity["preparation_probability_L1"], parity["sampling_probability_L1"]) > 1e-8:
            raise RuntimeError("Independent relabeled SDK/Qiskit distribution parity failed")
    record = {"spec": spec, "resources": resource_views,
              "independent_small_parity": parity, "native_export": member,
              "coefficient_slots": sum(len(g) for g in member["source_mode_group_Pauli_sequence"]),
              "n_optimized_variables": 0, "time_hyperparameters": 1,
              "paper_author_driver_reproduced": False,
              "source_state_parity_is_checked_only_for_registered_small_fixtures": True}
    save_json(filename, record)
    return directory, record


def allocations(budget, seed, length=20):
    result = []
    for i, k in enumerate((1, 2, 3)):
        shots = budget//3 + (i < budget % 3)
        count = (shots+127)//128
        for j in range(count):
            M = shots//count + (j < shots % count)
            member_seed = int(digest(["paper-common-SqDRIFT-v1", seed, budget, k, j])[:8], 16)
            result.append(({"time": float(k), "num_terms": int(length), "seed": member_seed,
                            "k": k, "member_index": j}, int(M)))
    if sum(M for _, M in result) != budget or max(M for _, M in result) > 128:
        raise RuntimeError("Frozen SqDRIFT allocation does not reconcile")
    return result


def run(case, budget, seed, lane="S3_ENGINEERING_PILOT"):
    import fcntl
    method = "D_SQDRIFT_N20"
    identity = run_identity(case, method, budget, seed, lane)
    directory = ROOT/"common-runs"/lane/case/method/(str(budget)+"-"+str(seed))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/"result.json").exists():
            return json.loads((directory/"result.json").read_text())
        meta, h1, eri = load_physical(case)
        binding = {"run_id": identity, "case_id": case, "problem_id": meta["problem_id"], "method_id": method,
                   "budget": budget, "seed": seed, "lane": lane,
                   "adapter_sha256": sha256(Path(__file__).read_bytes()),
                   "native_worker_sha256": sha256(Path(__file__).with_name("sqdrift_native_worker.py").read_bytes()),
                   "scientific_plan_sha256": sha256((PLAN_ROOT/"plan.json").read_bytes()),
                   "paper_author_driver_reproduced": False,
                   "implementation_label": "P1-inspired native SDK grouped SqDRIFT/common-budget adaptation"}
        if (directory/"request.json").exists() and json.loads((directory/"request.json").read_text()) != binding:
            raise RuntimeError("Frozen SqDRIFT run drift; use a versioned successor")
        save_json(directory/"request.json", binding)
        native_root = ROOT/"native-SqDRIFT"/case
        native_root.mkdir(parents=True, exist_ok=True)
        if not (native_root/"base.json").exists():
            save_json(native_root/"base.json", source_request(case))
        ledger = BudgetLedger(ROOT/"common-budget.sqlite")
        ledger.register(identity, budget)
        session = None
        tick, cpu = perf_counter(), process_time()
        merged = defaultdict(int)
        records = []
        try:
            for i, (spec, M) in enumerate(allocations(budget, seed)):
                event = digest([spec, M])
                saved = directory/"draws"/(event+".json")
                if saved.exists():
                    record = json.loads(saved.read_text())
                    blob = (directory/"draws"/(event+".npz")).read_bytes()
                    if sha256(blob) != record["counts_sha256"]:
                        raise RuntimeError("Committed SqDRIFT draw digest mismatch")
                    with np.load(BytesIO(blob), allow_pickle=False) as a:
                        ids, counts = a["ids"].copy(), a["counts"].copy()
                    ledger.complete(identity, event, int(counts.sum()), sha256(blob))
                else:
                    if session is None:
                        session = Session(native_root/"base.json", native_root)
                    member_dir, record = prepare_member(case, spec, session)
                    state, numerical = member_state(meta, record["native_export"])
                    request = {"member_id": digest(spec), "problem_id": meta["problem_id"], "shots": M,
                               "readout_seed": int(digest(["final", seed, i, budget])[:16], 16)}
                    status = ledger.reserve(identity, event, request, M)
                    if status != "NEW_RESERVATION":
                        raise RuntimeError("Unknown SqDRIFT draw outcome; do not resample")
                    ids = np.array(sorted(state), np.uint64)
                    probabilities = np.array([abs(state[int(x)])**2 for x in ids])
                    counts = np.random.default_rng(request["readout_seed"]).multinomial(M, probabilities/probabilities.sum())
                    ids, counts = ids[counts > 0], counts[counts > 0].astype(np.uint64)
                    blob = array_bytes({"ids": ids, "counts": counts})
                    atomic_bytes(directory/"draws"/(event+".npz"), blob)
                    record = {"member_id": digest(spec), "spec": spec, "shots": M, "request": request,
                              "counts_sha256": sha256(blob), "resources": record["resources"],
                              "coefficient_slots": record["coefficient_slots"], "state_numerics": numerical}
                    save_json(saved, record)
                    ledger.complete(identity, event, int(counts.sum()), sha256(blob))
                records.append(record)
                for x, count in zip(ids, counts):
                    merged[int(x)] += int(count)
                if (i+1) % 20 == 0:
                    save_json(directory/"progress.json", {"members_completed": i+1, "members_planned": len(allocations(budget, seed)),
                             "ledger": ledger.snapshot(identity), "pid": os.getpid()})
            ids = np.array([interleaved_to_blocked(x, meta["norb"]) for x in sorted(merged)], np.uint64)
            counts = np.array([merged[x] for x in sorted(merged)], np.uint64)
            atomic_bytes(directory/"final-counts.npz", array_bytes({"blocked_ids": ids, "counts": counts}))
            readouts = final_readouts(directory, meta, h1, eri, ids, counts)
            weighted = {field: sum(r["shots"]*r["resources"]["Z_sampling_equivalent"][field] for r in records)/budget
                        for field in ("cnot_count", "cnot_depth", "total_unitary_depth")}
            maxima = {field: max(r["resources"]["Z_sampling_equivalent"][field] for r in records) for field in weighted}
            report = {**binding, "status": "COMPLETED", "budget_ledger": ledger.snapshot(identity),
                      "independent_final_model_shots": int(counts.sum()), "pilot_model_shots": 0, "actual_QPU_shots": 0,
                      "final_CI": readouts, "member_count": len(records), "member_shot_weighted_mean_resources": weighted,
                      "member_max_resources": maxima, "total_model_cnot_applications": int(sum(r["shots"]*r["resources"]["Z_sampling_equivalent"]["cnot_count"] for r in records)),
                      "member_records": records, "VQE_optimized_variables": 0,
                      "this_attempt_wall_seconds": perf_counter()-tick, "this_attempt_CPU_seconds": process_time()-cpu,
                      "is_primary_run": lane == "C_PRIMARY", "whole_study_complete": False}
            if report["budget_ledger"]["known_actual_shots"] != budget:
                raise RuntimeError("SqDRIFT paid-shot budget mismatch")
            save_json(directory/"result.json", report)
            print(case, method, seed, "COMPLETED", [r["energy_Ha"] for r in readouts], flush=True)
            return report
        finally:
            if session is not None:
                session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        for seed in (11003, 11027, 11047):
            run("N2-32-R1p2-P2", 10000, seed)
