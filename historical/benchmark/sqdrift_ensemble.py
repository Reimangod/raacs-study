"""Native multi-time/multi-randomization SqDRIFT sampler construction.

Small-system qualification backend. Actual grouped SDK circuits and physical
states, not an average-probability single-circuit surrogate. No FCI selection.
"""
from __future__ import annotations
import argparse
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json, save_binary, save_npz
from .chemistry import load_reference
from .ansatz import compile_logical, measurement_equivalent_circuit, resources, circuit_bytes, verify_circuit_state
from .sqdrift_qualify import request_from_integrals
from .study.core import rss_peak
from .time_evolution_primitives import apply_sequence


def specs(mode):
    lengths, nr, seeds, times = ((8,), (2,), (101,), (.5, 1.)) if mode == "quick" else ((8, 32), (3, 9), (101, 211, 307), (.5, 1., 1.5))
    return [{"name": f"SqDRIFT_L{L}_Nr{r}_s{s}", "num_terms": L, "randomizations_per_time": r,
             "ensemble_seed": s, "times": list(times), "partition": "magnitude-refined SDK electronic groups; diagonals retained"}
            for L in lengths for r in nr for s in seeds]


def member_specs(problem_id, spec):
    members = []
    for time in spec["times"]:
        for rep in range(spec["randomizations_per_time"]):
            _, seed = rng_for("native-SqDRIFT-ensemble-member-v1", problem_id, spec, time, rep)
            # SDK accepts ordinary nonnegative integer seeds. Record actual input.
            seed = seed % (2**32)
            m = {"time": time, "num_terms": spec["num_terms"], "randomization_index": rep, "seed": seed}
            m["member_id"] = digest([problem_id, spec, m])
            members.append(m)
    return members


class SDKSession:
    def __init__(self, base, root):
        self.root = Path(root); self.started = perf_counter()
        env = dict(os.environ, PYTHONPATH=str(ROOT / ".runtime-overlays/sqdrift-qiskit25-v1"))
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        self.log = (self.root / "sdk-session-stderr.log").open("a")
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "benchmark.sqdrift_sdk_session", "--base", str(base), "--root", str(root)],
                                     cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        ack = self.read(120)
        if not ack.get("ready"):
            raise RuntimeError("SDK ready handshake failed")
        self.startup_seconds = perf_counter()-self.started

    def read(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("Native member boundary timeout; preserve completed prefix")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Native exporter exited (returncode {self.proc.poll()}); see preserved stderr")
        return json.loads(line)

    def export(self, member, directory):
        request = {k: member[k] for k in ("time", "num_terms", "seed")}
        request["relative_directory"] = str(directory.relative_to(self.root))
        self.proc.stdin.write(json.dumps(request)+"\n"); self.proc.stdin.flush()
        result = self.read(600)
        if result.get("request") != request:
            raise RuntimeError("SDK response/request identity mismatch")
        return result

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"stop":true}\n'); self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.terminate(); self.proc.wait(timeout=10)
        self.log.close()


def boundary(root, config, known_total=None):
    if shutil.disk_usage(root).free < config["free_floor_bytes"] or rss_peak() > config["parent_ram_bytes"]:
        raise RuntimeError("Resource boundary; preserve completed prefix")
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if known_total is None else known_total
    if total > config["output_cap_bytes"]:
        raise RuntimeError("Registered output boundary; preserve completed prefix")
    return total


def build_member(ref, member, directory, session):
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / "result.json"
    if path.exists():
        result = json.loads(path.read_text())
        assert result["member_id"] == member["member_id"]
        return result
    started = perf_counter()
    result = {"member_id": member["member_id"], "spec": member, "hardware_shots": 0, "FCI_used_for_selection": False}
    # Exporting is deterministic. Completed native export is reused on resume;
    # prior work time remains in its event, not relabelled zero.
    event_file = directory / "sdk-export-event.json"
    event = json.loads(event_file.read_text()) if event_file.exists() else session.export(member, directory)
    result["SDK_export_event"] = event
    try:
        if event["status"] != "SDK_MEMBER_EXPORTED":
            raise RuntimeError("Native export failed; no probability surrogate substituted")
        for filename, field in (("sdk-circuit.qpy.gz", "sdk_circuit_gzip_sha256"), ("sdk-result.json.gz", "sdk_result_gzip_sha256")):
            assert file_digest(directory / filename) == event[field]
        sdk = json.loads(gzip.decompress((directory / "sdk-result.json.gz").read_bytes()))
        loaded = qpy.load(BytesIO(gzip.decompress((directory / "sdk-circuit.qpy.gz").read_bytes())))[0]
        n = ref["meta"]["n_qubits"]
        qc = QuantumCircuit(n); qc.compose(loaded, qubits=list(reversed(range(n))), inplace=True)
        preparation = compile_logical(qc); sampler, removed = measurement_equivalent_circuit(qc)
        initial = np.zeros(1 << n, complex); initial[ref["meta"]["hf_id"]] = 1
        expected = apply_sequence(initial, sdk["sequence_for_independent_source_basis_parity"])
        parity = verify_circuit_state(preparation, expected)
        probability = abs(expected)**2
        if not np.allclose(abs(Statevector.from_instruction(sampler).data)**2, probability, atol=1e-9):
            raise RuntimeError("Sampling distribution does not match physical SDK circuit")
        valid = float(probability[ref["ids"]].sum())
        if abs(valid-1) > 1e-8:
            raise RuntimeError("Native fermionic member lost registered N/Sz sector")
        for name, circuit in (("preparation", preparation), ("sampling", sampler)):
            save_binary(directory / f"{name}.qpy.gz", gzip.compress(circuit_bytes(circuit), mtime=0))
        save_npz(directory / "probability.npz", probability=probability)
        result.update(status="ACTUAL_SDK_MEMBER_PARITY_VALIDATED", resources=resources(sampler, 1, member["num_terms"]),
                      preparation_resources=resources(preparation, 1, member["num_terms"]),
                      probability_sha256=file_digest(directory / "probability.npz"), state_parity_error=parity,
                      valid_N_Sz_probability=valid, removed_terminal_diagonal_gates=removed,
                      construction_time_parameter_count=1, n_optimized_variables=0,
                      group_first_moment_error=sdk["max_group_first_moment_error"],
                      unit_group_norm_upper_bound=sdk["max_unit_group_Pauli_L1_norm_upper_bound"])
    except Exception as exc:
        import traceback
        result.update(status="FAILED_PRESERVED", error=repr(exc), traceback=traceback.format_exc())
    result["parent_member_wall_seconds"] = perf_counter()-started
    result["parent_peak_rss_bytes"] = rss_peak()
    save_json(path, result)
    return result


def run(root, source, mode):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents:
        raise RuntimeError("Protect immutable source")
    root.mkdir(exist_ok=True, parents=True)
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source / "references").iterdir() if p.is_dir())
    methods = specs(mode)
    files = ["benchmark/sqdrift_ensemble.py", "benchmark/sqdrift_sdk_session.py", "benchmark/sqdrift_sdk_export.py", "benchmark/sqdrift_qualify.py", "benchmark/time_evolution_primitives.py", "benchmark/ansatz.py", "study-v1/BASELINES_B3_SQDRIFT.md"]
    config = {"parent_ram_bytes": 2*2**30, "free_floor_bytes": 3*2**30, "output_cap_bytes": 2*2**30, "worker_count": 1, "numeric_threads": 1}
    manifest = {"schema": "native-SqDRIFT-ensemble-v2", "mode": mode, "source": str(source), "case_ids": cases, "methods": methods,
                "implementation": {p: file_digest(ROOT / p) for p in files}, "resource_config": config,
                "source_dataset_id": json.loads((source / "REPRODUCIBILITY.json").read_text())["dataset_id"],
                "budget": {"M_grid": [1000, 10000, 100000], "R_grid": [8, 16, 32, 64, 128, 256, 512], "evaluation_repeats": 4 if mode == "quick" else 20},
                "allocation": "equal member shots + one deterministic remainder shot to earliest members; fixed time/randomization order",
                "scope": "small statevector qualification; NOT large-system scaling or an exact published hardware setting"}
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Frozen code/config drift; create explicit successor")
    if mode == "standard":
        quick = ROOT / "results/SqDRIFT-ensemble-quick-v2/validation.json"
        if not quick.exists() or json.loads(quick.read_text())["status"] != "QUICK_SQDRIFT_ENSEMBLE_VALIDATED":
            raise RuntimeError("Quick actual-circuit and allocation audit required before standard")
    save_json(path, manifest)
    import fcntl
    with (root / "run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in cases:
            t = perf_counter(); ref = load_reference(source / "references" / case)
            case_root = root / "native" / case; case_root.mkdir(exist_ok=True, parents=True)
            base_path = case_root / "base-request.json"
            base = request_from_integrals(ref, .5, 8, 0)
            if base_path.exists():
                assert json.loads(base_path.read_text()) == base
            else:
                save_json(base_path, base)
            setup_seconds = perf_counter()-t
            session = None
            try:
                for spec in methods:
                    # Full total once per ensemble. Increment only this member's
                    # new files below: avoid an O(all files × members) stat scan.
                    total_bytes = boundary(root, config)
                    directory = root / "cells" / case / spec["name"]; directory.mkdir(exist_ok=True, parents=True)
                    task_id = digest([base["problem_id"], spec, manifest["implementation"], "native-SqDRIFT-ensemble-v1"])
                    if (directory / "result.json").exists():
                        assert json.loads((directory / "result.json").read_text())["task_id"] == task_id
                        continue
                    started = perf_counter(); members = member_specs(base["problem_id"], spec); records = []
                    for member in members:
                        boundary(root, config, total_bytes)
                        md = case_root / "members" / member["member_id"]
                        previous_bytes = sum(p.stat().st_size for p in md.glob("*") if p.is_file()) if md.exists() else 0
                        if session is None:
                            session = SDKSession(base_path, case_root)
                            save_json(case_root / "session-setup.json", {"input_setup_seconds": setup_seconds,
                                "SDK_cold_start_seconds": session.startup_seconds, "shared_once_per_case": True,
                                "first_member_imports_may_add_first_member_time": True})
                        records.append(build_member(ref, member, md, session))
                        total_bytes += sum(p.stat().st_size for p in md.glob("*") if p.is_file())-previous_bytes
                        boundary(root, config, total_bytes)
                    failures = [r for r in records if r["status"] == "FAILED_PRESERVED"]
                    result = {"case_id": case, "method": spec["name"], "task_id": task_id, "spec": spec,
                              "problem_id": base["problem_id"], "member_ids": [r["member_id"] for r in records],
                              "member_count": len(records), "construction_wall_seconds": perf_counter()-started,
                              "construction_pilot_shots": 0, "hardware_shots": 0, "FCI_used_for_selection": False,
                              "status": "FAILED_MEMBERS_PRESERVED" if failures else "FIXED_ACTUAL_SQDRIFT_ENSEMBLE",
                              "failed_member_ids": [r["member_id"] for r in failures], "switch_latency_seconds": None,
                              "resource_scope": "ensemble member maxima/unweighted means; actual per-shot means depend on readout allocation"}
                    if not failures:
                        result["member_max_resources"] = {k: max(r["resources"][k] for r in records) for k in ("cnot_count", "cnot_depth", "total_depth")}
                        result["member_mean_resources"] = {k: float(np.mean([r["resources"][k] for r in records])) for k in ("cnot_count", "cnot_depth", "total_depth")}
                    save_json(directory / "result.json", result)
                    print(f"{case} {spec['name']}: {result['status']} ({len(records)} actual circuits)", flush=True)
            finally:
                if session is not None:
                    session.close()
            save_json(root / "case-status" / f"{case}.json", {"case_id": case, "terminal_ensembles": len(methods), "whole_study_complete": False})
    status = {"status": "ENSEMBLE_CONSTRUCTION_COMPLETE_PENDING_READOUT", "terminal_cells": len(cases)*len(methods),
              "actual_member_circuits": sum(len(member_specs("count-only", s)) for s in methods)*len(cases), "whole_study_complete": False}
    save_json(root / "construction-status.json", status)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick"); p.add_argument("--output")
    p.add_argument("--source", default=str(ROOT / "results/standard-v1")); args = p.parse_args()
    print(json.dumps(run(args.output or ROOT / f"results/SqDRIFT-ensemble-{args.mode}-v2", args.source, args.mode)))
