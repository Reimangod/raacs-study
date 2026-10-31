"""Actual isolated Aer / routed logical-bit / synthetic-channel qualification."""
from __future__ import annotations
import argparse
from io import BytesIO
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json, rng_for, digest
from .ansatz import circuit_bytes
from .fermionic_sqd_durable import encode
from .gate_noise_runtime import load_sampler, compile_sampler, logical_probabilities, simulate_counts, PROFILES, THERMAL
from .full_sqd_durable_cohort import METHODS
from .study.core import rss_peak

CASES = ["H2_R0.7", "H4_R1.1", "H6_R1.1", "LiH_R1.5", "BeH2_R1.3", "N2_R1.1"]


def channel_tests():
    t, T1, T2 = 1e-6, 100e-6, 80e-6
    channel = thermal_relaxation_error(T1, T2, t, 0.).to_quantumchannel()
    excited = DensityMatrix.from_label("1").evolve(channel)
    plus = DensityMatrix.from_label("+").evolve(channel)
    p1_error = abs(float(excited.probabilities()[1])-np.exp(-t/T1))
    coherence_error = abs(2*float(plus.data[0, 1].real)-np.exp(-t/T2))
    lam = .02
    dep = DensityMatrix.from_label("00").evolve(depolarizing_error(lam, 2).to_quantumchannel())
    dep_error = abs(float(dep.probabilities()[0])-(1-3*lam/4))
    assert max(p1_error, coherence_error, dep_error) < 1e-12
    q = QuantumCircuit(1); q.measure_all()
    counts, metadata = simulate_counts(q, "readout", 10000, 101)
    rate = counts.get(1, 0)/10000
    assert abs(rate-.01) < .005
    return {"T1_population_formula_error": p1_error, "T2_coherence_formula_error": coherence_error,
        "depolarizing_2q_formula_error": dep_error, "readout_empirical_flip_rate": rate,
        "readout_counts": counts, "readout_model_shots": 10000, "hardware_shots": 0}


def qualify(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    cfg = {"schema": "synthetic-gate-noise-runtime-qualification-v1", "cases": CASES, "methods": METHODS,
        "targets": ["all_to_all_cx", "line_cx"], "profiles": ["ideal", "combined"], "shots": 32,
        "all_noise_profiles": PROFILES, "thermal_parameters": THERMAL, "seed_namespace": "gate-noise-qualification-v1",
        "free_floor": int(2.75*2**30), "output_cap": 64*2**20, "RSS_cap": 2*2**30,
        "implementation": {"benchmark/"+name: file_digest(ROOT/"benchmark"/name) for name in ("gate_noise_runtime.py", "gate_noise_qualification.py")},
        "packages": {p: importlib.metadata.version(p) for p in ("qiskit-aer", "qiskit", "numpy", "scipy")},
        "base_lock_sha256": file_digest(ROOT/"uv.lock"), "external_hardware": False,
        "base_environment_unchanged": True, "runtime_overlay": ".runtime-overlays/aer-noise-v1",
        "PYTHONPATH_for_base_dependencies": str(ROOT/".venv/lib/python3.11/site-packages")}
    manifest = root/"manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == cfg
        if (root/"validation.json").exists(): return json.loads((root/"validation.json").read_text())
        raise RuntimeError("Partial runtime qualification preserved; do not silently retry")
    if shutil.disk_usage(root).free < cfg["free_floor"]+cfg["output_cap"]: raise RuntimeError("Qualification capacity reserve")
    save_json(manifest, cfg)
    capsule = root/"code-capsule"; capsule.mkdir()
    for name in cfg["implementation"]: (capsule/Path(name).name).write_bytes((ROOT/name).read_bytes())
    db = sqlite3.connect(root/"qualification.sqlite"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE circuits(id TEXT PRIMARY KEY,case_id TEXT,method TEXT,target TEXT,record BLOB,qpy BLOB)")
    db.execute("CREATE TABLE draws(id TEXT PRIMARY KEY,circuit TEXT,profile TEXT,shots INTEGER,seed TEXT,counts BLOB,metadata BLOB)"); db.commit()
    rows = []; max_error = 0.; actual_shots = 0; started = perf_counter()
    with threadpool_limits(limits=1):
        primitive = channel_tests(); actual_shots += primitive["readout_model_shots"]
        save_json(root/"channel-tests.json", primitive)
        try:
            for case in CASES:
                for method in METHODS:
                    qc, source = load_sampler(case, method)
                    if qc is None: raise RuntimeError("Qualification requires available representative samplers")
                    original = Statevector(qc).probabilities()
                    for target in cfg["targets"]:
                        if shutil.disk_usage(root).free < cfg["free_floor"] or rss_peak() > cfg["RSS_cap"]:
                            raise RuntimeError("Qualification resource boundary")
                        begin = perf_counter(); compiled, resource = compile_sampler(qc, target)
                        physical = Statevector(compiled.remove_final_measurements(inplace=False)).probabilities()
                        reconstructed = logical_probabilities(physical, resource["measurement_physical_to_logical_clbit"])
                        error = float(max(abs(original-reconstructed)))
                        if error > 1e-9: raise RuntimeError("Routed logical-bit distribution mismatch")
                        max_error = max(max_error, error)
                        cid = digest([case, method, target, source, cfg["implementation"]])
                        record = {"case": case, "method": method, "source": source, "resources": resource,
                            "max_logical_Z_probability_error": error, "compile_and_independent_parity_seconds": perf_counter()-begin,
                            "full_statevector_scope": "independent <=16q compiler qualification only, not adaptive algorithm feature"}
                        db.execute("INSERT INTO circuits VALUES(?,?,?,?,?,?)", (cid, case, method, target, encode(record), zlib.compress(circuit_bytes(compiled), 6))); db.commit()
                        for profile in cfg["profiles"]:
                            did = digest([cid, profile, cfg["shots"]]); seed = rng_for(cfg["seed_namespace"], did)[1]
                            before = perf_counter(); counts, backend = simulate_counts(compiled, profile, cfg["shots"], seed)
                            actual_shots += sum(counts.values())
                            db.execute("INSERT INTO draws VALUES(?,?,?,?,?,?,?)", (did, cid, profile, cfg["shots"], str(seed), encode(counts), encode(backend))); db.commit()
                            rows.append({"case": case, "method": method, "target": target, "profile": profile,
                                "actual_model_shots": cfg["shots"], "seconds": perf_counter()-before,
                                "unique_observations": len(counts), "native_2q_count": resource["native_2q_count"],
                                "native_2q_depth": resource["native_2q_depth"], "inserted_swaps": resource["inserted_explicit_swaps_before_decomposition"],
                                "logical_distribution_parity_max_error": error})
                        save_json(root/"execution-status.json", {"status": "RUNNING", "compiled_circuits": len(rows)//2, "planned_circuits": 48,
                            "actual_model_shots": actual_shots, "hardware_shots": 0})
        except Exception as exc:
            save_json(root/"failure.json", {"status": "QUALIFICATION_FAILURE_PRESERVED", "error": repr(exc), "completed_draws": len(rows), "actual_model_shots": actual_shots})
            db.close(); raise
    db.close()
    import pandas as pd
    pd.DataFrame(rows).to_csv(root/"all-runtime-draws.csv", index=False)
    out = {"status": "AER_SYNTHETIC_NOISE_AND_ROUTED_LOGICAL_MAPPING_QUALIFIED", "actual_compiled_circuits": 48,
        "actual_sampler_draws": len(rows), "new_qualification_model_shots": actual_shots, "hardware_shots": 0,
        "maximum_routed_logical_Z_distribution_error": max_error, "channel_tests_passed": True,
        "actual_total_seconds": perf_counter()-started, "process_RSS_HWM_bytes": rss_peak(),
        "database_sha256": file_digest(root/"qualification.sqlite"), "qualifier_sha256": file_digest(__file__),
        "noise_is_synthetic_not_device_calibration": True, "idle_noise_not_included": True,
        "main_noise_cohort_not_executed_by_qualification": True, "whole_phase_F_complete": False, "whole_study_complete": False}
    assert len(rows) == 96
    save_json(root/"validation.json", out); return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", default=str(ROOT/"results/gate-noise-runtime-qualification-v1")); a = p.parse_args()
    print(json.dumps(qualify(a.output)))
