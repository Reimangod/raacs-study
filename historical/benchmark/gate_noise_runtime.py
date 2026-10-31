"""Synthetic gate-noise/native-CX qualification, with explicit logical readout map.

Aer is optional and isolated in .runtime-overlays/aer-noise-v1. Nothing imports
Aer in the frozen primary environment unless these entrypoints are requested.
"""
from __future__ import annotations
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import zlib
import numpy as np
from qiskit import qpy, transpile
from qiskit.transpiler import CouplingMap
from .common import ROOT, file_digest, resolve_artifact

NATIVE_BASIS = ["rz", "sx", "x", "cx"]
PROFILES = {
    "ideal": {"readout_probability": 0., "depolarizing_lambda": 0., "thermal": False},
    "readout": {"readout_probability": .01, "depolarizing_lambda": 0., "thermal": False},
    "depolarizing_2q": {"readout_probability": 0., "depolarizing_lambda": .001, "thermal": False},
    "thermal_gate_local": {"readout_probability": 0., "depolarizing_lambda": 0., "thermal": True},
    "combined": {"readout_probability": .01, "depolarizing_lambda": .001, "thermal": True},
}
THERMAL = {"T1_seconds": 100e-6, "T2_seconds": 80e-6, "sx_seconds": 35e-9,
           "x_seconds": 70e-9, "cx_seconds": 300e-9, "RZ_virtual_seconds": 0.,
           "excited_state_population": 0., "idle_and_readout_relaxation_included": False}


def load_sampler(case, method):
    folder = ROOT/"results/RAACS-independent-comparison-standard-sharded-v1/readouts"/case
    db = sqlite3.connect(f"file:{folder/'readouts.sqlite'}?mode=ro", uri=True)
    row = db.execute("SELECT metadata FROM cells WHERE method=?", (method,)).fetchone(); db.close()
    if row is None: return None, {"status": "MISSING_SOURCE_METHOD", "case": case, "method": method}
    metadata = json.loads(row[0]); path = Path(metadata["result_path"])
    if not path.exists(): raise RuntimeError("Missing referenced source result")
    assert file_digest(path) == metadata["result_sha256"]
    record = json.loads(path.read_text())
    provenance = {"source_result": str(path), "source_result_sha256": metadata["result_sha256"]}
    if method.startswith("RAACS:"):
        dbpath = path.parent/"events.sqlite"; assert file_digest(dbpath) == record["event_database_sha256"]
        assert len(record["final_circuit_allocations"]) == 1
        cid = record["final_circuit_allocations"][0]["circuit_id"]
        db = sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True)
        raw = zlib.decompress(db.execute("SELECT sampling_qpy FROM circuits WHERE id=?", (cid,)).fetchone()[0]); db.close()
        provenance.update(source_events_sha256=record["event_database_sha256"], circuit_id=cid)
    else:
        qcpath = resolve_artifact(ROOT/"results/standard-v1", record["sampling_circuit_qpy"]) if "sampling_circuit_qpy" in record else path.parent/"sampling.qpy"
        raw = qcpath.read_bytes(); provenance.update(sampling_QPY=str(qcpath), sampling_QPY_sha256=file_digest(qcpath))
        if "sampling_circuit_sha256" in record: assert file_digest(qcpath) == record["sampling_circuit_sha256"]
    qc = qpy.load(BytesIO(raw))[0]
    if qc.num_clbits or qc.num_qubits > 16: raise ValueError("Qualified unmeasured small-system physical sampler only")
    return qc, provenance


def readout_map(qc):
    pairs = [(qc.find_bit(item.qubits[0]).index, qc.find_bit(item.clbits[0]).index)
             for item in qc.data if item.operation.name == "measure"]
    if len(pairs) != qc.num_qubits or len(set(p for p, c in pairs)) != qc.num_qubits or len(set(c for p, c in pairs)) != qc.num_clbits:
        raise RuntimeError("Incomplete/ambiguous logical readout map")
    return pairs


def compile_sampler(qc, target):
    if qc.num_clbits or target not in ("all_to_all_cx", "line_cx"): raise ValueError("Registered input/target only")
    measured = qc.copy(); measured.measure_all()
    coupling = CouplingMap.from_line(qc.num_qubits, bidirectional=True) if target == "line_cx" else None
    # Measurement is bound BEFORE routing, preserving the logical bit identities.
    routed = transpile(measured, basis_gates=NATIVE_BASIS+["swap"], coupling_map=coupling,
        initial_layout=list(range(qc.num_qubits)), layout_method="trivial", routing_method="sabre",
        optimization_level=1, seed_transpiler=8137)
    swaps = int(routed.count_ops().get("swap", 0))
    compiled = transpile(routed, basis_gates=NATIVE_BASIS, optimization_level=0, seed_transpiler=8137)
    mapping = readout_map(compiled)
    if target == "line_cx":
        for item in compiled.data:
            if item.operation.num_qubits == 2:
                a, b = [compiled.find_bit(q).index for q in item.qubits]
                if abs(a-b) != 1: raise RuntimeError("Routed gate violates synthetic line")
    resource = {"target": target, "synthetic_not_calibrated_hardware": True, "native_2q_family": "CX",
        "native_2q_count": int(compiled.count_ops().get("cx", 0)),
        "native_2q_depth": int(compiled.depth(filter_function=lambda item: item.operation.num_qubits == 2)),
        "native_total_gate_depth_without_measurement": int(compiled.remove_final_measurements(inplace=False).depth()),
        "inserted_explicit_swaps_before_decomposition": swaps, "measurement_physical_to_logical_clbit": mapping,
        "gate_counts": dict(compiled.count_ops()), "seed_transpiler": 8137, "basis_gates": NATIVE_BASIS,
        "compile_policy": "trivial initial layout, SABRE routing, opt1 with swap then opt0 native decomposition"}
    return compiled, resource


def logical_probabilities(physical_probabilities, mapping):
    p = np.asarray(physical_probabilities, float); result = np.zeros_like(p)
    for physical in np.flatnonzero(p):
        logical = sum(((int(physical) >> q) & 1) << c for q, c in mapping)
        result[logical] += p[physical]
    return result


def noise_model(profile):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
    if profile not in PROFILES: raise ValueError("Unknown synthetic noise profile")
    cfg = PROFILES[profile]; model = NoiseModel(basis_gates=NATIVE_BASIS)
    ro = cfg["readout_probability"]
    if ro: model.add_all_qubit_readout_error(ReadoutError([[1-ro, ro], [ro, 1-ro]]))
    error = None
    if cfg["depolarizing_lambda"]: error = depolarizing_error(cfg["depolarizing_lambda"], 2)
    if cfg["thermal"]:
        t = THERMAL
        for gate in ("sx", "x"):
            model.add_all_qubit_quantum_error(thermal_relaxation_error(t["T1_seconds"], t["T2_seconds"], t[gate+"_seconds"], 0.), [gate])
        one = thermal_relaxation_error(t["T1_seconds"], t["T2_seconds"], t["cx_seconds"], 0.)
        two = one.tensor(one)
        error = two if error is None else error.compose(two)
    if error is not None: model.add_all_qubit_quantum_error(error, ["cx"])
    return model


def simulate_counts(compiled, profile, shots, seed):
    from qiskit_aer import AerSimulator
    if compiled.num_qubits > 16 or shots < 1: raise ValueError("Bounded small-system physical noise backend")
    simulator = AerSimulator(method="statevector", device="CPU", noise_model=noise_model(profile),
        max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1,
        max_memory_mb=1024, fusion_enable=False)
    result = simulator.run(compiled, shots=shots, seed_simulator=int(seed) % (2**31-1)).result()
    if not result.success: raise RuntimeError(str(result))
    counts = {int(k.replace(" ", ""), 2): int(v) for k, v in result.get_counts().items()}
    if sum(counts.values()) != shots: raise RuntimeError("Aer count accounting mismatch")
    return counts, {"method": "statevector", "device": "CPU", "maximum_qubits": 16,
        "full_statevector_only_in_small_noise_simulator_not_sampler_algorithm": True,
        "max_memory_mb": 1024, "max_parallel_threads": 1, "hardware_shots": 0,
        "result_metadata": {k: v for k, v in result.results[0].metadata.items()
                            if k in ("method", "device", "num_qubits", "required_memory_mb", "noise", "measure_sampling")}}
