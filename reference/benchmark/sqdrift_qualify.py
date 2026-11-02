"""Actual official-SDK grouped circuit qualification in an isolated overlay."""
from __future__ import annotations
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from .common import ROOT, file_digest, save_json, save_binary, save_npz
from .chemistry import load_reference
from .integral_hamiltonian import IntegralHamiltonian
from .time_evolution_primitives import apply_sequence
from .ansatz import compile_logical, measurement_equivalent_circuit, resources, circuit_bytes, verify_circuit_state


def request_from_integrals(ref, time, num_terms, seed):
    meta = ref["meta"]
    backend = IntegralHamiltonian.from_saved_integrals(ref["h1"], ref["eri"], meta["hf_id"], meta["rhf_energy"])
    one, two = spinorb_from_spatial(ref["h1"], ref["eri"].transpose(0, 2, 3, 1))
    fermion = of.get_fermion_operator(of.InteractionOperator(backend.core, one, .5*two))
    return {"problem_id": meta["request_digest"], "n_qubits": meta["n_qubits"], "hf_id": meta["hf_id"],
            "time": time, "num_terms": num_terms, "seed": seed,
            "fermion_terms": [{"term": [[q, creation] for q, creation in term], "coefficient": [float(c.real), float(c.imag)]}
                              for term, c in sorted(fermion.terms.items()) if term and abs(c) > 1e-12],
            "scalar_identity_omitted": backend.core, "FCI_used": False}


def qualify(case, root, source, time=1., num_terms=8, seed=101):
    from threadpoolctl import threadpool_limits
    root = Path(root); root.mkdir(exist_ok=True, parents=True)
    directory = root / case; directory.mkdir(exist_ok=True)
    ref = load_reference(Path(source) / "references" / case)
    request = request_from_integrals(ref, time, num_terms, seed)
    save_json(directory / "request.json", request)
    overlay = ROOT / ".runtime-overlays/sqdrift-qiskit25-v1"
    env = dict(os.environ); env["PYTHONPATH"] = str(overlay)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    started = perf_counter()
    command = [sys.executable, "-m", "benchmark.sqdrift_sdk_export", "--input", str((directory / "request.json").resolve()), "--output", str(directory.resolve())]
    proc = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
    (directory / "sdk-stdout.log").write_text(proc.stdout)
    (directory / "sdk-stderr.log").write_text(proc.stderr)
    if proc.returncode:
        result = {"case_id": case, "status": "SDK_EXPORT_FAILED_PRESERVED", "returncode": proc.returncode,
                  "export_wall_seconds": perf_counter()-started, "whole_study_complete": False}
        save_json(directory / "qualification.json", result)
        return result
    export_seconds = perf_counter()-started
    sdk = json.loads((directory / "sdk-result.json").read_text())
    n, hf = ref["meta"]["n_qubits"], ref["meta"]["hf_id"]
    with threadpool_limits(limits=1):
        loaded = qpy.load(BytesIO((directory / "sdk-circuit.qpy").read_bytes()))[0]
        qc = QuantumCircuit(n); qc.compose(loaded, qubits=list(reversed(range(n))), inplace=True)
        preparation = compile_logical(qc)
        sampler, removed = measurement_equivalent_circuit(qc)
        initial = np.zeros(1 << n, complex); initial[hf] = 1
        expected = apply_sequence(initial, sdk["sequence_for_independent_source_basis_parity"])
        error = verify_circuit_state(preparation, expected)
        actual = Statevector.from_instruction(sampler).data
        assert np.allclose(abs(actual)**2, abs(expected)**2, atol=1e-9)
        valid_probability = float((abs(expected[ref["ids"]])**2).sum())
        if abs(valid_probability-1) > 1e-8:
            raise RuntimeError("Grouped fermionic evolution failed physical N/Sz preservation")
        save_binary(directory / "source-basis-preparation.qpy", circuit_bytes(preparation))
        save_binary(directory / "source-basis-sampling.qpy", circuit_bytes(sampler))
        save_npz(directory / "final.npz", state=expected, probability=abs(expected)**2)
    result = {"case_id": case, "status": "SDK_GROUPED_FERMION_CIRCUIT_PARITY_VALIDATED", "export_wall_seconds": export_seconds,
              "state_parity_error": error, "valid_N_Sz_sector_probability": valid_probability,
              "resources": resources(sampler, 1, num_terms), "preparation_resources": resources(preparation, 1, num_terms),
              "removed_terminal_diagonal": removed, "runtime": {"export_qiskit": "2.5.0", "main_common_compiler_qiskit": "2.6.0", "qiskit_fermions": "0.1.0"},
              "sdk_export_code_sha256": file_digest(ROOT / "benchmark/sqdrift_sdk_export.py"),
              "adapter_sha256": file_digest(__file__), "group_first_moment_error": sdk["max_group_first_moment_error"],
              "partition": sdk["partition"], "raw_SDK_first_moment_mismatch_before_refinement": sdk["raw_SDK_first_moment_mismatch_before_refinement"],
              "max_unit_group_Pauli_L1_norm_bound": sdk["max_unit_group_Pauli_L1_norm_upper_bound"],
              "Hamiltonian_diagonal_terms_retained": True, "hardware_shots": 0,
              "scope": "one randomized circuit; NOT the full multi-time/randomization SqDRIFT benchmark", "whole_study_complete": False}
    save_json(directory / "qualification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--case", default="H2_R0.7"); p.add_argument("--terms", type=int, default=8)
    p.add_argument("--output", default=str(ROOT / "results/SqDRIFT-SDK-qualification-v3")); p.add_argument("--source", default=str(ROOT / "results/standard-v1"))
    a = p.parse_args(); result = qualify(a.case, a.output, a.source, num_terms=a.terms)
    print(json.dumps(result))
    if result["status"] != "SDK_GROUPED_FERMION_CIRCUIT_PARITY_VALIDATED":
        raise SystemExit(1)
