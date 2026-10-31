"""Independent official-SDK fermion/JW-to-source Hamiltonian entry audit."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def export_only(request_file, output_file):
    import qiskit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.mappers.library import jordan_wigner
    assert qiskit.__version__ == "2.5.0"
    request = json.loads(Path(request_file).read_text())
    terms = [(tuple((bool(a), int(q)) for q, a in row["term"]), complex(*row["coefficient"])) for row in request["fermion_terms"]]
    operator = FermionOperator.from_terms(terms).normal_ordered().simplify(atol=1e-12)
    pauli = SparsePauliOp.from_sparse_observable(jordan_wigner(operator, request["n_qubits"])).simplify(atol=1e-12)
    if max(abs(pauli.coeffs.imag)) > 1e-10:
        raise RuntimeError("Non-Hermitian physical Hamiltonian")
    Path(output_file).write_text(json.dumps({"qiskit": qiskit.__version__, "terms": [(label[::-1], float(c.real)) for label, c in pauli.to_list()]})+"\n")


def audit(root, source):
    from .common import ROOT, file_digest, save_json
    from .chemistry import load_reference
    from .sqdrift_qualify import request_from_integrals
    from .nonvariational_audit import named_pauli_entries
    root, source = Path(root), Path(source); root.mkdir(exist_ok=True, parents=True)
    env = dict(os.environ); env["PYTHONPATH"] = str(ROOT / ".runtime-overlays/sqdrift-qiskit25-v1")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = "1"
    records = []
    for directory in sorted((source / "references").iterdir()):
        if not directory.is_dir():
            continue
        ref = load_reference(directory); meta = ref["meta"]; out = root / directory.name; out.mkdir(exist_ok=True)
        request = request_from_integrals(ref, 1., 8, 101)
        save_json(out / "request.json", request)
        subprocess.run([sys.executable, "-m", "benchmark.sqdrift_hamiltonian_audit", "--sdk-export-only", "--input", str((out / "request.json").resolve()), "--output", str((out / "pauli.json").resolve())], env=env, cwd=ROOT, check=True)
        terms = json.loads((out / "pauli.json").read_text())["terms"]
        rng = np.random.default_rng(7214)
        ids = np.unique(np.r_[ref["ids"][rng.integers(0, len(ref["ids"]), size=48)], rng.integers(0, 1 << meta["n_qubits"], size=48), [meta["hf_id"]]])
        actual = named_pauli_entries(terms, request["scalar_identity_omitted"], ids)
        expected = ref["h"][ids][:, ids].toarray()
        error = float(abs(actual-expected).max())
        if error > 1e-9:
            raise RuntimeError("SDK physical Hamiltonian/mode/phase mismatch: "+directory.name)
        records.append({"case_id": directory.name, "named_entries": len(ids)**2, "max_error_Ha": error})
        print("SDK original Hamiltonian parity "+directory.name, flush=True)
    result = {"status": "ALL_30_SDK_PHYSICAL_HAMILTONIANS_VALIDATED", "records": records,
              "implementation_sha256": file_digest(__file__), "hardware_shots": 0, "whole_study_complete": False}
    save_json(root / "validation.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--sdk-export-only", action="store_true"); p.add_argument("--input")
    p.add_argument("--output", default="results/SqDRIFT-SDK-Hamiltonian-audit-v1"); p.add_argument("--source", default="results/standard-v1")
    a = p.parse_args()
    if a.sdk_export_only:
        export_only(a.input, a.output)
    else:
        print(json.dumps(audit(a.output, a.source)))
