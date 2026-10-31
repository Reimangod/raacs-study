from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = ROOT.parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path: Path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def assert_safe_physical_input(meta):
    allowed = {"schema", "problem_id", "case_id", "norb", "n_qubits", "nalpha", "nbeta",
               "hf_id", "hf_energy_Ha", "core_energy", "mapping", "orbital_order",
               "physical_arrays_sha256", "geometry", "basis", "source", "qualification"}
    if set(meta) - allowed:
        raise ValueError("Forbidden/unrecognized physical input fields: " + str(sorted(set(meta) - allowed)))
    if any(x in json.dumps(meta).lower() for x in ("fci_vec", "fci_energy", "sci_vec", "evaluation_reference")):
        raise ValueError("Evaluation reference found in physical input")
