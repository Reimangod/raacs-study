from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_artifact(root,recorded_path):
    """Relocate stored absolute paths without editing immutable checkpoint JSON.

    Never fall back to an unrelated old output directory that happens to exist.
    """
    p=Path(recorded_path);parts=p.parts
    for anchor in ("distributions","checkpoints"):
        positions=[i for i,x in enumerate(parts) if x==anchor]
        if positions:return Path(root).resolve().joinpath(*parts[positions[-1]:])
    if not p.is_absolute():return Path(root).resolve()/p
    raise ValueError("Unknown artifact layout: "+str(p))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(raw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_binary(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(raw)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def provenance():
    packages = {}
    for name in ("numpy", "scipy", "pyscf", "qiskit", "ffsim", "qiskit-addon-sqd",
                 "openfermion", "openfermionpyscf", "quimb", "matplotlib", "pandas", "pyarrow"):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {"version": dist.version,
                              "license": dist.metadata.get("License-Expression") or dist.metadata.get("License")}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"status": "UNAVAILABLE"}
    vendor = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / "vendor/ceo").rglob("*.py"))}
    code = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / "benchmark").rglob("*.py"))}
    return {"packages": packages, "code_hashes": code, "vendor_hashes": vendor,
            "vendor_origin": "https://github.com/mafaldaramoa/ceo-adapt-vqe",
            "vendor_source": "/Users/rei/Documents/ceo-adapt-vqe/ceo-adapt-vqe/adaptvqe",
            "vendor_license": "Apache-2.0; vendor/ceo/LICENSE",
            "vendor_identity": "FILE_HASH_SNAPSHOT_NOT_INDEPENDENT_GIT_CHECKOUT",
            "lock_sha256": file_digest(ROOT / "uv.lock")}


def rng_for(*identity):
    seed = int(digest(list(identity))[:16], 16)
    return np.random.default_rng(seed), seed

