"""Fresh molecular integrals without a ground-state solver or FCI labels.

The optional small-system JW Hamiltonian is qualification only (<=16 qubits).
The integral-only route never allocates a Fock-space vector, matrix or full
determinant-sector enumeration. It is the physical problem boundary for C/D/E.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from math import comb
from pathlib import Path
from time import perf_counter
import numpy as np
from pyscf import ao2mo, gto, lib, mcscf, scf
from .common import digest, file_digest, save_json, save_npz


@dataclass(frozen=True)
class MolecularSpec:
    molecule: str
    distance_A: float
    frozen_core: int = 0
    active_orbitals: int | None = None
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0

    def geometry(self):
        r = self.distance_A
        if not np.isfinite(r) or r <= 0:
            raise ValueError("Positive finite distance required")
        if self.molecule.startswith("H") and self.molecule[1:].isdigit():
            return [("H", (0., 0., i*r)) for i in range(int(self.molecule[1:]))]
        if self.molecule == "LiH":
            return [("Li", (0., 0., 0.)), ("H", (0., 0., r))]
        if self.molecule == "BeH2":
            return [("H", (0., 0., -r)), ("Be", (0., 0., 0.)), ("H", (0., 0., r))]
        if self.molecule in ("N2", "C2"):
            return [(self.molecule[:-1], (0., 0., 0.)), (self.molecule[:-1], (0., 0., r))]
        if self.molecule == "H2O":
            angle = np.deg2rad(104.5)/2
            return [("O", (0., 0., 0.)), ("H", (r*np.sin(angle), 0., r*np.cos(angle))),
                    ("H", (-r*np.sin(angle), 0., r*np.cos(angle)))]
        raise ValueError("Unregistered molecular geometry")


def sector_ids(norb, na, nb):
    """Interleaved big-endian IDs; only call for bounded small qualification."""
    return np.array(sorted(sum(1 << (2*norb-1-2*i) for i in a)+sum(1 << (2*norb-2-2*j) for j in b)
                           for a in combinations(range(norb), na) for b in combinations(range(norb), nb)), dtype=np.int64)


def build_problem(spec, directory, small_full_H=False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    config = {"schema": "molecular-no-FCI-integrals-v1", "spec": asdict(spec), "geometry": spec.geometry(),
              "small_full_H": bool(small_full_H), "pyscf": __import__("pyscf").__version__,
              "implementation_sha256": file_digest(__file__), "max_integral_bytes": 256*2**20,
              "MO": "canonical RHF/ROHF energy order", "mapping": "JW interleaved alpha,beta; big-endian IDs",
              "FCI_solve": False, "FCI_labels": False}
    config_id = digest(config); path = directory/"problem.json"
    if path.exists():
        meta = json.loads(path.read_text())
        if meta["config_digest"] != config_id:
            raise RuntimeError("Molecular physical problem/code drift; use successor directory")
        for name, sha in meta["artifact_sha256"].items():
            if file_digest(directory/name) != sha:
                raise RuntimeError("Molecular problem artifact mismatch")
        return load_problem(directory)
    lib.num_threads(1); start = perf_counter()
    mol = gto.M(atom=spec.geometry(), unit="Angstrom", basis=spec.basis, charge=spec.charge, spin=spec.spin, verbose=0)
    mf = scf.RHF(mol) if spec.spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-11; mf.max_cycle = 150; mf.kernel(); method = type(mf).__name__
    if not mf.converged:
        mf = mf.newton(); mf.conv_tol = 1e-11; mf.kernel(); method += "+Newton"
    if not mf.converged:
        raise RuntimeError("SCF did not converge; not a ground-state sampler failure")
    ncore = spec.frozen_core
    norb = mf.mo_coeff.shape[1]-ncore if spec.active_orbitals is None else spec.active_orbitals
    nelec = mol.nelectron-2*ncore; na = (nelec+spec.spin)//2; nb = nelec-na
    if not isinstance(ncore, int) or not isinstance(norb, int) or norb < 1 or ncore < 0 or ncore+norb > mf.mo_coeff.shape[1] or not 0 <= nb <= na <= norb:
        raise ValueError("Invalid frozen/active-space/electron specification")
    if 2*norb > 62 or 8*norb**4 > config["max_integral_bytes"]:
        raise RuntimeError("Bounded uint64/integral-storage scope exceeded")
    if small_full_H and 2*norb > 16:
        raise ValueError("Full-Fock qualification is limited to16qubits; use integral-only route")
    h1, ecore = mcscf.casci.h1e_for_cas(mf, mf.mo_coeff, norb, ncore)
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff[:, ncore:ncore+norb]), norb)
    hf = sum(1 << (2*norb-1-2*i) for i in range(na))+sum(1 << (2*norb-2-2*i) for i in range(nb))
    save_npz(directory/"integrals.npz", h1=h1, eri=eri, mo_coeff=mf.mo_coeff, ecore=np.array(float(ecore)))
    artifacts = ["integrals.npz"]; n = 2*norb; sector_dimension = comb(norb, na)*comb(norb, nb)
    if small_full_H:
        import openfermion as of
        from openfermion.chem.molecular_data import spinorb_from_spatial
        from scipy.sparse import save_npz as save_sparse
        one, two = spinorb_from_spatial(h1, eri.transpose(0, 2, 3, 1))
        h = of.get_sparse_operator(of.InteractionOperator(ecore, one, .5*two), n_qubits=n).tocsr()
        if abs(h[hf, hf].real-mf.e_tot) > 1e-8:
            raise RuntimeError("Independent HF/JW matrix diagonal mismatch")
        save_sparse(directory/"small-hamiltonian.npz", h)
        save_npz(directory/"sector.npz", ids=sector_ids(norb, na, nb)); artifacts += ["small-hamiltonian.npz", "sector.npz"]
    physical_id = digest({"spec": asdict(spec), "h1": file_digest(directory/"integrals.npz"), "mapping": config["mapping"]})
    meta = {"config": config, "config_digest": config_id, "problem_id": physical_id,
            "n_orbitals": norb, "n_qubits": n, "n_electrons": nelec, "nalpha": na, "nbeta": nb,
            "hf_id": hf, "hf_energy_Ha": float(mf.e_tot), "core_constant_Ha": float(ecore),
            "sector_dimension": sector_dimension, "SCF_solver": method, "SCF_converged": True,
            "construction_wall_seconds": perf_counter()-start, "FCI_computed": False,
            "ground_state_reference_energy_Ha": None, "chemical_accuracy": "UNKNOWN_NO_GROUND_STATE_REFERENCE",
            "full_Fock_H_constructed": bool(small_full_H), "full_sector_enumerated": bool(small_full_H),
            "hardware_shots": 0, "artifact_sha256": {name: file_digest(directory/name) for name in artifacts}}
    save_json(path, meta)
    return load_problem(directory)


def load_problem(directory):
    directory = Path(directory); meta = json.loads((directory/"problem.json").read_text())
    with np.load(directory/"integrals.npz") as z: arrays = {name: z[name] for name in z.files}
    if meta["full_Fock_H_constructed"]:
        from scipy.sparse import load_npz
        arrays["h"] = load_npz(directory/"small-hamiltonian.npz")
        with np.load(directory/"sector.npz") as z: arrays["ids"] = z["ids"]
    return {"meta": meta, "directory": directory, **arrays}
