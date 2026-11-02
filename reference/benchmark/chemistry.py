"""RHF canonical active-space Hamiltonian and independent PySCF/JW parity."""
from __future__ import annotations
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy import sparse
from pyscf import gto, scf, ao2mo, fci, mcscf, lib
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
from .common import digest, save_json, save_npz, file_digest


def determinant_ids(norb, nalpha, nbeta):
    aa = list(fci.cistring.make_strings(range(norb), nalpha))
    bb = list(fci.cistring.make_strings(range(norb), nbeta))
    ids, signs = [], []
    for a in aa:
        aocc = [i for i in range(norb) if int(a) >> i & 1]
        for b in bb:
            bocc = [i for i in range(norb) if int(b) >> i & 1]
            ids.append(sum(1 << (2 * norb - 1 - 2*i) for i in aocc)
                       + sum(1 << (2 * norb - 2 - 2*i) for i in bocc))
            # PySCF alpha creators precede beta creators. JW interleaves them.
            signs.append((-1) ** sum(j < i for i in aocc for j in bocc))
    return np.array(ids, dtype=np.int64), np.array(signs), aa, bb


def build_reference(case, root, threads=1):
    lib.num_threads(threads)
    directory = Path(root) / "references" / case.id
    meta_path = directory / "reference.json"
    spec = case.specification()
    version = {"schema": "reference-1", "case": spec, "pyscf": __import__("pyscf").__version__}
    key = digest(version)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["request_digest"] != key:
            raise RuntimeError("Reference config changed; use a new results directory")
        for name, sha in meta["artifact_sha256"].items():
            if file_digest(directory / name) != sha:
                raise RuntimeError("Reference cache is incomplete or corrupted: " + name)
        return load_reference(directory)
    start = perf_counter()
    mol = gto.M(atom=case.geometry(), unit="Angstrom", basis=case.basis,
                charge=case.charge, spin=case.spin, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 150
    mf.kernel()
    scf_solver = "DIIS_RHF"
    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = 1e-11
        mf.kernel()
        scf_solver = "NEWTON_RHF_AFTER_DIIS"
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    ncore = case.frozen_core
    norb = mf.mo_coeff.shape[1] - ncore
    nelec = mol.nelectron - 2 * ncore
    na, nb = nelec // 2, nelec // 2
    h1, ecore = mcscf.casci.h1e_for_cas(mf, mf.mo_coeff, norb, ncore)
    mo_active = mf.mo_coeff[:, ncore:]
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_active), norb)
    solver = fci.direct_spin1.FCI(mol)
    solver.conv_tol = 1e-12
    # Energy-difference convergence alone permits a ~1e-6 CI residual on N2.
    # Tighten the actual eigensolver, never relax the independent JW check.
    solver.conv_tol_residual = 1e-8
    solver.lindep = 1e-18
    solver.max_space = 50
    solver.max_cycle = 200
    e_fci, coeff = solver.kernel(h1, eri, norb, (na, nb), ecore=ecore)
    if not solver.converged:
        raise RuntimeError("FCI did not converge")
    one, two = spinorb_from_spatial(h1, eri.transpose(0, 2, 3, 1))
    h = of.get_sparse_operator(of.InteractionOperator(ecore, one, .5 * two),
                               n_qubits=2*norb).tocsr()
    ids, signs, aa, bb = determinant_ids(norb, na, nb)
    ci_vector = coeff.ravel() * signs
    h_sector = h[ids][:, ids]
    residual = float(np.linalg.norm(h_sector @ ci_vector - e_fci * ci_vector))
    hf_id = sum(1 << (2*norb - 1 - q) for q in range(nelec))
    hf_energy = float(h[hf_id, hf_id].real)
    if abs(hf_energy - mf.e_tot) > 1e-8 or residual > 1e-7:
        raise RuntimeError(f"HF/FCI bit/phase parity failed: {hf_energy-mf.e_tot}, residual {residual}")
    weights = np.abs(ci_vector) ** 2
    if not np.isclose(weights.sum(), 1., atol=1e-10):
        raise RuntimeError("Unnormalized FCI reference")
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "hamiltonian.npz", h)
    save_npz(directory / "reference.npz", ids=ids, amplitudes=ci_vector,
             weights=weights, h1=h1, eri=eri, mo_coeff=mf.mo_coeff)
    records = []
    for idx, (state_id, amp, w) in enumerate(zip(ids, ci_vector, weights)):
        a, b = int(aa[idx // len(bb)]), int(bb[idx % len(bb)])
        records.append({"determinant": format(int(state_id), f"0{2*norb}b"),
                        "state_index": int(state_id), "alpha_bits": format(a, f"0{norb}b"),
                        "beta_bits": format(b, f"0{norb}b"), "fci_amplitude": float(amp),
                        "fci_weight": float(w), "is_hf": int(state_id) == hf_id,
                        "electron_number": nelec, "spin_sector": f"Nalpha={na};Nbeta={nb}"})
    table = pd.DataFrame(records).sort_values(["fci_weight", "state_index"], ascending=[False, True])
    table["rank"] = np.arange(1, len(table)+1)
    table.to_csv(directory / "determinants.csv", index=False)
    table.to_parquet(directory / "determinants.parquet", index=False)
    meta = {"request_digest": key, **version, "n_orbitals": norb, "n_qubits": 2*norb,
            "n_electrons": nelec, "nalpha": na, "nbeta": nb, "hf_id": hf_id,
            "rhf_energy": float(mf.e_tot), "fci_energy": float(e_fci),
            "hf_weight_fci": float(weights[ids == hf_id][0]),
            "correlation_energy": float(e_fci-mf.e_tot), "fci_residual": residual,
            "fci_weight_sum": float(weights.sum()), "scf_solver": scf_solver,
            "fci_solver_energy_tolerance":1e-12,"fci_solver_residual_tolerance":1e-8,
            "fci_solver_lindep":1e-18,"fci_solver_max_space":50,
            "scf_converged": True, "fci_converged": True, "seconds": perf_counter()-start}
    meta["artifact_sha256"] = {name: file_digest(directory / name) for name in
                               ("hamiltonian.npz", "reference.npz", "determinants.csv", "determinants.parquet")}
    save_json(meta_path, meta)
    return load_reference(directory)


def load_reference(directory):
    directory = Path(directory)
    meta = json.loads((directory / "reference.json").read_text())
    with np.load(directory / "reference.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return {"meta": meta, "directory": directory,
            "h": sparse.load_npz(directory / "hamiltonian.npz"), **arrays}

