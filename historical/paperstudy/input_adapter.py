"""Whitelist original catalog inputs; no SCF/FCI/candidate optimization is run."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import lzma
from pathlib import Path
from time import perf_counter
import urllib.request

import numpy as np
from pyscf import ao2mo

from .io import ROOT, PLAN_ROOT, atomic_bytes, save_json, digest, sha256, assert_safe_physical_input

C1_COMMIT = "b52bca05650e60d890b06ac0a3b6d167303d0a90"
BASE = "https://raw.githubusercontent.com/kevinsung/lucj/" + C1_COMMIT + "/"


def hf_energy(h1, eri, core, na, nb):
    occ = np.zeros(len(h1), float)
    occ[:na] += 1
    occ[:nb] += 1
    energy = float(core) + float(np.dot(occ, np.diag(h1)))
    energy += .5 * float(np.einsum("i,j,iijj->", occ, occ, eri))
    for nspin in (na, nb):
        energy -= .5 * sum(float(eri[i, j, j, i]) for i in range(nspin) for j in range(nspin))
    return energy


def raw_to_components(raw, case):
    """Do not return raw catalog or reference coefficient arrays to algorithms."""
    norb = int(raw["norb"])
    na, nb = map(int, raw["nelec"])
    if not 0 <= na <= norb or not 0 <= nb <= norb:
        raise ValueError("Invalid spin electron counts")
    if (na + nb, norb, 2 * norb) != (case["n_electrons"], case["n_spatial_orbitals"], case["n_qubits_jw"]):
        raise ValueError("Physical active-space identity mismatch")
    h1 = np.asarray(raw["one_body_integrals"], dtype=float)
    eri = np.ascontiguousarray(ao2mo.restore(1, np.asarray(raw["two_body_integrals"], dtype=float), norb))
    core = float(raw["core_energy"])
    if h1.shape != (norb, norb) or eri.shape != (norb,) * 4:
        raise ValueError("Integral shapes")
    if not np.isfinite(h1).all() or not np.isfinite(eri).all() or not np.isfinite(core):
        raise ValueError("Nonfinite physical integrals")
    symmetry_errors = [float(np.max(abs(h1-h1.T))), float(np.max(abs(eri-eri.transpose(1, 0, 2, 3)))),
                       float(np.max(abs(eri-eri.transpose(2, 3, 0, 1))))]
    if max(symmetry_errors) > 1e-8:
        raise ValueError("Real chemist-order integral symmetry mismatch")
    supplied_hf = float(raw["hf_energy"])
    if not np.isfinite(supplied_hf):
        raise ValueError("Nonfinite physical HF identity")
    raw_gauge_hf = hf_energy(h1, eri, core, na, nb)
    computed_hf = raw_gauge_hf
    orbital_conversion = {"kind": "NONE", "source_integrals_HF_determinant_energy_Ha": raw_gauge_hf}
    if abs(computed_hf-supplied_hf) > 1e-8 and raw.get("mo_coeff") is not None:
        # The Fe catalog stores integrals in the original orthonormal orbital
        # gauge, but HF/CISD data in its supplied MO gauge. Do not infer a new
        # scalar offset from the HF label. Transform both integral tensors by
        # the actual supplied orthogonal C, identically for every method.
        c = np.asarray(raw["mo_coeff"], float)
        if c.shape != (norb, norb) or not np.isfinite(c).all() or np.max(abs(c.T@c-np.eye(norb))) > 1e-8:
            raise ValueError("Source MO transform is not square orthogonal")
        h1 = np.ascontiguousarray(c.T@h1@c)
        eri = np.ascontiguousarray(ao2mo.incore.full(eri, c, compact=False).reshape((norb,)*4))
        computed_hf = hf_energy(h1, eri, core, na, nb)
        orbital_conversion.update(kind="SOURCE_MO_COEFF_ORTHOGONAL_TRANSFORM", matrix_sha256=sha256(c.astype("<f8").tobytes()),
                                  orthogonality_error=float(np.max(abs(c.T@c-np.eye(norb)))),
                                  all_comparison_methods_use_same_transformed_integrals=True,
                                  raw_paper_script_reproduction_is_separate=True)
    if not np.isfinite(computed_hf) or abs(computed_hf-supplied_hf) > 1e-8:
        raise ValueError(f"HF/reference determinant mismatch: {computed_hf} vs {supplied_hf}")
    hf_id = sum(1 << (2*norb-1-2*i) for i in range(na)) + sum(1 << (2*norb-2-2*i) for i in range(nb))
    arrays = {"h1": h1, "eri": eri}
    array_hashes = {k: sha256(v.astype("<f8").tobytes()) for k, v in arrays.items()}
    physical_identity = dict(norb=norb, nalpha=na, nbeta=nb, hf_id=hf_id, core_energy=core,
                             physical_arrays_sha256=array_hashes, mapping="Jordan-Wigner",
                             orbital_order="interleaved alpha/beta, orbital0 at most-significant logical bit")
    meta = dict(schema="paper-physical-input-1", problem_id=digest(physical_identity), case_id=case["id"],
                n_qubits=2*norb, hf_energy_Ha=supplied_hf, geometry=raw.get("atom"), basis=raw.get("basis"),
                source={"repository": "kevinsung/lucj", "commit": C1_COMMIT, "path": case["source_path"],
                        "archive_sha256": case["raw_input_sha256"]}, **physical_identity,
                qualification={"HF_reconstructed_Ha": computed_hf, "HF_absolute_difference_Ha": abs(computed_hf-supplied_hf),
                               "integral_symmetry_max_errors": symmetry_errors, "orbital_conversion": orbital_conversion,
                               "no_SCF_or_FCI_run": True})
    assert_safe_physical_input(meta)
    # Only initialization data, never FCI/SCI coefficients. Provenance of saved
    # CCSD convergence is not manufactured from the existence of an array.
    initialization = {k: np.asarray(raw[k], dtype=float) for k in ("ccsd_t1", "ccsd_t2", "cisd_vec", "mp2_t2") if raw.get(k) is not None}
    refs = {k: float(raw[k]) if raw.get(k) is not None else None for k in ("hf_energy", "fci_energy", "sci_energy", "cisd_energy", "ccsd_energy")}
    reference = {"schema": "evaluation-reference-1", "problem_id": meta["problem_id"], "source": meta["source"],
                 "energies_Ha": refs, "coefficient_arrays_exported": False,
                 "sector_and_reference_precision_verified": False, "algorithm_access": "FORBIDDEN"}
    return meta, arrays, initialization, reference


def array_bytes(arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def prepare(output=ROOT / "data"):
    plan = json.loads((PLAN_ROOT / "plan.json").read_text())
    output = Path(output)
    records = []
    for case in plan["primary_cases"]:
        if case.get("source_id") != "C1":
            records.append({"case_id": case["id"], "status": "INPUT_UNRESOLVED", "reason": case["geometry_reason"]})
            continue
        started = perf_counter()
        archive = output / "raw-source" / (case["raw_input_sha256"] + ".json.xz")
        if archive.exists():
            data = archive.read_bytes()
            downloaded = False
        else:
            with urllib.request.urlopen(BASE + case["source_path"], timeout=30) as response:
                data = response.read(8 * 2**20)
            downloaded = True
        if sha256(data) != case["raw_input_sha256"]:
            raise ValueError("Pinned source hash mismatch")
        if downloaded:
            atomic_bytes(archive, data)
        raw = json.loads(lzma.decompress(data))
        meta, arrays, init, reference = raw_to_components(raw, case)
        physical = output / "physical" / case["id"]
        physical.mkdir(parents=True, exist_ok=True)
        payload = array_bytes(arrays)
        atomic_bytes(physical / "integrals.npz", payload)
        save_json(physical / "problem.json", meta)
        init_dir = output / "initialization" / case["id"]
        atomic_bytes(init_dir / "amplitudes.npz", array_bytes(init))
        save_json(init_dir / "initialization.json", {"problem_id": meta["problem_id"], "fields": sorted(init),
                  "source_archive_sha256": case["raw_input_sha256"], "saved_optimizer_state": "UNKNOWN",
                  "saved_CCSD_convergence": "UNKNOWN", "original_creation_CPU_seconds": None,
                  "original_creation_cost_reason": "Not specified by downloaded archive; must measure C-lane initialization separately"})
        save_json(output / "evaluation-only" / (case["id"] + ".json"), reference)
        records.append({"case_id": case["id"], "problem_id": meta["problem_id"],
                        "status": "PHYSICAL_INPUT_RECONSTRUCTED_HF_PARITY_PASSED", "source_sha256": sha256(data),
                        "integral_archive_sha256": sha256(payload), "qualification": meta["qualification"],
                        "data_preparation_wall_seconds_this_call": perf_counter()-started, "source_downloaded_this_call": downloaded})
    report = {"schema": "paper-input-preparation-1", "cases": records, "new_SCF_FCI_or_candidates_executed": 0,
              "physical_inputs_prepared": sum(x["status"].startswith("PHYSICAL") for x in records),
              "whole_experiment_complete": False}
    save_json(output / "input-preparation.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "data")
    args = p.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))
