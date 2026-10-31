"""Public, legitimate supplementary-data retrieval; no guessed geometry."""
from io import BytesIO
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter, process_time
import urllib.request
import zipfile

from .io import ROOT, atomic_bytes, digest, save_json, sha256, assert_safe_physical_input


def fetch():
    directory = ROOT/"sources/Ghosh-2017"
    directory.mkdir(parents=True, exist_ok=True)
    filename = "SC-008-C6SC05036K-s001.pdf"
    target = directory/filename
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5433034/supplementaryFiles"
    if not target.exists():
        data = urllib.request.urlopen(url, timeout=60).read()
        archive = zipfile.ZipFile(BytesIO(data))
        if filename not in archive.namelist():
            raise RuntimeError("Expected public source supplement absent")
        payload = archive.read(filename)
        if not payload.startswith(b"%PDF"):
            raise RuntimeError("Source response is not PDF")
        atomic_bytes(target, payload)
        save_json(directory/"source.json", {"source_URL": url, "public_archive_member": filename,
                  "article_DOI": "10.1039/C6SC05036K", "zip_sha256": sha256(data),
                  "pdf_sha256": sha256(payload), "geometry_extracted": False,
                  "paper_P1_input_equality_not_yet_verified": True})
    if not (directory/"supplement.txt").exists():
        subprocess.run(["rtk", "proxy", "pdftotext", "-layout", str(target), str(directory/"supplement.txt")], check=True)
    return str(directory)


def geometry():
    directory = Path(fetch())
    text = (directory/"supplement.txt").read_text()
    section = text.split("Singlet Geometries (B3LYP/6-31G**)", 1)[1].split("Anthracene", 1)[0]
    rows = re.findall(r"^\s*([CH])\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", section, re.M)
    atoms = [[symbol, [float(x), float(y), float(z)]] for symbol, x, y, z in rows]
    if len(atoms) != 18 or sum(a[0] == "C" for a in atoms) != 10 or any(a[1][2] != 0 for a in atoms):
        raise RuntimeError("Naphthalene source coordinate extraction failed")
    record = {"atoms_angstrom": atoms, "geometry_digest": digest(atoms),
              "pdf_sha256": sha256((directory/"SC-008-C6SC05036K-s001.pdf").read_bytes()),
              "source_page": "S-10", "source_section": "Singlet Geometries (B3LYP/6-31G**)",
              "source_spelling": "Napthalene", "visual_coordinate_table_checked": True,
              "geometry_reoptimized": False}
    save_json(directory/"naphthalene-geometry.json", record)
    return record


def prepare():
    import numpy as np
    from pyscf import ao2mo, gto, mcscf, scf
    from pyscf.mcscf import avas
    from .input_adapter import array_bytes, hf_energy
    folder = ROOT/"data/physical/NAPH20-P1"
    if (folder/"problem.json").exists():
        return json.loads((folder/"problem.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    record = geometry()
    tick, cpu = perf_counter(), process_time()
    source = ROOT/"sources/Ghosh-2017"
    settings = {"basis": "cc-pvdz", "spin": 0, "charge": 0, "unit": "Angstrom",
                "AVAS_AO_labels": ["C 2pz"], "AVAS_threshold": 0.2, "AVAS_minao": "minao",
                "AVAS_with_IAO": False, "AVAS_canonicalize": True,
                "RHF_conv_tol": 1e-10, "RHF_max_cycle": 100, "max_memory_MB": 512,
                "original_P1_versions": {"pyscf": "2.4.0", "ffsim": "0.0.57"},
                "current_pyscf": version("pyscf"),
                "unreported_AVAS_options_use_explicit_current_defaults_not_author_verified": True}
    save_json(folder/"preparation-started.json", {"geometry": record, "settings": settings})
    mol = gto.M(atom=record["atoms_angstrom"], basis="cc-pvdz", unit="Angstrom", spin=0,
                charge=0, verbose=3, max_memory=512, output=str(folder/"pyscf.log"))
    mf = scf.RHF(mol)
    mf.max_memory = 512
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Source-geometry RHF did not converge; no geometry tuning")
    av = avas.AVAS(mf, ["C 2pz"], threshold=.2, minao="minao", with_iao=False, canonicalize=True)
    ncas, nelecas, mo = av.kernel()
    if (int(ncas), int(nelecas)) != (10, 10):
        raise RuntimeError(f"Published AVAS space not recovered: {ncas} orbitals/{nelecas} electrons")
    cas = mcscf.CASCI(mf, 10, (5, 5))
    cas.mo_coeff = mo
    h1, core = cas.get_h1eff(mo_coeff=mo)
    eri = ao2mo.restore(1, cas.get_h2eff(mo_coeff=mo), 10)
    h1, eri = np.ascontiguousarray(h1), np.ascontiguousarray(eri)
    reconstructed = hf_energy(h1, eri, float(core), 5, 5)
    if abs(reconstructed-mf.e_tot) > 1e-8:
        raise RuntimeError("AVAS active-space/core reference does not reconstruct physical RHF")
    hf_id = sum(1 << (19-2*i) for i in range(5))+sum(1 << (18-2*i) for i in range(5))
    hashes = {k: sha256(v.astype("<f8").tobytes()) for k, v in (("h1", h1), ("eri", eri))}
    physical = {"norb": 10, "nalpha": 5, "nbeta": 5, "hf_id": hf_id, "core_energy": float(core),
                "physical_arrays_sha256": hashes, "mapping": "Jordan-Wigner",
                "orbital_order": "interleaved alpha/beta, orbital0 at most-significant logical bit"}
    meta = {"schema": "paper-physical-input-1", **physical, "problem_id": digest(physical),
            "case_id": "NAPH20-P1", "n_qubits": 20, "hf_energy_Ha": float(mf.e_tot),
            "geometry": record["atoms_angstrom"], "basis": "cc-pvdz",
            "source": {"kind": "P1_PROCEDURE_RECONSTRUCTION_CURRENT_RUNTIME", "geometry": record,
                       "paper": "https://arxiv.org/html/2508.02578v3", "settings": settings,
                       "author_integral_byte_equality_verified": False},
            "qualification": {"HF_reconstructed_Ha": reconstructed,
                    "HF_absolute_difference_Ha": abs(reconstructed-mf.e_tot),
                    "mo_orthogonality_max_error": float(np.max(abs(mo.T@mf.get_ovlp()@mo-np.eye(mo.shape[1])))),
                    "occupied_weights": np.asarray(av.occ_weights).tolist(), "virtual_weights": np.asarray(av.vir_weights).tolist(),
                    "RHF_converged": bool(mf.converged), "FCI_run_during_input_generation": False}}
    assert_safe_physical_input(meta)
    atomic_bytes(folder/"integrals.npz", array_bytes({"h1": h1, "eri": eri}))
    atomic_bytes(folder/"orbital-basis.npz", array_bytes({"AVAS_MO": mo, "AO_overlap": mf.get_ovlp()}))
    save_json(folder/"problem.json", meta)
    save_json(folder/"preparation-cost.json", {"wall_seconds": perf_counter()-tick, "CPU_seconds": process_time()-cpu,
              "source_geometry": record, "SCF_AVAS_integrals_included": True, "FCI_used": False,
              "full_Fock_vector_prepared": False, "model_quantum_shots": 0, "actual_QPU_shots": 0})
    return meta


if __name__ == "__main__":
    import sys
    print(json.dumps(prepare() if "--prepare" in sys.argv else geometry(), indent=2))
