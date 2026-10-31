"""Measure CCSD/CISD initialization from physical-only fixed integrals.

No reference energies/coefficient archives are read. Noncanonical fixed orbitals
are recorded, not silently reoptimized or rejected by an arbitrary strict test.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from pyscf import ao2mo, cc, ci, gto, scf

from .input_adapter import array_bytes, hf_energy
from .io import ROOT, atomic_bytes, assert_safe_physical_input, save_json, sha256


def run(case_id, seconds=180.):
    folder = ROOT / "data/initialization-fresh" / case_id
    if (folder/"result.json").exists():
        return json.loads((folder/"result.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    physical = ROOT/"data/physical"/case_id
    meta = json.loads((physical/"problem.json").read_text())
    assert_safe_physical_input(meta)
    with np.load(physical/"integrals.npz") as z:
        h1, eri = z["h1"].copy(), z["eri"].copy()
    n, na, nb, core = meta["norb"], meta["nalpha"], meta["nbeta"], meta["core_energy"]
    if na != nb:
        raise ValueError("Registered primary closed-shell initialization only")
    mol = gto.M(verbose=0)
    mol.nelectron = na+nb
    mol.spin = 0
    mol.nao_nr = lambda *a: n
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a: h1
    mf.get_ovlp = lambda *a: np.eye(n)
    mf.energy_nuc = lambda *a: core
    mf._eri = ao2mo.restore(8, eri, n)
    mf.mo_coeff = np.eye(n)
    mf.mo_occ = np.array([2.]*na+[0.]*(n-na))
    mf.e_tot = hf_energy(h1, eri, core, na, nb)
    fock = mf.get_fock(dm=mf.make_rdm1())
    mf.mo_energy = np.diag(fock)
    fov = float(np.linalg.norm(fock[:na, na:]))
    mf.converged = fov < 1e-6
    if abs(mf.e_tot-meta["hf_energy_Ha"]) > 1e-8:
        raise RuntimeError("Fixed physical reference energy mismatch")
    attempts = []
    start, cpu_start = perf_counter(), process_time()
    save_json(folder/"started.json", {"case_id": case_id, "pid": os.getpid(), "problem_id": meta["problem_id"],
               "physical_input_only": True, "source_FCI_or_saved_amplitudes_read": False,
               "wall_limit_per_method_seconds": seconds})
    final = {}
    for method in ("CCSD", "CISD"):
        tick = perf_counter()
        solver = cc.CCSD(mf) if method == "CCSD" else ci.CISD(mf)
        solver.max_cycle = 150
        solver.conv_tol = 1e-8
        solver.max_memory = 768
        boundaries = []

        def callback(env):
            boundaries.append(len(boundaries)+1)
            if perf_counter()-tick > seconds:
                raise TimeoutError("Initialization iteration-boundary wall cap")

        solver.callback = callback
        try:
            solver.kernel()
            attempt = {"method": method, "converged": bool(solver.converged),
                       "total_energy_Ha": float(solver.e_tot), "iteration_callbacks": len(boundaries)}
            if not solver.converged:
                raise RuntimeError(method+" did not converge within registered iteration budget")
            if method == "CCSD":
                arrays = {"ccsd_t1": solver.t1, "ccsd_t2": solver.t2}
            else:
                arrays = {"cisd_vec": solver.ci}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise RuntimeError("Nonfinite initialized amplitudes")
            payload = array_bytes(arrays)
            atomic_bytes(folder/"amplitudes.npz", payload)
            final = {"status": "FRESH_INITIALIZATION_COMPLETED", "selected_method": method,
                     "amplitudes_sha256": sha256(payload)}
            attempt["status"] = "COMPLETED"
        except Exception as exc:
            attempt = {"method": method, "status": "FAILED_PRESERVED", "error": repr(exc),
                       "converged": False, "iteration_callbacks": len(boundaries)}
        attempt["wall_seconds"] = perf_counter()-tick
        attempts.append(attempt)
        save_json(folder/"attempts.json", {"attempts": attempts})
        if final:
            break
    if not final:
        final = {"status": "INITIALIZATION_FAILED_BOTH_METHODS"}
    final.update(case_id=case_id, problem_id=meta["problem_id"], attempts=attempts,
                 occupied_virtual_Fock_norm=fov, SCF_stationary_under_1e6_rule=bool(mf.converged),
                 SCF_orbital_optimization_executed=False, physical_orbital_gauge_unchanged=True,
                 total_wall_seconds=perf_counter()-start, CPU_process_seconds=process_time()-cpu_start,
                 original_saved_amplitudes_used=False, FCI_or_SCI_reference_used=False,
                 actual_QPU_shots=0, model_quantum_shots=0, whole_experiment_complete=False)
    save_json(folder/"result.json", final)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="N2-32-R1p2-P2")
    p.add_argument("--seconds", type=float, default=180.)
    a = p.parse_args()
    print(json.dumps(run(a.case, a.seconds), indent=2))
