"""Evaluation-only scalar reference, kept outside all algorithm requests."""
import json
from time import perf_counter, process_time
import numpy as np
from pyscf.fci import direct_spin1

from .common_execution import load_physical
from .io import ROOT, save_json


def naphthalene():
    path = ROOT/"data/evaluation-only/NAPH20-P1.json"
    if path.exists():
        return json.loads(path.read_text())
    meta, h1, eri = load_physical("NAPH20-P1")
    tick, cpu = perf_counter(), process_time()
    solver = direct_spin1.FCI()
    solver.verbose = 0
    solver.max_memory = 512
    solver.max_cycle = 150
    solver.conv_tol = 1e-11
    E, vector = solver.kernel(h1, eri, 10, (5, 5), ecore=meta["core_energy"])
    contracted = solver.contract_2e(solver.absorb_h1e(h1, eri, 10, (5, 5), .5), vector, 10, (5, 5))
    residual = float(np.linalg.norm(contracted-(E-meta["core_energy"])*vector))
    S2, multiplicity = solver.spin_square(vector, 10, (5, 5))
    record = {"schema": "evaluation-reference-1", "problem_id": meta["problem_id"],
              "energies_Ha": {"hf_energy": meta["hf_energy_Ha"], "fci_energy": float(E)},
              "source": "Independent PySCF direct_spin1 FCI on the frozen reconstructed physical integrals",
              "sector": {"nalpha": 5, "nbeta": 5}, "reference_spin_square": float(S2),
              "solver_converged": bool(solver.converged), "internal_residual_norm": residual,
              "sector_and_reference_precision_verified": bool(solver.converged and residual < 1e-5),
              "FCI_coefficient_arrays_exported": False, "algorithm_access": "FORBIDDEN",
              "CPU_seconds": process_time()-cpu, "wall_seconds": perf_counter()-tick,
              "reference_vector_size": int(vector.size), "full_Fock_dimension_not_allocated": True,
              "reference_not_used_to_select_or_tune_sampling": True}
    if not record["sector_and_reference_precision_verified"]:
        raise RuntimeError("Evaluation reference solver did not reach registered residual check")
    save_json(path, record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    naphthalene()
