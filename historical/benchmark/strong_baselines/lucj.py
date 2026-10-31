"""Official ffsim LUCJ with bounded, explicitly costed optimizations."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from time import perf_counter

import ffsim
import numpy as np
from pyscf import gto, lib, mp, scf
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..ansatz import circuit_bytes, resources, verify_circuit_state
from ..chemistry import determinant_ids
from ..common import digest, file_digest, rng_for, save_binary, save_json, save_npz
from ..lucj import embedded_lucj_circuits, lucj_pair_lists
from ..study.core import RitzCache, open_db, ranks, rss_peak


class BudgetReached(RuntimeError):
    pass


def method_specs():
    rows = [dict(name="lucj_mp2_2_fixed", init="mp2", layers=2, objective="fixed"),
            dict(name="lucj_ccsd_2_compressed", init="ccsd", layers=2, objective="fixed", compressed=True)]
    for init, layers, trunc in (("ccsd", 1, False), ("ccsd", 2, False),
                                ("ccsd", 2, True), ("mp2", 2, False)):
        name = f"lucj_{init}_{'truncated' if trunc else layers}_vqe"
        rows.append(dict(name=name, init=init, layers=layers, objective="vqe", truncated=trunc))
    for seed in (101, 211, 307):
        for objective in ("fixed", "vqe"):
            rows.append(dict(name=f"lucj_random_2_{objective}_s{seed}", init="random", layers=2,
                             objective=objective, seed=seed))
    rows.append(dict(name="lucj_ccsd_2_qsci", init="ccsd", layers=2, objective="qsci"))
    return rows


def mp2_initialization(case, ref, root):
    """New MP2 in the exact source MO gauge; never writes source reference files."""
    directory = Path(root) / "initializations" / case.id
    meta_path = directory / "mp2.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["problem_id"] != ref["meta"]["request_digest"] or file_digest(directory / "mp2.npz") != meta["sha256"]:
            raise RuntimeError("MP2 cache mismatch")
        with np.load(directory / "mp2.npz") as z:
            return z["t1"], z["t2"], {**meta, "cache_reused": True}
    started = perf_counter()
    mol = gto.M(atom=case.geometry(), basis=case.basis, unit="Angstrom", spin=case.spin, charge=case.charge, verbose=0)
    mf = scf.RHF(mol)
    mo = ref["mo_coeff"]
    occ = np.zeros(mo.shape[1]); occ[:mol.nelectron//2] = 2
    density = mf.make_rdm1(mo, occ)
    fock = mf.get_fock(dm=density)
    gauge_fock = mo.T @ fock @ mo
    offdiag = gauge_fock - np.diag(np.diag(gauge_fock))
    if np.max(abs(offdiag)) > 2e-5:
        raise RuntimeError("Source MOs are not canonical for reconstructed RHF")
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = mo, occ, np.diag(gauge_fock)
    mf.e_tot = ref["meta"]["rhf_energy"]
    mf.converged = True
    solver = mp.MP2(mf, frozen=case.frozen_core)
    solver.kernel()
    t2 = solver.t2
    if not np.all(np.isfinite(t2)):
        raise RuntimeError("Non-finite MP2 amplitudes")
    t1 = np.zeros((t2.shape[0], t2.shape[2]))
    save_npz(directory / "mp2.npz", t1=t1, t2=t2)
    meta = {"problem_id": ref["meta"]["request_digest"], "sha256": file_digest(directory / "mp2.npz"),
            "seconds": perf_counter()-started, "energy": float(solver.e_tot), "fci_used": False,
            "frozen_core": case.frozen_core, "mo_gauge_max_offdiag": float(abs(offdiag).max())}
    save_json(meta_path, meta)
    return t1, t2, {**meta, "cache_reused": False}


def initial_parameters(case, ref, spec, source, root):
    m = ref["meta"]; norb = m["n_orbitals"]
    pairs = lucj_pair_lists(norb, spec.get("truncated", False))
    started = perf_counter()
    if spec["init"] == "random":
        size = ffsim.UCJOpSpinBalanced.n_params(norb, spec["layers"], interaction_pairs=pairs, with_final_orbital_rotation=True)
        rng, seed = rng_for("strong-LUCJ-random-v1", case.id, spec["seed"])
        params = rng.uniform(-np.pi, np.pi, size)
        return params, pairs, {"kind": "random", "seed": str(seed), "seconds": perf_counter()-started,
                               "dependency_total_seconds": 0., "dependency_new_seconds": 0.}
    if spec["init"] == "ccsd":
        d = Path(source) / "references" / case.id
        if not (d / "ccsd.json").exists():
            raise RuntimeError("SOURCE_CCSD_INITIALIZATION_UNAVAILABLE; original failure preserved")
        meta = json.loads((d / "ccsd.json").read_text())
        if meta["problem"] != m["request_digest"] or file_digest(d / "ccsd.npz") != meta["sha256"]:
            raise RuntimeError("Source CCSD binding mismatch")
        with np.load(d / "ccsd.npz") as z:
            t1, t2 = z["t1"], z["t2"]
        meta = {**meta, "cache_reused": True}
    else:
        t1, t2, meta = mp2_initialization(case, ref, root)
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=spec["layers"],
             interaction_pairs=pairs, optimize=spec.get("compressed", False), options={"maxiter": 50})
    # Explicit final identity rotation if an implementation omitted zero t1.
    if op.final_orbital_rotation is None:
        op = ffsim.UCJOpSpinBalanced(op.diag_coulomb_mats, op.orbital_rotations, np.eye(norb))
    params = op.to_parameters(interaction_pairs=pairs)
    return params, pairs, {"kind": spec["init"], "amplitudes_sha256": meta["sha256"],
                           "dependency_total_seconds": meta["seconds"],
                           "dependency_new_seconds": 0. if meta["cache_reused"] else meta["seconds"],
                           "seconds": perf_counter()-started,
                           "factorization_optimized": spec.get("compressed", False)}


class Objective:
    """Durable actual-call ledger; finite-difference probes count as evaluations."""
    def __init__(self, ref, spec, pairs, directory, config):
        self.meta, self.spec, self.pairs, self.config = ref["meta"], spec, pairs, config
        self.norb = self.meta["n_orbitals"]
        self.nelec = (self.meta["nalpha"], self.meta["nbeta"])
        self.ids, self.signs, _, _ = determinant_ids(self.norb, *self.nelec)
        self.h = ref["h"][self.ids][:, self.ids].multiply(self.signs[:, None]*self.signs[None, :]).tocsr()
        self.directory = Path(directory)
        self.db = sqlite3.connect(self.directory / "objective.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (i INTEGER PRIMARY KEY, parameter_sha TEXT, loss REAL, seconds REAL, shots INTEGER, counts BLOB)")
        if self.db.execute("SELECT count(*) FROM calls").fetchone()[0]:
            raise RuntimeError("Partial cell has durable work; requires explicit successor restart, not silent free retry")
        self.cache_db = open_db(self.directory / "pilot-ci.sqlite")
        self.ritz = RitzCache(ref["h"], self.ids, self.meta["request_digest"], self.cache_db)
        self.start = perf_counter(); self.calls = 0; self.gradients = 0; self.iterations = 0
        self.best_loss = float("inf"); self.best_x = None
        self.gradient_records = {}

    def op(self, x):
        return ffsim.UCJOpSpinBalanced.from_parameters(x, norb=self.norb, n_reps=self.spec["layers"],
                   interaction_pairs=self.pairs, with_final_orbital_rotation=True)

    def state(self, x):
        return ffsim.apply_unitary(ffsim.hartree_fock_state(self.norb, self.nelec), self.op(x), norb=self.norb, nelec=self.nelec)

    def __call__(self, x):
        if self.calls >= self.config["max_evals"]:
            raise BudgetReached("ACTUAL_OBJECTIVE_EVALUATION_CAP")
        if perf_counter()-self.start > self.config["cell_seconds"] or rss_peak() > self.config["worker_ram_bytes"]:
            raise BudgetReached("WALL_OR_MEMORY_CAP")
        started = perf_counter(); vec = self.state(x); counts_blob = None; shots = 0
        if self.spec["objective"] == "qsci":
            shots = self.config["pilot_shots"]
            rng, _ = rng_for("strong-LUCJ-QSCI-pilot-v1", self.meta["request_digest"], self.spec["name"], self.calls)
            counts = rng.multinomial(shots, abs(vec)**2 / np.linalg.norm(vec)**2)
            valid = counts > 0
            selected = ranks(self.ids[valid], counts[valid], self.ids, self.meta["hf_id"], [self.spec["name"], self.calls])["frequency"][:32]
            _, loss, _, _ = self.ritz.solve(selected)
            from ..study.core import pack_pairs
            counts_blob = pack_pairs(self.ids[valid], counts[valid])
            self.cache_db.commit()
        else:
            loss = float(np.vdot(vec, self.h @ vec).real)
        if not np.isfinite(loss):
            raise RuntimeError("Non-finite objective")
        key = digest(np.asarray(x).tolist())
        self.db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?)", (self.calls, key, loss, perf_counter()-started, shots, counts_blob))
        self.db.commit(); self.calls += 1
        if loss < self.best_loss:
            self.best_loss, self.best_x = loss, np.array(x, copy=True)
            save_npz(self.directory / "best-observed.npz", parameters=x, loss=np.array(loss))
        return loss

    def gradient(self, x):
        eps = 1e-6; grad = np.zeros(len(x))
        for i in range(len(x)):
            y = x.copy(); y[i] += eps; fp = self(y)
            y[i] -= 2*eps; fm = self(y)
            grad[i] = (fp-fm)/(2*eps)
        self.gradients += 1
        self.gradient_records[digest(np.asarray(x).tolist())] = float(abs(grad).max())
        return grad


def run_cell(case, ref, spec, source, root, config):
    directory = Path(root) / "cells" / case.id / spec["name"]
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    identity = digest([ref["meta"]["request_digest"], spec, config])
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old["task_id"] != identity:
            raise RuntimeError("Frozen task changed")
        return old
    start = perf_counter()
    result = {"task_id": identity, "case_id": case.id, "method": spec["name"], "spec": spec,
              "problem_id": ref["meta"]["request_digest"], "fci_used_for_selection": False, "hardware_shots": 0}
    objective = None
    try:
        x0, pairs, init = initial_parameters(case, ref, spec, source, root)
        save_npz(directory / "initial.npz", parameters=x0)
        objective = Objective(ref, spec, pairs, directory, config)
        objective(x0)
        success, message = None, "FIXED_INITIALIZATION"
        optimizer_iterations = 0
        try:
            if spec["objective"] == "vqe":
                def callback(_):
                    objective.iterations += 1
                opt = minimize(objective, x0, jac=objective.gradient, method="L-BFGS-B", callback=callback,
                               options={"maxiter": config["maxiter"], "maxfun": config["max_evals"], "gtol": 1e-6, "ftol": 1e-12})
                success, message = bool(opt.success), str(opt.message)
                optimizer_iterations = int(opt.nit)
            elif spec["objective"] == "qsci":
                rng, _ = rng_for("strong-LUCJ-QSCI-proposal-v1", case.id)
                for _ in range(config["qsci_proposals"]):
                    trial = objective.best_x.copy()
                    trial[int(rng.integers(len(trial)))] += rng.normal(0, .1)
                    objective(trial)
                    objective.iterations += 1
                message = "REGISTERED_QSCI_PROPOSAL_CAP_NOT_CONVERGENCE"
                optimizer_iterations = objective.iterations
        except BudgetReached as exc:
            success, message = False, str(exc)
            optimizer_iterations = objective.iterations
        x = objective.best_x
        vec = objective.state(x)
        state = np.zeros(1 << ref["meta"]["n_qubits"], complex)
        state[objective.ids] = vec*objective.signs
        preparation, sampler, removed = embedded_lucj_circuits(objective.op(x), objective.norb, objective.nelec)
        parity = verify_circuit_state(preparation, state)
        sampling = Statevector.from_instruction(sampler).data
        if not np.allclose(abs(state)**2, abs(sampling)**2, atol=1e-9):
            raise RuntimeError("LUCJ sampler changed Z probabilities")
        energy = float(np.vdot(state, ref["h"] @ state).real)
        save_npz(directory / "final.npz", parameters=x, state=state, probability=abs(state)**2)
        save_binary(directory / "preparation.qpy", circuit_bytes(preparation))
        save_binary(directory / "sampling.qpy", circuit_bytes(sampler))
        gradient = objective.gradient_records.get(digest(x.tolist()))
        status = "FIXED" if spec["objective"] == "fixed" else "OPTIMIZER_TERMINATED" if success else "BOUNDED_SEARCH_TERMINATED"
        result.update(status=status, initialization=init, optimizer_success=success, optimizer_message=message,
                      optimizer_iterations=optimizer_iterations, objective_evaluations=objective.calls,
                      complete_gradient_evaluations=objective.gradients, gradient_inf_at_returned_point=gradient,
                      stationary=None if gradient is None else gradient <= 1e-6,
                      point_selection="best actually evaluated objective; includes finite-difference probes",
                      objective_min_observed=objective.best_loss,
                      qsci_pilot_selection_bias=spec["objective"] == "qsci",
                      vqe_energy=energy, energy_error_mHa=(energy-ref["meta"]["fci_energy"])*1000,
                      fci_scope="post-construction evaluation only", simulation="exact fixed Nalpha/Nbeta sector for small-system baseline; not scaling route",
                      state_parity_error=parity, resources=resources(sampler, len(x), spec["layers"]),
                      preparation_resources=resources(preparation, len(x), spec["layers"]),
                      removed_terminal_diagonal_gates=removed,
                      sampler_scope="Z-distribution equivalent; VQE energy belongs to preparation",
                      simulated_construction_shots=int(objective.db.execute("SELECT coalesce(sum(shots),0) FROM calls").fetchone()[0]),
                      vqe_objective_qpu_shots=None, vqe_objective_qpu_shots_reason="exact classical objective calls are not a QPU measurement budget",
                      final_sha256=file_digest(directory / "final.npz"), peak_process_rss_bytes=rss_peak())
    except Exception as exc:
        import traceback
        result.update(status="FAILED_PRESERVED", error=repr(exc), traceback=traceback.format_exc(),
                      objective_evaluations=objective.calls if objective else 0)
        if (directory / "objective.sqlite").exists():
            with sqlite3.connect(directory / "objective.sqlite") as prior:
                result["durable_objective_evaluations_including_prior_prefix"] = prior.execute("SELECT count(*) FROM calls").fetchone()[0]
                result["durable_pilot_shots_including_prior_prefix"] = int(prior.execute("SELECT coalesce(sum(shots),0) FROM calls").fetchone()[0])
    finally:
        if objective:
            objective.db.close(); objective.cache_db.close()
    result["construction_wall_seconds"] = perf_counter()-start
    save_json(result_path, result)
    return result
