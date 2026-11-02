"""Official ffsim LUCJ, fixed CCSD initialization and explicitly mapped JW basis."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from pyscf import gto,scf,cc,lib
import ffsim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .common import save_json,save_npz,save_binary,digest,file_digest,rng_for
from .chemistry import determinant_ids
from .ansatz import (compile_logical,measurement_equivalent_circuit,resources,
                     verify_circuit_state,circuit_bytes)
from .metrics import analytic_metrics

LUCJ_METHODS=("lucj_1","lucj_2","lucj_truncated","lucj_random_101")


def ccsd_amplitudes(case,reference,root):
    directory=Path(root)/"references"/case.id
    path=directory/"ccsd.npz"
    manifest=directory/"ccsd.json"
    failed=directory/"ccsd-nonconvergence-v1.json"
    settings={"problem":reference["meta"]["request_digest"],"mo_digest":digest(reference["mo_coeff"].tolist()),
              "frozen_core":case.frozen_core,"conv_tol":1e-9,"max_cycle":150}
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta["problem"]!=reference["meta"]["request_digest"] or file_digest(path)!=meta["sha256"]:
            raise RuntimeError("CCSD cache mismatch")
        with np.load(path) as z:
            return z["t1"],z["t2"],{**meta,"cache_reused":True}
    if failed.exists():
        record=json.loads(failed.read_text())
        if record["settings"]!=settings:raise RuntimeError("CCSD failed-cache settings changed")
        raise RuntimeError("Shared CCSD initialization did not converge; reused failure evidence "+str(failed))
    start=perf_counter()
    mol=gto.M(atom=case.geometry(),basis=case.basis,unit="Angstrom",spin=0,verbose=0)
    mf=scf.RHF(mol);mf.conv_tol=1e-11;mf.max_cycle=150;mf.kernel()
    if not mf.converged:
        mf=mf.newton();mf.conv_tol=1e-11;mf.kernel()
    if not mf.converged:
        raise RuntimeError("CCSD initialization: RHF failed")
    # Fix precisely the same MO gauge as the reference, not a newly chosen orbital basis.
    mf.mo_coeff=reference["mo_coeff"]
    solver=cc.CCSD(mf,frozen=case.frozen_core)
    solver.conv_tol=1e-9;solver.max_cycle=150
    solver.kernel()
    if not solver.converged:
        save_json(failed,{"status":"NONCONVERGED_NOT_FOR_ANSATZ_INITIALIZATION","settings":settings,
                         "seconds":perf_counter()-start,"energy":float(solver.e_tot) if np.isfinite(solver.e_tot) else None,
                         "cost_scope":"one shared classical CCSD attempt; not repeated for each LUCJ variant"})
        raise RuntimeError("CCSD initialization did not converge")
    save_npz(path,t1=solver.t1,t2=solver.t2)
    meta={"problem":reference["meta"]["request_digest"],"sha256":file_digest(path),
          "energy":float(solver.e_tot),"converged":True,"seconds":perf_counter()-start,
          "fci_used_for_initialization":False,"frozen_core":case.frozen_core}
    save_json(manifest,meta)
    return solver.t1,solver.t2,{**meta,"cache_reused":False}


def lucj_pair_lists(norb,truncated=False):
    # Linear same-spin nearest neighbors; all on-site alpha-beta interactions.
    aa=[(i,i+1) for i in range(norb-1)]
    ab=[(i,i) for i in range(norb)]
    if truncated:
        aa=aa[::2]
    return aa,ab


def embedded_lucj_circuits(op,norb,nelec):
    blocked=QuantumCircuit(2*norb)
    blocked.append(ffsim.qiskit.PrepareHartreeFockJW(norb,nelec),range(2*norb))
    blocked.append(ffsim.qiskit.UCJOpSpinBalancedJW(op),range(2*norb))
    # Qubit permutation alone is not fermionic reordering. Add the determinant signs.
    wires=[2*norb-1-2*i for i in range(norb)]+[2*norb-2-2*i for i in range(norb)]
    preparation=QuantumCircuit(2*norb)
    preparation.compose(blocked,qubits=wires,inplace=True)
    for a in range(norb):
        for b in range(a):
            preparation.cz(2*norb-1-2*a,2*norb-2-2*b)
    # Those terminal phase corrections change VQE energy but not Z counts. Remove
    # them explicitly for sampling, not silently from the preparation's resource count.
    sampler,removed=measurement_equivalent_circuit(preparation)
    return compile_logical(preparation),sampler,removed


def run_lucj(case,reference,method,root):
    if method not in LUCJ_METHODS:
        raise ValueError(method)
    directory=Path(root)/"checkpoints"/case.id/method
    directory.mkdir(parents=True,exist_ok=True)
    cp_path=directory/"step-0000.json"
    if cp_path.exists():
        cp=json.loads(cp_path.read_text())
        if cp["problem_id"]!=reference["meta"]["request_digest"]:
            raise RuntimeError("LUCJ problem mismatch")
        return {"status":"FIXED_CCSD_INITIALIZED" if "random" not in method else "FIXED_RANDOM", "checkpoints":1}
    start=perf_counter()
    m=reference["meta"];norb=m["n_orbitals"];nelec=(m["nalpha"],m["nbeta"])
    nreps=1 if method=="lucj_1" else 2
    pairs=lucj_pair_lists(norb,method=="lucj_truncated")
    random_seed=None
    if method=="lucj_random_101":
        # A random fixed-topology control does not require converged CCSD.
        # Same registered shape/seed as the previous implementation; existing
        # successful control vectors remain byte-identical and are not replaced.
        ccmeta=None
        size=ffsim.UCJOpSpinBalanced.n_params(norb,nreps,interaction_pairs=pairs,with_final_orbital_rotation=True)
        rng,random_seed=rng_for("lucj-random",case.id,101)
        parameters=rng.uniform(-np.pi,np.pi,size)
        op=ffsim.UCJOpSpinBalanced.from_parameters(parameters,norb=norb,n_reps=nreps,
                interaction_pairs=pairs,with_final_orbital_rotation=True)
    else:
        t1,t2,ccmeta=ccsd_amplitudes(case,reference,root)
        op=ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2,t1=t1,n_reps=nreps,interaction_pairs=pairs)
        parameters=op.to_parameters(interaction_pairs=pairs)
    vector=ffsim.apply_unitary(ffsim.hartree_fock_state(norb,nelec),op,norb=norb,nelec=nelec)
    ids,signs,_,_=determinant_ids(norb,*nelec)
    state=np.zeros(1<<(2*norb),dtype=complex);state[ids]=vector*signs
    preparation,sampler,removed=embedded_lucj_circuits(op,norb,nelec)
    parity=verify_circuit_state(preparation,state)
    sampling_state=Statevector.from_instruction(sampler).data
    probability=np.abs(state)**2
    if not np.allclose(probability,np.abs(sampling_state)**2,atol=1e-9):
        raise RuntimeError("Sampling-equivalent compiler changed probabilities")
    dist_path=Path(root)/"distributions"/case.id/method/"step-0000.npz"
    save_npz(dist_path,state=state,sampling_state=sampling_state,probability=probability)
    prep_path=directory/"preparation.qpy";sample_path=directory/"sampling.qpy"
    save_binary(prep_path,circuit_bytes(preparation));save_binary(sample_path,circuit_bytes(sampler))
    energy=float(np.vdot(state,reference["h"]@state).real)
    config={"schema":"ffsim-lucj-1","method":method,"problem":m["request_digest"],
            "ccsd_sha":ccmeta["sha256"] if ccmeta else None,"n_reps":nreps,"interaction_pairs":pairs,
            "parameters":parameters.tolist(),"random_seed":random_seed}
    cp={"schema":"checkpoint-1","config_digest":digest(config),"config":config,
        "case_id":case.id,"problem_id":m["request_digest"],"method":method,"step":0,
        "selected_operators":[],"optimized_parameters":parameters.tolist(),
        "parameter_origin":"CCSD_FACTORIZATION_FIXED" if random_seed is None else "REGISTERED_RANDOM_CONTROL",
        "vqe_energy":energy,"vqe_energy_error":energy-m["fci_energy"],
        "resources":resources(sampler,len(parameters),nreps),
        "preparation_resources":resources(preparation,len(parameters),nreps),
        "sampling_state_energy":float(np.vdot(sampling_state,reference["h"]@sampling_state).real),
        "sampling_equivalence":"exact computational-basis distribution; not VQE-state identity",
        "removed_terminal_diagonal_gates":removed,"qiskit_state_error":parity,
        "optimization_seconds":0.,"objective_evaluations":0,"gradient_evaluations":0,
        "ccsd_initialization_seconds":ccmeta["seconds"] if ccmeta else 0.,
        "ccsd_initialization_cache_reused":ccmeta["cache_reused"] if ccmeta else None,
        "ccsd_new_classical_work_seconds":(0. if ccmeta["cache_reused"] else ccmeta["seconds"]) if ccmeta else 0.,
        "ccsd_initialization_required":ccmeta is not None,"optimizer_success":None,
        "optimizer_message":"FIXED_INITIALIZATION_NO_VQE_TRAINING",
        "distribution":str(dist_path.resolve()),"distribution_sha256":file_digest(dist_path),
        "circuit_qpy":str(prep_path.resolve()),"circuit_sha256":file_digest(prep_path),
        "sampling_circuit_qpy":str(sample_path.resolve()),"sampling_circuit_sha256":file_digest(sample_path),
        "trajectory_seconds":perf_counter()-start,"analytic_metrics":analytic_metrics(probability,reference)}
    cp["checkpoint_id"]=digest(config)
    save_json(cp_path,cp)
    status={"status":"FIXED_CCSD_INITIALIZED" if random_seed is None else "FIXED_RANDOM","checkpoints":1,"config":config}
    save_json(directory/"status.json",status)
    print(f"{case.id} {method}: E={energy:.10f}, sampler CNOT={cp['resources']['cnot_count']}",flush=True)
    return status
