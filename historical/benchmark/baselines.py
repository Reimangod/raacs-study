from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from .ansatz import (make_pool,hf_state,state_from_parameters,build_circuit,
                      measurement_equivalent_circuit,verify_circuit_state,resources,circuit_bytes)
from .common import save_json,save_npz,save_binary,digest,file_digest,rng_for
from .metrics import analytic_metrics


def run_baseline(case,reference,method,root):
    directory=Path(root)/"checkpoints"/case.id/method
    directory.mkdir(parents=True,exist_ok=True)
    cp_path=directory/"step-0000.json"
    m=reference["meta"]
    pool=make_pool("ceo_ovp",m["n_qubits"])
    parent_id=None
    revision_step=0
    if method=="hf":
        indices=[];theta=np.array([]);seed=None
        if cp_path.exists():
            return json.loads((directory/"status.json").read_text())
    elif method=="ceo_random_101":
        originals=sorted((Path(root)/"checkpoints"/case.id/"ceo_ovp").glob("step-*.json"))
        if not originals:
            raise RuntimeError("Random CEO requires same-run CEO topology; do not use historical compressed circuits")
        parent=json.loads(originals[-1].read_text())
        indices=parent["selected_operators"]
        parent_id=parent["checkpoint_id"]
        if cp_path.exists():
            saved=json.loads(cp_path.read_text())
            if saved["config"]["parent_checkpoint_id"]==parent_id:
                return json.loads((directory/"status.json").read_text())
            # Preserve the previous control and add a parent-bound revision.
            revision_step=parent["step"]
            cp_path=directory/f"step-{revision_step:04d}.json"
            if cp_path.exists():
                if json.loads(cp_path.read_text())["config"]["parent_checkpoint_id"]!=parent_id:
                    raise RuntimeError("Random control parent drift at the same revision")
                return json.loads((directory/"status.json").read_text())
        rng,seed=rng_for("random-ceo-final-topology",case.id,101)
        theta=rng.uniform(-np.pi,np.pi,len(indices))
    else:
        raise ValueError(method)
    state=state_from_parameters(pool,indices,theta,hf_state(m["n_qubits"],m["hf_id"]))
    preparation=build_circuit(pool,indices,theta,m["hf_id"])
    parity=verify_circuit_state(preparation,state)
    sampler,removed=measurement_equivalent_circuit(preparation)
    probability=np.abs(state)**2
    sampling_state=Statevector(sampler).data
    if not np.allclose(probability,np.abs(sampling_state)**2,atol=1e-9):
        raise RuntimeError("Baseline measurement compiler mismatch")
    dist=Path(root)/"distributions"/case.id/method/f"step-{revision_step:04d}.npz"
    save_npz(dist,state=state,sampling_state=sampling_state,probability=probability)
    prep_path=directory/f"preparation-{revision_step:04d}.qpy";sampler_path=directory/f"sampling-{revision_step:04d}.qpy"
    save_binary(prep_path,circuit_bytes(preparation));save_binary(sampler_path,circuit_bytes(sampler))
    config={"schema":"baseline-1","problem":m["request_digest"],"method":method,
            "indices":indices,"theta":theta.tolist(),"seed":seed,"parent_checkpoint_id":parent_id}
    energy=float(np.vdot(state,reference["h"]@state).real)
    cp={"schema":"checkpoint-1","config_digest":digest(config),"config":config,
        "checkpoint_id":digest(config),"case_id":case.id,"method":method,"step":revision_step,
        "fixed_control_revision_not_adapt_growth":True,
        "problem_id":m["request_digest"],"selected_operators":indices,"optimized_parameters":theta.tolist(),
        "vqe_energy":energy,"vqe_energy_error":energy-m["fci_energy"],
        "resources":resources(sampler,len(theta),len(indices)),
        "preparation_resources":resources(preparation,len(theta),len(indices)),
        "sampling_equivalence":"exact computational-basis distribution; not VQE-state identity",
        "removed_terminal_diagonal_gates":removed,"qiskit_state_error":parity,
        "distribution":str(dist.resolve()),"distribution_sha256":file_digest(dist),
        "circuit_qpy":str(prep_path.resolve()),"circuit_sha256":file_digest(prep_path),
        "sampling_circuit_qpy":str(sampler_path.resolve()),"sampling_circuit_sha256":file_digest(sampler_path),
        "optimization_seconds":0.,"objective_evaluations":0,"optimizer_success":None,
        "analytic_metrics":analytic_metrics(probability,reference)}
    save_json(cp_path,cp)
    status={"status":"FIXED_BASELINE","checkpoints":len(list(directory.glob('step-*.json'))),"config":config}
    save_json(directory/"status.json",status)
    return status
