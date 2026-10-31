"""Independent author/Qiskit reconstruction of CUDA output, never a blind file copy."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from qiskit.quantum_info import Statevector
from .ansatz import (make_pool,pool_catalog,hf_state,state_from_parameters,build_circuit,
                     verify_circuit_state,measurement_equivalent_circuit,resources,circuit_bytes)
from .common import digest,file_digest,save_binary,save_json,save_npz,provenance,resolve_artifact
from .metrics import analytic_metrics


def import_execution(reference,bundle,execution,output):
    start=perf_counter();bundle=Path(bundle);execution=Path(execution);output=Path(output)
    request=json.loads((bundle/"request.json").read_text());terminal=json.loads((execution/"terminal.json").read_text())
    if terminal["request_digest"]!=digest(request) or terminal["backend"]!="cupy":raise RuntimeError("Execution/request mismatch")
    if "A100" not in terminal["device"] or not terminal["slurm_job"]:raise RuntimeError("Missing actual A100 evidence")
    m=reference["meta"];method=request["method"];case=request["case_id"];cfg=request["config"]
    if request["problem_id"]!=m["request_digest"] or request["hf_id"]!=m["hf_id"]:raise RuntimeError("Problem mismatch")
    for name,sha in request["files"].items():
        if file_digest(bundle/name)!=sha:raise RuntimeError("Bundle changed")
    if file_digest(reference["directory"]/"hamiltonian.npz")!=request["files"]["hamiltonian.npz"]:raise RuntimeError("Hamiltonian changed")
    pool=make_pool(method,m["n_qubits"]);catalog=pool_catalog(pool)
    if catalog["catalog_digest"]!=json.loads((bundle/"pool.json").read_text())["catalog_digest"]:raise RuntimeError("Pool drift")
    pool_setup_seconds=perf_counter()-start
    config={"schema":"adapt-2-cuda-execution-1","method":method,"problem":m["request_digest"],
            "catalog_digest":catalog["catalog_digest"],**cfg,
            "optimizer":"BFGS;full-ansatz;analytic-adjoint-gradient;cold-identity-Hessian",
            "selection":"maximum absolute current gradient; smallest index tie-break",
            "no_fci_for_training_or_stopping":True,"execution_backend":"CuPy complex128 exact two-level Pauli generators"}
    config_id=digest(config);root=output/"checkpoints"/case/method;root.mkdir(parents=True,exist_ok=True)
    save_json(root/"pool.json",catalog);initial=hf_state(m["n_qubits"],m["hf_id"])
    imported=0
    for path in sorted(execution.glob("step-*.json")):
        raw=json.loads(path.read_text());step=raw["step"];cp_path=root/path.name
        if cp_path.exists():
            prior=json.loads(cp_path.read_text())
            if prior.get("source_execution_id")!=terminal["execution_id"]:raise RuntimeError("Do not overwrite another trajectory")
            if (prior.get("source_raw_checkpoint_sha256")!=file_digest(path) or
                prior["selected_operators"]!=raw["selected_operators"] or
                prior["optimized_parameters"]!=raw["optimized_parameters"]):raise RuntimeError("Imported checkpoint changed")
            for key in ("distribution","circuit_qpy","sampling_circuit_qpy"):
                sha=key+"_sha256" if key=="distribution" else key.replace("_qpy","_sha256")
                if file_digest(resolve_artifact(output,prior[key]))!=prior[sha]:raise RuntimeError("Imported artifact changed: "+key)
            imported+=1;continue
        dist=path.with_suffix(".npz")
        if file_digest(dist)!=raw["state_sha256"]:raise RuntimeError("GPU state hash drift")
        with np.load(dist) as z:state=z["state"]
        indices=raw["selected_operators"];theta=raw["optimized_parameters"]
        gradients=np.array(raw["pool_gradients_before_append"])
        if len(indices)!=len(theta) or gradients.shape!=(pool.size,):raise RuntimeError("Parameter/gradient shape mismatch")
        if step:
            previous=json.loads((execution/f"step-{step-1:04d}.json").read_text())
            selected=int(np.argmax(np.abs(gradients)))
            additions=([i for i in range(pool.size) if pool.get_qubits(i)==pool.get_qubits(selected) and abs(gradients[i])>1e-10]
                       if method=="ceo_mvp" else [selected])
            if (raw["selected_this_step"]!=selected or raw["selected_group"]!=additions or
                indices!=previous["selected_operators"]+additions or
                raw["initial_parameters"]!=previous["optimized_parameters"]+[0.]*len(additions)):
                raise RuntimeError("Native selection/warm-start semantics mismatch")
        native=state_from_parameters(pool,indices,theta,initial)
        if np.linalg.norm(native-state)>1e-8:raise RuntimeError("Author/GPU state mismatch")
        energy=float(np.vdot(state,reference["h"]@state).real)
        if abs(energy-raw["energy"])>1e-9:raise RuntimeError("CPU/GPU energy mismatch")
        qc=build_circuit(pool,indices,theta,m["hf_id"]);parity=verify_circuit_state(qc,state)
        sampler,removed=measurement_equivalent_circuit(qc);sampling=Statevector.from_instruction(sampler).data
        if not np.allclose(np.abs(state)**2,np.abs(sampling)**2,atol=1e-9):raise RuntimeError("Sampler probability changed")
        dp=output/"distributions"/case/method/path.with_suffix(".npz").name
        save_npz(dp,state=state,sampling_state=sampling,probability=np.abs(state)**2)
        qpy=cp_path.with_suffix(".qpy");sqpy=cp_path.with_suffix(".sampling.qpy")
        save_binary(qpy,circuit_bytes(qc));save_binary(sqpy,circuit_bytes(sampler))
        gradients=raw["pool_gradients_before_append"]
        cp={"schema":"checkpoint-1","config_digest":config_id,"predecessor_config_digest":None,
            "case_id":case,"problem_id":m["request_digest"],"method":method,"step":step,
            "selected_operators":indices,"optimized_parameters":theta,"initial_parameters":raw["initial_parameters"],
            "selected_this_step":raw["selected_this_step"],"selected_group":raw["selected_group"],
            "pool_gradients_before_append":gradients,"pool_gradient_norm_before_append":float(np.linalg.norm(gradients)),
            "pool_gradient_components_computed":pool.size,"vqe_energy":energy,"vqe_energy_error":energy-m["fci_energy"],
            "resources":resources(sampler,len(theta),len(indices)),"qiskit_state_error":parity,
            "preparation_resources":resources(qc,len(theta),len(indices)),
            "sampling_equivalence":"exact computational-basis distribution; not VQE-state identity",
            "removed_terminal_diagonal_gates":removed,"sampling_circuit_qpy":str(sqpy.resolve()),
            "sampling_circuit_sha256":file_digest(sqpy),"distribution":str(dp.resolve()),"distribution_sha256":file_digest(dp),
            "circuit_qpy":str(qpy.resolve()),"circuit_sha256":file_digest(qpy),
            "trajectory_seconds":raw["trajectory_seconds"],"analytic_metrics":analytic_metrics(np.abs(state)**2,reference),
            "source_execution_id":terminal["execution_id"],"source_raw_checkpoint_sha256":file_digest(path),
            "execution_backend":terminal["backend"],"execution_device":terminal["device"],"slurm_job":terminal["slurm_job"],
            "timing_scope":"GPU numerical trajectory; Mac reconstruction/import separately recorded"}
        for k in ("optimization_seconds","objective_evaluations","gradient_evaluations","optimizer_iterations","optimizer_success","optimizer_message","final_parameter_gradient_norm"):
            cp[k]=raw[k]
        cp["checkpoint_id"]=digest({"config":config_id,"operators":indices,"theta":theta})
        save_json(cp_path,cp);imported+=1
    status={"status":terminal["status"],"config":config,"config_digest":config_id,"checkpoints":imported,
            "elapsed_seconds":terminal["trajectory_seconds"],"source_execution_id":terminal["execution_id"],
            "runtime":terminal,"mac_validation_and_import_seconds":perf_counter()-start,"provenance":provenance()}
    status["cpu_pool_setup_seconds"]=pool_setup_seconds
    status["cpu_post_pool_validation_and_import_seconds"]=perf_counter()-start-pool_setup_seconds
    save_json(root/"status.json",status);return status
