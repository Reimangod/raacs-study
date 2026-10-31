"""Bounded P2-inspired TN surrogate control, never exact-paper reproduction.

Keep the paper's MPS(chi50, cutoff1e-10), 10k surrogate samples, and PSD-MADS.
Use the registered common CI readout (r32) instead of the unavailable original
Dice objective. Final exact-state readouts are independent and run elsewhere.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
from time import perf_counter,process_time

import ffsim
import numpy as np
import PyNomad
from qiskit import QuantumCircuit
import quimb.tensor as qtn
from qiskit_quimb import quimb_circuit
from paperstudy.ci import solve


def save(path,data):
    p=path.with_suffix(".pending");p.write_text(json.dumps(data,indent=2)+"\n");p.replace(path)


def run(config_path):
    config=json.loads(Path(config_path).read_text());root=Path(config_path).parent
    if (root/"optimization-result.json").exists():return
    n=config["norb"];ne=tuple(config["nelec"]);pairs=([tuple(p) for p in config["pairs"][0]],[tuple(p) for p in config["pairs"][1]])
    with np.load(root/"integrals.npz",allow_pickle=False) as z:h,g=z["h1"].copy(),z["eri"].copy()
    with np.load(root/"initial-operator.npz",allow_pickle=False) as z:
        initial=ffsim.UCJOpSpinBalanced(z["diag_coulomb_mats"],z["orbital_rotations"],z["final_orbital_rotation"])
    for name,expected in config["input_sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError("TN physical/initial input changed")
    theta=initial.to_parameters(interaction_pairs=pairs)
    start,cpu=perf_counter(),process_time();lock=threading.Lock();records=[];best=[None,None];rejected=[0]
    directory=root/"surrogate-evaluations";directory.mkdir(exist_ok=True)

    def objective(point):
        x=np.array([point.get_coord(i) for i in range(point.size())])
        with lock:
            if len(records)>=500 or perf_counter()-start>3000:
                rejected[0]+=1
                return 0
            index=len(records);records.append({"index":index,"status":"RUNNING"})
        tick,cc=perf_counter(),process_time()
        try:
            op=ffsim.UCJOpSpinBalanced.from_parameters(x,norb=n,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
            q=QuantumCircuit(2*n);q.append(ffsim.qiskit.PrepareHartreeFockJW(n,ne),range(2*n));q.append(ffsim.qiskit.UCJOpSpinBalancedJW(op),range(2*n))
            circuit=quimb_circuit(q.decompose(reps=2),quimb_circuit_class=qtn.CircuitMPS,max_bond=50,cutoff=1e-10,progbar=False)
            observed=list(circuit.sample(10000,seed=config["surrogate_seed"]))
            ids,counts=np.unique(np.array([int(s[::-1],2) for s in observed],np.uint64),return_counts=True)
            count_file=directory/(str(index)+"-counts.npz")
            with count_file.open("wb") as f:np.savez_compressed(f,blocked_ids=ids,counts=counts.astype(np.uint64))
            ci=solve(h,g,config["core_energy"],ne,ids,counts.astype(np.uint64),32,seconds=120)
            value=ci["energy_Ha"]
            record={"index":index,"status":"COMPLETED","theta_sha256":hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
                "surrogate_samples":10000,"actual_QPU_shots":0,"CI":ci,"counts_sha256":hashlib.sha256(count_file.read_bytes()).hexdigest(),
                "wall_seconds":perf_counter()-tick,"process_CPU_delta_concurrent_not_additive":process_time()-cc}
            with lock:
                records[index]=record
                if best[0] is None or value<best[0]:best[:]=[value,x.copy()]
                save(directory/(str(index)+".json"),record)
                save(root/"progress.json",{"completed":sum(r["status"]=="COMPLETED" for r in records),"started":len(records),
                    "best_surrogate_energy_Ha":best[0],"elapsed_wall_seconds":perf_counter()-start,"CPU_seconds":process_time()-cpu,
                    "not_independent_final_result":True})
            point.setBBO(str(value).encode());return 1
        except Exception as exc:
            with lock:
                records[index]={"index":index,"status":"FAILED_PRESERVED","error":repr(exc),"wall_seconds":perf_counter()-tick}
                save(directory/(str(index)+".json"),records[index])
            return 0

    options=["BB_OUTPUT_TYPE OBJ","MAX_BB_EVAL 500","DISPLAY_DEGREE 0","PSD_MADS_OPTIMIZATION true",
        "PSD_MADS_NB_VAR_IN_SUBPROBLEM 20","PSD_MADS_SUBPROBLEM_MAX_BB_EVAL 20","PSD_MADS_NB_SUBPROBLEM 3",
        "NB_THREADS_PARALLEL_EVAL 4","SEED "+str(config["optimizer_seed"]),"CACHE_FILE "+str(root/"nomad-cache.txt")]
    result=PyNomad.optimize(objective,theta.tolist(),[],[],options)
    if best[1] is None:raise RuntimeError("No valid surrogate evaluations; do not substitute unoptimized state")
    op=ffsim.UCJOpSpinBalanced.from_parameters(best[1],norb=n,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
    with (root/"optimized-operator.npz").open("wb") as f:np.savez_compressed(f,diag_coulomb_mats=op.diag_coulomb_mats,orbital_rotations=op.orbital_rotations,final_orbital_rotation=op.final_orbital_rotation)
    save(root/"optimization-result.json",{"status":"TN_COMMON_READOUT_SURROGATE_RETURNED","problem_id":config["problem_id"],
        "configuration":config,"options":options,"optimizer_result":result,"objective_records":records,
        "primitive_boundary_rejections":rejected[0],"best_surrogate_energy_Ha":best[0],
        "source_initial_coefficients":theta.tolist(),"selected_coefficients":best[1].tolist(),
        "surrogate_samples":10000*sum(r["status"]=="COMPLETED" for r in records),
        "attempted_surrogate_samples_in_failed_calls":None,"failed_call_samples_unknown_not_zero":True,
        "actual_QPU_shots":0,"independent_final_energy_evaluated":False,"model_QPU_shots_for_classical_surrogate":0,
        "final_exact_sampling_still_required":True,"MPS_approximation_used_only_during_training":True,
        "paper_native_Dice_objective_reproduced":False,"registered_variant":"P2-inspired PSD-MADS/MPS with common r32 objective",
        "CPU_seconds":process_time()-cpu,"wall_seconds":perf_counter()-start,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "operator_sha256":hashlib.sha256((root/"optimized-operator.npz").read_bytes()).hexdigest()})


if __name__=="__main__":
    import sys;run(sys.argv[1])
