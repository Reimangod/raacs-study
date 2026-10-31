"""Official configuration recovery with an explicit-determinant qubit-H solver.

This is NOT the alpha×beta Cartesian expansion used by full fermionic SQD.
Every iteration reuses the original raw counts, updating occupancies only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import eigh
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from .chemistry import load_reference
from .common import digest,rng_for,save_json,file_digest
from .qsci import CHEMICAL_ACCURACY_HA,select_configurations,solve_selected


def ids_to_blocked(ids,norb):
    ids=np.asarray(ids,dtype=np.int64)
    beta=[((ids >> (2*norb-2-2*j))&1).astype(bool) for j in reversed(range(norb))]
    alpha=[((ids >> (2*norb-1-2*j))&1).astype(bool) for j in reversed(range(norb))]
    return np.stack(beta+alpha,axis=1)


def blocked_to_ids(bits):
    bits=np.asarray(bits,dtype=bool);norb=bits.shape[1]//2
    result=np.zeros(len(bits),dtype=np.int64)
    for j in range(norb):
        result|=bits[:,2*norb-1-j].astype(np.int64) << (2*norb-1-2*j)
        result|=bits[:,norb-1-j].astype(np.int64) << (2*norb-2-2*j)
    return result


def ci_occupancies(h,selected,norb):
    hs=h.tocsr()[selected][:,selected].toarray()
    eig,vec=eigh(hs,subset_by_index=(0,0))
    weight=np.abs(vec[:,0])**2
    alpha=np.array([np.dot(weight,(selected >> (2*norb-1-2*j))&1) for j in range(norb)])
    beta=np.array([np.dot(weight,(selected >> (2*norb-2-2*j))&1) for j in range(norb)])
    return float(eig[0]),(alpha,beta),float(np.linalg.norm(hs@vec[:,0]-eig[0]*vec[:,0]))


def recover_readout(reference,ids,counts,size,seed,iterations=3):
    m=reference["meta"];norb=m["n_orbitals"]
    valid=set(map(int,reference["ids"]))
    raw_matrix=ids_to_blocked(ids,norb)
    probabilities=np.array(counts,dtype=float)/np.sum(counts)
    occupancies=(np.r_[np.ones(m["nalpha"]),np.zeros(norb-m["nalpha"])],
                 np.r_[np.ones(m["nbeta"]),np.zeros(norb-m["nbeta"])])
    records=[]
    for iteration in range(iterations):
        start=perf_counter()
        corrected,weights=recover_configurations(raw_matrix,probabilities,occupancies,
                            m["nalpha"],m["nbeta"],rand_seed=np.random.default_rng(seed+iteration))
        recovered_ids=blocked_to_ids(corrected)
        # Returned weights are real normalized masses, NOT fabricated integer counts.
        ranked=sorted(zip(recovered_ids,weights),key=lambda x:(-float(x[1]),int(x[0])))
        chosen=[m["hf_id"]]
        for state_id,_ in ranked:
            if int(state_id) in valid and int(state_id) not in chosen and len(chosen)<size:
                chosen.append(int(state_id))
        selected=np.array(sorted(chosen),dtype=np.int64)
        energy,occupancies,residual=ci_occupancies(reference["h"],selected,norb)
        if residual>1e-8 or energy<m["fci_energy"]-1e-8:
            raise RuntimeError("SQD projection/sector verification failed")
        records.append({"iteration":iteration+1,"energy":energy,
                        "energy_error_mHa":1000*(energy-m["fci_energy"]),
                        "dimension":len(selected),"selected_ids":selected.tolist(),
                        "valid_recovered_mass":float(sum(w for i,w in zip(recovered_ids,weights) if int(i) in valid)),
                        "recovered_unique":len(recovered_ids),"recovery_and_solver_seconds":perf_counter()-start,
                        "occupancies":[occupancies[0].tolist(),occupancies[1].tolist()],
                        "residual":residual,"new_quantum_shots":0})
    return records


def run_sqd(root,iterations=3,shot_values=(1000,10000,100000)):
    root=Path(root);summaries=[]
    for case_dir in sorted((root/"checkpoints").iterdir()):
        if not case_dir.is_dir():
            continue
        reference=load_reference(root/"references"/case_dir.name)
        for method_dir in sorted(case_dir.iterdir()):
            status=method_dir/"status.json"
            if not status.exists() or json.loads(status.read_text()).get("status")=="RUNNING":continue
            cps=sorted(method_dir.glob("step-*.json"))
            if not cps:
                continue
            # Prospective, outcome-independent rule: final saved checkpoint of each method.
            cp=json.loads(cps[-1].read_text())
            raw_path=root/"qsci"/case_dir.name/method_dir.name/cps[-1].name
            if not raw_path.exists():
                continue
            context={"schema":"official-recovery-qubit-1","checkpoint_id":cp["checkpoint_id"],
                     "source_qsci_sha256":file_digest(raw_path),"iterations":iterations,
                     "shot_values":list(shot_values),"primary_iteration":"last-not-best",
                     "selection":"same HF/frequency/R as QSCI; no alpha-beta Cartesian completion",
                     "occupancy_initialization":"same HF for all arms; no FCI information"}
            output=root/"sqd"/case_dir.name/f"{method_dir.name}.json"
            if output.exists():
                saved=json.loads(output.read_text())
                if saved["context_id"]!=digest(context):
                    raise RuntimeError("SQD settings changed; preserve old results and use new output")
                summaries.extend(saved["rows"])
                continue
            rows=[]
            for raw in json.loads(raw_path.read_text())["rows"]:
                if not raw["include_hf"] or raw["shots"] not in shot_values:
                    continue
                _,seed=rng_for("official-sqd-recovery-v1",raw["raw_evaluation_id"])
                # numpy's Generator seed+iteration is valid up to arbitrary Python int.
                history=recover_readout(reference,raw["raw_count_ids"],raw["raw_counts"],raw["R_cap"],seed,iterations)
                final=history[-1]
                rows.append({"case_id":cp["case_id"],"method":cp["method"],"step":cp["step"],
                             "checkpoint_id":cp["checkpoint_id"],"shots":raw["shots"],"replicate":raw["replicate"],
                             "source_raw_evaluation_id":raw["raw_evaluation_id"],"R_cap":raw["R_cap"],
                             "energy_without_recovery":raw["energy"],"energy_with_recovery":final["energy"],
                             "energy_error_without_recovery_mHa":raw["energy_error_mHa"],
                             "energy_error_with_recovery_mHa":final["energy_error_mHa"],
                             "dimension_before":raw["dimension"],"dimension_after":final["dimension"],
                             "valid_raw_rate":raw["valid_shots"]/raw["shots"],
                             "valid_recovered_mass":final["valid_recovered_mass"],
                             "recovery_seconds":sum(x["recovery_and_solver_seconds"] for x in history),
                             "iterations":history,"new_quantum_shots":0,
                             "workflow":"official SQD recovery + explicit determinant qubit-H diagonalization",
                             "status":"COMPLETED","seed":seed})
            save_json(output,{"context":context,"context_id":digest(context),"rows":rows})
            summaries.extend(rows)
    save_json(root/"sqd"/"summary.json",summaries)
    return len(summaries)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="results/quick")
    p.add_argument("--iterations",type=int,default=3)
    args=p.parse_args();print("SQD readouts:",run_sqd(args.output,args.iterations))


if __name__=="__main__":
    main()
