"""Public P3 distribution -> official CR/batching -> bounded matrix-free CI.

No additional QPU shots, no ideal sampler pretending to be hardware counts,
no integer rounding of the original floating distribution weights.
"""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from pyscf.fci import selected_ci
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_and_subsample


def save(path,value):
    tmp=path.with_suffix(".pending");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)


class BoundedCI(selected_ci.SelectedCI):
    def __init__(self,deadline):
        super().__init__();self.deadline=deadline;self.calls=0;self.verbose=0;self.max_memory=1024
    def contract_2e(self,*args,**kwargs):
        if perf_counter()>self.deadline:raise TimeoutError("H-lane CI matvec boundary")
        self.calls+=1;return super().contract_2e(*args,**kwargs)


def projected(h,g,core,ne,bits,deadline,Dcap):
    n=len(h);powers=np.array([1<<i for i in range(n-1,-1,-1)],np.uint64)
    strings=np.union1d(bits[:,:n]@powers,bits[:,n:]@powers).astype(np.int64)
    d=len(strings)**2
    if not len(strings):raise ValueError("No valid spin strings; do not invent an HF replacement")
    if d>Dcap:raise MemoryError("Actual Cartesian space exceeds registered H resource cap; do not silently trim")
    solver=BoundedCI(deadline);tick=perf_counter()
    e,v=solver.kernel_fixed_space(h,g,n,ne,(strings,strings),tol=1e-9,max_cycle=200,max_space=20,
        davidson_only=True,max_memory=1024,ecore=core,verbose=0)
    h2=solver.absorb_h1e(h,g,n,ne,.5)
    residual=float(np.linalg.norm(solver.contract_2e(h2,v,n,ne)-(e-core)*v))
    dm=solver.make_rdm1s(v,n,ne);occ=(np.diag(dm[0]),np.diag(dm[1]))
    return {"energy_Ha":float(e),"actual_d":d,"spin_strings":len(strings),"converged":bool(solver.converged),
            "residual_norm":residual,"spin_square":float(solver.spin_square(v,n,ne)[0]),"matvec_calls":solver.calls,
            "wall_seconds":perf_counter()-tick,"occupancies":[x.tolist()for x in occ]},occ


def run(config_path):
    root=Path(config_path).parent;config=json.loads(Path(config_path).read_text())
    if (root/"result.json").exists():return
    for name,expected in config["input_sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError("H input digest changed")
    with np.load(root/"integrals.npz",allow_pickle=False)as z:h,g=z["h1"].copy(),z["eri"].copy()
    with np.load(root/"distribution.npz",allow_pickle=False)as z:ids=z["blocked_ids_source_order"].copy();weights=z["source_weights"].copy()
    if np.any(weights<0)or not np.isfinite(weights).all()or weights.sum()<=0:raise ValueError("Unsigned probability distribution required; no silent quasi-probability projection")
    weights=weights/weights.sum();n=config["norb"];ne=tuple(config["nelec"])
    bits=((ids[:,None]>>np.arange(2*n-1,-1,-1,dtype=np.uint64))&1).astype(bool)
    valid=(bits[:,:n].sum(axis=1)==ne[1])&(bits[:,n:].sum(axis=1)==ne[0])
    valid_mass=float(weights[valid].sum())
    initial_bit_occ=np.average(bits[valid].astype(float),axis=0,weights=weights[valid])
    occupations=(initial_bit_occ[n:][::-1],initial_bit_occ[:n][::-1])
    start,cpu=perf_counter(),process_time();deadline=start+config["wall_cap_seconds"]
    records=[]
    save(root/"started.json",{"pid":os.getpid(),"input_unique":len(ids),"valid_sector_weight":valid_mass,
        "initial_occupancies":[a.tolist()for a in occupations],"not_integer_counts":True})
    try:
        # No-CR uses all available correct-sector configurations, as in P3.
        batches=postselect_and_subsample(bits,weights,hamming_right=ne[0],hamming_left=ne[1],
            samples_per_batch=int(valid.sum()),num_batches=1,rand_seed=41001)
        try:
            off,occ=projected(h,g,config["core_energy"],ne,batches[0],deadline,config["no_CR_D_cap"])
            save(root/"no-CR.json",{"status":"COMPLETED","CI":off,"all_correct_sector_strings_used":True,
                "batches":1,"source_shots_reconstructed":False,"new_QPU_shots":0})
        except MemoryError as exc:
            # This arm is an independent resource-limited control, not permission
            # to replace its full correct-sector space by an easier subset.
            off={"status":"NO_CR_RESOURCE_CAPPED_NOT_EVALUATED","failure":repr(exc),
                 "all_correct_sector_unique":int(valid.sum()),"no_CR_D_cap":config["no_CR_D_cap"]}
            save(root/"no-CR.json",off)
        for iteration in range(10):
            if perf_counter()>deadline:raise TimeoutError("H-lane whole-run wall boundary")
            save(root/"phase.json",{"phase":"OFFICIAL_CONFIGURATION_RECOVERY","iteration":iteration,
                "raw_distribution_unique":len(ids),"started_wall_seconds":perf_counter()-start})
            tick=perf_counter()
            recovered,probabilities=recover_configurations(bits,weights,occupations,ne[0],ne[1],rand_seed=41001+iteration)
            cr_seconds=perf_counter()-tick
            batches=postselect_and_subsample(recovered,probabilities,hamming_right=ne[0],hamming_left=ne[1],
                samples_per_batch=config["CR_samples_per_batch"],num_batches=10,rand_seed=42001+iteration)
            results=[];occs=[]
            for j,batch in enumerate(batches):
                save(root/"phase.json",{"phase":"MATRIX_FREE_CI","iteration":iteration,"batch":j,
                    "started_wall_seconds":perf_counter()-start})
                item,o=projected(h,g,config["core_energy"],ne,batch,deadline,config["CR_D_cap"])
                item.update(iteration=iteration,batch=j);save(root/("CI-"+str(iteration)+"-"+str(j)+".json"),item)
                results.append(item);occs.append(o)
            # Equal batch mean, not lowest-energy batch selection, updates occupations.
            occupations=tuple(np.mean([o[s] for o in occs],axis=0)for s in (0,1))
            values=[x["energy_Ha"]for x in results]
            row={"iteration":iteration,"mean_energy_Ha":float(np.mean(values)),"minimum_energy_Ha":float(min(values)),
                "maximum_energy_Ha":float(max(values)),"batch_energies_Ha":values,"actual_d":[x["actual_d"]for x in results],
                "occupancies":[o.tolist()for o in occupations],"CR_wall_seconds":cr_seconds,
                "total_wall_seconds":perf_counter()-start,"CPU_seconds":process_time()-cpu,
                "raw_distribution_unique":len(ids),"recovered_unique":len(recovered),"new_quantum_observations":0,
                "all_CI_converged":all(x["converged"]for x in results),
                "max_residual_norm":max(x["residual_norm"]for x in results)}
            records.append(row);save(root/"progress.json",{"iterations":records,"no_CR":off})
        status="PUBLIC_DISTRIBUTION_CURRENT_RUNTIME_CR_COMPLETED"
        failure=None
    except (TimeoutError,MemoryError)as exc:
        status="PAPER_POSTPROCESSING_RESOURCE_CAPPED_PRESERVED";failure=repr(exc)
    save(root/"result.json",{"status":status,"case_id":config["case_id"],"problem_id":config["problem_id"],
        "configuration":config,"iterations":records,"failure":failure,"new_QPU_shots":0,"new_model_quantum_shots":0,
        "original_float_weights_preserved":True,"original_SQD_0p1_runtime_reproduced":False,
        "CI_dimension_is_actual_Cartesian_dimension_not_number_of_shots":True,
        "batch_range_is_not_confidence_interval":True,"CPU_seconds":process_time()-cpu,
        "wall_seconds":perf_counter()-start,"slurm_job_id":os.environ.get("SLURM_JOB_ID")})


if __name__=="__main__":
    import sys;run(sys.argv[1])
