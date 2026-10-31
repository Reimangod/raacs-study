"""Secondary synthetic readout noise, NOT Aer gate-noise or hardware evidence.

Independent symmetric bit-flip readout probability0.01 is fixed for every arm.
The same raw counts feed sector-filtered QSCI and official SQD recovery.
No native-device noise model, zero-noise extrapolation or FCI-guided correction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .common import digest,file_digest,rng_for,save_json,resolve_artifact
from .chemistry import load_reference
from .qsci import select_configurations,solve_selected
from .sqd import recover_readout


def bitflip_distribution(probability,rate):
    p=np.asarray(probability,dtype=float).copy()
    if not 0<=rate<=1 or len(p)&(len(p)-1):raise ValueError("Invalid rate/dimension")
    indices=np.arange(len(p))
    for bit in range(len(p).bit_length()-1):
        p=(1-rate)*p+rate*p[indices ^ (1<<bit)]
    if p.min()<-1e-14 or abs(p.sum()-1)>1e-9:raise ValueError("Invalid transformed distribution")
    return np.maximum(p,0)/p.sum()


def run(root,rate=.01):
    root=Path(root);total=0
    for case in sorted((root/"checkpoints").iterdir()):
        if not case.is_dir():continue
        ref=load_reference(root/"references"/case.name);valid=set(map(int,ref["ids"]))
        for method in sorted(case.iterdir()):
            status=method/"status.json"
            if not status.exists() or json.loads(status.read_text()).get("status")=="RUNNING":continue
            paths=sorted(method.glob("step-*.json"))
            if not paths:continue
            cp=json.loads(paths[-1].read_text());target=root/"readout-noise-v1"/case.name/(method.name+".json")
            contract={"schema":"synthetic-readout-noise-1","checkpoint_id":cp["checkpoint_id"],
                      "distribution_sha256":cp["distribution_sha256"],"bit_flip_probability":rate,
                      "shots":[1000,10000,100000],"repeats":20,"R_cap":32,"recovery_iterations":3,
                      "initial_occupancies":"HF; identical for all arms","checkpoint_rule":"last saved; not best outcome",
                      "scope":"synthetic independent readout errors only; NOT gate-noise or device model"}
            context=digest(contract)
            if target.exists():
                old=json.loads(target.read_text())
                if old["context_id"]!=context:raise RuntimeError("Noise experiment drift")
                total+=len(old["rows"]);continue
            distribution=resolve_artifact(root,cp["distribution"])
            if file_digest(distribution)!=cp["distribution_sha256"]:raise RuntimeError("Distribution drift")
            with np.load(distribution) as z:p=bitflip_distribution(z["probability"],rate)
            rows=[]
            for shots in contract["shots"]:
                for rep in range(20):
                    rng,seed=rng_for("readout-noise-v1",context,shots,rep)
                    counts_all=rng.multinomial(shots,p);ids=np.flatnonzero(counts_all);counts=counts_all[ids]
                    selected=select_configurations(ids,counts,valid,ref["meta"]["hf_id"],32,True)
                    before=solve_selected(ref["h"],selected)
                    started=perf_counter();history=recover_readout(ref,ids,counts,32,seed,3);last=history[-1]
                    rows.append({"case_id":case.name,"method":method.name,"checkpoint_id":cp["checkpoint_id"],
                                 "shots":shots,"replicate":rep,"seed":seed,"raw_count_ids":ids.tolist(),"raw_counts":counts.tolist(),
                                 "valid_raw_rate":float(sum(c for i,c in zip(ids,counts) if int(i) in valid))/shots,
                                 "dimension_before":before["dimension"],"dimension_after":last["dimension"],
                                 "energy_before":before["energy"],"energy_after":last["energy"],
                                 "error_before_mHa":1000*(before["energy"]-ref["meta"]["fci_energy"]),
                                 "error_after_mHa":last["energy_error_mHa"],"recovery_seconds":perf_counter()-started,
                                 "valid_recovered_mass":last["valid_recovered_mass"],"history":history,
                                 "new_simulated_raw_shots":shots,"new_hardware_shots":0,
                                 "resources":cp["resources"]})
            save_json(target,{"contract":contract,"context_id":context,"rows":rows})
            total+=len(rows);print(case.name,method.name,"synthetic readout",len(rows),flush=True)
    save_json(root/"readout-noise-v1/status.json",{"readouts":total,"scope":"SYNTHETIC_READOUT_ONLY_NOT_HARDWARE_NOT_AER"})
    return total


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();print(run(a.output))
