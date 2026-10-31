"""Explicit alternate cumulative readout, never mixed with independent primary data.

Reuses saved independent counts. The hypothetical cumulative shot budget at step k
is M times the number of consumed checkpoints, NOT M. Fixed-control revisions are
not growth and are excluded. This mode does not generate new quantum samples.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .analysis import ADAPT_METHODS
from .chemistry import load_reference
from .common import digest,file_digest,save_json
from .qsci import select_configurations,solve_selected


def merge_counts(accumulator,ids,counts):
    accumulator.update({int(i):int(c) for i,c in zip(ids,counts)})


def run(root,cases=None):
    root=Path(root);outputs=[]
    for case in sorted((root/"checkpoints").iterdir()):
        if not case.is_dir() or (cases and case.name not in cases):continue
        ref=load_reference(root/"references"/case.name)
        for method in ADAPT_METHODS:
            paths=sorted((root/"qsci"/case.name/method).glob("step-*.json"))
            if not paths:continue
            contract={"schema":"alternate-cumulative-readout-1","case_id":case.name,"method":method,
                      "source_sha256":[file_digest(p) for p in paths],"HF":True,
                      "scope":"derived reuse of independent primary draws; separate cumulative budget, not a new sampler experiment"}
            context=digest(contract);target=root/"cumulative-qsci-v1"/case.name/method/(context+".json")
            if target.exists():outputs.append(str(target));continue
            accumulators={};source_ids={};gate_cost={};rows=[]
            for path in paths:
                data=json.loads(path.read_text())
                for raw in data["rows"]:
                    if not raw["include_hf"]:continue
                    key=(raw["shots"],raw["replicate"])
                    counts=accumulators.setdefault(key,Counter());merge_counts(counts,raw["raw_count_ids"],raw["raw_counts"])
                    ids=source_ids.setdefault(key,[]);ids.append(raw["raw_evaluation_id"])
                    cost=gate_cost.setdefault(key,[0,0]);cost[0]+=raw["cnot_shot_proxy"];cost[1]+=raw["depth_shot_proxy"]
                    selected=select_configurations(list(counts),list(counts.values()),ref["ids"],ref["meta"]["hf_id"],raw["R_cap"],True)
                    result=solve_selected(ref["h"],selected)
                    total=sum(counts.values())
                    if total!=raw["shots"]*len(ids):raise RuntimeError("Cumulative shot accounting mismatch")
                    rows.append({"case_id":case.name,"method":method,"step":raw["step"],"shots_per_checkpoint":raw["shots"],
                                 "cumulative_raw_shots":total,"replicate":raw["replicate"],"consumed_checkpoints":len(ids),
                                 "source_raw_evaluation_ids":ids.copy(),"R_cap":raw["R_cap"],"dimension":len(selected),
                                 "energy":result["energy"],"energy_error_mHa":1000*(result["energy"]-ref["meta"]["fci_energy"]),
                                 "cumulative_cnot_shot_proxy":cost[0],"cumulative_depth_shot_proxy":cost[1],
                                 "new_quantum_shots":0,"selected_ids":selected.tolist()})
            save_json(target,{"contract":contract,"context_id":context,"rows":rows});outputs.append(str(target))
    save_json(root/"cumulative-qsci-v1/status.json",{"mode":"ALTERNATE_NOT_PRIMARY","outputs":outputs,"case_subset":cases,
               "new_quantum_shots":0,"note":"total cumulative budget grows with prefix; not an equal-M independent-checkpoint comparison"})
    print("alternate cumulative method files",len(outputs));return outputs


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--cases",nargs="+")
    a=p.parse_args();run(a.output,a.cases)
