"""Secondary exact-realized-d readout using existing counts, no extra shots.

For each already-declared resource-constrained pair and replicate, truncate both
frequency-ranked sets to min(d_A,d_B). Pair choice remains retrospective; this
diagnostic removes unequal CI dimension, NOT post-selection or training cost.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .analysis import table_export
from .chemistry import load_reference
from .common import digest,file_digest,save_json
from .qsci import select_configurations,solve_selected


def run(root):
    root=Path(root);pairs=json.loads((root/"analysis/resource_constrained_comparisons.json").read_text())
    refs={};raw_cache={};summaries=[];total=0
    for pair in pairs:
        case=pair["case_id"]
        if case not in refs:refs[case]=load_reference(root/"references"/case)
        ref=refs[case]
        a=root/"qsci"/case/pair["ceo"]/f"step-{pair['ceo_step']:04d}.json"
        b=root/"qsci"/case/pair["baseline"]/f"step-{pair['baseline_step']:04d}.json"
        contract={"schema":"paired-exact-realized-d-1","pair":pair,"source_sha256":[file_digest(a),file_digest(b)],
                  "rule":"per-replicate min realized d; HF first then observed count descending/state integer ascending",
                  "new_quantum_shots":0,"scope":"retrospective selected pairs; NOT independent policy validation"}
        context=digest(contract);target=root/"matched-dimension"/"pairs"/(context+".json")
        if target.exists():saved=json.loads(target.read_text())
        else:
            for path in (a,b):
                if path not in raw_cache:raw_cache[path]=json.loads(path.read_text())["rows"]
            ar={r["replicate"]:r for r in raw_cache[a] if r["include_hf"] and r["shots"]==pair["shots"]}
            br={r["replicate"]:r for r in raw_cache[b] if r["include_hf"] and r["shots"]==pair["shots"]}
            rows=[]
            for rep in sorted(set(ar)&set(br)):
                x,y=ar[rep],br[rep];dimension=min(x["dimension"],y["dimension"])
                values=[]
                for raw in (x,y):
                    selected=select_configurations(raw["raw_count_ids"],raw["raw_counts"],ref["ids"],ref["meta"]["hf_id"],dimension,True)
                    result=solve_selected(ref["h"],selected)
                    if result["dimension"]!=dimension:raise RuntimeError("Exact-d matching failed")
                    values.append(result["energy"])
                rows.append({"replicate":rep,"dimension":dimension,"ceo_energy":values[0],"baseline_energy":values[1],
                             "delta_error_mHa":1000*(values[0]-values[1]),"source_raw_evaluation_ids":[x["raw_evaluation_id"],y["raw_evaluation_id"]]})
            saved={"contract":contract,"context_id":context,"rows":rows};save_json(target,saved)
        rows=saved["rows"];total+=len(rows)
        summaries.append({**pair,"same_realized_dimension":True,"replicates":len(rows),
                          "mean_common_dimension":float(np.mean([r["dimension"] for r in rows])),
                          "matched_d_delta_error_mHa":float(np.mean([r["delta_error_mHa"] for r in rows])),
                          "new_quantum_shots":0,"pair_context_id":context})
    table_export(root/"matched-dimension","summary",summaries)
    save_json(root/"matched-dimension/status.json",{"pairs":len(summaries),"readout_pairs":total,"new_quantum_shots":0})
    return len(summaries)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();print(run(a.output))
