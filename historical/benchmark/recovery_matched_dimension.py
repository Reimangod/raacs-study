"""Four-way same-realized-d readout: CEO/base × before/after recovery.

Reuses raw counts and recorded occupancies. For invalid-sector data only, replay
the third official recovery iteration with its ORIGINAL seed and input occupancy.
Final dimensions match in every replicate, not just a favorable equal-d subset.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from .analysis import table_export
from .chemistry import load_reference
from .common import digest,file_digest,rng_for,save_json
from .qsci import solve_selected
from .sqd import ids_to_blocked,blocked_to_ids


def rank_with_hf(ids,weights,valid,hf,size=None):
    ranked=[hf];seen={hf}
    pairs=((int(i),w) for i,w in zip(ids,weights) if w>0 and int(i) in valid)
    for i,w in sorted(pairs,key=lambda x:(-float(x[1]),x[0])):
        if size is not None and len(ranked)>=size:break
        if i not in seen:ranked.append(i);seen.add(i)
    return ranked


def run(root):
    root=Path(root);pairs=json.loads((root/"analysis/resource_constrained_comparisons.json").read_text())
    groups={};references={}
    @lru_cache(maxsize=4)
    def read(path):return json.loads(Path(path).read_text())
    requests=set()
    for p in pairs:
        requests|={(p["case_id"],p["ceo"],p["ceo_step"],p["shots"]),(p["case_id"],p["baseline"],p["baseline_step"],p["shots"])}
    for case,method,step,shots in sorted(requests):
        if case not in references:references[case]=load_reference(root/"references"/case)
        ref=references[case];m=ref["meta"];valid=set(map(int,ref["ids"]))
        raw=read(str(root/"qsci"/case/method/f"step-{step:04d}.json"))["rows"]
        raw={r["replicate"]:r for r in raw if r["include_hf"] and r["shots"]==shots}
        source=root/"sqd-selected-v1/groups"/case/method/f"step-{step:04d}-M{shots}.json"
        selected=read(str(source));rows={}
        for r in selected["rows"]:
            x=raw[r["replicate"]];before=rank_with_hf(x["raw_count_ids"],x["raw_counts"],valid,m["hf_id"],x["R_cap"])
            if x["valid_shots"]==shots:
                after=before
            else:
                history=r.get("history")
                if history is None:
                    final=read(str(root/"sqd"/case/(method+".json")))
                    history=next(z["iterations"] for z in final["rows"] if z["source_raw_evaluation_id"]==x["raw_evaluation_id"])
                _,seed=rng_for("official-sqd-recovery-v1",x["raw_evaluation_id"])
                occupancy=tuple(np.array(z) for z in history[-2]["occupancies"])
                corrected,weights=recover_configurations(ids_to_blocked(x["raw_count_ids"],m["n_orbitals"]),
                            np.array(x["raw_counts"],dtype=float)/shots,occupancy,m["nalpha"],m["nbeta"],
                            rand_seed=np.random.default_rng(seed+2))
                after=rank_with_hf(blocked_to_ids(corrected),weights,valid,m["hf_id"],x["R_cap"])
                if sorted(after[:x["R_cap"]])!=history[-1]["selected_ids"]:raise RuntimeError("Official recovery replay mismatch")
            rows[x["replicate"]]={"before":before[:x["R_cap"]],"after":after[:x["R_cap"]],
                                  "source_raw_evaluation_id":x["raw_evaluation_id"]}
        groups[(case,method,step,shots)]=(rows,file_digest(source))
    summaries=[]
    for p in pairs:
        a,sha=groups[(p["case_id"],p["ceo"],p["ceo_step"],p["shots"])];b,shb=groups[(p["case_id"],p["baseline"],p["baseline_step"],p["shots"])]
        context={"pair":p,"recovery_group_sha256":[sha,shb],"rule":"min of four realized dimensions per replicate; original frequency/HF rule",
                 "scope":"retrospective final-d matching; intermediate recovery still used registered R cap32","new_quantum_shots":0}
        key=digest(context);target=root/"sqd-matched-dimension/pairs"/(key+".json")
        if target.exists():rows=json.loads(target.read_text())["rows"]
        else:
            rows=[]
            for rep in sorted(set(a)&set(b)):
                size=min(len(a[rep][s]) for s in ("before","after"))
                size=min(size,*(len(b[rep][s]) for s in ("before","after")))
                values={}
                for label,data in (("ceo",a),("base",b)):
                    for stage in ("before","after"):
                        ids=np.array(sorted(data[rep][stage][:size]),dtype=np.int64)
                        values[label+"_"+stage]=solve_selected(references[p["case_id"]]["h"],ids)["energy"]
                rows.append({"replicate":rep,"common_d":size,
                             "delta_before_mHa":1000*(values["ceo_before"]-values["base_before"]),
                             "delta_after_mHa":1000*(values["ceo_after"]-values["base_after"]),
                             "raw_evaluation_ids":[a[rep]["source_raw_evaluation_id"],b[rep]["source_raw_evaluation_id"]]})
            save_json(target,{"context":context,"context_id":key,"rows":rows})
        summaries.append({**p,"matched_four_way_d":True,"repeats":len(rows),
                          "mean_common_d":float(np.mean([r["common_d"] for r in rows])),
                          "matched_delta_before_mHa":float(np.mean([r["delta_before_mHa"] for r in rows])),
                          "matched_delta_after_mHa":float(np.mean([r["delta_after_mHa"] for r in rows])),"new_quantum_shots":0})
    table_export(root/"sqd-matched-dimension","summary",summaries)
    result={"pairs":len(summaries),"all_replicates_four_way_equal_d":True,"new_quantum_shots":0,
            "scope":"final dimensions matched; not a matched-total-classical-work or independent-policy test"}
    save_json(root/"sqd-matched-dimension/status.json",result);print(result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)
