"""Does the declared retrospective QSCI comparison survive the same recovery?

No new quantum samples. Final-checkpoint recovery is reused by raw evaluation ID;
other already-selected early checkpoints run exactly the same official settings.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .analysis import table_export
from .chemistry import load_reference
from .common import digest,file_digest,rng_for,save_json
from .sqd import recover_readout


def reusable_final_rows(data):
    context=data.get("context",{})
    if context.get("iterations")!=3 or context.get("primary_iteration")!="last-not-best":return {}
    return {r["source_raw_evaluation_id"]:r for r in data["rows"]}


def run(root):
    root=Path(root);pairs=json.loads((root/"analysis/resource_constrained_comparisons.json").read_text())
    requests=set()
    for p in pairs:
        requests.add((p["case_id"],p["ceo"],p["ceo_step"],p["shots"]))
        requests.add((p["case_id"],p["baseline"],p["baseline_step"],p["shots"]))
    @lru_cache(maxsize=4)
    def raw_file(path):return json.loads(Path(path).read_text())
    @lru_cache(maxsize=1)
    def reference(case):return load_reference(root/"references"/case)
    groups={};executed=reused=0
    for case,method,step,shots in sorted(requests):
        source=root/"qsci"/case/method/f"step-{step:04d}.json"
        context={"schema":"selected-official-recovery-1","source_raw_sha256":file_digest(source),"shots":shots,
                 "iterations":3,"initial_occupancies":"HF","primary_iteration":"last-not-best",
                 "selection":"retrospective QSCI resource-constrained pair; not a new policy validation"}
        target=root/"sqd-selected-v1"/"groups"/case/method/f"step-{step:04d}-M{shots}.json"
        if target.exists():
            saved=json.loads(target.read_text())
            if saved["context_id"]!=digest(context):raise RuntimeError("Selected recovery drift")
        else:
            final=root/"sqd"/case/(method+".json")
            available=reusable_final_rows(raw_file(str(final))) if final.exists() else {}
            rows=[]
            for raw in raw_file(str(source))["rows"]:
                if not raw["include_hf"] or raw["shots"]!=shots:continue
                identity=raw["raw_evaluation_id"]
                if identity in available:
                    r=available[identity]
                    rows.append({"replicate":raw["replicate"],"source_raw_evaluation_id":identity,
                                 "energy":r["energy_with_recovery"],"dimension":r["dimension_after"],
                                 "reused_final_recovery":True,"new_recovery_calls":0})
                else:
                    _,seed=rng_for("official-sqd-recovery-v1",identity)
                    history=recover_readout(reference(case),raw["raw_count_ids"],raw["raw_counts"],raw["R_cap"],seed,3)
                    rows.append({"replicate":raw["replicate"],"source_raw_evaluation_id":identity,
                                 "energy":history[-1]["energy"],"dimension":history[-1]["dimension"],
                                 "reused_final_recovery":False,"new_recovery_calls":3,"history":history})
            saved={"context":context,"context_id":digest(context),"rows":rows};save_json(target,saved)
        groups[(case,method,step,shots)]={r["replicate"]:r for r in saved["rows"]}
        executed+=sum(r["new_recovery_calls"] for r in saved["rows"])
        reused+=sum(r["reused_final_recovery"] for r in saved["rows"])
    rows=[]
    for p in pairs:
        a=groups[(p["case_id"],p["ceo"],p["ceo_step"],p["shots"])];b=groups[(p["case_id"],p["baseline"],p["baseline_step"],p["shots"])]
        reps=sorted(set(a)&set(b));delta=[1000*(a[r]["energy"]-b[r]["energy"]) for r in reps]
        equal=[r for r in reps if a[r]["dimension"]==b[r]["dimension"]]
        rows.append({**p,"delta_after_recovery_mHa":float(np.mean(delta)),"repeats":len(reps),
                     "equal_realized_d_repeats_after":len(equal),
                     "delta_after_on_equal_d_repeats_mHa":float(np.mean([1000*(a[r]["energy"]-b[r]["energy"]) for r in equal])) if equal else None,
                     "ceo_after_dimension_mean":float(np.mean([a[r]["dimension"] for r in reps])),
                     "baseline_after_dimension_mean":float(np.mean([b[r]["dimension"] for r in reps])),
                     "new_quantum_shots":0,"selection_scope":"same retrospective pair; not independent confirmation"})
    table_export(root/"sqd-selected-v1","pair-summary",rows)
    result={"comparison_pairs":len(rows),"unique_case_method_step_M_groups":len(groups),
            "stored_new_recovery_iterations":executed,"reused_final_readouts":reused,"new_quantum_shots":0}
    save_json(root/"sqd-selected-v1/status.json",result);print(result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)
