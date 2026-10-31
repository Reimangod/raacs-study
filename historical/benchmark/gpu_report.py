"""Combine actual M4 and allocated-A100 evidence without declaring premature adoption."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT,save_json


def main():
    records=[]
    for path in sorted((ROOT/"results/gpu-remote").glob("*/result/terminal.json")):
        case=path.parent.parent.name
        local=ROOT/"results/gpu-qualification/profiles"/case/"m4"
        terminal=json.loads(path.read_text())
        cuda=json.loads((path.parent/"cuda/profile.json").read_text())
        mac=json.loads((local/"profile.json").read_text())
        launch=json.loads((local/"launch.json").read_text())
        transfer=json.loads((path.parent.parent/"submission-started.json").read_text())["fixture_upload_seconds"]
        with np.load(local/"values.npz") as c,np.load(path.parent/"cuda/values.npz") as g:
            parity={k:{"max_abs_error":float(np.max(np.abs(c[k]-g[k]))),
                       "passed":bool(np.allclose(c[k],g[k],rtol=1e-8,atol=1e-9))} for k in c.files}
        records.append({"case_id":case,"slurm_job_id":terminal["slurm_job_id"],"gpu_device":cuda["device"],
                        "fixture_sha256":cuda["fixture_sha256"],"m4_gpu_parity":parity,
                        "components":[{"component":k,"m4_seconds":v["median_seconds"],
                                       "a100_seconds":cuda["warm_components"][k]["median_seconds"],
                                       "m4_over_a100_warm_speedup":v["median_seconds"]/cuda["warm_components"][k]["median_seconds"]}
                                      for k,v in mac["warm_components"].items()],
                        "m4_cold_process_seconds":launch["process_wall_seconds"],
                        "a100_cold_process_seconds":terminal["children"]["cuda"]["process_wall_seconds"],
                        "fixture_upload_seconds":transfer,
                        "m4_over_gpu_upload_plus_cold_speedup":launch["process_wall_seconds"]/(transfer+terminal["children"]["cuda"]["process_wall_seconds"]),
                        "m4_repeated_workload_seconds":mac["objective_workload_30_seconds"],
                        "gpu_repeated_workload_seconds":cuda["objective_workload_30_seconds"],
                        "decision":"COMPONENTS_QUALIFIED_FULL_TRAJECTORY_ADOPTION_NOT_YET_PROVEN",
                        "scientific_molecular_production_executed_on_gpu":False,
                        "notes":["GPU job completed, but this was a numerical/timing fixture, not full molecular production",
                                 "Cold wall includes process launch/import/input; upload additionally charged; queue wait excluded and separate",
                                 "Small CI is faster on CPU; do not offload merely to claim GPU utilization"]})
    destination=ROOT/"results/gpu-remote"
    save_json(destination/"qualification-summary.json",records)
    lines=["# A100 component qualification (not full trajectory speedup)","",
           "Each case used its own one-A100 Slurm job. M4 comparison uses the identical numeric fixture.","",
           "| Case | Component | M4 ms | A100 ms | Warm speedup |","|---|---|---:|---:|---:|"]
    for record in records:
        for c in record["components"]:
            lines.append(f"| {record['case_id']} | {c['component']} | {1000*c['m4_seconds']:.4f} | {1000*c['a100_seconds']:.4f} | {c['m4_over_a100_warm_speedup']:.3f} |")
    lines += ["","No full production GPU adoption claim is made. Next: a frozen short full-trajectory CPU/CUDA parity and startup/transfer-inclusive speed comparison, with shared task ownership, then assign separate larger cases to the two A100 workers only where justified."]
    (destination/"PROFILE_REPORT.md").write_text("\n".join(lines)+"\n")
    print("Profile cases:",len(records),"; all M4-CUDA component parity:",all(all(v["passed"] for v in r["m4_gpu_parity"].values()) for r in records))


if __name__=="__main__":main()
