"""Evidence-derived adoption of the numeric CUDA route, never a sampler-win gate."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT,save_json,file_digest


def run():
    p=ROOT/"results/engine-v2-qualification"
    read=lambda path:json.loads((p/path).read_text())
    checks=read("all-pool-check/cuda-parity.json")
    if len(checks)!=5 or any(c["status"]!="PASSED" for c in checks):raise RuntimeError("Missing native/CUDA pool checks")
    native=sorted((p/"m4-native/checkpoints/N2_R0.9/ceo_ovp").glob("step-*.json"))
    cuda=sorted((p/"N2_R0.9-long/result").glob("step-*.json"))
    if len(native)!=65 or len(cuda)!=65:raise RuntimeError("Need full identical 64-step trial")
    differences=[]
    for a,b in zip(native,cuda):
        x=json.loads(a.read_text());y=json.loads(b.read_text())
        if x["selected_operators"]!=y["selected_operators"]:raise RuntimeError("Selection trajectory drift")
        differences.append(abs(x["vqe_energy"]-y["energy"]))
    if max(differences)>1e-9:raise RuntimeError("Full trajectory numerical mismatch")
    timing=read("import-timing.json")
    cold=read("N2_R0.9-long/result/cold-launch.json")["cold_wall_seconds"]
    upload=read("N2_R0.9-long/intent.json")["input_upload_seconds"]
    download=read("N2_R0.9-long/download-timing.json")["output_download_seconds"]
    # CPU lower bound: recorded cold pool setup + only steps9..64. GPU does0..64.
    cpu_bound=read("m4-native/extension-launch.json")["wall_after_import_seconds"]
    # Same persistent CPU controller creates one author pool/catalog, reuses it for
    # export and independent import. Cold CUDA process includes imports/H2D/D2H.
    components={"cpu_cold_pool_and_catalog_setup":timing["cold_pool_setup_seconds"],
                "input_network_transfer":upload,"cuda_cold_process_all_64_steps":cold,
                "output_network_transfer":download,
                "cpu_cached_pool_validation_all_65_checkpoints":timing["warm_validation_seconds"]+timing["warm_pool_setup_seconds"]}
    total=sum(components.values())
    if total>=cpu_bound:raise RuntimeError("No startup/transfer-inclusive speedup on the qualified route")
    result={"schema":"cuda-adoption-1","status":"QUALIFIED_N2_CUDA_NUMERICAL_ROUTE",
            "five_pools_native_cuda_checks":checks,"full_trajectory_case":"N2_R0.9",
            "steps":64,"selected_sequence_identical":True,"max_energy_difference_Ha":max(differences),
            "gpu_route_components_seconds":components,"gpu_route_total_seconds":total,
            "cpu_native_conservative_wall_bound_seconds":cpu_bound,"measured_conservative_speed_ratio":cpu_bound/total,
            "comparison_scope":"persistent Mac CPU controller; one shared cold author-pool/catalog setup; CUDA cold launch and both network directions and independent reconstruction included",
            "excluded_from_both":"one-time package installation; scheduler queue wait; one-time controller imports",
            "deployment_conditions":["persistent CPU pool cache used for export/import","one A100 per independent N2 geometry; at most two jobs",
                                     "all imported checkpoints checked against native author generator and Qiskit",
                                     "CI/chemistry/LUCJ/counts/aggregation stay CPU","method-specific speedup beyond OVP not separately established"],
            "not_a_scientific_sampler_advantage_claim":True,
            "code_hashes":{n:file_digest(ROOT/"benchmark"/n) for n in ("cuda_backend.py","engine_worker.py","gpu_profile_worker.py")}}
    save_json(p/"adoption.json",result)
    print(result["status"],"conservative speed ratio",cpu_bound/total)
    return result


if __name__=="__main__":run()
