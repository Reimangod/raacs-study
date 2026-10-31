"""Coverage/accounting audit, not a new research authorization bureaucracy."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from .common import save_json,save_binary,file_digest,provenance,resolve_artifact
from .config import cases_for

METHODS=("ceo_ovp","ceo_mvp","qeb","qubit","gsd","lucj_1","lucj_2","lucj_truncated","lucj_random_101","hf","ceo_random_101")
FIGURES=("qsci_vs_cnot","qsci_vs_depth","qsci_vs_shots","nonhf_coverage","hf_probability","vqe_error","qsci_vs_iteration","topk_recall","shots_to_chemical_vs_depth")
FIGURES=FIGURES+tuple(stem+f"_M{m}" for stem in ("qsci_vs_cnot","qsci_vs_depth","qsci_vs_iteration") for m in (10000,100000))


def audit(root,hash_artifacts=True):
    root=Path(root);issues=[];pending=[];refs=[];cells=[];checkpoints=[];raw_draws=virtual_shots=0;finite=0
    for case in cases_for("full"):
        meta=root/"references"/case.id/"reference.json"
        if not meta.exists():pending.append("reference:"+case.id);continue
        m=json.loads(meta.read_text());refs.append(m)
        if abs(m["fci_weight_sum"]-1)>1e-9 or m["fci_residual"]>1e-7:issues.append("reference validation:"+case.id)
        if hash_artifacts:
            for name,sha in m["artifact_sha256"].items():
                if file_digest(meta.parent/name)!=sha:issues.append("reference hash:"+case.id+"/"+name)
        for method in METHODS:
            directory=root/"checkpoints"/case.id/method;status=directory/"status.json";failure=root/"failures"/case.id/(method+".json")
            s=json.loads(status.read_text()).get("status") if status.exists() else ("FAILED_BEFORE_CHECKPOINT" if failure.exists() else "NOT_STARTED")
            cells.append({"case_id":case.id,"method":method,"status":s})
            if s in ("NOT_STARTED","RUNNING"):pending.append("method:"+case.id+"/"+method)
            for path in sorted(directory.glob("step-*.json")):
                c=json.loads(path.read_text());checkpoints.append(c)
                if c["problem_id"]!=m["request_digest"]:issues.append("problem binding:"+str(path))
                if c["qiskit_state_error"]>1e-8:issues.append("Qiskit parity:"+str(path))
                if len(c["analytic_metrics"])!=4:issues.append("analytic M grid:"+str(path))
                if hash_artifacts:
                    for key,sha in (("distribution","distribution_sha256"),("circuit_qpy","circuit_sha256"),("sampling_circuit_qpy","sampling_circuit_sha256")):
                        if file_digest(resolve_artifact(root,c[key]))!=c[sha]:issues.append("checkpoint hash:"+str(path)+"/"+key)
                q=root/"qsci"/case.id/method/path.name
                if not q.exists():pending.append("finite:"+case.id+"/"+method+"/"+path.name);continue
                d=json.loads(q.read_text());rows=d["rows"];finite+=1
                primary=[r for r in rows if r["include_hf"]]
                if len(rows)!=120 or len(primary)!=60:issues.append("finite trial grid:"+str(q))
                for r in primary:
                    if sum(r["raw_counts"])!=r["shots"]:issues.append("raw-count sum:"+str(q))
                    if len(r["selected_ids"])!=r["dimension"] or r["dimension"]>r["R_cap"]:issues.append("realized d:"+str(q))
                    if r["new_simulated_raw_shots"]!=r["shots"]:issues.append("raw shot accounting:"+str(q))
                    raw_draws+=1;virtual_shots+=r["new_simulated_raw_shots"]
                if any(r["new_simulated_raw_shots"] for r in rows if not r["include_hf"]):issues.append("ablation double charge:"+str(q))
        for name in FIGURES:
            for ext in ("png","svg","pdf"):
                if not (root/"figures"/case.id/(name+"."+ext)).exists():pending.append("figure:"+case.id+"/"+name+"."+ext)
    final_cp={}
    for c in checkpoints:
        key=(c["case_id"],c["method"])
        if key not in final_cp or final_cp[key]["step"]<c["step"]:final_cp[key]=c
    sqd_count=noise_count=0
    for (case,method),cp in final_cp.items():
        for prefix in ("sqd","readout-noise-v1"):
            path=root/prefix/case/(method+".json")
            if not path.exists():pending.append(prefix+":"+case+"/"+method);continue
            data=json.loads(path.read_text())
            context=data.get("context",data.get("contract",{}))
            if context["checkpoint_id"]!=cp["checkpoint_id"]:issues.append("stale final readout:"+str(path))
            if len(data["rows"])!=60:issues.append("recovery trial grid:"+str(path))
            if prefix=="sqd":sqd_count+=len(data["rows"])
            else:noise_count+=len(data["rows"])
    native_path=root/"native_resources/ecr-line/summary.json"
    native=json.loads(native_path.read_text()) if native_path.exists() else []
    native_ids={r["checkpoint_id"] for r in native}
    for c in checkpoints:
        if c["checkpoint_id"] not in native_ids:pending.append("native:"+c["checkpoint_id"])
    # Scientific failures and caps are exposed outcomes, not infrastructure zero-fill.
    result={"schema":"benchmark-requirements-audit-1","status":"COMPLETE_WITH_EXPLICIT_LIMITS" if not issues and not pending else "IN_PROGRESS_OR_REPAIR_REQUIRED",
            "reference_cases":len(refs),"planned_cases":30,"planned_method_cells":330,"cell_status_counts":dict(Counter(c["status"] for c in cells)),
            "checkpoints":len(checkpoints),"finite_evaluated_checkpoints":finite,"independent_raw_draws":raw_draws,
            "simulated_raw_shots":virtual_shots,"hardware_shots":0,"sqd_readouts":sqd_count,"synthetic_noise_readouts":noise_count,
            "native_profile_rows":len(native_ids),"issues":issues,"pending_count":len(pending),"pending":pending,
            "all_cells":cells,"scope_limits":["ideal classical multinomial sampling; not quantum hardware data",
            "synthetic ECR target and readout-only noise; no actual device calibration or Aer gate-noise experiment",
            "CCSD-initialized fixed LUCJ, not globally optimized LUCJ","common-loop CEO pools, not full CEO* subroutine reproduction",
            "retrospective checkpoint/frontier selection; no independent online-policy validation",
            "single ADAPT initialization, geometry points not independent molecule replicates","end-to-end quantum training Measurement Cost not established"]}
    save_json(root/"requirements-audit.json",result);save_json(root/"environment-final.json",provenance())
    lines=["# Requirement coverage audit","",f"Status: `{result['status']}`.","",
           f"References {len(refs)}/30; method cells {len(cells)}/330; checkpoints {len(checkpoints)}; finite evaluated {finite}.",
           f"Official recovery {sqd_count}; synthetic readout recovery {noise_count}; real hardware shots0.",
           f"Issues {len(issues)}; pending checks {len(pending)}.","","## Cell status counts",""]
    lines += [f"- {k}: {v}" for k,v in result["cell_status_counts"].items()]
    lines += ["","## Claim boundaries",""]+["- "+x for x in result["scope_limits"]]
    lines += ["","## Requirement mapping","",
              "| Objective | Implementation / evidence |","|---|---|",
              "|0–3 environment, references, pools | uv.lock, THIRD_PARTY_NOTICES, chemistry/ansatz/lucj, references/* |",
              "|4 checkpoints | checkpoints/*/*/step-*.json/QPY, distributions, per-method status |",
              "|5 analytic discovery | four-M analytic_metrics per checkpoint; all top-K fields |",
              "|6 finite QSCI | qsci raw counts, HF ablation, R cap/actual d,20 repeats and intervals |",
              "|7 recovery | official sqd records; separate synthetic readout-noise-v1; no Aer claim |",
              "|8 resources | logical sampler/preparation resources; declared synthetic ECR derivative |",
              "|9 optima/frontiers | analysis/checkpoint_optima and pareto_frontiers; retrospective labels |",
              "|10 plots | nine required types per case,3 formats, cross-case figures |",
              "|11 fairness | phase/energy/sector checks; matched-dimension secondary readout; missingness |",
              "|12–14 execution/data | cache hashes, case locks, atomic saves, CLI; CPU-owned GPU fleet |",
              "|15 tests | benchmark/tests; record actual pytest output separately |",
              "|16 questions | report.md eight-question evidence and limits |",
              "|17 milestones | actual30-case coverage above; capped/failed cases remain visible |",
              "|A100 addendum | component profiles, engine-v2 parity/timing/adoption, per-case Slurm evidence |"]
    if issues:lines += ["","## Issues",""]+["- "+x for x in issues]
    if pending:lines += ["","## Pending (first100; full JSON preserves all)",""]+["- "+x for x in pending[:100]]
    save_binary(root/"REQUIREMENTS_AUDIT.md",("\n".join(lines)+"\n").encode())
    print(result["status"],"cases",len(refs),"checkpoints",len(checkpoints),"pending",len(pending),"issues",len(issues))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--no-hash",action="store_true")
    a=p.parse_args();audit(a.output,not a.no_hash)
