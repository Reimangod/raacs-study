"""Registered SqDRIFT S5 variants. Main frozen sources/outcomes are unchanged."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from .sqdrift_execution import Session, source_request, prepare_member, member_state
from .budget import BudgetLedger
from .ci import interleaved_to_blocked
from .common_execution import final_readouts, load_physical, run_identity
from .input_adapter import array_bytes
from .io import ROOT, PLAN_ROOT, atomic_bytes, digest, save_json, sha256

def allocations(budget, seed, length, shots_per_member):
    if length not in (5,20,50) or shots_per_member not in (128,512):
        raise ValueError("Unregistered sensitivity")
    result = []
    for i,k in enumerate((1,2,3)):
        shots = budget//3+(i < budget%3)
        count = (shots+shots_per_member-1)//shots_per_member
        for j in range(count):
            M = shots//count+(j < shots%count)
            member_seed = int(digest(["paper-S5-SqDRIFT-v1",seed,budget,k,j,length,shots_per_member])[:8],16)
            result.append(({"time":float(k),"num_terms":length,"seed":member_seed,"k":k,"member_index":j},int(M)))
    if sum(M for _,M in result) != budget or max(M for _,M in result)>shots_per_member:
        raise RuntimeError("Allocation mismatch")
    return result

def run(case, budget, seed, length=5, shots_per_member=128, lane="S5_SENSITIVITY"):
    import fcntl
    method = "D_SQDRIFT_N"+str(length)+("-M512" if shots_per_member == 512 else "")
    identity = run_identity(case, method, budget, seed, lane)
    directory = ROOT/"common-runs"/lane/case/method/(str(budget)+"-"+str(seed))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/"result.json").exists():
            return json.loads((directory/"result.json").read_text())
        meta, h1, eri = load_physical(case)
        binding = {"run_id": identity, "case_id": case, "problem_id": meta["problem_id"], "method_id": method,
                   "budget": budget, "seed": seed, "lane": lane,
                   "adapter_sha256": sha256(Path(__file__).read_bytes()),
                   "native_worker_sha256": sha256(Path(__file__).with_name("sqdrift_native_worker.py").read_bytes()),
                   "scientific_plan_sha256": sha256((PLAN_ROOT/"plan.json").read_bytes()),
                   "paper_author_driver_reproduced": False,
                   "n_excitations": length, "shots_per_member": shots_per_member,
                   "implementation_label": "P1-inspired native SDK grouped SqDRIFT/common-budget adaptation"}
        if (directory/"request.json").exists() and json.loads((directory/"request.json").read_text()) != binding:
            raise RuntimeError("Frozen SqDRIFT run drift; use a versioned successor")
        save_json(directory/"request.json", binding)
        native_root = ROOT/"native-SqDRIFT"/case
        native_root.mkdir(parents=True, exist_ok=True)
        if not (native_root/"base.json").exists():
            save_json(native_root/"base.json", source_request(case))
        ledger = BudgetLedger(ROOT/"common-budget.sqlite")
        ledger.register(identity, budget)
        session = None
        tick, cpu = perf_counter(), process_time()
        merged = defaultdict(int)
        records = []
        try:
            for i, (spec, M) in enumerate(allocations(budget, seed, length, shots_per_member)):
                event = digest([spec, M])
                saved = directory/"draws"/(event+".json")
                if saved.exists():
                    record = json.loads(saved.read_text())
                    blob = (directory/"draws"/(event+".npz")).read_bytes()
                    if sha256(blob) != record["counts_sha256"]:
                        raise RuntimeError("Committed SqDRIFT draw digest mismatch")
                    with np.load(BytesIO(blob), allow_pickle=False) as a:
                        ids, counts = a["ids"].copy(), a["counts"].copy()
                    ledger.complete(identity, event, int(counts.sum()), sha256(blob))
                else:
                    if session is None:
                        session = Session(native_root/"base.json", native_root)
                    member_dir, record = prepare_member(case, spec, session)
                    state, numerical = member_state(meta, record["native_export"])
                    request = {"member_id": digest(spec), "problem_id": meta["problem_id"], "shots": M,
                               "readout_seed": int(digest(["final", seed, i, budget])[:16], 16)}
                    status = ledger.reserve(identity, event, request, M)
                    if status != "NEW_RESERVATION":
                        raise RuntimeError("Unknown SqDRIFT draw outcome; do not resample")
                    ids = np.array(sorted(state), np.uint64)
                    probabilities = np.array([abs(state[int(x)])**2 for x in ids])
                    counts = np.random.default_rng(request["readout_seed"]).multinomial(M, probabilities/probabilities.sum())
                    ids, counts = ids[counts > 0], counts[counts > 0].astype(np.uint64)
                    blob = array_bytes({"ids": ids, "counts": counts})
                    atomic_bytes(directory/"draws"/(event+".npz"), blob)
                    record = {"member_id": digest(spec), "spec": spec, "shots": M, "request": request,
                              "counts_sha256": sha256(blob), "resources": record["resources"],
                              "coefficient_slots": record["coefficient_slots"], "state_numerics": numerical}
                    save_json(saved, record)
                    ledger.complete(identity, event, int(counts.sum()), sha256(blob))
                records.append(record)
                for x, count in zip(ids, counts):
                    merged[int(x)] += int(count)
                if (i+1) % 20 == 0:
                    save_json(directory/"progress.json", {"members_completed": i+1, "members_planned": len(allocations(budget, seed, length, shots_per_member)),
                             "ledger": ledger.snapshot(identity), "pid": os.getpid()})
            ids = np.array([interleaved_to_blocked(x, meta["norb"]) for x in sorted(merged)], np.uint64)
            counts = np.array([merged[x] for x in sorted(merged)], np.uint64)
            atomic_bytes(directory/"final-counts.npz", array_bytes({"blocked_ids": ids, "counts": counts}))
            readouts = final_readouts(directory, meta, h1, eri, ids, counts)
            weighted = {field: sum(r["shots"]*r["resources"]["Z_sampling_equivalent"][field] for r in records)/budget
                        for field in ("cnot_count", "cnot_depth", "total_unitary_depth")}
            maxima = {field: max(r["resources"]["Z_sampling_equivalent"][field] for r in records) for field in weighted}
            report = {**binding, "status": "COMPLETED", "budget_ledger": ledger.snapshot(identity),
                      "independent_final_model_shots": int(counts.sum()), "pilot_model_shots": 0, "actual_QPU_shots": 0,
                      "final_CI": readouts, "member_count": len(records), "member_shot_weighted_mean_resources": weighted,
                      "member_max_resources": maxima, "total_model_cnot_applications": int(sum(r["shots"]*r["resources"]["Z_sampling_equivalent"]["cnot_count"] for r in records)),
                      "member_records": records, "VQE_optimized_variables": 0,
                      "this_attempt_wall_seconds": perf_counter()-tick, "this_attempt_CPU_seconds": process_time()-cpu,
                      "is_primary_run": lane == "C_PRIMARY", "whole_study_complete": False}
            if report["budget_ledger"]["known_actual_shots"] != budget:
                raise RuntimeError("SqDRIFT paid-shot budget mismatch")
            save_json(directory/"result.json", report)
            print(case, method, seed, "COMPLETED", [r["energy_Ha"] for r in readouts], flush=True)
            return report
        finally:
            if session is not None:
                session.close()

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--case",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--length",type=int,choices=[5,20,50],required=True)
    p.add_argument("--shots-per-member",type=int,choices=[128,512],default=128)
    a=p.parse_args()
    print(json.dumps(run(a.case,100000,a.seed,a.length,a.shots_per_member)))
