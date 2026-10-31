"""Fresh HF-to-sampler construction with online coefficient controls.

No inherited compressed circuits or future operator sequences. Compact durable
normal forms, raw counts and all objective work, including rejected trials.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import QuantumCircuit
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .ansatz import circuit_bytes, measurement_equivalent_circuit, resources
from .integral_connections import IntegralConnections
from .local_sampler_primitives import LocalQE, local_proposals, observed_ci, pack_counts64
from .parameter_control_primitives import vqe_update
from .resource_adaptive_sampler import integral_problem
from .sparse_qe_circuit import SparseQECircuit
from .study.core import rss_peak

MODES = ("fixed_pi4", "one_parameter_vqe", "blockwise_vqe", "full_vqe", "qsci_local_grid")
NAMESPACE = "RAACS-online-parameter-discovery-v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS cells(id TEXT PRIMARY KEY,task TEXT,status TEXT,result TEXT);
CREATE TABLE IF NOT EXISTS circuits(id TEXT PRIMARY KEY,problem TEXT,actions BLOB,resources TEXT);
CREATE TABLE IF NOT EXISTS draws(id TEXT PRIMARY KEY,cell TEXT,circuit TEXT,role TEXT,step INTEGER,extra TEXT,shots INTEGER,seed TEXT,counts BLOB,energy REAL,selected TEXT);
CREATE TABLE IF NOT EXISTS optimizations(id TEXT PRIMARY KEY,cell TEXT,step INTEGER,circuit TEXT,trace BLOB);
CREATE TABLE IF NOT EXISTS decisions(cell TEXT,step INTEGER,record BLOB,PRIMARY KEY(cell,step));
CREATE TABLE IF NOT EXISTS final_qpy(cell TEXT PRIMARY KEY,preparation BLOB,sampling BLOB);
"""


def compressed(value):
    return zlib.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), 6)


def decoded(blob): return json.loads(zlib.decompress(blob))


def setting(mode):
    source = ROOT/"results/standard-v1"
    cases = ["H2_R0.7", "H4_R1.1"] if mode == "quick" else sorted(p.name for p in (source/"references").iterdir() if p.is_dir())
    files = ("online_parameter_controls.py", "parameter_control_primitives.py", "sparse_qe_circuit.py", "local_sampler_primitives.py",
             "integral_hamiltonian.py", "integral_connections.py", "resource_adaptive_sampler.py", "ansatz.py", "common.py")
    return {"schema": NAMESPACE, "mode": mode, "source": str(source), "cases": cases, "seeds": [101] if mode == "quick" else [101,211,307],
            "methods": list(MODES), "M": 10000, "R": 64, "readout_R": [8,16,32,64,128,256,512],
            "final_repeats": 4 if mode == "quick" else 20, "probe": 32, "steps": 8, "final_reserve": 5000,
            "source_parent_cap": 128, "target_cap": 128, "candidate_cap": 32, "measured_candidates": 4,
            "support_cap": 512, "gain_threshold_Ha": 1e-6,
            "BFGS": {"max_calls": 300, "max_iterations": 80, "seconds": 30., "gtol": 1e-6},
            "implementation": {str(Path("benchmark")/p): file_digest(ROOT/"benchmark"/p) for p in files},
            "vendor_circuit_sha256": file_digest(ROOT/"vendor/ceo/adaptvqe/circuits.py"),
            "spec_sha256": file_digest(ROOT/"study-v1/ONLINE_PARAMETER_CONTROLS.md"),
            "runtime": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "qiskit")},
            "free_floor": 3*2**30, "output_cap": 48*2**20, "RSS_cap": 2*2**30,
            "FCI_selection": False, "hardware_shots": 0}


def task_identity(cfg, info, seed, mode):
    # No raw reference-file hash / FCI evaluation label in algorithm RNG.
    return {"schema": NAMESPACE, "problem": info, "seed": seed, "mode": mode,
            "algorithm": {k: cfg[k] for k in ("M","R","readout_R","final_repeats","probe","steps","final_reserve","source_parent_cap","target_cap","candidate_cap","measured_candidates","support_cap","gain_threshold_Ha","BFGS")},
            "implementation": cfg["implementation"], "vendor": cfg["vendor_circuit_sha256"], "runtime": cfg["runtime"]}


def construct(db, cfg, case, seed, mode):
    if mode not in MODES: raise ValueError("Unregistered online parameter mode")
    backend, info = integral_problem(Path(cfg["source"])/"references"/case)
    task = task_identity(cfg, info, seed, mode); tid = digest(task)
    existing = db.execute("SELECT status,result FROM cells WHERE id=?", (tid,)).fetchone()
    if existing:
        if existing[0] == "COMPLETED": return json.loads(existing[1])
        raise RuntimeError("Preserved partial online task; no automatic free retry")
    db.execute("INSERT INTO cells VALUES(?,?,?,NULL)", (tid, json.dumps(task), "RUNNING")); db.commit()
    started = perf_counter(); spent = 0; objective_calls = 0; gradient_calls = 0; circuits = {}; sims = {}
    def physical(actions, hamiltonian=False):
        key = digest([info["problem_id"], actions]); ops = [LocalQE(tuple(a["source"]), tuple(a["target"]), a["n_qubits"]) for a in actions]
        if key not in circuits:
            prep = QuantumCircuit(info["n_qubits"], name="online-raacs-"+key[:12])
            for q in range(info["n_qubits"]):
                if info["hf_id"] >> q & 1: prep.x(q)
            for op, a in zip(ops, actions): prep = op.append(prep, a["theta"])
            sample, _ = measurement_equivalent_circuit(prep); res = resources(sample, len(ops), len(ops))
            circuits[key] = (prep, sample, res)
            db.execute("INSERT OR IGNORE INTO circuits VALUES(?,?,?,?)", (key, info["problem_id"], compressed(actions), json.dumps(res)))
        skey = (tuple(op.identity for op in ops), hamiltonian)
        if skey not in sims:
            sims[skey] = SparseQECircuit(info["n_qubits"], info["hf_id"], ops, cfg["support_cap"], backend if hamiltonian else None)
        return key, sims[skey], circuits[key]
    def measure(actions, role, step, extra, shots):
        nonlocal spent
        cid, sim, (_, _, res) = physical(actions)
        did = digest([tid, cid, role, step, extra, shots]); rngseed = rng_for(NAMESPACE, did)[1]
        ids, ns = sim.counts([a["theta"] for a in actions], shots, rngseed)
        obs = dict(zip(map(int, ids), map(int, ns)))
        ci = observed_ci(backend, obs, info["hf_id"], info["nalpha"], info["nbeta"], cfg["R"])
        db.execute("INSERT INTO draws VALUES(?,?,?,?,?,?,?,?,?,?,?)", (did,tid,cid,role,step,json.dumps(extra),shots,str(rngseed),pack_counts64(ids,ns),ci["energy_Ha"],json.dumps(ci["selected_ids"]))); db.commit()
        if role != "final": spent += shots
        return {"id": did, "circuit": cid, "actions": actions, "ci": ci, "counts": obs, "resources": res}
    initial = measure([], "initial", 0, "HF", cfg["probe"]); current = initial; actions = []; stop = "STEP_CAP"; decisions = []
    for step in range(1, cfg["steps"]+1):
        connections = IntegralConnections(backend, parent_cap=cfg["source_parent_cap"])
        proposals, scope = local_proposals(backend, connections, current["ci"], cfg["target_cap"], cfg["candidate_cap"])
        if not proposals: stop = "NO_LOCAL_PROPOSALS"; break
        parent_cnot = current["resources"]["cnot_count"]; parent_depth = current["resources"]["cnot_depth"]
        ranks = []
        for p in proposals:
            op = p["operator"]; nominal = actions+[{**asdict(op), "theta": float(np.pi/8)}]
            _, _, (_, _, res) = physical(nominal)
            cost = 1+max(0,res["cnot_count"]-parent_cnot)+.5*max(0,res["cnot_depth"]-parent_depth)
            target = set(op.action(dict(zip(current["ci"]["selected_ids"],current["ci"]["coefficients"]))))
            novelty = len(target-set(current["counts"])-{info["hf_id"]})/max(1,len(target))
            ranks.append({"operator": op, "score": p["residual_priority"]*(1+novelty)/cost})
        ranks.sort(key=lambda p: (-p["score"],p["operator"]))
        measured = []; no_budget = False
        for p in ranks[:cfg["measured_candidates"]]:
            op = p["operator"]
            angles = (-np.pi/4,-np.pi/8,np.pi/8,np.pi/4) if mode == "qsci_local_grid" else (np.pi/4,)
            for angle in angles:
                if spent+cfg["probe"] > cfg["M"]-cfg["final_reserve"]: no_budget = True; break
                trial = actions+[{**asdict(op), "theta": float(angle)}]; before_optimization = [a["theta"] for a in trial]
                if mode.endswith("_vqe"):
                    initial_cid, sim, _ = physical(trial, True)
                    control = vqe_update(sim, before_optimization, mode, **cfg["BFGS"])
                    oid = digest([tid, step, op.identity, before_optimization])
                    db.execute("INSERT INTO optimizations VALUES(?,?,?,?,?)", (oid,tid,step,initial_cid,compressed(control))); db.commit()
                    objective_calls += control["actual_objective_evaluations"]; gradient_calls += control["actual_complete_gradient_evaluations"]
                    trial = [{**a, "theta": t} for a,t in zip(trial,control["theta"])]
                work_before = backend.matrix_entry_requests
                candidate = measure(trial, "candidate", step, [op.identity,float(angle)], cfg["probe"])
                work = backend.matrix_entry_requests-work_before
                gain = current["ci"]["energy_Ha"]-candidate["ci"]["energy_Ha"]
                dc = max(0,candidate["resources"]["cnot_count"]-parent_cnot)
                candidate.update(gain=gain,utility=gain/(1+dc+work/64),operator=op,work=work)
                measured.append(candidate)
            if no_budget: break
        eligible = [p for p in measured if p["gain"] > cfg["gain_threshold_Ha"]]
        winner = max(eligible,key=lambda p:(p["utility"],p["gain"],-p["resources"]["cnot_count"],p["operator"].source,p["operator"].target,tuple(a["theta"] for a in p["actions"]))) if eligible else None
        decision = {"parent_draw": current["id"], "parent_actions": actions, "scope": scope,
                    "ranked": [{"operator": asdict(p["operator"]), "score": p["score"]} for p in ranks],
                    "measured": [{"draw": p["id"],"gain":p["gain"],"utility":p["utility"],"CI_matrix_requests":p["work"]} for p in measured],
                    "winner": winner["id"] if winner else None, "pilot_spent": spent, "no_more_budget": no_budget}
        db.execute("INSERT INTO decisions VALUES(?,?,?)", (tid,step,compressed(decision))); db.commit(); decisions.append(decision)
        if winner is None: stop = "PILOT_BUDGET_CAP" if no_budget else "NO_PILOT_ENERGY_IMPROVEMENT"; break
        current = winner; actions = winner["actions"]
        if no_budget: stop = "PILOT_BUDGET_CAP"; break
    remaining = cfg["M"]-spent
    assert remaining >= cfg["final_reserve"]
    final = []
    for rep in range(cfg["final_repeats"]):
        draw = measure(actions,"final",len(actions),rep,remaining)
        final.append({"draw":draw["id"],"energy_Ha":draw["ci"]["energy_Ha"],"actual_d":draw["ci"]["actual_dimension"]})
    cid, _, (prep, sample, res) = physical(actions)
    db.execute("INSERT INTO final_qpy VALUES(?,?,?)", (tid,zlib.compress(circuit_bytes(prep),6),zlib.compress(circuit_bytes(sample),6)))
    result = {"case":case,"seed":seed,"mode":mode,"task_id":tid,"status":"COMPLETED","construction_stop":stop,
              "accepted_operators":actions,"final_circuit":cid,"decisions":len(decisions),"sampling_resources":res,
              "pilot_shots":spent,"final_shots_per_rep":remaining,"final_repeats":cfg["final_repeats"],"final":final,
              "new_actual_model_shots":spent+remaining*cfg["final_repeats"],"known_branch_budget":cfg["M"],
              "actual_classical_objective_calls":objective_calls,"actual_full_gradient_calls":gradient_calls,
              "VQE_QPU_objective_shots":None if mode.endswith("_vqe") else 0,
              "named_H_requests":backend.matrix_entry_requests,"wall_seconds":perf_counter()-started,"peak_RSS_bytes":rss_peak(),
              "FCI_used":False,"full_Fock_vector_used":False,"hardware_shots":0,"whole_study_complete":False}
    raw_result = json.dumps(result)
    db.execute("UPDATE cells SET status='COMPLETED',result=? WHERE id=?",(raw_result,tid)); db.commit()
    return json.loads(raw_result)  # Same JSON-normalized object on first run/resume.


def run(root, mode, min_free_gib=3.):
    root = Path(root).resolve()
    source = ROOT/"results/standard-v1"
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Protect the old physical-source dataset")
    if any((p/"REPRODUCIBILITY.json").exists() for p in (root,*root.parents)):
        raise ValueError("Do not write inside any sealed experiment")
    root.mkdir(parents=True,exist_ok=True)
    with (root/"run.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not 2.75 <= min_free_gib <= 1000: raise ValueError("Explicit bounded free-space configuration")
        cfg = setting(mode); cfg["free_floor"] = int(min_free_gib*2**30); mf = root/"manifest.json"
        if mf.exists(): assert json.loads(mf.read_text()) == cfg, "Frozen online config drift"
        else:
            if shutil.disk_usage(root).free < cfg["free_floor"]+cfg["output_cap"]: raise RuntimeError("Online output headroom insufficient")
            if mode == "standard":
                quick = ROOT/"results/online-parameter-quick-v1/validation.json"
                assert quick.exists() and json.loads(quick.read_text())["status"] == "ONLINE_PARAMETER_COUNTS_CIRCUITS_DECISIONS_VALIDATED"
            save_json(mf,cfg)
        # Reverify every immutable source artifact; no writes to old output.
        inventory = ROOT/"results/parameter-controls-standard-v1/checkpoint-inventory.json"
        old_audit = ROOT/"results/parameter-controls-standard-v1/source-audit.json"
        if not (root/"source-audit.json").exists():
            pointer = json.loads((Path(cfg["source"])/"REPRODUCIBILITY.json").read_text())
            manifest_path = Path(cfg["source"])/pointer["manifest"]
            assert file_digest(manifest_path) == pointer["manifest_sha256"]
            artifacts = json.loads(manifest_path.read_text())["artifacts"]
            total = 0
            for name, record in artifacts.items():
                p = Path(cfg["source"])/name; h = hashlib.sha256()
                with p.open("rb") as stream:
                    for chunk in iter(lambda:stream.read(2**20),b""): h.update(chunk)
                assert h.hexdigest() == record["sha256"] and p.stat().st_size == record["bytes"], name
                total += record["bytes"]
            save_json(root/"source-audit.json", {"checked":len(artifacts),"bytes":total,"all_match":True,"source_modified":False,
                      "manifest_sha256":pointer["manifest_sha256"],"shared_inventory_sha256":file_digest(inventory),
                      "prior_audit_sha256_for_provenance_only":file_digest(old_audit)})
        db = sqlite3.connect(root/"events.sqlite"); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL"); db.executescript(SCHEMA); db.commit()
        planned = len(cfg["cases"])*len(cfg["seeds"])*len(MODES)
        for case in cfg["cases"]:
            for seed in cfg["seeds"]:
                for method in MODES:
                    size = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
                    if shutil.disk_usage(root).free < cfg["free_floor"] or size > cfg["output_cap"] or rss_peak() > cfg["RSS_cap"]:
                        db.close(); raise RuntimeError("Preserved online resource boundary; no free retry")
                    try: construct(db,cfg,case,seed,method)
                    except Exception as exc:
                        db.execute("UPDATE cells SET status='FAILED_PRESERVED',result=? WHERE id=(SELECT id FROM cells WHERE status='RUNNING' ORDER BY rowid DESC LIMIT 1)",(json.dumps({"error":repr(exc),"cost_unknown_not_zero":True}),)); db.commit()
                    states = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status"))
                    save_json(root/"execution-status.json",{"status":"RUNNING","planned":planned,"terminal":sum(states.values()),"states":states})
                print(json.dumps({"case":case,"seed":seed,"states":states}),flush=True)
        states = dict(db.execute("SELECT status,count(*) FROM cells GROUP BY status")); db.close()
        result = {"status":"ONLINE_PARAMETER_EXECUTION_PENDING_AUDIT","planned":planned,"terminal":sum(states.values()),"states":states,
                  "events_sha256":file_digest(root/"events.sqlite"),"whole_study_complete":False}
        save_json(root/"execution-status.json",result); return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["quick","standard"],default="quick");p.add_argument("--output");p.add_argument("--min-free-gib",type=float,default=3.)
    a=p.parse_args(); print(json.dumps(run(a.output or ROOT/f"results/online-parameter-{a.mode}-v1",a.mode,a.min_free_gib)))
