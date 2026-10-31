"""Same-bank paid allocation controls with explicit inherited construction cost."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import QuantumCircuit
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,rng_for,save_json
from .ansatz import circuit_bytes,measurement_equivalent_circuit,resources
from .local_sampler_primitives import LocalQE,pack_counts64,valid_sector
from .resource_adaptive_sampler import integral_problem,allocation_counts
from .sparse_qe_circuit import SparseQECircuit
from .parameter_controls import source_queue
from .study.core import rss_peak

METHODS=("single_last","uniform","greedy","discovery","energy_guided","single_deep_no_pilot")
NAMESPACE="RAACS-fixed-shallow-bank-allocation-v1"
SCHEMA="""
CREATE TABLE IF NOT EXISTS bases(id TEXT PRIMARY KEY,task TEXT,status TEXT,result TEXT);
CREATE TABLE IF NOT EXISTS circuits(id TEXT PRIMARY KEY,problem TEXT,actions TEXT,resources TEXT,prep BLOB,sampling BLOB);
CREATE TABLE IF NOT EXISTS draws(id TEXT PRIMARY KEY,base TEXT,circuit TEXT,role TEXT,method TEXT,rep INTEGER,member INTEGER,shots INTEGER,seed TEXT,counts BLOB);
CREATE TABLE IF NOT EXISTS policies(base TEXT,method TEXT,record TEXT,PRIMARY KEY(base,method));
CREATE TABLE IF NOT EXISTS finals(base TEXT,method TEXT,rep INTEGER,record TEXT,PRIMARY KEY(base,method,rep));
"""


def allocation_weights(method,pilots,hf,hf_energy):
    if not pilots:raise ValueError("Nonempty physical bank")
    count=len(pilots)
    if method=="single_last":return [0.]*(count-1)+[1.]
    if method=="uniform":return [1.]*count
    if method=="greedy":
        winner=min(range(count),key=lambda i:(pilots[i]["energy_Ha"],i))
        return [float(i==winner) for i in range(count)]
    if method=="discovery":
        supports=[set(p["observed_ids"])-{hf} for p in pilots]
        return [float(len(s-set().union(*(v for j,v in enumerate(supports) if j!=i)))) for i,s in enumerate(supports)]
    if method=="energy_guided":return [max(0.,hf_energy-p["energy_Ha"]) for p in pilots]
    raise ValueError("Unregistered allocation method")


def setting(mode):
    source,queue=source_queue(mode)
    files=("fixed_bank_allocation.py","sparse_qe_circuit.py","parameter_controls.py","local_sampler_primitives.py","integral_hamiltonian.py","resource_adaptive_sampler.py","ansatz.py","common.py")
    return {"schema":NAMESPACE,"mode":mode,"source":str(ROOT/"results/standard-v1"),"construction":str(source),"queue":queue,"methods":list(METHODS),
            "M":10000,"R":64,"bank_prefix_cap":4,"pilot_each":64,"repeats":4 if mode=="quick" else 20,"support_cap":512,
            "free_floor":int(2.75*2**30),"output_cap":64*2**20,"RSS_cap":2*2**30,
            "implementation":{str(Path("benchmark")/p):file_digest(ROOT/"benchmark"/p) for p in files},
            "spec_sha256":file_digest(ROOT/"study-v1/FIXED_BANK_ALLOCATION.md"),"hardware_shots":0}


def run_base(db,cfg,item):
    source=Path(item["source_cell"]);parent=json.loads((source/"result.json").read_text())
    assert file_digest(source/"events.sqlite")==parent["event_database_sha256"]
    backend,info=integral_problem(Path(cfg["source"])/"references"/item["case"])
    all_actions=parent["accepted_operators"];inherited=parent["pilot_raw_model_shots"]
    task={"problem":info,"source_seed":item["seed"],"accepted_source_actions":all_actions,"source_pilot_shots":inherited,
          "settings":{k:cfg[k] for k in ("M","R","bank_prefix_cap","pilot_each","repeats","support_cap")},"implementation":cfg["implementation"]}
    bid=digest(task)
    existing=db.execute("SELECT status,result FROM bases WHERE id=?",(bid,)).fetchone()
    if existing:
        if existing[0]=="COMPLETED":return json.loads(existing[1])
        raise RuntimeError("Partial allocation base preserved, no free retry")
    db.execute("INSERT INTO bases VALUES(?,?,?,NULL)",(bid,json.dumps(task),"RUNNING"));db.commit()
    started=perf_counter();bank=[];prepared={};ci_cache={};unique_solves=0
    def prepare(actions):
        cid=digest([info["problem_id"],actions])
        if cid not in prepared:
            ops=[LocalQE(tuple(a["source"]),tuple(a["target"]),a["n_qubits"]) for a in actions]
            sim=SparseQECircuit(info["n_qubits"],info["hf_id"],ops,cfg["support_cap"])
            prep=QuantumCircuit(info["n_qubits"],name="fixed-bank-"+cid[:12])
            for q in range(info["n_qubits"]):
                if info["hf_id"]>>q&1:prep.x(q)
            for op,a in zip(ops,actions):prep=op.append(prep,a["theta"])
            sample,_=measurement_equivalent_circuit(prep);res=resources(sample,len(ops),len(ops))
            prepared[cid]=(sim,[a["theta"] for a in actions],res)
            db.execute("INSERT OR IGNORE INTO circuits VALUES(?,?,?,?,?,?)",(cid,info["problem_id"],json.dumps(actions),json.dumps(res),
                       zlib.compress(circuit_bytes(prep),6),zlib.compress(circuit_bytes(sample),6)))
        return cid
    for k in range(1,min(len(all_actions),cfg["bank_prefix_cap"])+1):bank.append(prepare(all_actions[:k]))
    if not bank:bank=[prepare([])]
    deep=prepare(all_actions)
    def ci(obs):
        nonlocal unique_solves
        ranked=sorted((x for x in obs if x!=info["hf_id"] and valid_sector(x,info["n_qubits"],info["nalpha"],info["nbeta"])),key=lambda x:(-obs[x],x))
        ids=tuple(sorted([info["hf_id"]]+ranked[:cfg["R"]-1]))
        if ids not in ci_cache:ci_cache[ids]=float(np.linalg.eigvalsh(backend.matrix(ids,ids))[0]);unique_solves+=1
        return {"energy_Ha":ci_cache[ids],"selected_ids":list(ids),"actual_d":len(ids)}
    def draw(cid,role,method,rep,member,shots):
        did=digest([NAMESPACE,bid,cid,role,method,rep,member,shots]);seed=rng_for(NAMESPACE,did)[1]
        sim,theta,res=prepared[cid];ids,ns=sim.counts(theta,shots,seed)
        db.execute("INSERT INTO draws VALUES(?,?,?,?,?,?,?,?,?,?)",(did,bid,cid,role,method,rep,member,shots,str(seed),pack_counts64(ids,ns)));db.commit()
        return did,dict(zip(map(int,ids),map(int,ns)))
    pilots=[];pilot_ids=[]
    for j,cid in enumerate(bank):
        did,obs=draw(cid,"shared_pilot","SHARED",-1,j,cfg["pilot_each"]);pilot_ids.append(did)
        pilots.append({**ci(obs),"observed_ids":sorted(obs)})
    common_pilot=cfg["pilot_each"]*len(bank);new_final=0
    for method in METHODS:
        if method=="single_deep_no_pilot":members=[deep];weights=[1.];pilot_charge=0
        else:members=bank;weights=allocation_weights(method,pilots,info["hf_id"],info["HF_energy_Ha"]);pilot_charge=common_pilot
        remaining=cfg["M"]-inherited-pilot_charge
        assert remaining>0
        allocation=allocation_counts(remaining,weights)
        used=[prepared[cid][2] for cid,n in zip(members,allocation) if n]
        record={"members":members,"weights":weights,"allocation":allocation,"pilot_ids_used":pilot_ids if pilot_charge else [],
                "source_pilot_charge":inherited,"new_pilot_charge":pilot_charge,"final_shots":remaining,"known_branch_budget":cfg["M"],
                "maximum_final_CNOT":max(r["cnot_count"] for r in used),"maximum_final_depth":max(r["cnot_depth"] for r in used),
                "weighted_final_CNOT":sum(n*prepared[cid][2]["cnot_count"] for cid,n in zip(members,allocation))/remaining,
                "weighted_final_depth":sum(n*prepared[cid][2]["cnot_depth"] for cid,n in zip(members,allocation))/remaining,
                "final_gate_shots":sum(n*prepared[cid][2]["cnot_count"] for cid,n in zip(members,allocation)),
                "pilot_gate_shots_charged":sum(cfg["pilot_each"]*prepared[cid][2]["cnot_count"] for cid in bank) if pilot_charge else 0,
                "final_switches":max(0,len(used)-1),"hardware_switch_latency_seconds":None,
                "source_full_growth_NOT_avoided":True}
        db.execute("INSERT INTO policies VALUES(?,?,?)",(bid,method,json.dumps(record)));db.commit()
        for rep in range(cfg["repeats"]):
            merged={};drawids=[]
            for j,(cid,shots) in enumerate(zip(members,allocation)):
                if not shots:continue
                did,obs=draw(cid,"final_component",method,rep,j,shots);drawids.append(did)
                for x,n in obs.items():merged[x]=merged.get(x,0)+n
                new_final+=shots
            assert sum(merged.values())==remaining
            result={**ci(merged),"component_draws":drawids,"raw_shots":remaining}
            db.execute("INSERT INTO finals VALUES(?,?,?,?)",(bid,method,rep,json.dumps(result)));db.commit()
    result={"status":"COMPLETED","case":item["case"],"source_seed":item["seed"],"base_id":bid,"source_events_sha256":parent["event_database_sha256"],
            "source_result_sha256":file_digest(source/"result.json"),"shallow_members":bank,"deep_member":deep,"source_pilot_shots":inherited,
            "new_shared_pilot_shots_actual_once":common_pilot,"new_actual_final_shots":new_final,"new_total_model_shots":common_pilot+new_final,
            "physical_source_growth_NOT_avoided":True,"unique_CI_solves":unique_solves,"classical_wall_seconds":perf_counter()-started,"RSS_HWM_bytes":rss_peak(),
            "FCI_used":False,"full_Fock_vector_used":False,"hardware_shots":0,"whole_study_complete":False}
    raw=json.dumps(result);db.execute("UPDATE bases SET status='COMPLETED',result=? WHERE id=?",(raw,bid));db.commit();return json.loads(raw)


def run(root,mode):
    root=Path(root).resolve();source=ROOT/"results/standard-v1"
    if root==source or source in root.parents or root in source.parents:raise ValueError("Protect source")
    if any((p/"REPRODUCIBILITY.json").exists() for p in (root,*root.parents)):raise ValueError("Protect sealed experiments")
    root.mkdir(parents=True,exist_ok=True)
    with (root/"run.lock").open("a") as lock,threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=setting(mode);path=root/"manifest.json"
        if path.exists():assert json.loads(path.read_text())==cfg
        else:
            if shutil.disk_usage(root).free<cfg["free_floor"]+cfg["output_cap"]:raise RuntimeError("Insufficient bounded allocation storage")
            if mode=="standard":assert json.loads((ROOT/"results/fixed-bank-allocation-quick-v1/validation.json").read_text())["status"]=="FIXED_BANK_COUNTS_ALLOCATION_CIRCUITS_VALIDATED"
            save_json(path,cfg)
        if not (root/"source-audit.json").exists():
            pointer=json.loads((source/"REPRODUCIBILITY.json").read_text());manifest=source/pointer["manifest"]
            assert file_digest(manifest)==pointer["manifest_sha256"]
            artifacts=json.loads(manifest.read_text())["artifacts"];total=0
            for name,record in artifacts.items():
                p=source/name;h=hashlib.sha256()
                with p.open("rb") as stream:
                    for chunk in iter(lambda:stream.read(2**20),b""):h.update(chunk)
                assert h.hexdigest()==record["sha256"] and p.stat().st_size==record["bytes"],name
                total+=record["bytes"]
            save_json(root/"source-audit.json",{"checked":len(artifacts),"bytes":total,"all_match":True,"source_modified":False,"manifest_sha256":pointer["manifest_sha256"]})
        db=sqlite3.connect(root/"events.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL");db.executescript(SCHEMA)
        for item in cfg["queue"]:
            if shutil.disk_usage(root).free<cfg["free_floor"] or sum(p.stat().st_size for p in root.iterdir() if p.is_file())>cfg["output_cap"] or rss_peak()>cfg["RSS_cap"]:
                db.close();raise RuntimeError("Preserved allocation resource boundary")
            run_base(db,cfg,item)
            n=db.execute("SELECT count(*) FROM bases WHERE status='COMPLETED'").fetchone()[0]
            save_json(root/"execution-status.json",{"status":"RUNNING","bases":n,"planned_bases":len(cfg["queue"]),"policy_cells":n*len(METHODS)})
            print(json.dumps({"case":item["case"],"seed":item["seed"],"bases":n}),flush=True)
        db.close();result={"status":"FIXED_BANK_ALLOCATION_PENDING_AUDIT","bases":len(cfg["queue"]),"policy_cells":len(cfg["queue"])*len(METHODS),
                          "events_sha256":file_digest(root/"events.sqlite"),"whole_study_complete":False}
        save_json(root/"execution-status.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["quick","standard"],default="quick");p.add_argument("--output");a=p.parse_args()
    print(json.dumps(run(a.output or ROOT/f"results/fixed-bank-allocation-{a.mode}-v1",a.mode)))
