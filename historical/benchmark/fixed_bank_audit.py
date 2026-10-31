"""All shared/component draws, physical banks and allocation accounting audit."""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from scipy.linalg import eigvalsh
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,rng_for,save_json
from .local_sampler_primitives import LocalQE,unpack_counts64,valid_sector
from .resource_adaptive_sampler import integral_problem
from .sparse_qe_circuit import SparseQECircuit
from .ansatz import resources
from .fixed_bank_allocation import NAMESPACE,METHODS
from .parameter_controls_audit import close
from .raacs_counts_audit import circuit_signature

SUCCESS="FIXED_BANK_COUNTS_ALLOCATION_CIRCUITS_VALIDATED"


def allocation(total,weights):
    w=np.maximum(np.array(weights,float),0)
    if not w.sum():w=np.ones(len(w))
    ideal=total*w/w.sum();result=np.floor(ideal).astype(int)
    for i in sorted(range(len(w)),key=lambda i:(-(ideal[i]-result[i]),i))[:total-int(result.sum())]:result[i]+=1
    return result.tolist()


def audit(root):
    root=Path(root).resolve();cfg=json.loads((root/"manifest.json").read_text());status=json.loads((root/"execution-status.json").read_text());start=perf_counter()
    assert file_digest(root/"events.sqlite")==status["events_sha256"]
    for name,sha in cfg["implementation"].items():assert file_digest(ROOT/name)==sha,name
    assert file_digest(ROOT/"study-v1/FIXED_BANK_ALLOCATION.md")==cfg["spec_sha256"]
    sourceaudit=json.loads((root/"source-audit.json").read_text());assert sourceaudit["checked"]==40944 and sourceaudit["all_match"]
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True)
    rows=[];total_pilot=0;total_final=0;draw_ids=set();physical_checks=set();all_old_sources={};eigen_checks=0
    with threadpool_limits(limits=1):
        for item in cfg["queue"]:
            source=Path(item["source_cell"]);parent=json.loads((source/"result.json").read_text());assert file_digest(source/"events.sqlite")==parent["event_database_sha256"]
            backend,info=integral_problem(Path(cfg["source"])/"references"/item["case"])
            acts=parent["accepted_operators"]
            task={"problem":info,"source_seed":item["seed"],"accepted_source_actions":acts,"source_pilot_shots":parent["pilot_raw_model_shots"],
                  "settings":{k:cfg[k] for k in ("M","R","bank_prefix_cap","pilot_each","repeats","support_cap")},"implementation":cfg["implementation"]}
            bid=digest(task);raw,statuscell,result=db.execute("SELECT task,status,result FROM bases WHERE id=?",(bid,)).fetchone()
            assert json.loads(raw)==task and statuscell=="COMPLETED";r=json.loads(result)
            assert r["source_events_sha256"]==parent["event_database_sha256"] and r["source_result_sha256"]==file_digest(source/"result.json")
            assert r["case"]==item["case"] and r["source_seed"]==item["seed"] and r["base_id"]==bid
            expected=[digest([info["problem_id"],acts[:k]]) for k in range(1,min(len(acts),cfg["bank_prefix_cap"])+1)] or [digest([info["problem_id"],[]])]
            assert r["shallow_members"]==expected and r["deep_member"]==digest([info["problem_id"],acts])
            circuits={};draws={};ci_cache={}
            def physical(cid):
                if cid not in circuits:
                    pid,a,res,prep,sample=db.execute("SELECT problem,actions,resources,prep,sampling FROM circuits WHERE id=?",(cid,)).fetchone();a=json.loads(a);res=json.loads(res)
                    assert pid==info["problem_id"] and cid==digest([pid,a]);ops=[LocalQE(tuple(x["source"]),tuple(x["target"]),x["n_qubits"]) for x in a]
                    sim=SparseQECircuit(info["n_qubits"],info["hf_id"],ops,cfg["support_cap"])
                    qc=QuantumCircuit(info["n_qubits"],name="fixed-bank-"+cid[:12])
                    for q in range(info["n_qubits"]):
                        if info["hf_id"]>>q&1:qc.x(q)
                    for op,x in zip(ops,a):qc=op.append(qc,x["theta"])
                    savedprep=qpy.load(BytesIO(zlib.decompress(prep)))[0];savedsample=qpy.load(BytesIO(zlib.decompress(sample)))[0]
                    assert circuit_signature(qc)==circuit_signature(savedprep)
                    assert resources(savedsample,len(ops),len(ops))==res
                    if cid not in physical_checks:
                        exact=Statevector.from_instruction(savedprep).data;vector=np.zeros_like(exact);vector[sim.ids.astype(np.int64)]=sim.state([x["theta"] for x in a])
                        close(exact,vector,1e-9);close(abs(Statevector.from_instruction(savedsample).data)**2,abs(exact)**2,1e-9);physical_checks.add(cid)
                    circuits[cid]=(sim,[x["theta"] for x in a],res)
                return circuits[cid]
            def solve(obs):
                nonlocal eigen_checks
                order=sorted((x for x in obs if x!=info["hf_id"]),key=lambda x:(-obs[x],x))
                ids=tuple(sorted([info["hf_id"]]+order[:cfg["R"]-1]))
                if ids not in ci_cache:
                    h=backend.matrix(ids,ids);E=float(np.linalg.eigvalsh(h)[0]);close(E,float(eigvalsh(h,check_finite=False)[0]),1e-9)
                    ci_cache[ids]=E;eigen_checks+=1
                return ci_cache[ids],list(ids)
            for did,cid,role,method,rep,member,shots,seed,blob in db.execute("SELECT id,circuit,role,method,rep,member,shots,seed,counts FROM draws WHERE base=? ORDER BY rowid",(bid,)):
                assert did not in draw_ids;draw_ids.add(did)
                assert did==digest([NAMESPACE,bid,cid,role,method,rep,member,shots]) and seed==str(rng_for(NAMESPACE,did)[1])
                sim,theta,res=physical(cid);ids,ns=unpack_counts64(blob);p=abs(sim.state(theta))**2;p/=p.sum();predicted=np.random.default_rng(int(seed)).multinomial(shots,p)
                assert np.array_equal(ids,sim.ids[predicted>0]) and np.array_equal(ns,predicted[predicted>0]) and int(ns.sum())==shots
                assert all(valid_sector(int(x),info["n_qubits"],info["nalpha"],info["nbeta"]) for x in ids)
                obs=dict(zip(map(int,ids),map(int,ns)));draws[did]={"cid":cid,"role":role,"method":method,"rep":rep,"member":member,"shots":shots,"obs":obs}
                if role=="shared_pilot":assert method=="SHARED" and rep==-1;total_pilot+=shots
                else:assert role=="final_component";total_final+=shots
            pilot_ids=[did for did,d in draws.items() if d["role"]=="shared_pilot"]
            assert len(pilot_ids)==len(expected)
            pilots=[]
            for j,did in enumerate(pilot_ids):
                d=draws[did];assert d["cid"]==expected[j] and d["shots"]==cfg["pilot_each"] and d["member"]==j
                E,_=solve(d["obs"]);pilots.append((E,set(d["obs"])-{info["hf_id"]}))
            final_ids=[]
            for mode in METHODS:
                record=json.loads(db.execute("SELECT record FROM policies WHERE base=? AND method=?",(bid,mode)).fetchone()[0])
                if mode=="single_deep_no_pilot":members=[r["deep_member"]];weights=[1.];pilotcharge=0
                else:
                    members=expected;pilotcharge=cfg["pilot_each"]*len(expected)
                    if mode=="single_last":weights=[0.]*(len(expected)-1)+[1.]
                    elif mode=="uniform":weights=[1.]*len(expected)
                    elif mode=="greedy":winner=min(range(len(pilots)),key=lambda i:(pilots[i][0],i));weights=[float(i==winner) for i in range(len(expected))]
                    elif mode=="discovery":weights=[float(len(s-set().union(*(v[1] for j,v in enumerate(pilots) if j!=i)))) for i,(_,s) in enumerate(pilots)]
                    else:weights=[max(0.,info["HF_energy_Ha"]-v[0]) for v in pilots]
                close(weights,record["weights"],1e-11);assert record["members"]==members
                left=cfg["M"]-parent["pilot_raw_model_shots"]-pilotcharge;alloc=allocation(left,record["weights"])
                assert record["allocation"]==alloc and record["final_shots"]==left and record["known_branch_budget"]==cfg["M"]
                assert record["source_pilot_charge"]==parent["pilot_raw_model_shots"] and record["new_pilot_charge"]==pilotcharge
                assert record["pilot_ids_used"]==(pilot_ids if pilotcharge else [])
                res=[physical(cid)[2] for cid in members];used=[v for v,n in zip(res,alloc) if n]
                assert record["maximum_final_CNOT"]==max(v["cnot_count"] for v in used)
                assert record["maximum_final_depth"]==max(v["cnot_depth"] for v in used)
                close(record["weighted_final_CNOT"],sum(n*v["cnot_count"] for n,v in zip(alloc,res))/left)
                close(record["weighted_final_depth"],sum(n*v["cnot_depth"] for n,v in zip(alloc,res))/left)
                assert record["final_gate_shots"]==sum(n*v["cnot_count"] for n,v in zip(alloc,res))
                assert record["pilot_gate_shots_charged"]==(sum(cfg["pilot_each"]*physical(cid)[2]["cnot_count"] for cid in expected) if pilotcharge else 0)
                assert record["final_switches"]==max(0,len(used)-1) and record["hardware_switch_latency_seconds"] is None
                for rep in range(cfg["repeats"]):
                    f=json.loads(db.execute("SELECT record FROM finals WHERE base=? AND method=? AND rep=?",(bid,mode,rep)).fetchone()[0]);obs={}
                    components=[did for did,d in draws.items() if d["method"]==mode and d["rep"]==rep]
                    assert components==f["component_draws"] and len(components)==sum(n>0 for n in alloc)
                    for did in components:
                        d=draws[did];j=d["member"];assert d["cid"]==members[j] and d["shots"]==alloc[j]
                        final_ids.append(did)
                        for x,n in d["obs"].items():obs[x]=obs.get(x,0)+n
                    assert sum(obs.values())==left==f["raw_shots"];E,ids=solve(obs);close(E,f["energy_Ha"]);assert ids==f["selected_ids"] and len(ids)==f["actual_d"]
            assert set(final_ids)=={did for did,d in draws.items() if d["role"]=="final_component"} and len(final_ids)==len(set(final_ids))
            actual_pilot=sum(d["shots"] for d in draws.values() if d["role"]=="shared_pilot");actual_final=sum(d["shots"] for d in draws.values() if d["role"]=="final_component")
            assert actual_pilot==r["new_shared_pilot_shots_actual_once"] and actual_final==r["new_actual_final_shots"]
            assert actual_pilot+actual_final==r["new_total_model_shots"]
            assert r["source_pilot_shots"]==parent["pilot_raw_model_shots"] and r["unique_CI_solves"]==len(ci_cache)
            assert not r["FCI_used"] and not r["full_Fock_vector_used"] and r["hardware_shots"]==0 and r["physical_source_growth_NOT_avoided"]
            all_old_sources[item["source_cell"]]=parent["pilot_raw_model_shots"]
            rows.append({"case":item["case"],"seed":item["seed"],"base_id":bid,"draws":len(draws),"new_pilot_shots_once":actual_pilot,"new_final_shots":actual_final,"source_pilot_charge_not_new":parent["pilot_raw_model_shots"]})
            print(json.dumps({"case":item["case"],"seed":item["seed"],"validated_bases":len(rows)}),flush=True)
    assert len(draw_ids)==db.execute("SELECT count(*) FROM draws").fetchone()[0]
    assert len(rows)==status["bases"]==len(cfg["queue"]) and status["policy_cells"]==len(rows)*len(METHODS)
    db.close();result={"status":SUCCESS,"bases":len(rows),"policy_cells":len(rows)*len(METHODS),"raw_component_and_pilot_draws_rebuilt":len(draw_ids),
                      "physical_QPY_states_and_sampling_distributions_checked":2*len(physical_checks),"unique_independent_CI_checks":eigen_checks,
                      "new_pilot_shots_actual_once":total_pilot,"new_final_model_shots":total_final,"inherited_pilot_shots_unique_NOT_new":sum(all_old_sources.values()),
                      "rows":rows,"events_sha256":status["events_sha256"],"manifest_sha256":file_digest(root/"manifest.json"),"auditor_sha256":file_digest(Path(__file__)),
                      "hardware_shots":0,"FCI_used":False,"wall_seconds":perf_counter()-start,"whole_study_complete":False}
    path=root/"validation.json"
    if path.exists():path=root/f"validation-recheck-{digest(result)[:16]}.json"
    save_json(path,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();r=audit(a.root);print(json.dumps({k:v for k,v in r.items() if k!="rows"}))
