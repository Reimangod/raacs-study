"""Read-only all-record online audit; no references/FCI labels or free retries."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits
from .common import ROOT, digest, file_digest, rng_for, save_json
from .online_parameter_controls import decoded, task_identity, NAMESPACE
from .local_sampler_primitives import LocalQE, local_proposals, unpack_counts64, valid_sector
from .integral_connections import IntegralConnections
from .resource_adaptive_sampler import integral_problem
from .sparse_qe_circuit import SparseQECircuit
from .ansatz import resources, measurement_equivalent_circuit
from .parameter_controls_audit import close
from .raacs_counts_audit import circuit_signature

SUCCESS = "ONLINE_PARAMETER_COUNTS_CIRCUITS_DECISIONS_VALIDATED"


def ops_from(actions): return [LocalQE(tuple(a["source"]),tuple(a["target"]),a["n_qubits"]) for a in actions]


def audit(root):
    root = Path(root).resolve(); started = perf_counter()
    cfg = json.loads((root/"manifest.json").read_text()); status = json.loads((root/"execution-status.json").read_text())
    assert file_digest(root/"events.sqlite") == status["events_sha256"]
    for name, sha in cfg["implementation"].items(): assert file_digest(ROOT/name) == sha, name
    assert file_digest(ROOT/"vendor/ceo/adaptvqe/circuits.py") == cfg["vendor_circuit_sha256"]
    assert file_digest(ROOT/"study-v1/ONLINE_PARAMETER_CONTROLS.md") == cfg["spec_sha256"]
    audit_source = json.loads((root/"source-audit.json").read_text())
    assert audit_source["all_match"] and audit_source["checked"] == 40944
    rows = []; failures = []; all_draw_ids = set(); total_circuits = 0; all_trace_calls = 0; raw_pilots = 0; raw_final = 0; final_qpy = 0; physical_quick = 0
    db = sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True)
    with threadpool_limits(limits=1):
        for case in cfg["cases"]:
            backend, info = integral_problem(Path(cfg["source"])/"references"/case)
            circuits = {}; sims = {}; ci_cache = {}; physical_checked = set()
            def circuit(cid, with_h=False):
                if cid not in circuits:
                    pid, blob, res_raw = db.execute("SELECT problem,actions,resources FROM circuits WHERE id=?",(cid,)).fetchone()
                    acts = decoded(blob); assert pid == info["problem_id"]
                    assert cid == digest([pid,acts])
                    qc = QuantumCircuit(info["n_qubits"],name="online-raacs-"+cid[:12])
                    for q in range(info["n_qubits"]):
                        if info["hf_id"] >> q & 1: qc.x(q)
                    ops = ops_from(acts)
                    for op,a in zip(ops,acts): qc = op.append(qc,a["theta"])
                    sample,_ = measurement_equivalent_circuit(qc)
                    res = resources(sample,len(ops),len(ops)); assert res == json.loads(res_raw)
                    circuits[cid] = (acts,ops,qc,sample,res)
                acts,ops,qc,sample,res = circuits[cid]
                key = (tuple(o.identity for o in ops),with_h)
                if key not in sims: sims[key] = SparseQECircuit(info["n_qubits"],info["hf_id"],ops,cfg["support_cap"],backend if with_h else None)
                return acts,ops,qc,sample,res,sims[key]
            def ci_from(obs):
                ranked = sorted((x for x in obs if x != info["hf_id"] and valid_sector(x,info["n_qubits"],info["nalpha"],info["nbeta"])),key=lambda x:(-obs[x],x))
                ids = tuple(sorted([info["hf_id"]]+ranked[:cfg["R"]-1]))
                if ids not in ci_cache:
                    h = backend.matrix(ids,ids); close(h,h.T.conj(),1e-10)
                    values,vecs = np.linalg.eigh(h)
                    ci_cache[ids] = {"energy_Ha":float(values[0]),"selected_ids":list(ids),"coefficients":vecs[:,0]}
                return ci_cache[ids]
            for seed in cfg["seeds"]:
                for mode in cfg["methods"]:
                    task = task_identity(cfg,info,seed,mode); tid = digest(task)
                    saved = db.execute("SELECT task,status,result FROM cells WHERE id=?",(tid,)).fetchone()
                    if not saved or saved[1] != "COMPLETED":
                        failures.append({"case":case,"seed":seed,"mode":mode,"preserved_status":saved[1] if saved else "MISSING"}); continue
                    assert json.loads(saved[0]) == task
                    r = json.loads(saved[2]); assert (r["case"],r["seed"],r["mode"],r["task_id"]) == (case,seed,mode,tid)
                    draws = {}; pilots = 0; final = []; trace_calls = 0
                    for did,cid,role,step,extra_raw,shots,rngseed,blob,E,selected_raw in db.execute("SELECT id,circuit,role,step,extra,shots,seed,counts,energy,selected FROM draws WHERE cell=? ORDER BY rowid",(tid,)):
                        acts,ops,qc,sample,res,sim = circuit(cid)
                        theta = [a["theta"] for a in acts]; extra = json.loads(extra_raw)
                        assert role in ("initial","candidate","final") and did not in all_draw_ids
                        all_draw_ids.add(did)
                        assert did == digest([tid,cid,role,step,extra,shots]) and str(rng_for(NAMESPACE,did)[1]) == rngseed
                        ids, ns = unpack_counts64(blob)
                        assert len(ids) == len(set(map(int,ids))) and int(ns.sum()) == shots and all(int(n)>0 for n in ns)
                        p = abs(sim.state(theta))**2; p /= p.sum()
                        predicted = np.random.default_rng(int(rngseed)).multinomial(shots,p)
                        assert np.array_equal(ids,sim.ids[predicted>0]) and np.array_equal(ns,predicted[predicted>0])
                        obs = dict(zip(map(int,ids),map(int,ns)))
                        assert all(valid_sector(x,info["n_qubits"],info["nalpha"],info["nbeta"]) for x in obs)
                        ci = ci_from(obs); close(ci["energy_Ha"],E); assert ci["selected_ids"] == json.loads(selected_raw)
                        draws[did] = {"circuit":cid,"actions":acts,"ops":ops,"ci":ci,"counts":obs,"res":res,"role":role,"step":step,"extra":extra,"shots":shots}
                        if role == "final": final.append((did,ci))
                        else: pilots += shots; assert shots == cfg["probe"]
                        if cfg["mode"] == "quick" and cid not in physical_checked:
                            exact = Statevector.from_instruction(qc).data
                            fresh = np.zeros_like(exact);fresh[sim.ids.astype(np.int64)] = sim.state(theta)
                            close(exact,fresh,1e-9);close(abs(Statevector.from_instruction(sample).data)**2,abs(exact)**2,1e-9)
                            physical_checked.add(cid);physical_quick += 2
                    optrecords = {}
                    for oid,step,initialcid,raw in db.execute("SELECT id,step,circuit,trace FROM optimizations WHERE cell=?",(tid,)):
                        assert mode.endswith("_vqe")
                        control = decoded(raw); acts,ops,_,_,_,sim = circuit(initialcid,True)
                        initial = [a["theta"] for a in acts]; close(initial,control["initial_theta"],0)
                        assert oid == digest([tid,step,ops[-1].identity,initial]) and initial[-1] == np.pi/4
                        indices = list(range(len(ops))) if mode == "full_vqe" else list(range(max(0,len(ops)-(2 if mode=="blockwise_vqe" else 1)),len(ops)))
                        assert indices == control["optimized_variable_indices"]
                        assert control["n_optimized_variables"] == len(indices) and control["n_independent_ansatz_parameters"] == len(ops)
                        assert control["optimizer_initial_state"] == "RESET_IDENTITY_BFGS" and control["QPU_objective_measurement_shots"] is None
                        assert control["actual_objective_evaluations"] == control["actual_complete_gradient_evaluations"] == len(control["trace"]) <= cfg["BFGS"]["max_calls"]
                        close(control["trace"][0]["theta"],initial,0)
                        fixed = [j for j in range(len(ops)) if j not in indices]
                        for index,t in enumerate(control["trace"]):
                            assert t["call"] == index+1; close(np.array(t["theta"])[fixed],np.array(initial)[fixed],0)
                            energy,g = sim.energy_gradient(t["theta"])
                            close(energy,t["energy_Ha"]);close(max(abs(g[indices]),default=0.),t["active_gradient_inf"])
                            close(max(abs(g),default=0.),t["full_gradient_inf"])
                        best = min(control["trace"],key=lambda t:t["energy_Ha"])
                        close(best["theta"],control["theta"],0)
                        energy,g = sim.energy_gradient(control["theta"])
                        assert control["active_stationary"] == (max(abs(g[indices]),default=0.) <= cfg["BFGS"]["gtol"])
                        assert control["full_stationary"] == (max(abs(g),default=0.) <= cfg["BFGS"]["gtol"])
                        theta = np.array(control["theta"])
                        for j in indices:
                            plus=theta.copy();minus=theta.copy();plus[j]+=1e-6;minus[j]-=1e-6
                            close((sim.energy_gradient(plus)[0]-sim.energy_gradient(minus)[0])/2e-6,g[j],4e-7)
                        optrecords[(step,ops[-1].identity)] = (acts,control)
                        trace_calls += len(control["trace"])
                    initial_draws = [d for d in draws if draws[d]["role"] == "initial"]
                    assert len(initial_draws) == 1
                    current = initial_draws[0]; assert draws[current]["actions"] == [] and draws[current]["step"] == 0
                    measured_ids = []; optkeys = set(); step_count = 0
                    for step,blob in db.execute("SELECT step,record FROM decisions WHERE cell=? ORDER BY step",(tid,)):
                        d=decoded(blob);step_count+=1;assert step == step_count and d["parent_draw"] == current
                        parent = draws[current];assert d["parent_actions"] == parent["actions"]
                        proposals,scope = local_proposals(backend,IntegralConnections(backend,parent_cap=cfg["source_parent_cap"]),parent["ci"],cfg["target_cap"],cfg["candidate_cap"])
                        assert scope == d["scope"]
                        ranks=[]
                        for proposal in proposals:
                            op=proposal["operator"]
                            nominal=parent["actions"]+[{**asdict(op),"theta":float(np.pi/8)}]
                            nominalid=digest([info["problem_id"],nominal]);_,_,_,_,res,_=circuit(nominalid)
                            cost=1+max(0,res["cnot_count"]-parent["res"]["cnot_count"])+.5*max(0,res["cnot_depth"]-parent["res"]["cnot_depth"])
                            action=set(op.action(dict(zip(parent["ci"]["selected_ids"],parent["ci"]["coefficients"]))))
                            novelty=len(action-set(parent["counts"])-{info["hf_id"]})/max(1,len(action))
                            ranks.append((op,proposal["residual_priority"]*(1+novelty)/cost))
                        # Independent NumPy vs producer SciPy eigenvectors agree
                        # to ~1e-15 but can reverse nearly tied residual scores.
                        # Verify every score independently within numerical
                        # tolerance, then replay the EXACT stored-score order.
                        # Do not demand bit-identical order from two LAPACK APIs.
                        fresh_scores = {op:score for op,score in ranks}
                        registered_ranks = []
                        assert len(ranks)==len(d["ranked"])
                        for savedrank in d["ranked"]:
                            x=savedrank["operator"];op=LocalQE(tuple(x["source"]),tuple(x["target"]),x["n_qubits"])
                            assert op in fresh_scores
                            close(fresh_scores[op],savedrank["score"],1e-11)
                            registered_ranks.append((op,savedrank["score"]))
                        assert len(set(op for op,_ in registered_ranks))==len(ranks)
                        assert registered_ranks==sorted(registered_ranks,key=lambda x:(-x[1],x[0]))
                        ranks=registered_ranks
                        eligible=[]
                        expected_pairs=[(op,float(angle)) for op,_ in ranks[:cfg["measured_candidates"]] for angle in ((-np.pi/4,-np.pi/8,np.pi/8,np.pi/4) if mode=="qsci_local_grid" else (np.pi/4,))]
                        assert len(d["measured"])<=len(expected_pairs)
                        for (op,angle),entry in zip(expected_pairs,d["measured"]):
                            did=entry["draw"];m=draws[did]; measured_ids.append(did)
                            assert m["role"]=="candidate" and m["step"]==step and m["extra"]==[op.identity,angle]
                            initial=parent["actions"]+[{**asdict(op),"theta":angle}]
                            if mode.endswith("_vqe"):
                                optacts,control=optrecords[(step,op.identity)];optkeys.add((step,op.identity))
                                assert optacts==json.loads(json.dumps(initial))
                                expected=[{**a,"theta":t} for a,t in zip(initial,control["theta"])]
                            else: expected=initial
                            assert m["actions"]==json.loads(json.dumps(expected))
                            gain=parent["ci"]["energy_Ha"]-m["ci"]["energy_Ha"]
                            work=len(m["ci"]["selected_ids"])**2
                            utility=gain/(1+max(0,m["res"]["cnot_count"]-parent["res"]["cnot_count"])+work/64)
                            close(gain,entry["gain"]);close(utility,entry["utility"]);assert work==entry["CI_matrix_requests"]
                            # As for proposal ranking: independently verify the
                            # physical numeric values, then replay exact saved
                            # utilities/ties, not a different LAPACK rounding.
                            if entry["gain"]>cfg["gain_threshold_Ha"]:eligible.append((did,entry["utility"],entry["gain"],op))
                        best=max(eligible,key=lambda a:(a[1],a[2],-draws[a[0]]["res"]["cnot_count"],a[3].source,a[3].target,tuple(x["theta"] for x in draws[a[0]]["actions"]))) if eligible else None
                        assert d["winner"]==(best[0] if best else None)
                        if best:current=best[0]
                        else:assert step==r["decisions"]
                    assert set(optrecords)==optkeys
                    assert len(measured_ids)==len(set(measured_ids)) and set(measured_ids)=={did for did,d in draws.items() if d["role"]=="candidate"}
                    assert r["accepted_operators"]==draws[current]["actions"] and r["final_circuit"]==draws[current]["circuit"]
                    assert r["decisions"]==step_count and step_count<=cfg["steps"]
                    assert pilots==r["pilot_shots"] and r["final_shots_per_rep"]+pilots==cfg["M"]
                    assert r["final_shots_per_rep"]>=cfg["final_reserve"] and len(final)==cfg["final_repeats"]==r["final_repeats"]
                    assert r["new_actual_model_shots"]==pilots+cfg["final_repeats"]*r["final_shots_per_rep"]
                    for rep,((did,ci),entry) in enumerate(zip(final,r["final"])):
                        assert draws[did]["circuit"]==r["final_circuit"] and draws[did]["extra"]==rep
                        assert draws[did]["shots"]==r["final_shots_per_rep"] and did==entry["draw"]
                        close(ci["energy_Ha"],entry["energy_Ha"]);assert len(ci["selected_ids"])==entry["actual_d"]
                    assert trace_calls==r["actual_classical_objective_calls"]==r["actual_full_gradient_calls"]
                    assert (r["VQE_QPU_objective_shots"] is None)==mode.endswith("_vqe")
                    prep_blob,sample_blob=db.execute("SELECT preparation,sampling FROM final_qpy WHERE cell=?",(tid,)).fetchone()
                    prep=qpy.load(BytesIO(zlib.decompress(prep_blob)))[0];sample=qpy.load(BytesIO(zlib.decompress(sample_blob)))[0]
                    acts,ops,qc,_,res,sim=circuit(r["final_circuit"])
                    assert circuit_signature(prep)==circuit_signature(qc)
                    exact=Statevector.from_instruction(prep).data;fresh=np.zeros_like(exact)
                    fresh[sim.ids.astype(np.int64)]=sim.state([a["theta"] for a in acts])
                    close(exact,fresh,1e-9);close(abs(Statevector.from_instruction(sample).data)**2,abs(exact)**2,1e-9)
                    assert resources(sample,len(ops),len(ops))==res==r["sampling_resources"]
                    assert not r["FCI_used"] and not r["full_Fock_vector_used"] and r["hardware_shots"]==0
                    final_qpy+=2;raw_pilots+=pilots;raw_final+=r["final_shots_per_rep"]*cfg["final_repeats"];all_trace_calls+=trace_calls
                    rows.append({"case":case,"seed":seed,"mode":mode,"draws":len(draws),"pilots":pilots,"objective_calls":trace_calls,"accepted_steps":len(acts)})
            total_circuits+=len(circuits)
            print(json.dumps({"case":case,"cells_checked":len(rows),"failures":len(failures)}),flush=True)
    assert len(rows)+len(failures)==status["planned"]==len(cfg["cases"])*len(cfg["seeds"])*len(cfg["methods"])
    assert len(all_draw_ids)==db.execute("SELECT count(*) FROM draws").fetchone()[0]
    db.close()
    result={"status":SUCCESS if not failures else "ONLINE_PARAMETER_VALIDATION_MISSING_OR_FAILED_CELLS",
            "validated":len(rows),"failures":failures,"rows":rows,"raw_draws_rebuilt":len(all_draw_ids),"circuits_resources_rebuilt":total_circuits,
            "final_QPY_states_and_distributions_checked":final_qpy,"extra_quick_physical_QPY_checks":physical_quick,
            "objective_trace_calls_recomputed":all_trace_calls,"new_pilot_model_shots":raw_pilots,"new_final_model_shots":raw_final,
            "events_sha256":status["events_sha256"],"manifest_sha256":file_digest(root/"manifest.json"),"auditor_sha256":file_digest(Path(__file__)),
            "full_vectors_small_auditor_only_not_policy":True,"FCI_used":False,"hardware_shots":0,"wall_seconds":perf_counter()-started,"whole_study_complete":False}
    result["independent_ranking_score_tolerance"] = 1e-11
    result["exact_recorded_score_order_verified"] = True
    path=root/"validation.json"
    if path.exists():path=root/f"validation-recheck-{digest(result)[:16]}.json"
    save_json(path,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();r=audit(a.root)
    print(json.dumps({k:v for k,v in r.items() if k!="rows"}))
    if r["failures"]:raise SystemExit(1)
