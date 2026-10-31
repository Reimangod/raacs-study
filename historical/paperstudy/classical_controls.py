"""HF and native threshold-HCI controls on the same physical tensors.

No saved CI coefficients or reference energies enter selection. HCI is the
bounded variational stage, not optimized SHCI/PT2 or a matched-shot algorithm.
"""
import json
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from benchmark.integral_hamiltonian import IntegralHamiltonian
from benchmark.threshold_hci import HeatBathConnections,HCISpec,variational
from .budget import BudgetLedger
from .ci import solve,interleaved_to_blocked
from .common_execution import load_physical
from .input_adapter import array_bytes
from .io import ROOT,atomic_bytes,save_json,digest,sha256

CASES=("NAPH20-P1","N2-32-R1p2-P2","N2-32-R2p4-P2","FE2S2-40-P2")


def run():
    root=ROOT/"classical-controls-v1"
    spec={"cases":list(CASES),"HCI_epsilon1_Ha":[.01,.003,.001],"HCI_dimension_cap":512,
          "HCI_max_iterations":32,"HCI_wall_cap_seconds":120,"is_SHCI_or_PT2":False,
          "source_sha256":sha256((ROOT.parents[2]/"ceo-sampler-benchmark/benchmark/threshold_hci.py").read_bytes()),
          "HF_budgets":[10000,100000,1000000],"HF_seeds":[101,211,307,401,503,601,701,809,907,1009],
          "source":"Existing qualified integral-driven threshold-HCI implementation; all threshold-qualified parents participate",
          "claim_scope":"Classical variational threshold controls, not optimized Dice/SHCI and not a ground-state reference"}
    frozen=root/"spec.json"
    if frozen.exists() and json.loads(frozen.read_text())!=spec:raise RuntimeError("Classical control spec changed")
    save_json(frozen,spec)
    rows=[]
    for case in CASES:
        m,h,g=load_physical(case)
        hf=interleaved_to_blocked(m["hf_id"],m["norb"])
        if (hf&((1<<m["norb"])-1)).bit_count()!=m["nalpha"] or (hf>>m["norb"]).bit_count()!=m["nbeta"]:
            raise RuntimeError("HF bit convention failed")
        ledger=BudgetLedger(ROOT/"common-budget.sqlite")
        for B in spec["HF_budgets"]:
            for seed in spec["HF_seeds"]:
                out=root/case/"HF"/(str(B)+"-"+str(seed))
                if (out/"result.json").exists():continue
                identity=digest(["paper-HF-common-v1",m["problem_id"],B,seed])
                ledger.register(identity,B)
                request={"problem_id":m["problem_id"],"HF_blocked_id":hf,"shots":B,"seed":seed}
                status=ledger.reserve(identity,"HF-final",request,B)
                filename=out/"final-counts.npz"
                if not filename.exists():
                    if status!="NEW_RESERVATION":raise RuntimeError("Unknown HF draw; no duplicate model cost")
                    atomic_bytes(filename,array_bytes({"blocked_ids":np.array([hf],np.uint64),"counts":np.array([B],np.uint64)}))
                ledger.complete(identity,"HF-final",B,sha256(filename.read_bytes()))
                readouts=[solve(h,g,m["core_energy"],(m["nalpha"],m["nbeta"]),np.array([hf],np.uint64),np.array([B],np.uint64),r) for r in (8,16,32)]
                if max(abs(x["energy_Ha"]-m["hf_energy_Ha"]) for x in readouts)>1e-8:raise RuntimeError("HF energy mismatch")
                save_json(out/"result.json",{"status":"COMPLETED","case_id":case,"problem_id":m["problem_id"],
                    "method_id":"HF_ONLY","budget":B,"seed":seed,"final_CI":readouts,"budget_ledger":ledger.snapshot(identity),
                    "actual_QPU_shots":0,"CNOT":0,"CNOT_depth":0,"coefficient_slots":0,
                    "sampling":"Analytic delta distribution of the known Hartree-Fock determinant; no stochastic uncertainty",
                    "X_gates":m["nalpha"]+m["nbeta"],"total_unitary_depth":1})
        backend=IntegralHamiltonian(h,g,m["core_energy"])
        tick,cpu=perf_counter(),process_time();connections=HeatBathConnections(backend)
        save_json(root/case/"HCI-setup.json",{"problem_id":m["problem_id"],"setup":connections.setup,
                  "wall_seconds":perf_counter()-tick,"CPU_seconds":process_time()-cpu,"shared_once":True})
        for epsilon in spec["HCI_epsilon1_Ha"]:
            output=root/case/("HCI-epsilon-"+str(epsilon));filename=output/"result.json"
            if filename.exists():rows.append(json.loads(filename.read_text()));continue
            tick,cpu=perf_counter(),process_time()
            result=variational(backend,m["hf_id"],HCISpec(epsilon1_Ha=epsilon,dimension_cap=512,max_iterations=32,
                wall_seconds=120,RSS_cap_bytes=2*2**30),connections=connections,
                observer=lambda history:save_json(output/"checkpoint.json",{"history":history,"problem_id":m["problem_id"]}))
            result.update(case_id=case,problem_id=m["problem_id"],CPU_seconds=process_time()-cpu,
                          total_wall_seconds=perf_counter()-tick,method_id="THRESHOLD_HCI_VARIATIONAL_ONLY",
                          final_energy_Ha=result["history"][-1]["variational_energy_Ha"] if result["history"] else None,
                          coefficient_ids_are_interleaved=True,ground_state_quality_certified=False,
                          source_code_sha256=spec["source_sha256"],data_selection_uses_quantum_counts=False)
            save_json(filename,result);rows.append(result)
            print("HCI",case,epsilon,result["status"],result["final_energy_Ha"],flush=True)
    save_json(root/"summary.json",{"HF_planned_and_completed":120,"HCI_planned":12,"HCI_results":rows,
              "actual_QPU_shots":0,"SHCI_PT2_or_ground_reference_claim":False})


if __name__=="__main__":run()
