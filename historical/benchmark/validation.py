"""Non-optimizing post-run checks; immutable checkpoints remain unchanged."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from pyscf import gto
from .ansatz import make_pool,pool_catalog,selection_gradients
from .chemistry import load_reference
from .common import digest,file_digest,save_json,resolve_artifact
from .config import cases_for


def run(root):
    root=Path(root);rows=[];constants=[];issues=[];pool_checks=0
    cases={c.id:c for c in cases_for("full")}
    refs={}
    for case_id,case in cases.items():
        if not (root/"references"/case_id/"reference.json").exists():continue
        r=load_reference(root/"references"/case_id);refs[case_id]=r
        mol=gto.M(atom=case.geometry(),basis=case.basis,unit="Angstrom",charge=case.charge,spin=case.spin,verbose=0)
        constant=float(r["h"][0,0].real);nuclear=float(mol.energy_nuc())
        constants.append({"case_id":case_id,"problem_id":r["meta"]["request_digest"],
                          "active_hamiltonian_constant_Ha":constant,"nuclear_repulsion_Ha":nuclear,
                          "frozen_core_electronic_constant_Ha":constant-nuclear,
                          "derivation":"H[active-vacuum,active-vacuum]; nuclear repulsion from registered geometry; no new SCF/FCI",
                          "frozen_spatial_orbitals":case.frozen_core})
    tasks=[]
    for path in (root/"checkpoints").glob("*/*/status.json"):
        case=path.parent.parent.name;method=path.parent.name
        if method not in ("ceo_ovp","ceo_mvp","qeb","qubit","gsd"):continue
        if case in refs:tasks.append((refs[case]["meta"]["n_qubits"],method,case,path))
    for n,method,case,path in sorted(tasks):
        status=json.loads(path.read_text());cps=sorted(path.parent.glob("step-*.json"))
        if not cps:continue
        history=[json.loads(p.read_text()) for p in cps];last=history[-1]
        largest=max(np.diff([c["vqe_energy"] for c in history]),default=0.)
        if largest>1e-8:issues.append("Nested growth energy increased: "+case+"/"+method)
        row={"case_id":case,"method":method,"terminal":status["status"],"last_step":last["step"],
             "largest_nested_growth_energy_increase_Ha":float(largest),
             "final_is_observed_vqe_minimum_within_1e-8_Ha":last["vqe_energy"]<=min(c["vqe_energy"] for c in history)+1e-8,
             "terminal_pool_gradient_norm":None,"convergence_check":"NOT_REQUIRED_FOR_CAPPED_OR_UNFINISHED_RUN"}
        if status["status"]=="CONVERGED":
            context={"last_checkpoint_sha256":file_digest(cps[-1]),"distribution_sha256":last["distribution_sha256"],
                     "method":method,"problem":last["problem_id"],"criterion":status["config"]["gradient_tolerance"]}
            target=root/"validation/convergence"/case/(method+".json")
            if target.exists() and json.loads(target.read_text())["context_id"]==digest(context):
                check=json.loads(target.read_text())
            else:
                pool=make_pool(method,n);distribution=resolve_artifact(root,last["distribution"])
                if file_digest(distribution)!=last["distribution_sha256"]:raise RuntimeError("State hash mismatch")
                with np.load(distribution) as z:state=z["state"]
                started=perf_counter();g=selection_gradients(pool,state,refs[case]["h"])
                check={"context_id":digest(context),"context":context,"norm":float(np.linalg.norm(g)),
                       "seconds":perf_counter()-started,"pool_size":len(g),"catalog_digest":pool_catalog(pool)["catalog_digest"],
                       "work_scope":"one audit-only classical gradient vector; NOT additional optimizer training or hardware shots"}
                save_json(target,check)
            pool_checks+=1
            row["terminal_pool_gradient_norm"]=check["norm"]
            row["convergence_check"]="PASSED" if check["norm"]<=context["criterion"]+1e-9 else "FAILED"
            if row["convergence_check"]=="FAILED":issues.append("Convergence not reproduced: "+case+"/"+method)
        rows.append(row)
    save_json(root/"validation/reference-constants.json",constants)
    save_json(root/"validation/trajectory-checks.json",rows)
    result={"reference_constants":len(constants),"trajectory_rows":len(rows),"convergence_checks":pool_checks,
            "issues":issues,"hardware_shots":0,"optimizer_starts":0,"fci_runs":0}
    save_json(root/"validation/status.json",result);print(result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);run(p.parse_args().output)
