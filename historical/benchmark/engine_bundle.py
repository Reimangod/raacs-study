"""Export only Hamiltonian, author pool and fixed optimizer settings; no FCI labels."""
from __future__ import annotations
import argparse
from pathlib import Path
from .ansatz import make_pool,pool_catalog
from .chemistry import load_reference
from .common import save_json,save_binary,file_digest,digest


def create_bundle(root,case_id,method,destination,max_steps=64,task_seconds=600):
    root=Path(root);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    r=load_reference(root/"references"/case_id);m=r["meta"]
    catalog=pool_catalog(make_pool(method,m["n_qubits"]))
    save_json(destination/"pool.json",catalog)
    save_binary(destination/"hamiltonian.npz",(r["directory"]/"hamiltonian.npz").read_bytes())
    request={"schema":"engine-bundle-1","case_id":case_id,"method":method,"hf_id":m["hf_id"],
             "problem_id":m["request_digest"],"config":{"max_steps":max_steps,"task_seconds":task_seconds,
             "gradient_tolerance":1e-3,"optimizer_gtol":1e-7,"optimizer_maxiter":200},
             "no_fci_coefficients_or_energy_in_bundle":True,
             "files":{n:file_digest(destination/n) for n in ("pool.json","hamiltonian.npz")}}
    old=destination/"request.json"
    if old.exists():
        import json
        if digest(json.loads(old.read_text()))!=digest(request):raise RuntimeError("Immutable bundle changed")
    save_json(old,request);return destination


def extend_bundle(source,destination,max_steps):
    """New versioned request, same immutable numeric Hamiltonian/catalog."""
    import json
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if (destination/"request.json").exists():raise RuntimeError("Destination already frozen")
    request=json.loads((source/"request.json").read_text())
    for name,sha in request["files"].items():
        if file_digest(source/name)!=sha:raise RuntimeError("Source changed")
        save_binary(destination/name,(source/name).read_bytes())
    request["config"]["max_steps"]=max_steps
    save_json(destination/"request.json",request)
    return destination


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--case",required=True)
    p.add_argument("--method",default="ceo_ovp");p.add_argument("--destination",required=True)
    p.add_argument("--max-steps",type=int,default=64);a=p.parse_args()
    print(create_bundle(a.output,a.case,a.method,a.destination,a.max_steps))


if __name__=="__main__":main()
