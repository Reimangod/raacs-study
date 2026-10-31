"""Export small numeric-only fixtures for CPU/CUDA qualification, no training labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .ansatz import make_pool
from .chemistry import load_reference
from .common import save_npz,save_json,file_digest,resolve_artifact


def create_fixture(root,case_id,method="ceo_ovp"):
    root=Path(root)
    path=sorted((root/"checkpoints"/case_id/method).glob("step-*.json"))[-1]
    cp=json.loads(path.read_text());ref=load_reference(root/"references"/case_id)
    pool=make_pool(method,ref["meta"]["n_qubits"])
    if method not in ("ceo_ovp","ceo_mvp","qeb","gsd"):
        raise ValueError("This qualification fixture uses author A^3=-A excitation kernels only")
    arrays={"theta":np.array(cp["optimized_parameters"]),"hf_id":np.array(ref["meta"]["hf_id"]),
            "n_qubits":np.array(ref["meta"]["n_qubits"])}
    h=ref["h"].tocsr()
    arrays.update(h_data=h.data,h_indices=h.indices,h_indptr=h.indptr)
    for j,index in enumerate(cp["selected_operators"]):
        a=pool.get_imp_op(index).tocsr()
        arrays[f"a{j}_data"]=a.data;arrays[f"a{j}_indices"]=a.indices;arrays[f"a{j}_indptr"]=a.indptr
    terms={};entries=[]
    for i in range(pool.size):
        for term,c in pool.get_q_op(i).terms.items():
            pos=terms.setdefault(term,len(terms))
            entries.append((i,pos,c))
    masks=[]
    for term in terms:
        flip=phase=ny=0
        for q,p in term:
            mask=1 << (pool.n-1-q)
            if p in ("X","Y"):flip|=mask
            if p in ("Y","Z"):phase|=mask
            ny+=p=="Y"
        masks.append((flip,phase,ny))
    coef=sparse.csr_matrix(([e[2] for e in entries],([e[0] for e in entries],[e[1] for e in entries])),
                           shape=(pool.size,len(terms)))
    arrays.update(coef_data=coef.data,coef_indices=coef.indices,coef_indptr=coef.indptr,
                  masks=np.array(masks,dtype=np.uint64),pool_size=np.array(pool.size))
    # A fixed CI subspace for solver profiling, chosen without FCI coefficients.
    with np.load(resolve_artifact(root,cp["distribution"])) as z:
        prob=z["probability"]
    order=sorted(ref["ids"],key=lambda i:(-prob[i],int(i)))[:32]
    arrays["ci_h"]=h[order][:,order].toarray()
    directory=root/"profiles"/case_id;directory.mkdir(parents=True,exist_ok=True)
    fixture=directory/"fixture.npz"
    save_npz(fixture,**arrays)
    save_json(directory/"fixture.json",{"case_id":case_id,"method":method,"source_checkpoint":cp["checkpoint_id"],
              "problem":ref["meta"]["request_digest"],"fixture_sha256":file_digest(fixture),
              "purpose":"CPU/GPU numerical parity and component timing; not a new molecular comparison outcome",
              "ci_selection":"top probability for timing fixture only; no FCI amplitudes; not primary finite-shot QSCI"})
    return fixture


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--case",required=True)
    a=p.parse_args();print(create_fixture(a.output,a.case))


if __name__=="__main__":main()
