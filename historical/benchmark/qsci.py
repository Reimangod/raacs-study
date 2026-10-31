"""Finite raw multinomial measurements and explicit-determinant Ritz readout."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.stats import t
from .common import digest, file_digest, rng_for, save_json,resolve_artifact

CHEMICAL_ACCURACY_HA = 0.00159360143764


def select_configurations(ids, counts, valid_ids, hf_id, size, include_hf=True):
    if size < 1:
        raise ValueError("CI cap must be positive")
    valid = set(map(int, valid_ids))
    pairs = [(int(i),int(c)) for i,c in zip(ids,counts) if c > 0 and int(i) in valid]
    pairs.sort(key=lambda x:(-x[1],x[0]))
    chosen = [hf_id] if include_hf else []
    for state_id,_ in pairs:
        if state_id not in chosen and len(chosen) < size:
            chosen.append(state_id)
    return np.array(sorted(chosen),dtype=np.int64)


def solve_selected(h, selected):
    if not len(selected):
        return {"status": "EMPTY_VALID_SUBSPACE", "energy": None, "residual": None,
                "dimension": 0, "seconds": 0.}
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Duplicate configuration IDs")
    start = perf_counter()
    projected=h.tocsr()[selected][:,selected]
    difference=projected-projected.conj().T
    if difference.nnz and np.max(np.abs(difference.data))>1e-10:
        raise RuntimeError("Non-Hermitian projected H")
    if len(selected)>=512:
        hs=projected
        initial=np.random.default_rng(7081).normal(size=len(selected))
        values,vectors=eigsh(hs,k=1,which="SA",tol=1e-12,maxiter=5000,v0=initial)
        solver="sparse eigsh; deterministic v0; tol1e-12"
        nonzeros=int(np.count_nonzero(np.abs(projected.data)>1e-14))
    else:
        hs=projected.toarray();values,vectors=eigh(hs,subset_by_index=(0,0))
        solver="dense eigh"
        nonzeros=int(np.count_nonzero(np.abs(hs)>1e-14))
    energy = float(values[0])
    residual = float(np.linalg.norm(hs @ vectors[:,0]-energy*vectors[:,0]))
    if residual > 1e-8:
        raise RuntimeError("Inaccurate Ritz solution")
    return {"status": "COMPLETED", "energy": energy, "residual": residual,
            "dimension": len(selected), "seconds": perf_counter()-start,
            "matrix_nonzeros":nonzeros,"solver":solver}


def evaluate_checkpoint(reference, checkpoint_path, root, shot_grid=(1000,10000,100000), repeats=20, ci_size=32):
    cp = json.loads(Path(checkpoint_path).read_text())
    distribution=resolve_artifact(root,cp["distribution"])
    if file_digest(distribution) != cp["distribution_sha256"]:
        raise RuntimeError("Distribution hash drift")
    path = Path(root)/"qsci"/cp["case_id"]/cp["method"]/f"step-{cp['step']:04d}.json"
    contract = {"schema": "finite-qsci-1", "checkpoint_id": cp["checkpoint_id"],
                "distribution_sha256": cp["distribution_sha256"],
                "problem": reference["meta"]["request_digest"],
                "shots": list(shot_grid), "repeats": repeats, "size_cap": ci_size,
                "sampling": "independent multinomial per checkpoint/M/replicate",
                "selection": "count descending; state integer ascending; include-HF and no-HF ablation",
                "namespace": "finite-evaluation-v1"}
    context_id = digest(contract)
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["context_id"] != context_id:
            raise RuntimeError("QSCI configuration changed; new output directory required")
        return prior["rows"]
    with np.load(distribution) as data:
        p = data["probability"]
    if np.min(p)<-1e-14 or abs(p.sum()-1)>1e-9:
        raise ValueError("Bad probability distribution")
    p = np.maximum(p,0); p /= p.sum()
    valid = set(map(int,reference["ids"]))
    rows = []
    for shots in shot_grid:
        for rep in range(repeats):
            rng, seed = rng_for("finite-evaluation-v1",cp["case_id"],cp["checkpoint_id"],shots,rep)
            counts_all = rng.multinomial(shots,p)
            ids = np.flatnonzero(counts_all)
            counts = counts_all[ids]
            raw_id = digest({"checkpoint":cp["checkpoint_id"],"shots":shots,"replicate":rep,"seed":seed})
            for include_hf in (True,False):
                selected = select_configurations(ids,counts,valid,reference["meta"]["hf_id"],ci_size,include_hf)
                result = solve_selected(reference["h"],selected)
                error = result["energy"]-reference["meta"]["fci_energy"] if result["energy"] is not None else None
                if error is not None and error < -1e-8:
                    raise RuntimeError("QSCI below reference: check H/sector parity")
                corr = reference["meta"]["rhf_energy"]-reference["meta"]["fci_energy"]
                rows.append({"case_id":cp["case_id"],"method":cp["method"],"step":cp["step"],
                             "checkpoint_id":cp["checkpoint_id"],"context_id":context_id,
                             "raw_evaluation_id":raw_id,"shots":shots,"replicate":rep,"seed":seed,
                             "include_hf":include_hf,"R_cap":ci_size,**result,
                             "energy_error_mHa": 1000*error if error is not None else None,
                             "correlation_energy_recovery": 1-error/corr if error is not None and corr>1e-12 else None,
                             "chemical_accuracy": error <= CHEMICAL_ACCURACY_HA if error is not None else None,
                             "valid_shots":int(sum(c for i,c in zip(ids,counts) if int(i) in valid)),
                             "unique_valid":sum(int(i) in valid for i in ids),
                             "raw_count_ids":ids.tolist() if include_hf else None,
                             "raw_counts":counts.tolist() if include_hf else None,
                             "counts_source_id":raw_id,"selected_ids":selected.tolist(),
                             "new_simulated_raw_shots":shots if include_hf else 0,
                             "counts_reused_for_ablation":not include_hf,
                             "cnot_shot_proxy":shots*cp["resources"]["cnot_count"],
                             "depth_shot_proxy":shots*cp["resources"]["cnot_depth"],
                             **cp["resources"],"vqe_energy":cp["vqe_energy"]})
    save_json(path,{"contract":contract,"context_id":context_id,"rows":rows,
                    "note":"Same counts reused across HF ablation; CNOT-shot is a logical proxy, not hardware time."})
    return rows


def aggregate_rows(rows):
    data = pd.DataFrame(rows)
    summaries = []
    for key,group in data.groupby(["case_id","method","step","shots","include_hf"],dropna=False):
        good = group[group.status == "COMPLETED"]
        mean, std, lo, hi = None,None,None,None
        if len(good):
            mean = float(good.energy_error_mHa.mean())
            std = float(good.energy_error_mHa.std(ddof=1)) if len(good)>1 else None
            if std is not None:
                delta = float(t.ppf(.95,len(good)-1)*std/np.sqrt(len(good)))
                lo, hi = mean-delta,mean+delta
        first = group.iloc[0]
        summaries.append({"case_id":key[0],"method":key[1],"step":int(key[2]),"shots":int(key[3]),
                          "include_hf":bool(key[4]),"planned_repeats":len(group),"completed_repeats":len(good),
                          "mean_energy_error_mHa":mean,"std_energy_error_mHa":std,
                          "ci90_low_mHa":lo,"ci90_high_mHa":hi,
                          "ci_method":"approximate Student-t across independent draws; not simultaneous",
                          "chemical_accuracy_success_rate":float(good.chemical_accuracy.mean()) if len(good) else None,
                          "actual_dimension_mean":float(good.dimension.mean()) if len(good) else None,
                          "cnot_count":int(first.cnot_count),"cnot_depth":int(first.cnot_depth),
                          "parameter_count":int(first.parameter_count),"R_cap":int(first.R_cap)})
    return summaries


def run_qsci(reference, root, methods, shots=(1000,10000,100000),repeats=20,ci_size=32):
    rows = []
    for method in methods:
        for path in sorted((Path(root)/"checkpoints"/reference["directory"].name/method).glob("step-*.json")):
            rows.extend(evaluate_checkpoint(reference,path,root,shots,repeats,ci_size))
    summaries = aggregate_rows(rows)
    save_json(Path(root)/"qsci"/reference["directory"].name/"summary.json",summaries)
    return {"readouts":len(rows),"new_raw_draws":sum(r["new_simulated_raw_shots"] for r in rows)}

