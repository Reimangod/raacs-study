import numpy as np


def discovery_probabilities(p, shots):
    p = np.asarray(p, dtype=float)
    if np.any(p < -1e-14) or np.any(p > 1+1e-14):
        raise ValueError("Invalid probability")
    p = np.clip(p, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.expm1(shots * np.log1p(-p))


def analytic_metrics(probability, reference, shots=(100, 1000, 10000, 100000), topks=(5, 10, 20, 50)):
    p = np.asarray(probability, float)
    if not np.isclose(p.sum(), 1, atol=1e-10):
        raise ValueError("Unnormalized probability")
    meta, ids, w = reference["meta"], reference["ids"], reference["weights"]
    hf = meta["hf_id"]
    nonhf = ids != hf
    denom = float(w[nonhf].sum())
    # Include available nonzero-weight determinants only; numerical zeros are not targets.
    targets = np.where(nonhf & (w > 1e-16))[0]
    targets = sorted(targets, key=lambda i: (-w[i], int(ids[i])))
    positive = p[p > 0]
    common = {"hf_probability": float(p[hf]), "nonhf_probability": float(1-p[hf]),
              "entropy_nats": float(-np.dot(positive, np.log(positive))),
              "valid_sector_probability": float(p[ids].sum()),
              "fci_nonhf_weight": denom}
    rows = []
    for m in shots:
        discovered = discovery_probabilities(p, m)
        row = {**common, "shots": m,
               "weighted_coverage": float(np.dot(w, discovered[ids])),
               "nonhf_weighted_coverage": float(np.dot(w[nonhf], discovered[ids[nonhf]])/denom) if denom > 1e-14 else None,
               "expected_unique": float(discovered.sum()),
               "expected_unique_valid": float(discovered[ids].sum()),
               "metric_type": "ANALYTIC_EXPECTATION_USING_FCI_FOR_EVALUATION_ONLY"}
        for k in topks:
            chosen = targets[:k]
            row[f"recall_top{k}"] = float(discovered[ids[chosen]].mean()) if chosen else None
            row[f"recall_top{k}_actual_targets"] = len(chosen)
        rows.append(row)
    return rows
