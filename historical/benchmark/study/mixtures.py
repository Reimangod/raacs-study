"""Post-growth checkpoint-bank diagnostics, with pilot counts inside budget.

No FCI label is passed to policy selection. The bank itself was grown in full;
this is NOT an online policy that avoids that historical construction cost.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..chemistry import load_reference
from ..common import digest, file_digest, rng_for, save_json
from ..qsci import CHEMICAL_ACCURACY_HA
from .core import ADAPT, SHOTS, RitzCache, open_db, pack_pairs, ranks, rss_peak


def allocation(total, count):
    base, extra = divmod(total, count)
    return [base + (i < extra) for i in range(count)]


def measure(p, amount, rng):
    if not amount:
        return Counter()
    v = rng.multinomial(amount, p)
    return Counter({int(i): int(v[i]) for i in np.flatnonzero(v)})


def selected_from_counts(counts, valid, hf, R=32):
    ids = sorted(counts)
    return ranks(ids, [counts[i] for i in ids], valid, hf, ["mixture-readout", ids])["frequency"][:R]


def choose_diverse(pilot_counts, allowed, max_circuits, hf):
    """Greedy new non-HF observed support. Pilot-only; no amplitudes/FCI."""
    selected, covered = [], set()
    while len(selected) < min(max_circuits, len(allowed)):
        candidates = [i for i in allowed if i not in selected]
        best = max(candidates, key=lambda i: (len((set(pilot_counts[i]) - {hf}) - covered), -i))
        selected.append(best)
        covered |= set(pilot_counts[best]) - {hf}
    return selected


def run_mixtures(source, output, records, options):
    source, output = Path(source), Path(output)
    historical = pd.read_csv(source / "qsci-summary.csv")
    historical = historical[historical.include_hf & historical.method.isin(ADAPT)]
    completed = []
    with threadpool_limits(limits=options["threads"]):
        for case in sorted({r["case_id"] for r in records}):
            folder = output / "mixtures" / case
            folder.mkdir(parents=True, exist_ok=True)
            status_path = folder / "status.json"
            if status_path.exists() and json.loads(status_path.read_text())["status"] == "COMPLETED":
                completed.append(json.loads(status_path.read_text()))
                continue
            ref = load_reference(source / "references" / case)
            db = open_db(folder / "cache.sqlite")
            cache = RitzCache(ref["h"], ref["ids"], ref["meta"]["request_digest"], db)
            rows, counts_records = [], []
            hf, valid = ref["meta"]["hf_id"], ref["ids"]
            for method in ADAPT:
                cps = sorted([r for r in records if r["case_id"] == case and r["method"] == method], key=lambda r: r["step"])
                if not cps:
                    continue
                distributions = []
                for cp in cps:
                    path = source / cp["distribution_path"]
                    if file_digest(path) != cp["distribution_sha"]:
                        raise RuntimeError("Mixture source probability hash drift")
                    with np.load(path) as z:
                        p = z["probability"].astype(float)
                    p = np.maximum(p, 0)
                    p /= p.sum()
                    distributions.append(p)
                # Exclude HF-only checkpoint from the fixed evenly spaced candidate bank.
                available = [i for i, cp in enumerate(cps) if cp["step"] > 0] or [0]
                bank_indices = sorted(set(available[int(i)] for i in np.linspace(0, len(available)-1, min(8, len(available))).round()))
                for M in SHOTS:
                    oldM = M if M in (1000, 10000, 100000) else 1000
                    old = historical[(historical.case_id == case) & (historical.method == method) & (historical.shots == oldM)]
                    old = old[old.step.isin([c["step"] for c in cps])]
                    best_step = int(old.sort_values(["mean_energy_error_mHa", "cnot_count", "step"]).iloc[0].step)
                    retrospective_best = next(i for i, cp in enumerate(cps) if cp["step"] == best_step)
                    for rep in range(options["repeats"]):
                        selected_all = []
                        for policy in ("best_single_retrospective", "final", "equal_three", "pilot_diversity", "pilot_qsci_energy"):
                            ts = perf_counter()
                            rng, seed = rng_for(options["namespace"], "multi-checkpoint-evaluation", case, method, M, rep, policy)
                            counts, amounts, pilot_shots, pilot_calls = Counter(), Counter(), 0, 0
                            def add(index, amount):
                                counts.update(measure(distributions[index], amount, rng))
                                amounts[index] += amount
                            if policy in ("best_single_retrospective", "final"):
                                add(retrospective_best if policy.startswith("best") else len(cps)-1, M)
                            elif policy == "equal_three":
                                chosen = sorted(set(bank_indices[int(i)] for i in np.linspace(0, len(bank_indices)-1, min(3, len(bank_indices))).round()))
                                for i, amount in zip(chosen, allocation(M, len(chosen))):
                                    add(i, amount)
                            else:
                                bank = bank_indices[:min(len(bank_indices), max(1, M // 5))]
                                pilot_shots = min(M // 2, 10 * len(bank))
                                pilot = {}
                                pilot_energy = {}
                                for i, amount in zip(bank, allocation(pilot_shots, len(bank))):
                                    pilot[i] = measure(distributions[i], amount, rng)
                                    counts.update(pilot[i])
                                    amounts[i] += amount
                                    if policy == "pilot_qsci_energy":
                                        selected = selected_from_counts(pilot[i], valid, hf)
                                        _, energy, _, _ = cache.solve(selected)
                                        pilot_energy[i] = energy
                                        pilot_calls += 1
                                chosen = choose_diverse(pilot, bank, 3, hf) if policy == "pilot_diversity" else [min(bank, key=lambda i: (pilot_energy[i], cps[i]["cnot"], i))]
                                for i, amount in zip(chosen, allocation(M - pilot_shots, len(chosen))):
                                    add(i, amount)
                            assert sum(counts.values()) == M and sum(amounts.values()) == M
                            selected = selected_from_counts(counts, valid, hf)
                            subspace_id, energy, hit, lookup = cache.solve(selected)
                            selected_all.append((policy, selected, counts, amounts, pilot_shots, pilot_calls, energy, seed, subspace_id, perf_counter()-ts))
                        common_d = min(len(x[1]) for x in selected_all)
                        for policy, selected, counts, amounts, pilot_shots, pilot_calls, energy, seed, subspace_id, seconds in selected_all:
                            # Match realized dimensions, preserving each policy's frequency order.
                            _, matched_energy, _, _ = cache.solve(selected[:common_d])
                            err = energy - ref["meta"]["fci_energy"]
                            if err < -1e-8:
                                raise RuntimeError("Mixture variational reference violation")
                            row_id = digest([options["namespace"], case, method, M, rep, policy])
                            rows.append({"row_id": row_id, "case_id": case, "method": method, "shots_total": M, "replicate": rep,
                                         "policy": policy, "seed": str(seed), "energy": energy, "energy_error_mHa": err*1000,
                                         "chemical_accuracy": err <= CHEMICAL_ACCURACY_HA, "R_cap": 32, "actual_dimension": len(selected),
                                         "matched_common_d": common_d, "matched_error_mHa": (matched_energy-ref["meta"]["fci_energy"])*1000,
                                         "pilot_shots_included": pilot_shots, "pilot_ci_calls": pilot_calls,
                                         "circuits_executed": len([n for n in amounts.values() if n]), "minimal_switch_count": max(0, len([n for n in amounts.values() if n])-1),
                                         "switch_latency_seconds": None, "switch_latency_status": "NOT_MEASURED_NO_QPU",
                                         "gate_shot_proxy": sum(cps[i]["cnot"]*n for i, n in amounts.items()),
                                         "depth_shot_proxy": sum(cps[i]["depth"]*n for i, n in amounts.items()),
                                         "max_cnot": max(cps[i]["cnot"] for i, n in amounts.items() if n),
                                         "max_depth": max(cps[i]["depth"] for i, n in amounts.items() if n),
                                         "allocation_checkpoint_ids": [cps[i]["checkpoint_id"] for i in sorted(amounts)],
                                         "allocation_shots": [amounts[i] for i in sorted(amounts)], "subspace_cache_id": subspace_id,
                                         "seconds": seconds, "retrospective_calibration_M": oldM if policy.startswith("best") else None,
                                         "retrospective_calibration_shots_equivalent": oldM*20*len(cps) if policy.startswith("best") else 0,
                                         "construction_scope": "full pre-grown checkpoint bank cost remains; not construction-free early stopping",
                                         "hardware_shots": 0, "new_simulated_shots": M})
                            ids = sorted(counts)
                            counts_records.append({"row_id": row_id, "counts": pack_pairs(ids, [counts[i] for i in ids])})
                        db.commit()
            pd.DataFrame(rows).to_parquet(folder / "readouts.parquet", index=False)
            pd.DataFrame(counts_records).to_parquet(folder / "raw-counts.parquet", index=False)
            db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.close()
            status = {"case": case, "status": "COMPLETED", "readouts": len(rows), "simulated_shots": sum(x["new_simulated_shots"] for x in rows),
                      "peak_process_rss_bytes": rss_peak(), "hardware_shots": 0,
                      "scope": "A5 post-growth bank; best-single historical calibration is retrospective; pilots inside budget"}
            save_json(status_path, status)
            completed.append(status)
    save_json(output / "mixtures-status.json", completed)
    print(f"Multi-checkpoint cases {len(completed)}, readouts {sum(x['readouts'] for x in completed)}", flush=True)
    return completed
