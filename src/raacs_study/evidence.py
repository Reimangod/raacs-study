"""Faithful saved-data conversion; no quantum/CI computations in this module."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from collections import defaultdict
from statistics import fmean


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def pick(row, *keys):
    """Preserve missing strings, including an explicitly empty first alias."""
    for key in keys:
        if key in row:
            return row[key]
    return ""


def number(value):
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def normalize(row, cohort, source_table, ordinal):
    """Do not infer a zero, convergence, independent parameter rank or reference."""
    small = cohort == "small-exact-singlet"
    identity = {"cohort": cohort, "source_table": source_table, "row_ordinal": ordinal, "original_row": row}
    record_id = hashlib.sha256(canonical(identity)).hexdigest()
    status = "COMPLETED" if small else pick(row, "status")
    result = {
        "record_id": record_id, "cohort": cohort,
        "case": pick(row, "case") or ("NAPH20-BLOCK-BOYS" if cohort == "naphthalene-additional" else ""),
        "historical_method_id": pick(row, "method"),
        "display_name": "RAACS" if pick(row, "method") == "B4" else pick(row, "method"),
        "replicate": pick(row, "replicate", "seed"), "historical_endpoint_id": pick(row, "id"),
        "context": pick(row, "view", "context") or "EXACT_D",
        "final_shots": pick(row, "final_shots", "used_final_shots"),
        "pilot_shots": pick(row, "pilot_shots", "acquired_pilot_model_samples"),
        "total_acquired_model_shots": pick(row, "actual_acquired_total_shots", "acquired_total_model_samples"),
        "QPU_shots": "0", "R_target": pick(row, "R_target", "target_R"),
        "D_target": pick(row, "D_target", "target_D"), "D_actual": pick(row, "D_actual", "actual_D"),
        "singlet_dimension": pick(row, "singlet_dimension"),
        "readout_status": status, "energy_Ha": pick(row, "energy_Ha"),
        "error_mHa": pick(row, "error_mHa"),
        "Recovery_percent": pick(row, "Recovery_percent", "Recovery_pct"),
        "cnot_count": pick(row, "cnot_count", "CNOT"),
        "cnot_depth": pick(row, "cnot_depth", "CNOT_Depth"),
        "total_depth": pick(row, "total_depth", "Total_Depth"),
        "independent_parameters": pick(row, "independent_parameters"),
        "angle_slots_mean": pick(row, "angle_slots_mean"),
        "CI_CPU_seconds_actual": pick(row, "CI_CPU_seconds_actual", "actual_new_CI_CPU_seconds", "CI_CPU_seconds"),
        "spin_rule": "exact selected-space S=0" if small else "H+0.1 S^2 root; physical-H expectation",
        "source_table": source_table, "source_row_ordinal": str(ordinal),
    }
    return result, {**identity, "record_id": record_id}


def safe_member_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name and "\0" not in name)


def verify_repository(repo: Path):
    checks = json.loads((repo / "data/manifest.json").read_text())
    if checks.get("schema") != "research-distribution-1":
        raise ValueError("Unsupported research-data manifest")
    errors = []
    names = [entry["path"] for entry in checks["files"]]
    if len(names) != len(set(names)):
        errors.append("duplicate manifest paths")
    for entry in checks["files"]:
        name = entry["path"]
        if not safe_member_name(name):
            errors.append("unsafe path: " + name)
            continue
        path = repo / name
        if not path.is_file():
            errors.append("missing: " + name)
        elif path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            errors.append("digest mismatch: " + name)
    records = [json.loads(line) for line in (repo / "data/original-rows.jsonl").read_text().splitlines()]
    rows = read_csv((repo / "data/catalog.csv").read_bytes())
    if len(rows) != len(records):
        errors.append("catalog/raw-row count mismatch")
    by_id = {}
    for item in records:
        normalized, expected = normalize(item["original_row"], item["cohort"], item["source_table"], item["row_ordinal"])
        if expected != item:
            errors.append("raw-row identity mismatch: " + item["record_id"])
        by_id[item["record_id"]] = normalized
    if len(by_id) != len(records):
        errors.append("duplicate record IDs")
    for row in rows:
        if row != by_id.get(row["record_id"]):
            errors.append("normalized value drift: " + row["record_id"])
    return {"passed": not errors, "catalog_rows": len(rows), "files_checked": len(checks["files"]), "errors": errors}


def aggregate(rows):
    """All registered endpoint rows count; completed/matched means are explicit."""
    groups = defaultdict(list)
    fields = ("cohort", "case", "historical_method_id", "context", "final_shots", "D_target")
    for row in rows:
        groups[tuple(row[key] for key in fields)].append(row)
    output = []
    metrics = ("energy_Ha", "error_mHa", "Recovery_percent", "cnot_count", "cnot_depth", "total_depth", "CI_CPU_seconds_actual")
    for key, group in sorted(groups.items()):
        matched = [r for r in group if r["readout_status"] == "COMPLETED" and
                   number(r["D_actual"]) == number(r["D_target"]) and number(r["D_target"]) is not None and
                   number(r["energy_Ha"]) is not None]
        unique = len({r["replicate"] for r in group})
        row = dict(zip(fields, key))
        row.update(display_name=group[0]["display_name"], n_endpoint_rows=len(group),
                   n_replicate_labels=unique, n_completed=sum(r["readout_status"] == "COMPLETED" for r in group),
                   n_D_matched=len(matched), all_recorded_replicates_D_matched=len(matched) == len(group),
                   mean_scope="CONDITIONAL_ON_COMPLETED_AND_ACTUAL_D_MATCH; inspect n and missing-source table")
        for metric in metrics:
            values = [number(r[metric]) for r in matched if number(r[metric]) is not None]
            row[metric + "_mean_D_matched"] = fmean(values) if values else ""
            row[metric + "_n"] = len(values)
        output.append(row)
    return output


def reproduce(repo: Path, output: Path):
    repo, output = repo.resolve(), output.resolve()
    if output == repo or output.is_relative_to(repo / "data") or output.is_relative_to(repo / "results"):
        raise ValueError("Use a separate output directory; never overwrite registered data/results")
    report = verify_repository(repo)
    if not report["passed"]:
        raise ValueError(report)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv((repo / "data/catalog.csv").read_bytes())
    summary = aggregate(rows)
    write_csv(output / "endpoint-summary.csv", summary)
    # No cross-cohort total cost is formed from repeated endpoint rows.
    write_json(output / "reproduction.json", {
        "schema": "saved-results-reproduction-1", "catalog_sha256": sha256_file(repo / "data/catalog.csv"),
        "endpoint_rows": len(rows), "summary_rows": len(summary), "new_model_samples": 0,
        "new_QPU_shots": 0, "new_CI_solves": 0,
        "unit_warning": "Endpoint/seed labels are not independent optimization trials. Costs are not summed over CI endpoints."
    })
    from .svg import write_figures
    figures = write_figures(summary, output)
    return {"endpoint_rows": len(rows), "summary_rows": len(summary), "figures": figures,
            "output": str(output), "new_samples": 0, "new_CI_solves": 0}
