"""Regenerate portable saved-data products without refreshing any evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

from raacs_study.evidence import reproduce, sha256_file
from check_docs import write_report


def products(folder: Path) -> set[str]:
    names = {p.relative_to(folder).as_posix() for p in (folder / "figures").glob("*.svg")}
    for name in ("endpoint-summary.csv", "reproduction.json"):
        if (folder / name).is_file():
            names.add(name)
    return names


def compare_products(generated: Path, committed: Path) -> dict:
    actual, expected = products(generated), products(committed)
    errors = [f"missing generated product: {p}" for p in sorted(expected - actual)]
    errors += [f"unexpected generated product: {p}" for p in sorted(actual - expected)]
    errors += [f"product differs: {p}" for p in sorted(actual & expected)
               if sha256_file(generated / p) != sha256_file(committed / p)]
    for name in ("endpoint-summary.csv", "reproduction.json"):
        if name not in expected:
            errors.append(f"required committed product missing: {name}")
    if not any(p.startswith("figures/") for p in expected):
        errors.append("no committed SVG figures")
    return {"passed": not errors, "files_checked": len(actual & expected),
            "figures_checked": len([p for p in actual & expected if p.startswith("figures/")]),
            "errors": errors}


def check_reproduction(repo: Path) -> dict:
    repo = repo.resolve()
    with tempfile.TemporaryDirectory(prefix="raacs-ci-reproduction-") as temporary:
        generated = Path(temporary) / "generated"
        report = reproduce(repo, generated)
        result = compare_products(generated, repo / "results")
    result.update(endpoint_rows=report["endpoint_rows"], summary_rows=report["summary_rows"],
                  new_model_samples=0, new_CI_solves=0, evidence_modified=False,
                  scope="all portable saved-data products; no new acquisition or evidence refresh")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_reproduction(args.repo)
    write_report(args.repo, args.output, result)
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
