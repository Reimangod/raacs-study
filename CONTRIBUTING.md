# Contributing

## Before changing anything

Read [scope](docs/SCOPE.md), [method identity](docs/METHOD.md), and
[limitations](docs/LIMITATIONS.md). This repository preserves an existing study;
a documentation or software change is not permission to resample molecules,
launch GPU/Slurm jobs, or retune a historical policy.

Keep current documentation, figures, code comments, and pull requests in English.
Do not translate or rewrite the bytes of archived evidence to satisfy that rule.

## Local checks

```sh
uv sync --locked --extra test
uv run --no-sync pytest -q
uv run --no-sync python scripts/check_docs.py --repo .
uv run --no-sync python scripts/check_reproduction.py --repo .
```

The optional scientific check is documented in
[REPRODUCTION.md](docs/REPRODUCTION.md#2-small-independent-saved-basis-ci-check).
All routine checks are CPU-only. Generated reports belong in `.ci-reports/`, not
in the historical verification records or scientific data directories.

## Protect the study record

- Do not overwrite `data-v0.1.0`, its assets, or another fixed paper/data tag.
- Keep `data/`, `configs/`, `historical/`, and registered `results/` unchanged for
  documentation/CI-only changes. Hash verification must still pass.
- Label any future corrected scientific product as a new derivative/version;
  do not reseal changed evidence just to make a test pass.
- Preserve missing values, failed runs, source IDs, readout settings, and the
  difference between endpoint rows and independent experiments.
- Do not execute arbitrary historical runners. Use the safe documented CLI.
- Never commit credentials, private correspondence, environments, or raw download
  caches. Keep the repository private until the separate rights review is complete.

## Pull requests

Explain the user-visible change, files affected, tests run, and scientific
boundaries. For numerical changes, identify the exact input and comparison scope.
For documentation changes, check relative links and inspect any changed figure.
Passing CI is necessary for the checks it covers, not proof of scientific validity.
