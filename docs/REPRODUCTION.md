# Reproducing the study

## Saved tables and figures

From the repository root, using Python 3.11 or newer:

```sh
uv sync --locked --extra test
uv run --no-sync raacs-study verify --repo .
uv run --no-sync raacs-study reproduce --repo . --output reproduced
uv run --no-sync pytest -q
```

The summary is `reproduced/endpoint-summary.csv`; the plots are in
`reproduced/figures/`. This operation uses the committed catalog. It requires
neither a GPU nor a raw-data download, and makes no new model samples or CI
solves. The generated summary records group denominators and explicitly labels
conditional means. Failed and dimension-unreached readouts are not dropped.

The scientific-file checksums in `data/manifest.json` retain the original
hashes of every included scientific payload. The full numerical catalog,
original comparison tables, three cohort protocols, saved CI inputs, and
registered result products are unchanged. Source files were moved from
`historical/` to `reference/` without altering their bytes. New path mappings
are documented in `data/source-files.json` and the policy registry.

## Small independent saved-basis CI check

```sh
uv sync --locked --extra science
uv run --no-sync raacs-study ci-check --repo . --output reproduced/saved-ci-check.json
```

This recomputes two existing LiH 1.5 Å bases with at most 16 determinants,
using the saved Hamiltonian, exact selected-space S=0, and explicit orbital-order
fermionic phases. It checks numerical energy/residual parity within 1e-8 Ha.
It does not compute a new FCI reference or certify every large-system CI root.

## Automated software checks

```sh
uv run --no-sync python scripts/check_docs.py --repo .
uv run --no-sync python scripts/check_reproduction.py --repo .
```

[GitHub Actions](../.github/workflows/test.yml) runs tests on Python 3.11, 3.12,
and 3.13; checks byte-identical regeneration of all 42 saved products including
40 SVG figures; runs the two small CI checks; checks documentation links; and
builds and installs the wheel in a clean environment. Actions have read-only
repository permissions and do not launch molecular acquisition or GPU jobs.
Transient reports are workflow artifacts, not files in the research data tree.
Passing these tests is not independent scientific review or a demonstration of
quantum advantage.

## Algorithm source and full molecular regeneration

```sh
uv run --no-sync raacs-study policy --repo .
```

This displays the policy and cohort bindings. The
[source reference](../reference/README.md) contains the registered algorithm
implementations and their static local import dependencies. These remain
original snapshots, not automatic production entrypoints: original physical
inputs, some external runtimes and path handling are required for fresh
molecular construction. Do not run their acquisition or dispatch functions
as a quick start. Full fresh molecular regeneration has not been certified by
the lightweight checks above.

## Original raw evidence

The fixed [data-v0.1.0 release](https://github.com/Reimangod/raacs-study/releases/tag/data-v0.1.0)
preserves the complete original archive. Its tag, assets, and checksums are not
modified by this curation. Access currently requires an authorized collaborator:

```sh
gh release download data-v0.1.0 --repo Reimangod/raacs-study --dir data/downloads
uv run --no-sync raacs-study archives --repo . --directory data/downloads --full
```

The download is about 4.4 GiB. Full extraction requires about 7.1 GiB more;
ordinary table reproduction needs neither. To access a single original file:

```sh
uv run --no-sync raacs-study locate --repo . --path outputs/b4-small-molecule-comparison-2026-10-28-v1/protocol.json
uv run --no-sync raacs-study extract --repo . --directory data/downloads --path outputs/b4-small-molecule-comparison-2026-10-28-v1/protocol.json --output reproduced/original-small-protocol.json
```

The archive index maps original workspace paths to exact-byte blobs. Extraction
checks size/hash and refuses overwrites; it does not execute extracted scripts.
Do not load untrusted pickles or executables. The full archive also contains
exploratory work and legacy files outside the curated code tree. Its
[redistribution restrictions](../THIRD_PARTY.md) remain unresolved.

Saved-table reproduction is deterministic. New adaptive trajectories can differ
with RNG implementations, linear algebra libraries, and CPU-boundary stopping.
They should not be presented as byte-identical repeats of the original study.
