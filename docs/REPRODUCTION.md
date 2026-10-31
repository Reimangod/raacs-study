# Reproducing the recorded study

## 1. Saved-result tables and figures (small CPU-only environment)

```sh
uv sync --locked --extra test
uv run raacs-study verify --repo .
uv run raacs-study reproduce --repo . --output reproduced
uv run pytest -q
```

`reproduced/endpoint-summary.csv` includes all recorded groups and explicit
completed/D-matched counts. The mean columns are named as conditional means.
Figures require all recorded rows to match D and at least three readout labels;
the separate missing-source table still has to be consulted. These SVG figures
are new saved-data views, not claims to pixel-identically redraw old slides.
No costs are summed across repeated CI endpoints.

## 2. Small independent saved-basis CI check

```sh
uv sync --locked --extra science
uv run raacs-study ci-check --repo . --output reproduced/saved-ci-check.json
```

This verifies two saved LiH 1.5-A selected bases with at most 16 determinants,
using the recorded Hamiltonian, exact projected S=0, and explicit orbital-order
fermionic phases. It does not run a new FCI reference, resample circuits, or
validate every large CI root. The report states its numerical tolerance/scope.

## 3. Original data archives

For the private GitHub delivery, authenticate as an authorized collaborator:

```sh
gh release download data-v0.1.0 --repo Reimangod/raacs-study --dir data/downloads
uv run raacs-study archives --repo . --directory data/downloads --full
```

Each content-addressed ZIP has `inventory.jsonl` mapping original workspace-
relative paths to exact-byte blobs. `data/source-inventory.jsonl.gz` is the
cross-archive lookup. `data/archive-manifest.json` binds ZIP sizes and SHA-256.
Repeated equal content occupies one blob, but all original path associations
remain. Storage deduplication does not deduplicate scientific execution costs.

Archive extraction must validate paths and avoid overwrites. Do not extract
untrusted pickle objects or execute binaries. Historical absolute references
can be resolved against the source inventory without editing original JSON.
Archived SQLite DB/WAL evidence is preserved as captured; use a writable copy
for forensic reconstruction, not the original source file.

Locate or extract just one original file without expanding all 7+GiB:

```sh
uv run raacs-study locate --repo . --path outputs/b4-small-molecule-comparison-2026-10-28-v1/protocol.json
uv run raacs-study extract --repo . --directory data/downloads --path outputs/b4-small-molecule-comparison-2026-10-28-v1/protocol.json --output reproduced/original-small-protocol.json
```

Extraction verifies the original blob's size/hash and refuses existing targets.
No extracted script is executed. The source inventory supports exact original
relative paths (and the specifically recorded original absolute-path prefix).

## 4. Historical policy source, not an automatic production dispatch

`uv run raacs-study policy --repo .` prints exact policy/source bindings.
The `historical-policy` extra pins the core known scientific packages, but a
lock file is not proof that every exploratory native backend/remote job works.
See the import/rights audits. Full fresh molecular reruns are a separate, costly
reproduction layer and are not invoked by CI or the packaging commands.

Do not run the old dispatchers or replace local absolute paths en masse. The
original protocols include hashes of old source files and locations. A portable
I/O successor must preserve original algorithm semantics and record its changed
execution identity; it must not pretend to be the original historical run.

## Platform and stochastic expectations

The saved-table conversion is deterministic. Small CI is compared numerically,
not bytewise. Fresh adaptive trajectories may differ with RNG implementations,
linear algebra libraries and CPU-boundary stopping on M4 vs A100 hosts. Retain
the original counts/choices so users can reproduce conclusions without rerunning
the entire costly acquisition. Every demonstrated layer is reported separately.
