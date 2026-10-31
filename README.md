# RAACS research archive

This repository packages **existing** RAACS policy implementations and simulation
evidence, with LUCJ and SqDRIFT controls. It does not report a new experiment.
The historical method ID **B4** is retained in raw records; **RAACS** is its
current display name, not a rename of ACTIVE or every earlier policy.

**Status:** private research-archive preparation; no Zenodo DOI is assigned.
See [Japanese guide](docs/README_JA.md), [scope](docs/SCOPE.md),
[method](docs/METHOD.md), [data definitions](docs/DATA_DICTIONARY.md),
[reproduction](docs/REPRODUCTION.md), and [limitations](docs/LIMITATIONS.md).
The [figure guide](results/README.md) lists all ten physical conditions without
selecting only favorable examples.
Method/source citations are listed in [references](docs/REFERENCES.md).
See [delivery status](docs/STATUS_JA.md) for the completed private archive and the
remaining public/Zenodo publication boundary.

## Start here

1. `data/catalog.csv`: normalized endpoint index, retaining original IDs/values.
2. `data/cases.json`: physical conditions and their source metadata.
3. `configs/policy-registry.json`: policy names and exact source/config bindings.
4. `data/archive-manifest.json`: archive locations, hashes, and scope.
5. `results/`: portable tables and SVG figures generated from saved records.
6. `data/coverage.json`: coverage, missingness and distinct counting units.

## CPU-only quick reproduction

```sh
uv sync --extra test --locked
uv run raacs-study verify --repo .
uv run raacs-study reproduce --repo . --output reproduced
uv run pytest -q
```

This layer does **not** resample circuits, solve new molecular ground states,
submit jobs, contact a QPU, or use A100s. The source numerical columns remain
strings in the faithful catalog; additional labels are separate fields.

See `docs/REPRODUCTION.md` for raw-archive download/verification and a separate
small saved-basis CI check. Archived historical runners can contain absolute
paths and side effects: do not invoke them as the portable API.

## Scientific interpretation

- All included study shots are noiseless **model samples**; actual QPU shots are
  zero. Finite-shot randomness is not a hardware-noise simulation.
- Energy is the physical-H expectation **after classical CI**, not necessarily
  the energy expectation of the sampling circuit alone.
- Recovery is correlation-energy recovery, not configuration-recovery success.
- Small-molecule exact-singlet readout and larger-case spin-penalty readout are
  separate cohorts. Equal raw determinant D does not imply equal CPU work.
- Resources describe sampling circuits. Ensemble means are shot weighted where
  registered; they are not a circuit preparing the final CI eigenvector.
- Missing/unreached/capped runs remain visible. Unknown independent parameter
  counts and incompletely validated references are not replaced by zero.
- Original source status and final readout status are separate: a historical
  `*_PENDING` source label is not evidence that completed readout is still live.

Original-data hash checks establish integrity, not scientific correctness or
independent peer review. See the machine-readable verification reports for the
layers actually exercised by this release.

Recorded cohort code/input bindings were checked: 451/451 match captured bytes.
The separately cloned saved-result package passes 20 tests and two tiny saved-
basis CI checks. These checks do not certify full fresh molecular regeneration.
