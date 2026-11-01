# RAACS

**Resource-aware adaptive sampling for quantum-selected configuration interaction.**

[![Verification](https://github.com/Reimangod/raacs-study/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Reimangod/raacs-study/actions/workflows/test.yml)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Original data: CC BY 4.0](https://img.shields.io/badge/original_data-CC_BY_4.0-blue.svg)](LICENSE-DATA.md)

RAACS builds a small bank of quantum sampling circuits. A classical selected-CI
calculation identifies useful electronic configurations; a resource-aware rule
uses that information to propose the next circuit operations. The final samples
are evaluated by classical configuration interaction (CI).

This repository provides the **recorded policy, simulation data, and reproducible
tables and figures**, including comparisons with LUCJ and SqDRIFT. It is a private
research archive, not a new experiment or a claim of universal superiority.
The fixed data release is [data-v0.1.0](https://github.com/Reimangod/raacs-study/releases/tag/data-v0.1.0).
No publication DOI has been assigned.

## How it works

![RAACS workflow: molecular inputs feed a classical proposal step, candidate circuits are sampled, a classical CI selects a circuit bank, and reserved final samples produce the final CI readout. The selected bank feeds back into the proposal step. Quantum steps in this archive were simulated.](docs/figures/raacs-workflow.svg)

1. **Start from the physical problem.** Use molecular integrals, electron counts,
   the Hartree-Fock reference, and a fixed sampling budget.
2. **Propose useful exchanges.** A small classical CI supplies a residual-based
   priority. The registered rule also penalizes additional CNOT count and depth.
3. **Test actual circuits.** Sample candidate exchange circuits at fixed angles,
   then select a bank of at most four using classical CI scores.
4. **Evaluate with fresh final samples.** Spend the reserved final budget on the
   selected bank and report the post-CI energy and sampling-circuit resources.

The implementation was historically called **B4**. Only that registered policy
is displayed as **RAACS**; ACTIVE and other exploratory policies are different
methods. See the [algorithm and source map](docs/METHOD.md).

## Choose a starting point

| You want to… | Open |
|---|---|
| Understand the study without reading code | [Study overview](docs/OVERVIEW.md) |
| Inspect results for a particular molecule | [All-condition figure guide](results/README.md) |
| Read individual numerical records | [Endpoint catalog](data/catalog.csv) and [field definitions](docs/DATA_DICTIONARY.md) |
| Check the physical inputs and method identity | [Cases](data/cases.json) and [policy registry](configs/policy-registry.json) |
| Rebuild the saved tables and plots | [Reproduction guide](docs/REPRODUCTION.md) |
| Understand what is and is not verified | [Verification guide](docs/VERIFICATION.md) and [limitations](docs/LIMITATIONS.md) |

## Reproduce in four commands

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run these commands from a clone of this repository:

```sh
uv sync --locked --extra test
uv run --no-sync raacs-study verify --repo .
uv run --no-sync raacs-study reproduce --repo . --output reproduced
uv run --no-sync pytest -q
```

The generated summary is `reproduced/endpoint-summary.csv`; the plots are in
`reproduced/figures/`. **No GPU, QPU, raw-data download, or new molecular sampling
is needed.** The optional [saved-basis CI check](docs/REPRODUCTION.md#2-small-independent-saved-basis-ci-check)
recomputes two small existing LiH bases, not an entire molecular study.

## What is included

There are **10 distinct physical conditions** and **6,261 saved endpoint rows**.
An endpoint row is a readout at a particular Shot/CI setting, **not an independent
experiment**. Naphthalene appears in two separate acquisition cohorts.

| Molecule | Recorded geometry | Qubits | Cohort |
|---|---|---:|---|
| LiH | 1.5 and 3.0 Å | 10 | Small molecules |
| BeH2 | 1.3 and 2.5 Å | 12 | Small molecules |
| Linear H6 | 1.4 and 2.2 Å | 12 | Small molecules |
| Naphthalene | Recorded block-Boys setup | 20 | Earlier large and additional naphthalene |
| N2 | 1.2 and 2.4 Å | 32 | Earlier large |
| Fe2S2 | Recorded cluster geometry | 40 | Earlier large |

The catalog contains RAACS, registered LUCJ initializations/layer counts, and
SqDRIFT5/20; SqDRIFT50 is available only in the additional naphthalene cohort.
Coverage is not a full cross-product of every method and condition.
See [coverage](data/coverage.json) and [missing sources](data/missing-sources.csv).

## Read the comparisons correctly

| Quantity | What it means here |
|---|---|
| Energy | Physical-Hamiltonian energy **after classical CI**, not necessarily the sampling circuit's expectation value |
| Recovery | Percentage of the reference correlation energy recovered; **not** a noise-repair success rate |
| Shot count | Noiseless model samples; actual hardware/QPU shots are **zero** |
| R and actual D | Spin-string cap and obtained determinant-space dimension; a requested D can be unreached |
| CNOT and depth | Resources of the **sampling circuits**, using the registered decomposition and ensemble weighting |
| Cost | Initialization, construction/pilot sampling, final sampling, and classical CI work are distinct components |

Compare methods within the same cohort and readout setting. Matching final shots
and actual D does **not** equalize total work, spin treatment, or initialization.
Small-molecule exact-singlet and larger-case spin-penalty readouts remain separate.
Missing values are not zero, and failed or D-unreached records are not discarded.
Reference-certification gaps, including Fe2S2, are documented in
[limitations](docs/LIMITATIONS.md).

## What GitHub Actions verifies

- **Core checks:** tests and archive integrity on Python 3.11, 3.12, and 3.13.
- **Saved-result parity:** regenerate every portable table/plot product and check
  it byte-for-byte against the committed products.
- **Numerical check:** two saved LiH selected-CI bases, with one CPU thread.
- **Documentation and packaging:** local links, English-language documentation,
  safe SVGs, and an installed-wheel CLI smoke test.

Each run retains small verification reports. Passing these checks establishes
the stated software/data checks, **not** independent scientific review, full
fresh molecular regeneration, or experimental quantum advantage.
See [checks, commands, and boundaries](docs/VERIFICATION.md).

## Data access and reuse

The Git repository contains the lightweight catalog, policy snapshots, inputs
for the tiny CI check, and plots. Approximately **4.4 GiB** of raw evidence is
stored separately in private release assets; it is not needed for the quick start.
See [archive access](docs/REPRODUCTION.md#3-original-data-archives),
[archive manifest](data/archive-manifest.json), and [delivery status](docs/STATUS.md).

New packaging code is [MIT licensed](LICENSE). Independently owned data is
[CC BY 4.0](LICENSE-DATA.md); [third-party materials](THIRD_PARTY.md) retain their
own terms. The broader archive has unresolved public-redistribution items and
must remain private pending [rights review](docs/RIGHTS_AND_PUBLICATION.md).

For attribution, use [CITATION.cff](CITATION.cff) and identify the exact data
release. [Method references](docs/REFERENCES.md) distinguish publications,
preprints, and implementation sources. Contributions should follow the
[contributing guide](CONTRIBUTING.md); historical evidence and release tags must
not be overwritten.
