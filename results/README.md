# Saved-data figure guide

All figures below are regenerated views of saved results, not new experiments.
Energy is **post-CI physical-H energy**. Sampling-circuit CNOT/Depth and raw
per-replicate values are available in `../data/catalog.csv`; source/construction
cost tables are retained in `../data/original-tables/`.

| Condition | Qubits | Main final-shot slice | Saved CI-energy curve |
|---|---:|---:|---|
| LiH 1.5 Å | 10 | 32,768 | [SVG](figures/small-exact-singlet__LiH_R1.5__SAME_FINAL_SHOTS__32768.svg) |
| LiH 3.0 Å | 10 | 32,768 | [SVG](figures/small-exact-singlet__LiH_R3.0__SAME_FINAL_SHOTS__32768.svg) |
| BeH2 1.3 Å | 12 | 32,768 | [SVG](figures/small-exact-singlet__BeH2_R1.3__SAME_FINAL_SHOTS__32768.svg) |
| BeH2 2.5 Å | 12 | 32,768 | [SVG](figures/small-exact-singlet__BeH2_R2.5__SAME_FINAL_SHOTS__32768.svg) |
| Linear H6 1.4 Å | 12 | 32,768 | [SVG](figures/small-exact-singlet__H6_R1.4__SAME_FINAL_SHOTS__32768.svg) |
| Linear H6 2.2 Å | 12 | 32,768 | [SVG](figures/small-exact-singlet__H6_R2.2__SAME_FINAL_SHOTS__32768.svg) |
| Naphthalene, earlier cohort | 20 | 85,120 | [SVG](figures/large-85120__NAPH20-BLOCK-BOYS__EXACT_D__85120.svg) |
| N2 1.2 Å | 32 | 85,120 | [SVG](figures/large-85120__N2-32-R1p2-P2__EXACT_D__85120.svg) |
| N2 2.4 Å | 32 | 85,120 | [SVG](figures/large-85120__N2-32-R2p4-P2__EXACT_D__85120.svg) |
| Fe2S2 | 40 | 85,120 | [SVG](figures/large-85120__FE2S2-40-P2__EXACT_D__85120.svg) |
| Naphthalene, additional cohort | 20 | 753,120 | [SVG](figures/naphthalene-additional__NAPH20-BLOCK-BOYS__EXACT_D__753120.svg) |

There are ten distinct physical conditions, but naphthalene has two separate
cohorts. Small cases use exact selected-space singlets; the larger cohorts use
the registered penalized-root readout. Do not merge these curves into one test.

Curves show means only where every recorded row in that group reached the same
target D and at least three replicate labels exist. Missing sources are listed
separately; an absent curve does not mean zero error or zero cost. Replicate
semantics differ between adaptive RAACS, fixed-source LUCJ, and SqDRIFT ensembles.

`endpoint-summary.csv` explicitly reports the group denominators and the scope of
conditional means. `figure-sources.csv` links each SVG to the exact catalog hash.
The other 29 SVG views retain further Shot/context slices. Original published-
presentation assets and the original analysis scripts remain in raw archives.
