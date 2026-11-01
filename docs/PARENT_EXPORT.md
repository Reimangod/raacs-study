# Earlier selected comparison export

This is the English guide to the **earlier selected export**, not the scope of
the entire current repository. The original-language guide remains in the fixed
data release's Git history. Historical paths below refer to that export.

## Included in that export

- Naphthalene: 20 qubits.
- N2: 32 qubits at 1.2 and 2.4 Å.
- Fe2S2: 40 qubits.
- B4, LUCJ layers 1/2/4, and SqDRIFT5/20/50. SqDRIFT50 occurs only in the
  additional naphthalene cohort.

The selected export did not include the small-molecule cohort, other proposed
policies, or unrelated CEO compression/Transformer experiments. It contained
saved comparison tables, samples, selected configurations, CI results, circuits,
construction records, physical inputs, and counting conventions. It was not a
backup of every older search version or discarded branch. The current archive's
broader scope is documented in [SCOPE.md](SCOPE.md).

## Entry points inside the export

| Historical export path | Contents |
|---|---|
| `tables/START_HERE_all_endpoint_index.csv` | Conditions, methods, seeds, shots, D, energy, Recovery, resources, CI CPU time |
| `tables/new_naphthalene_group_summary.csv` | Three-readout summary for additional naphthalene |
| `tables/existing_85120shots_group_summary.csv` | Summaries of the four earlier large conditions |
| `tables/new_naphthalene_cost_components.csv` | Construction, final sampling, and classical CPU components |

## Conditions that must not be merged

The earlier 85,120-final-model-shot and additional 753,120-final-model-shot
cohorts are separate. `PREFIX_CRN` records form another nested sampling series;
each row records its denominator. Samples are noiseless model draws; QPU shots
are zero.

Energy is the physical-H expectation after selected CI, not the VQE expectation
of the sampling circuit. The large-case readout solves an `H + 0.1 S^2` problem
and evaluates physical H; it is not exact S=0 projection. Recovery is recovered
correlation energy, not a configuration-repair probability. The Fe2S2 reference
is not independently certified.

In this study, R counts spin strings and the common-list Cartesian determinant
space has D = R². Other papers may define R differently. Unreached, missing, and
failed cases are not filled with zero energy or hidden in success-only averages.
Equal final shots/D do not equalize construction, initialization, pilot samples,
or classical work. Registered LUCJ controls do not stand for the strongest
fully VQE- or tensor-network-optimized LUCJ variant.

Additional-naphthalene LUCJ resource-v2 values use author-native simplification:
1/2/4-layer CNOT counts are 322/544/988. Older generic-compiler values in raw
records are not substituted into that table. Earlier cohorts retain their own
registered resource values. All numbers are this study's simulations, not
transcriptions of published tables or exact reproductions of every paper setup.

## Export layout and provenance

| Directory or file | Role |
|---|---|
| `tables/` | Selected-method CSV rows without numerical edits |
| `conditions/` | Filtered browsing views, not new protocol freezes |
| `raw/` | Exact original bytes for the selected experiments |
| `views/` | Explicitly filtered derivatives of multi-method records |
| `provenance/` | Source hashes, extraction operations, and path mappings |
| `MANIFEST.csv` | File sizes and SHA-256 digests |

Original absolute paths can be resolved with the export's
`provenance/SOURCE_PATH_MAP.csv`. Missing CI vectors or coefficients were not
invented; a saved vector digest alone remains a digest. Packaging did not alter
or delete research data, run new experiments, or optimize new sources. Private
contents require review before external sharing.
