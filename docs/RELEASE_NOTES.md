# RAACS research archive 0.1.0 — private preparation release

This release packages existing evidence. It is not a new performance result,
an independently reviewed paper, or a public Zenodo deposit.

## Contents

- Ten physical conditions: six small-molecule geometries plus naphthalene 20q,
  N2 32q at two distances, and Fe2S2 40q.
- 6,261 saved endpoint rows: 5,580 small, 468 earlier large, 213 additional
  naphthalene. Rows are not independent experiments.
- Explicit small-source population: 144 planned replicate slots, 132 available,
  12 CCSD-nonconvergence missing slots; no zero-energy imputation.
- Original B4/RAACS policy and inherited code, original configurations, historical
  failure/correction records and comparator data; old method IDs preserved.
- Eight content-addressed evidence archives with original-path mappings and
  exact-byte digests. Large archives are distributed in <=1GiB parts without
  deleting or modifying original research files.
- Deterministic saved-table reproduction, SVG views and a small independent
  saved-basis CI check. See release verification for actually tested layers.
- New packaging code: MIT. Independently owned data: CC BY 4.0. Third-party
  components retain their licenses; broad-history public redistribution review
  remains separate.

## Download

`gh release download data-v0.1.0 --repo Reimangod/raacs-study --dir data/downloads`

Authenticate as an authorized collaborator. `data/archive-manifest.json` binds
archive/part sizes and hashes. `raacs-study archives --full` can inspect multipart
ZIPs virtually without materializing another multi-GiB file.

## Boundaries

No new molecular sampling, source tuning, A100 jobs or FCI reference computation
was performed by this packaging task. Two tiny existing selected-CI bases may
be recomputed solely to validate saved-data reproduction. This does not validate
every historical molecular calculation. All QPU shots are zero.

No Zenodo DOI is assigned. This private release is not a public long-term data
deposit. Original absolute paths/native binaries remain in historical evidence;
do not blindly execute old dispatchers. Current portable CLI has separate scope.
