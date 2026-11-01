# Archive delivery status

This page records the **data-v0.1.0 delivery on 2026-10-31**. Test counts below
belong to that fixed release, not to later documentation or CI commits. For the
current branch, consult [GitHub Actions](https://github.com/Reimangod/raacs-study/actions/workflows/test.yml).

## Delivered

| Item | Recorded delivery |
|---|---|
| Repository | Reimangod/raacs-study, private |
| Data release | data-v0.1.0, private preparation release |
| Physical conditions | 10; additional naphthalene is a separate cohort |
| Saved endpoint rows | 6,261; not independent experiments |
| Endpoint breakdown | 5,580 small, 468 earlier large, 213 additional naphthalene |
| Small-source population | 144 planned replicate slots; 132 available; 12 missing because CCSD did not converge |
| Policy/configuration inventory | 187 configuration, freeze, and candidate-binding records; not 187 independent methods |
| Raw history | 153,794 files; about 7.1 GiB of original content stored in about 4.4 GiB |
| Release assets | 8 archives in 11 assets; large ZIPs split into parts no larger than 1 GiB |
| Derived products | 2,092 summary rows, 40 SVG plots, and a guide covering every physical condition |
| Source preservation | SHA-256 rechecked for all 153,794 files and 883 additional source files |
| Executed code/input bindings | 451 recorded bindings matched captured bytes |
| Fixed-release tests | 20 passed, including in a separate clone |
| Bounded numerical checks | 2 saved LiH CI bases matched the recorded energies |
| Asset integrity | All 11 GitHub asset sizes and server-side digests matched |
| Download check | One small-molecule archive downloaded; an original protocol extracted and checked |
| Licenses | MIT for new packaging code; CC BY 4.0 for independently owned data; third-party terms remain separate |

No source research file was modified or deleted during packaging. No new
molecular sampling, source optimization, or A100 job was started.

## Interpretation remains conditional

B4 is retained as the historical ID, with RAACS as its registered display alias.
ACTIVE remains separate. Small-case exact-singlet readout is not merged with
large-case spin-penalty readout. Model shots are not QPU shots; correlation-energy
Recovery is not configuration-recovery success.

Coefficient slots, independent parameters, and optimized variables are distinct.
Equal final shots and actual D do not equalize initialization, construction,
pilot sampling, or classical CPU work. Failed, missing, D-unreached, corrected,
and superseded records remain part of the archive.

## Not established by this delivery

1. **Public, DOI-backed archival publication.** No Zenodo DOI is assigned.
   Publisher supplementary material and historical vendor/binary redistribution
   require the [rights review](RIGHTS_AND_PUBLICATION.md).
2. **Fresh full-cohort reproduction.** Saved-result reproduction and two small CI
   checks do not rerun every molecular construction or validate every large root.
3. **Independent scientific certification.** Integrity checks and limited
   numerical parity are not independent peer review or proof of every claim.

The earlier selected export ZIP is a redundant derivative, not one of the eight
new archives. Its packaging code, original digest, extraction records, and
source tables are retained separately. See [the export guide](PARENT_EXPORT.md)
and [reproduction instructions](REPRODUCTION.md).
