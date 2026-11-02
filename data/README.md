# Research data

Start with [catalog.csv](catalog.csv) for numerical records and
[cases.json](cases.json) for physical problems. The
[data dictionary](../docs/DATA_DICTIONARY.md) defines every catalog field.

| File or directory | Contents |
|---|---|
| `catalog.csv` | All 6,261 normalized readouts, with original numerical strings |
| `original-tables/` | Original tables, including circuit resources, construction costs, and missing sources |
| `original-rows.jsonl` | Original row content and source-to-record identities |
| `cases.json` | Ten physical conditions, input metadata, and problem digests |
| `source-population.csv`, `missing-sources.csv`, `coverage.json` | Recorded source coverage and missing controls |
| `fixtures/` | Two small saved LiH bases for an independent numerical check |
| `manifest.json` | Checksums of the retained scientific files |
| `source-files.json` | Algorithm-source lineage and local import dependencies |
| `archive-manifest.json`, `source-inventory.jsonl.gz` | Indexes for accessing original raw evidence separately |

Records from the small exact-singlet study, the earlier large-molecule study,
and the additional naphthalene study are separate cohorts. Keep their readout
rules and budgets distinct. Missing values are not zero. Repeated readouts do
not represent independent circuit construction or additional acquisition costs.

The current distribution retains the full numerical catalog and original
comparison tables. The broader source inventory and fixed release also include
earlier exploratory work; those records are not additional confirmation trials.
See [reproduction](../docs/REPRODUCTION.md) for selective archive access.
