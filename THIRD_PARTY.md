# Third-party sources and retained rights

| Component | Origin and local evidence | Treatment |
|---|---|---|
| CEO implementation used by benchmark | https://github.com/mafaldaramoa/ceo-adapt-vqe ; exact local vendor bytes recorded in data/copied-source-manifest.json | Retained under historical/vendor/ceo with its Apache-2.0 LICENSE; not relicensed MIT |
| ffsim, PySCF, Qiskit, JAX, NumPy, SciPy | Exact package versions in uv.lock and original freezes | Dependencies, not wholesale copies of installed environments |
| LUCJ paper code included in historical evidence | Upstream project files/records in captured vendor directories | Preserve source license/notice where present; do not infer permission from a citation alone |
| Other historical benchmark/frontier code | Exact copied-source and archive inventories | Original rights and attribution retained; blanket MIT grant is not asserted here |

The current local CEO vendor snapshot is the executable-source authority.
An upstream URL or a currently checked-out upstream commit is **not** proof that
it is byte-identical to this historical vendor snapshot. Per-file hashes bind
the delivered version. Historical patches and unverified lineage remain visible.

Public distribution of the entire broad historical archive is pending a full
review of bundled vendor tools, executables and legacy materials. The core
new packaging code/data license decision is complete, but this is not an
assertion that every archived third-party file is cleared for public release.
