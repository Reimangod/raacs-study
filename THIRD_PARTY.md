# Third-party sources and retained rights

| Component | Origin and local evidence | Treatment |
|---|---|---|
| CEO exchange implementation | https://github.com/mafaldaramoa/ceo-adapt-vqe ; exact local bytes in data/source-files.json | Retained under reference/vendor/ceo with its Apache-2.0 LICENSE; not relicensed MIT |
| ffsim, PySCF, Qiskit, JAX, NumPy, SciPy | Exact package versions in uv.lock and original freezes | Dependencies, not wholesale copies of installed environments |
| LUCJ paper code in the full archive | Captured upstream metadata for kevinsung/lucj | Archive redistribution review is incomplete; a citation alone is not a license |
| RAACS and comparator source snapshots | data/source-files.json | Original rights and attribution retained; blanket MIT grant is not asserted here |

The current local CEO vendor snapshot is the executable-source authority.
An upstream URL or a currently checked-out upstream commit is **not** proof that
it is byte-identical to this historical vendor snapshot. Per-file hashes bind
the delivered version. Historical patches and unverified lineage remain visible.

Public redistribution of the full original archive remains subject to review.
It includes a publisher supplementary PDF with unestablished redistribution
permission, LUCJ upstream material requiring license review, and legacy native
binaries requiring classification. These are not needed by the quick start.
They remain in the existing private release, not in the curated source tree.

Removing files from the current branch does not remove them from Git history or
earlier releases. This cleanup is not a rights clearance, a history purge, or
authorization to make the entire repository and all release assets public.
