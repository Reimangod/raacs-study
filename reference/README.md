# Algorithm source reference

These are the original RAACS and comparator source files needed to inspect
the registered methods, together with their local import dependencies.
Their bytes are unchanged; paths are indexed in
[`data/source-files.json`](../data/source-files.json).

| Role | Source |
|---|---|
| RAACS selected-space singlet scoring | [frontierloop/spin_beam.py](frontierloop/spin_beam.py) |
| Candidate circuits and bank selection | [frontierloop/beam.py](frontierloop/beam.py) |
| Resource priorities and deterministic selection | [raacsnext/policy.py](raacsnext/policy.py) |
| Local exchange proposals | [benchmark/local_sampler_primitives.py](benchmark/local_sampler_primitives.py) |
| Singlet CI | [frontierloop/singlet.py](frontierloop/singlet.py) |
| LUCJ controls | [benchmark/lucj.py](benchmark/lucj.py), [paperstudy/lucj.py](paperstudy/lucj.py) |
| SqDRIFT controls | [benchmark/sqdrift_ensemble.py](benchmark/sqdrift_ensemble.py), [paperstudy/sqdrift_execution.py](paperstudy/sqdrift_execution.py) |
| Sampling-circuit resources | [benchmark/ansatz.py](benchmark/ansatz.py) |
| Exchange-gate implementation | [vendor/ceo/adaptvqe/circuits.py](vendor/ceo/adaptvqe/circuits.py) |

The package and class names retain their original identities. Only the
registered B4 policy is displayed as RAACS; other policies are not renamed.

These source snapshots are not a portable molecular execution service. Some
functions depend on original input files, runtimes, or absolute paths. Do not
execute acquisition or remote-dispatch functions as a quick start. Use the
[reproduction tools](../docs/REPRODUCTION.md) to rebuild the saved results.
The static import closure is checked, but it is not a guarantee that every
runtime file or external dependency is available. The complete original source
snapshot remains in the fixed data release.

Third-party code retains its original terms. The exchange implementation's
[Apache-2.0 license](vendor/ceo/LICENSE) is included; see
[third-party sources](../THIRD_PARTY.md) for the broader archive restrictions.
