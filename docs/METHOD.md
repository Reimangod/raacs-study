# Registered RAACS policy and its exact source

RAACS is the display name of the registered B4 `SpinBeamEngine` in this study.
It is a sampling-circuit construction policy. Earlier exploratory policies are
not renamed RAACS.

## Inputs and output

Inputs are physical one-/two-electron integrals, HF occupation/energy, electron
populations, a model-sampling budget, deterministic request seeds and registered
CPU/work limits. FCI/SCI solution vectors or evaluation-reference energies are
not inputs to its decisions. The output is a bank of at most four sampling
circuits and the actual paid measurements. A final classical CI readout is a
separate layer.

## Decision sequence

1. Measure the HF preparation with 32 pilot model shots.
2. Fit a small CI vector to the configurations actually observed in the paid
   archive. Use integral connections to propose bounded single/double exchanges.
3. For each current parent, rank active exchanges by residual priority times
   observed occupation activation, divided by a registered resource increment.
   The inherited code uses `1 + max(0, delta_CNOT) + 0.5 max(0, delta_CNOT_depth)`.
   It is a heuristic, not an exact predicted energy decrease.
4. Try up to two exchanges per parent at the four angles
   `-pi/4, -pi/8, pi/8, pi/4`, paying 64 pilot model shots for every new candidate.
   These are real author-QE circuits, not freely prepared classical CI vectors.
5. Score candidates in the registered exact selected-space singlet. Preselect at
   most eight candidates, then greedily assemble a bank of at most four using a
   larger selected-string cap. The B4 code uses R16 for candidate scoring and
   R64 for bank scoring. Numeric ties use the registered tolerance/resource keys
   and deterministic identity order.
6. Continue up to depth eight, including temporarily non-improving paths. Keep
   the best pilot bank subject to the original improvement/tie rules.
7. Allocate the reserved final budget across the selected bank. Pilot samples
   guide construction; independent final draws are not fabricated from a pilot
   energy or a classical state vector.

The executed cohort protocols bind the limits (e.g. 60 construction CPU seconds)
and final readout. The old exploratory B4 protocol also mentions a later orbital
optimization stage; that must **not** be assumed present in confirmation cohorts
that explicitly disabled it.

## Code-to-step map

| Role | Byte-identical source snapshot |
|---|---|
| B4 singlet scoring override | reference/frontierloop/spin_beam.py |
| Beam loop, candidate angles and bank selection | reference/frontierloop/beam.py |
| Numerical selection, boundaries and resource keys | reference/raacsnext/policy.py |
| Residual/local exchange proposals and actual QE | reference/benchmark/local_sampler_primitives.py |
| Exact selected-space S=0 construction | reference/frontierloop/singlet.py |
| Reuse only of the same classical CI basis solve | reference/frontierloop/singlet_cache.py |
| Paid model-draw identity and persistent counts | reference/frontierloop/journal.py |
| Physical sparse QE simulation | reference/paperstudy/sparse_raacs.py |

Code snapshots and original protocols, rather than this prose, determine exact
historical semantics. They are hash bound in data/manifest.json and indexed in
data/source-files.json. The [source guide](../reference/README.md) maps the
RAACS and comparator implementations.
No new claim of mathematical necessity, universal superiority or noise-robust
equivalence is made by packaging them.
