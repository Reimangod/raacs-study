"""Budget-filling selections from the SAME outcome-independent T1 request union.

All three arms retain min(R, union size), not an energy-dependent quota.
The frequency arm does NOT need CI screening operationally; do not charge it
four expensive screens merely because this development study already has them.
No claim that mean-CI-marginal ranking is a new scientific algorithm.
"""
import numpy as np
from qiskit_addon_sqd.fermion import SCIResult
from qiskit_addon_sqd.trim import TrimPolicy


def selections(calls, ids, counts, n, hf, r=128):
    screens = [x for x in calls if x['round'] == 0]
    if len(screens) != 4 or any(len(x['CI_strings']) > r for x in screens):
        raise ValueError('Exactly four bounded native screening requests required')
    universe = sorted({int(s) for x in screens for s in x['CI_strings']} | {int(hf)})
    weights = dict.fromkeys(universe, 0.)
    frequency = dict.fromkeys(universe, 0)
    mask = (1 << n)-1
    for x, c in zip(ids, counts):
        for s in [int(x) & mask, int(x) >> n]:
            if s in frequency:
                frequency[s] += int(c)
    result_rows = []
    for x in screens:
        strings = np.asarray(x['CI_strings'], np.int64)
        wa, wb = np.asarray(x['electron_marginal_weights_alpha']), np.asarray(x['electron_marginal_weights_beta'])
        if wa.shape != strings.shape or wb.shape != strings.shape or min(wa.min(), wb.min()) < -1e-14:
            raise ValueError('Missing/invalid endogenous CI marginal weights')
        if abs(wa.sum()-1) > 1e-8 or abs(wb.sum()-1) > 1e-8:
            raise ValueError('CI probability weights not normalized')
        for s, a, b in zip(strings, wa, wb):
            weights[int(s)] += float((a+b)/8)
        aa, bb = strings[np.argsort(-wa, kind='stable')], strings[np.argsort(-wb, kind='stable')]
        occ = ((strings[:, None] >> np.arange(n)) & 1).astype(float)
        # Native interface permits a solver to supply exactly ranked carryover
        # without retaining its full eigenvector. All strings survive ratio1.
        result_rows.append(SCIResult(energy=x['energy_Ha'], sci_state=None,
            orbital_occupancies=(wa@occ, wb@occ), carryover=(aa, bb)))
    native = TrimPolicy(trim_ratio=1., max_strings_per_trim=r).refine(result_rows, 0)
    if native is None or len(native) != 1:
        raise RuntimeError('Native no-local-trim refinement contract changed')
    # Exact caller-side symmetrize_spin/include_HF/max_dim semantics of the
    # installed native loop: preserve order of include lists then strings_a.
    ordered = list(dict.fromkeys([int(hf)]+list(map(int, native[0][0]))))[:r]
    rank = lambda scores: sorted([int(hf)]+sorted([s for s in universe if s != int(hf)],
        key=lambda s: (-scores[s], s))[:r-1])
    arms = {'GLOBAL_CI_MARGINAL': rank(weights), 'SAME_UNION_FREQUENCY': rank(frequency),
            'NATIVE_TRIM_RATIO1': sorted(ordered)}
    expected = min(r, len(universe))
    if any(len(s) != expected or not set(s) <= set(universe) or int(hf) not in s for s in arms.values()):
        raise RuntimeError('Fixed cardinality/source-only inclusion failed')
    return {'universe': universe, 'arms': arms,
        'mean_CI_marginal_weights': [[s, weights[s]] for s in universe],
        'same_union_frequency': [[s, frequency[s]] for s in universe],
        'energy_or_reference_ranking_used': False,
        'all_arms_have_same_actual_CI_dimension': expected**2,
        'screening_CI_required_operationally': len(universe) > r}
