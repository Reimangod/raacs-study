"""Outcome-blind joint-list selection from already paid native HCI coefficients."""
import math


def joint_pairs(native,n,nelec,hf,d_cap=16384):
    if d_cap!=16384 or not native or len({(a,b) for a,b,v in native})!=len(native):
        raise ValueError('Registered D16384 cap and unique native pairs required')
    if any(not math.isfinite(v) or a<0 or b<0 or a>=1<<n or b>=1<<n or a.bit_count()!=nelec[0] or b.bit_count()!=nelec[1] for a,b,v in native):
        raise ValueError('Native amplitude/particle-sector mismatch')
    norm=math.fsum(v*v for a,b,v in native)
    if abs(norm-1)>1e-8:raise ValueError('Incomplete native coefficient export')
    target=(hf,hf);scores={(a,b):v*v for a,b,v in native};scores.setdefault(target,0.)
    ordered=[target]+sorted((p for p in scores if p!=target),key=lambda p:(-scores[p],p))[:d_cap-1]
    pairs=sorted(ordered)
    return pairs,dict(native_norm=norm,retained_native_weight=math.fsum(scores[p] for p in pairs),
        actual_joint_dimension=len(pairs),native_joint_dimension=len(native),HF_included=True,
        rule='HF_ALWAYS_THEN_DESCENDING_NATIVE_AMPLITUDE_SQUARED_TIE_ALPHA_BETA_INTEGER_LEX_CANONICAL_SORT')
