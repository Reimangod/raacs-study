"""T3 outcome-free native disjoint screen requests, same T1/T2 policy."""
import numpy as np
from qiskit_addon_sqd.counts import bit_array_to_arrays
from qiskit_addon_sqd.fermion import SubspaceRequest
from qiskit_addon_sqd.trim import TrimPolicy
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left
from .trim_readout import bit_array
from .io import digest


def requests(ids,counts,n,nelec,hf,seed,r=128,samples=128,batches=4):
    bits,probs=bit_array_to_arrays(bit_array(ids,counts,n,nelec))
    bits,probs=postselect_by_hamming_right_and_left(bits,probs,hamming_right=nelec[0],hamming_left=nelec[1])
    request=SubspaceRequest(bitstrings=bits,probabilities=probs,carryover_strings_a=np.array([],np.int64),
        carryover_strings_b=np.array([],np.int64),norb=n,nelec=tuple(nelec),samples_per_batch=samples,
        num_batches=batches,rng=np.random.default_rng(seed),iteration=0,symmetrize_spin=True)
    raw=TrimPolicy(trim_ratio=1.,max_strings_per_trim=r).prepare_subspaces(request)
    result=[]
    for a,b in raw:
        # Exactly the native caller's symmetrize/include_HF/max_dim constraint.
        if not np.array_equal(a,b):raise ValueError('Native symmetric request mismatch')
        strings=sorted(list(dict.fromkeys([int(hf)]+list(map(int,a))))[:r])
        result.append(strings)
    if len(result)!=batches:raise ValueError('Native batch count drift')
    return result


def frequency(universe,ids,counts,n,hf,r=128):
    weights=dict.fromkeys(universe,0);mask=(1<<n)-1
    for x,c in zip(ids,counts):
        for s in [int(x)&mask,int(x)>>n]:
            if s in weights:weights[s]+=int(c)
    return sorted([int(hf)]+sorted([s for s in universe if s!=hf],key=lambda s:(-weights[s],s))[:r-1])


def make(ids,counts,n,nelec,hf,seed,r=128):
    screens=requests(ids,counts,n,nelec,hf,seed,r)
    universe=sorted({s for strings in screens for s in strings}|{int(hf)})
    return dict(seed=int(seed),screens=screens,universe=universe,
        same_union_frequency=frequency(universe,ids,counts,n,hf,r),
        screening_required=len(universe)>r,final_actual_CI_dimension=min(r,len(universe))**2,
        request_digest=digest([seed,screens,universe,r]),uses_energy_reference_or_old_winner=False)


def evaluation_key(context,strings):
    if list(strings)!=sorted(set(map(int,strings))):raise ValueError('Canonical unique sorted CI strings required')
    return digest(dict(context=context,CI_strings=list(map(int,strings))))
