"""Targeted Clifford matrix amplitudes via affine stabilizer support.

C|x> = (C X^x C†) C|0>. Stabilizers determine the amplitudes of C|0>
up to one common global phase, which cannot change output probabilities.
No exponentially large final Clifford state is materialized.
"""
from functools import lru_cache
import math
from time import process_time
import numpy as np
from qiskit.quantum_info import Pauli
from .clifford_frame_checker_v3 import masks


def affine_basis(frame):
    n=frame.num_qubits;rows=[Pauli(label) for label in frame.to_labels(mode='S')];pivots=[];rank=0
    for q in range(n):
        pivot=next((i for i in range(rank,n) if rows[i].x[q]),None)
        if pivot is None:continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank]
        for i in range(n):
            if i!=rank and rows[i].x[q]:rows[i]=rows[i].compose(rows[rank])
        pivots.append(q);rank+=1
    moves=[masks(row) for row in rows[:rank]];constraints=[]
    for row in rows[rank:]:
        x,z,phase=masks(row)
        if x or phase not in (1,-1):raise ValueError('Invalid diagonal stabilizer row')
        constraints.append([z,int(phase==-1)])
    constraints += [[1<<q,0] for q in pivots]
    rank_eq=0;pivot_eq=[]
    for q in range(n):
        pivot=next((i for i in range(rank_eq,n) if (constraints[i][0]>>q)&1),None)
        if pivot is None:raise ValueError('Nonunique stabilizer anchor')
        constraints[rank_eq],constraints[pivot]=constraints[pivot],constraints[rank_eq]
        for i in range(n):
            if i!=rank_eq and (constraints[i][0]>>q)&1:
                constraints[i][0]^=constraints[rank_eq][0];constraints[i][1]^=constraints[rank_eq][1]
        pivot_eq.append(q);rank_eq+=1
    anchor=sum(int(constraints[i][1])<<q for i,q in enumerate(pivot_eq));magnitude=2.**(-rank/2)

    @lru_cache(maxsize=100000)
    def amplitude(y):
        delta=y^anchor;current=anchor;phase=1.+0j
        for q,(x,z,factor) in zip(pivots,moves):
            if (delta>>q)&1:
                phase*=factor*(-1 if (z&current).bit_count()%2 else 1);current^=x
        if current!=y:return 0j
        return phase*magnitude
    return amplitude,rank,anchor


def project(frame,state,targets,deadline):
    n=frame.num_qubits;targets=list(map(int,targets))
    if not targets or len(targets)>100000 or len(set(targets))!=len(targets) or any(y<0 or y>=1<<n for y in targets):raise ValueError('Bounded unique target addresses')
    base,rank,anchor=affine_basis(frame);out={y:0j for y in targets};pairs=0
    for x,a in state.items():
        if process_time()>deadline:raise TimeoutError('Targeted Clifford CPU cap')
        p=Pauli((np.zeros(n,dtype=bool),np.array([(x>>q)&1 for q in range(n)],dtype=bool)))
        flip,z,factor=masks(p.evolve(frame,frame='s'))
        for j,y in enumerate(targets):
            if j%256==0 and process_time()>deadline:raise TimeoutError('Targeted Clifford CPU cap')
            source=y^flip;out[y]+=a*factor*(-1 if (z&source).bit_count()%2 else 1)*base(source);pairs+=1
    return out,dict(final_Clifford_support_rank=rank,final_Clifford_anchor=anchor,
        queried_amplitude_pairs=pairs,base_amplitude_cache=base.cache_info()._asdict(),
        full_norm_squared=math.fsum(abs(a)**2 for a in state.values()),
        final_global_phase_ignored_only=True,full_final_state_constructed=False)
