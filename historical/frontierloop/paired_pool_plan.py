"""Uniform equal-size initial determinant spaces, from source data only."""
import math
import numpy as np
from .small_initial_pool_plan import choose


def choose_pairs(ids,values,n,nelec,coefficient_source=False):
    strings=choose(ids,values,n,nelec,r=8,coefficient_source=coefficient_source)
    product=[(a,b) for a in strings for b in strings];target=len(product);hf=((1<<nelec[0])-1,)*2
    weights={};mask=(1<<n)-1
    for x,value in zip(ids,values):
        w=float(abs(value)**2) if coefficient_source else int(value)
        if w:weights[(int(x)&mask,int(x)>>n)]=w
    selected=[hf]
    for p in sorted(weights,key=lambda p:(-weights[p],p)):
        if len(selected)==target:break
        if p!=hf:selected.append(p)
    observed_selected=set(selected)
    # If the observed support is too small, do not silently compare fewer
    # determinants: complete from the product control in canonical order.
    for p in product:
        if len(selected)==target:break
        if p not in observed_selected:selected.append(p);observed_selected.add(p)
    selected=sorted(selected)
    if len(selected)!=target or len(set(selected))!=target or hf not in selected:raise RuntimeError('Equal-size HF-containing initial spaces required')
    total=math.fsum(map(float,weights.values()))
    info=dict(initial_dimension=target,observed_joint_support=len(weights),
        observed_weight_retained=math.fsum(float(weights.get(p,0)) for p in selected)/total,
        unobserved_selected_pairs=sum(p not in weights for p in selected),
        equal_product_and_joint_spaces=set(selected)==set(product),
        product_joint_overlap=len(set(selected)&set(product)),
        joint_spin_flip_closed=all((b,a) in observed_selected for a,b in selected),
        no_spin_singlet_guarantee=True)
    return strings,product,selected,info


def source_pairs(source,n,nelec):
    if source['kind']=='CLASSICAL_HF_ONLY':
        hf=(1<<nelec[0])-1
        return choose_pairs(np.array([hf+(hf<<n)],dtype=np.uint64),np.array([1],dtype=np.uint64),n,nelec)
    if source['kind'] not in ['QUANTUM_FULL_COUNTS','CLASSICAL_FULL_COUNTS','CLASSICAL_COEFFICIENT_MARGINAL']:
        raise ValueError('Unknown source kind')
    coeff=source['kind']=='CLASSICAL_COEFFICIENT_MARGINAL'
    with np.load(source['payload'],allow_pickle=False) as a:
        return choose_pairs(a['blocked_ids'],a['coefficients'] if coeff else a['counts'],n,nelec,coeff)
