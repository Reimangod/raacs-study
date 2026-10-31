"""Post-outcome N40 source diagnostic; privileged model probabilities, not a policy."""
from time import process_time,perf_counter
from pathlib import Path
import numpy as np
from . import direct_sample_pilot as s
from .io import load,save,sha256
from .paths import ROOT


def expected_seen(prob,shots):
    with np.errstate(divide='ignore'):
        return -np.expm1(shots*np.log1p(-np.clip(prob,0,1)))


def main():
    p=s.verified();t=load(s.RUN/'terminal.json')
    if t['failure']:raise ValueError('This is a post-complete diagnostic only')
    cpu,wall=process_time(),perf_counter();original=next(x for x in p['points'] if x['point']==[1.,1.])
    with np.load(original['files']['probabilities.npz']['path'],allow_pickle=False) as z:base=z['probabilities']
    rows=[]
    for q in p['points']:
        with np.load(q['files']['probabilities.npz']['path'],allow_pickle=False) as z:prob=z['probabilities']
        n=int(round(len(prob)**.5));joint=prob.reshape(n,n);half=.5*(joint.sum(axis=0)+joint.sum(axis=1))
        union=joint.sum(axis=0)+joint.sum(axis=1)-joint.diagonal()
        rank=np.lexsort((np.arange(n),-half));top=rank[:32]
        rows.append({'point':q['point'],'HF_probability':float(prob[0]),
                     'full_TV_from_original':float(.5*sum(abs(prob-base))),
                     'full_entropy_nats':float(-sum(prob[prob>0]*np.log(prob[prob>0]))),
                     'expected_unique_full':{str(M):float(sum(expected_seen(prob,M))) for M in (500,1000,5500)},
                     'expected_unique_half_with_HF_added':{str(M):float(1+sum(expected_seen(union[1:],M))) for M in (500,1000,5500)},
                     'expected_discovered_exact_top32_half':{str(M):float(sum(expected_seen(union[top],M))) for M in (500,1000,5500)}})
    save(ROOT/'analysis/N40-source-distribution-diagnostic.json',
         {'status':'POST_OUTCOME_NINE_FROZEN_PROBABILITY_SOURCES_ANALYZED','rows':rows,
          'privileged_exact_model_probabilities_not_hardware_accessible':True,
          'used_for_original_selection':False,'FCI_used':False,'new_shots_CI_energy':[0,0,0],
          'new_quantum_necessity_or_novelty_claim':False,'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-wall})
    print(rows)


if __name__=='__main__':main()
