"""Additive no-shot diagnostic; leaves D2 result and frozen solver unchanged."""
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,space
from . import spin_orbital


def main():
    dest=ROOT/'diagnostics/D2-N2-far-F25-C3-diagnostic-v1.json'
    if dest.exists():print('EXISTING_DIAGNOSTIC_NO_RERUN');return
    space();source=ROOT/'runs/D2-COMPRESSION-SCREEN/N2-32-R2p4-P2/F25'
    req=load(source/'C3-SPIN-OO/request.json');draw=load(source/'draw-result.json')
    assert req['counts_SHA256']==draw['counts_SHA256']==sha256((source/'final-counts.npz').read_bytes())
    p=OLD/'data/physical/N2-32-R2p4-P2';m=load(p/'problem.json')
    with np.load(p/'integrals.npz',allow_pickle=False) as a:h,g=a['h1'],a['eri']
    records=[];previous=[];original=spin_orbital.diagonalize
    spec=load(ROOT/'protocol-C3-SPIN-OO.json')
    def traced(ham,nelec,strings,settings):
        result,rdm=original(ham,nelec,strings,settings)
        item={'call':len(records),'result':result}
        if previous:
            old,old_rdm=previous[-1]
            bound=float(old_rdm.expectation(ham).real)+old['spin_penalty_energy_Ha']
            item.update(known_old_vector_penalized_Rayleigh_Ha=bound,
                returned_minus_known_vector_Ha=result['penalized_energy_Ha']-bound,
                original_monotonicity_tolerance_Ha=spec['monotonicity_tolerance_Ha'])
        records.append(item);previous.append((result,rdm));return result,rdm
    cpu,tick=process_time(),perf_counter();error=None
    spin_orbital.diagonalize=traced
    try:spin_orbital.optimize(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),np.array(req['CI_strings'],np.int64),spec)
    except Exception as exc:error=repr(exc)
    finally:spin_orbital.diagonalize=original
    result={'status':'ADDITIVE_DIAGNOSTIC_NOT_REPLACEMENT_RESULT','records':records,'exception':error,
        'counts_SHA256':req['counts_SHA256'],'source_request_SHA256':sha256((source/'C3-SPIN-OO/request.json').read_bytes()),
        'solver_file_SHA256':sha256((ROOT/'frontierloop/spin_orbital.py').read_bytes()),
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
        'new_model_shots':0,'source_results_or_tolerances_changed':False,'goal_complete':False}
    save(dest,result);print(result,flush=True)


if __name__=='__main__':main()
