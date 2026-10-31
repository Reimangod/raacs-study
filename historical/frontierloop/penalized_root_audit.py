"""Diagnose the saved D2 failure with an independent lowest-root solve, zero shots."""
from time import process_time,perf_counter
import numpy as np
import ffsim
import jax
from .paths import ROOT,OLD
from .io import load,save,arrays,write,sha256,space
from .spin_orbital import diagonalize
from .lowest_penalized import solve


def main():
    d=ROOT/'diagnostics/D2-N2-far-F25-C3-lowest-root-v1'
    if (d/'started.json').exists():raise RuntimeError('Preserve diagnostic prefix; do not repeat blindly')
    space();d.mkdir(exist_ok=True);source=ROOT/'runs/D2-COMPRESSION-SCREEN/N2-32-R2p4-P2/F25'
    req=load(source/'C3-SPIN-OO/request.json');counts_sha=sha256((source/'final-counts.npz').read_bytes())
    assert req['counts_SHA256']==counts_sha
    settings=load(ROOT/'protocol-C3-SPIN-OO.json');s=np.array(req['CI_strings'],np.int64)
    save(d/'started.json',{'counts_SHA256':counts_sha,'settings_SHA256':sha256((ROOT/'protocol-C3-SPIN-OO.json').read_bytes()),
        'diagnostic_code_SHA256':sha256(__import__('pathlib').Path(__file__).read_bytes()),
        'solver_code_SHA256':sha256((ROOT/'frontierloop/lowest_penalized.py').read_bytes()),
        'new_model_shots':0,'old_frozen_results_changed':False})
    p=OLD/'data/physical/N2-32-R2p4-P2';m=load(p/'problem.json');ne=m['nalpha'],m['nbeta']
    with np.load(p/'integrals.npz',allow_pickle=False) as a:ham=ffsim.MolecularHamiltonian(a['h1'],a['eri'],constant=m['core_energy'])
    jax.config.update('jax_enable_x64',True);cpu,tick=process_time(),perf_counter()
    old,rdm=diagonalize(ham,ne,s,settings)
    u,_=ffsim.optimize_orbitals(rdm,ham,real=True,method='L-BFGS-B',
        options={'maxiter':settings['optimizer_maxiter_per_outer'],'maxfun':settings['optimizer_maxfun_per_outer']},return_optimize_result=True)
    changed=ham.rotated(u.T.conj());bound=float(rdm.expectation(changed).real)+old['spin_penalty_energy_Ha']
    returned,_=diagonalize(changed,ne,s,settings)
    newer,vector,_=solve(changed,ne,s,seed=20261025)
    payload=arrays(h1=changed.one_body_tensor,eri=changed.two_body_tensor,CI_strings=s,vector=vector,rotation=u)
    write(d/'diagnostic-arrays.npz',payload)
    result={'status':'INDEPENDENT_LOWEST_ROOT_DIAGNOSTIC_NOT_PRIMARY_REPLACEMENT',
        'old_initial':old,'old_reset_Davidson':returned,'independent_SA':newer,
        'known_old_vector_bound_Ha':bound,'Davidson_minus_bound_Ha':returned['penalized_energy_Ha']-bound,
        'SA_minus_bound_Ha':newer['penalized_energy_Ha']-bound,
        'arrays_SHA256':sha256(payload),'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
        'new_model_shots':0,'old_frozen_results_changed':False,'goal_complete':False}
    save(d/'result.json',result);print(result,flush=True)


if __name__=='__main__':main()
