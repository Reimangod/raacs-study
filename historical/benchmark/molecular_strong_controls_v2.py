"""Versioned dependency injection of verified spin-blocked MPS into native v1.

No monkeypatched package/vendor files. Context-local runtime providers are
explicitly bound in the outer manifest and draw evidence. V1 never rewritten.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import ROOT,file_digest,save_json
from . import molecular_strong_controls as native
from . import resource_adaptive_sampler as adaptive
from .molecular_lucj_primitives import initial_amplitudes as original_amplitudes,fixed_integral_reference
from .spin_blocked_sampling import SpinBlockedCountSampler

original_configuration=native.configuration


def configuration(mode):
    cfg=original_configuration(mode);cfg['schema']='molecular-strong-controls-v2-spin-blocked'
    cfg['sampling']['max_bond']=128
    cfg['layout']='literal spin-blocked simulator qubit permutation; measured bits remapped'
    cfg['minimum_retained_norm_squared']=SpinBlockedCountSampler.minimum_retained_norm_squared
    cfg['qualification_manifest_sha256']=file_digest(ROOT/'results/molecular-MPS-layout-qualification-v1/manifest.json')
    cfg['qualification_result_sha256']=file_digest(ROOT/'results/molecular-MPS-layout-qualification-v1/execution-status.json')
    cfg['successor_spec_sha256']=file_digest(ROOT/'study-v1/MOLECULAR_STRONG_CONTROLS_V2.md')
    for f in ('molecular_strong_controls_v2.py','spin_blocked_sampling.py','mps_layout_qualification.py'):
        cfg['implementation'][f]=file_digest(ROOT/'benchmark'/f)
    return cfg


def checked_amplitudes(backend,info,folder,kind,cycles,seconds):
    t1,t2,record=original_amplitudes(backend,info,folder,kind,cycles,seconds)
    if kind=='mp2' and record['qualification'].get('UHF_stationary') is False:
        from pyscf import mp
        from time import perf_counter
        path=Path(folder)/'mp2-noncanonical-check-v2.json'
        if path.exists():
            check=json.loads(path.read_text());assert check['source_sha256']==file_digest(Path(folder)/'mp2.json')
        else:
            start=perf_counter();mf,_=fixed_integral_reference(backend,info);solver=mp.MP2(mf);solver.max_memory=768;solver.kernel()
            check={'source_sha256':file_digest(Path(folder)/'mp2.json'),'native_iterative_converged':bool(solver.converged),
                'maximum_t2_difference':max(float(np.max(abs(a-b))) for a,b in zip(t2,solver.t2)),
                'wall_seconds':perf_counter()-start,'new_model_shots':0,'FCI_used':False}
            save_json(path,check)
        if not check['native_iterative_converged'] or check['maximum_t2_difference']>=1e-7:
            raise RuntimeError('Preserved amplitude initialization failure: native noncanonical MP2 check')
    return t1,t2,record


def run(output,mode):
    if mode=='standard':
        p=ROOT/'results/molecular-strong-controls-quick-v2/physical-layout-validation-v2.json'
        if not p.exists() or json.loads(p.read_text())['status']!='SPIN_BLOCKED_NATIVE_QUICK_QUALIFIED':
            raise RuntimeError('Actual successor quick bitmapping/norm/native controls qualification required')
    # The imported native functions use these explicit runtime providers; the
    # unchanged RAACS algorithm also receives the same qualified count engine.
    with patch.object(native,'configuration',configuration),patch.object(native,'CircuitCountSampler',SpinBlockedCountSampler),\
         patch.object(native,'initial_amplitudes',checked_amplitudes),patch.object(adaptive,'CircuitCountSampler',SpinBlockedCountSampler):
        return native.run(output,mode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','standard'],default='quick');p.add_argument('--output')
    a=p.parse_args();r=run(a.output or ROOT/f'results/molecular-strong-controls-{a.mode}-v2',a.mode)
    print(json.dumps({'status':r.get('status') if r else 'PENDING','whole_goal_complete':False}))
