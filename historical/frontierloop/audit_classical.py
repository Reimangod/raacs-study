"""Check all fresh classical sources, full sparse spin action and 24 readouts."""
from collections import defaultdict
import sqlite3
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256
from .run_classical_control import freeze
from .singlet import spin_action
from .cisd_sparse import marginal_strings
from paperstudy.ci import common_strings


def main():
    assert freeze()==load(ROOT/'freeze-C6-CLASSICAL-CISD.json')
    spec=load(ROOT/'protocol-C6-CLASSICAL-CISD.json');sources=[];checks=[];draws=0
    for case in spec['cases']:
        d=ROOT/'runs/C6-CLASSICAL-CISD'/case/'source';r=load(d/'result.json')
        meta=load(OLD/'data/physical'/case/'problem.json')
        assert r['status']=='COMPLETED' and r['physical_problem_id']==meta['problem_id']
        assert r['freeze']==freeze() and r['converged'] and r['residual_norm']<1e-5
        assert r['quantum_model_shots']==r['actual_QPU_shots']==0 and not r['full_Fock_vector_used']
        payload=d/'sparse-CISD-state.npz';assert sha256(payload.read_bytes())==r['sparse_state_SHA256']
        with np.load(payload,allow_pickle=False) as a:ids,coeff=a['blocked_ids'],a['coefficients']
        assert abs(np.sum(abs(coeff)**2)-1)<1e-7
        action=defaultdict(complex);n=meta['norb'];mask=(1<<n)-1
        for x,v in zip(ids,coeff):
            x=int(x)
            assert (x&mask).bit_count()==meta['nalpha'] and (x>>n).bit_count()==meta['nbeta']
            if v==0:continue
            for y,c in spin_action(x,n).items():action[y]+=c*v
        full_spin_action_norm=float(np.sqrt(sum(abs(v)**2 for v in action.values())))
        assert full_spin_action_norm<1e-8
        p=d/'classical-counts.npz';assert sha256(p.read_bytes())==r['classical_counts_SHA256']
        with np.load(p,allow_pickle=False) as a:observed,counts=a['blocked_ids'],a['counts']
        assert int(counts.sum())==spec['classical_draws'];draws+=int(counts.sum())
        assert set(map(int,observed))<=set(int(x) for x,v in zip(ids,coeff) if v!=0)
        sources.append({'case':case,'source_S2_action_norm':full_spin_action_norm,
            'source_energy_Ha':r['energy_Ha'],'source_residual_norm':r['residual_norm'],
            'source_SHA256':sha256((d/'result.json').read_bytes())})
        for variant in spec['variants']:
            if variant=='DETERMINISTIC_CISD_MARGINAL':
                expected=marginal_strings(ids,coeff,n,meta['nalpha'],spec['r_cap']).tolist()
            else:
                expected=common_strings(observed,counts,n,(meta['nalpha'],meta['nbeta']),spec['r_cap'])[0].tolist()
            for mode in spec['readouts']:
                directory=d.parent/variant/mode;s=load(directory/'result.json')
                assert s['status']=='COMPLETED' and s['freeze']==freeze() and s['CI_strings']==expected
                assert s['source_result_SHA256']==sha256((d/'result.json').read_bytes())
                assert s['new_quantum_model_shots']==0 and not s['quantum_sampler_used']
                assert s['quantum_sampler_resources'] is None and not s['goal_complete']
                assert sha256((directory/'optimized-representation.npz').read_bytes())==s['representation_SHA256']
                with np.load(directory/'optimized-representation.npz',allow_pickle=False) as z:
                    assert z['CI_strings'].tolist()==expected
                    for u in z['rotations']:
                        np.testing.assert_allclose(u.T@u,np.eye(len(u)),atol=1e-9)
                field='penalized_energy_Ha' if mode=='C3-SPIN-OO' else 'energy_Ha'
                for row in s['history']:
                    assert row['CI_converged'] and row['residual_norm']<1e-5
                    if mode=='C3-SPIN-OO':
                        assert abs(row['penalized_energy_Ha']-row['energy_Ha']-row['spin_penalty_energy_Ha'])<1e-8
                    if mode=='C4-SINGLET-OO':assert abs(row['spin_square'])<1e-8
                assert all(b[field]<=a[field]+1e-8 for a,b in zip(s['history'],s['history'][1:]))
                checks.append({'case':case,'variant':variant,'readout':mode,'passed':True})
    with sqlite3.connect('file:'+str(ROOT/'production-shots.sqlite')+'?mode=ro',uri=True) as db:
        quantum,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL) FROM draws').fetchone()
    assert quantum==3500000 and unknown==0 and len(checks)==24
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert size<2**30
    save(ROOT/'CHECKPOINT_CLASSICAL_AUDIT.json',{'status':'VERIFIED_CLASSICAL_CONTROL_NOT_GOAL_COMPLETION',
        'sources':sources,'readouts':checks,'actual_classical_draws_generated_once':draws,
        'new_quantum_model_shots':0,'known_current_root_quantum_model_shots':quantum,
        'unknown_draws':unknown,'study_bytes':size,'goal_complete':False})
    print({'sources_checked':len(sources),'readouts_checked':len(checks),'classical_draws':draws,
        'new_quantum_model_shots':0,'study_MiB':size/2**20,'goal_complete':False})


if __name__=='__main__':main()
