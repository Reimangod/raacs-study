"""Bounded evidence checks for the 24 new zero-shot spin/OO analyses."""
import sqlite3
import numpy as np
from .paths import ROOT
from .io import load,save,sha256
from .run_spin_orbital import freeze as c3freeze
from .run_singlet_orbital import freeze as c4freeze


def main():
    assert c3freeze()==load(ROOT/'freeze-C3-SPIN-OO.json')
    assert c4freeze()==load(ROOT/'freeze-C4-SINGLET-OO.json')
    checks=[]
    for cycle in ('C3-SPIN-OO','C4-SINGLET-OO'):
        protocol=load(ROOT/('protocol-'+cycle+'.json'))
        for case in protocol['cases']:
            for method in protocol['methods']:
                directory=ROOT/'runs'/cycle/case/method
                r=load(directory/'result.json');request=load(directory/'request.json')
                assert all(r[k]==v for k,v in request.items())
                assert r['new_quantum_model_shots']==0 and not r['world_best_claim_authorized']
                assert sha256((directory/'optimized-representation.npz').read_bytes())==r['representation_SHA256']
                with np.load(directory/'optimized-representation.npz',allow_pickle=False) as a:
                    assert a['CI_strings'].tolist()==r['CI_strings']
                    assert a['rotations'].shape[0]==len(r['history'])-1
                    for rotation in a['rotations']:
                        np.testing.assert_allclose(rotation.T@rotation,np.eye(len(rotation)),atol=1e-9)
                previous=load(ROOT/'runs/C2-OO'/case/method/'request.json')
                assert r['source']==previous['source'] and r['CI_strings']==previous['CI_strings']
                source=r['source']
                if source.get('source_result'):
                    path=__import__('pathlib').Path(source['source_result'])
                    assert sha256(path.read_bytes())==source['source_result_SHA256']
                    assert sha256((path.parent/'final-counts.npz').read_bytes())==source['source_counts_SHA256']
                for row in r['history']:
                    assert row['CI_converged'] and row['residual_norm']<=1e-5
                    if cycle=='C3-SPIN-OO':
                        assert abs(row['penalized_energy_Ha']-row['energy_Ha']-row['spin_penalty_energy_Ha'])<1e-8
                    else:
                        assert abs(row['spin_square'])<1e-8 and row['exact_singlet_projection']
                field='penalized_energy_Ha' if cycle=='C3-SPIN-OO' else 'energy_Ha'
                assert all(b[field]<=a[field]+1e-8 for a,b in zip(r['history'],r['history'][1:]))
                checks.append({'cycle':cycle,'case':case,'method':method,'passed':True})
    with sqlite3.connect('file:'+str(ROOT/'production-shots.sqlite')+'?mode=ro',uri=True) as db:
        known,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL) FROM draws').fetchone()
    assert known==3100000 and unknown==0
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert size<2**30
    save(ROOT/'CHECKPOINT_SPIN_AUDIT.json',{
        'status':'VERIFIED_RESEARCH_PROGRESS_NOT_GOAL_COMPLETION','checked_new_cells':len(checks),
        'checks':checks,'known_production_model_shots':known,'unknown_draws':unknown,
        'additional_model_shots':0,'study_bytes':size,'goal_complete':False,
        'strongest_source_aligned_controls_complete':False,'independent_confirmation_complete':False})
    print({'checked_new_cells':len(checks),'additional_model_shots':0,'study_MiB':size/2**20,'goal_complete':False})


if __name__=='__main__':
    main()
