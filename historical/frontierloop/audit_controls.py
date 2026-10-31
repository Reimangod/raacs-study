"""Current-state C5 evidence audit; unfinished work stays unfinished."""
from collections import Counter
from pathlib import Path
from time import time
import sqlite3
import numpy as np
from .paths import ROOT
from .io import load,save,sha256
from .run_controls import freeze
from .control_sources import load_source


def main():
    assert freeze()==load(ROOT/'freeze-C5-CONTROLS.json')
    spec=load(ROOT/'protocol-C5-CONTROLS.json');manifest=load(ROOT/'manifest-C5-CONTROLS.json')
    checked=[];statuses=Counter();sources=0
    for cell in manifest['cells']:
        if cell['status']=='AVAILABLE':
            ids,counts=load_source(cell);sources+=1
            assert int(counts.sum())==cell['source_final_model_shots']
        selected_strings=[]
        for readout in spec['readouts']:
            d=ROOT/'runs/C5-CONTROLS'/cell['case']/cell['method']/readout
            p=d/'result.json';failure=d/'failure.json'
            r=load(p) if p.exists() else load(failure) if failure.exists() else None
            if not r:
                statuses['NOT_TERMINAL']+=1
                continue
            statuses[r['status']]+=1
            if r['status']!='COMPLETED':
                if r['status']=='MISSING_SAME_PROBLEM_SOURCE':
                    assert cell['status']==r['status'] and r['new_quantum_model_shots']==0
                continue
            assert r['freeze']==freeze() and r['source']==cell
            assert r['physical_problem_id']==cell['expected_problem_id']
            assert r['new_quantum_model_shots']==0 and not r['world_best_claim_authorized']
            request=load(d/'request.json');assert all(r[k]==v for k,v in request.items())
            assert sha256((d/'optimized-representation.npz').read_bytes())==r['representation_SHA256']
            with np.load(d/'optimized-representation.npz',allow_pickle=False) as z:
                assert z['CI_strings'].tolist()==r['CI_strings']
                for u in z['rotations']:
                    np.testing.assert_allclose(u.T@u,np.eye(len(u)),atol=1e-9)
            selected_strings.append(r['CI_strings'])
            field='penalized_energy_Ha' if readout=='C3-SPIN-OO' else 'energy_Ha'
            for row in r['history']:
                assert row['residual_norm']<=1e-5 and row['CI_converged']
                if readout=='C3-SPIN-OO':
                    assert abs(row['penalized_energy_Ha']-row['energy_Ha']-row['spin_penalty_energy_Ha'])<1e-8
                if readout=='C4-SINGLET-OO':
                    assert abs(row['spin_square'])<1e-8 and row['exact_singlet_projection']
            assert all(b[field]<=a[field]+1e-8 for a,b in zip(r['history'],r['history'][1:]))
            checked.append({'case':cell['case'],'method':cell['method'],'readout':readout,'passed':True})
        assert not selected_strings or all(s==selected_strings[0] for s in selected_strings)
    with sqlite3.connect('file:'+str(ROOT/'production-shots.sqlite')+'?mode=ro',uri=True) as db:
        known,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL) FROM draws').fetchone()
    total_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert total_bytes<2**30
    save(ROOT/'CHECKPOINT_CONTROLS_AUDIT.json',{'status':'VERIFIED_CURRENT_PROGRESS_NOT_GOAL_COMPLETION',
        'asof_unix':time(),'statuses':dict(statuses),'qualified_completed_readouts':len(checked),'checks':checked,
        'all_input_sources_hash_verified':sources,'known_global_production_shots':known,
        'unknown_global_draws':unknown,'C5_additional_model_shots':0,'study_bytes':total_bytes,
        'C5_finished':(ROOT/'cycle-C5-CONTROLS-runs-finished.json').exists(),'goal_complete':False})
    print({'statuses':dict(statuses),'source_hashes_checked':sources,'study_MiB':total_bytes/2**20,
        'C5_additional_model_shots':0,'goal_complete':False})


if __name__=='__main__':
    main()
