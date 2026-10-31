"""Reconstruct the four new sampler runs from paid draws and real circuits."""
from io import BytesIO
import json
import sqlite3
import numpy as np
from qiskit import qpy
from benchmark.ansatz import resources,measurement_equivalent_circuit
from paperstudy.ci import interleaved_to_blocked
from .paths import ROOT,OLD
from .io import load,save,sha256
from .run_b4 import freeze


def main():
    assert freeze()==load(ROOT/'freeze-B4.json')
    spec=load(ROOT/'protocol-B4.json');checks=[]
    with sqlite3.connect('file:'+str(ROOT/'production-shots.sqlite')+'?mode=ro',uri=True) as db:
        ledger={(run,event):(actual,artifact) for run,event,actual,artifact in
            db.execute('SELECT run,event,actual,artifact_digest FROM draws')}
        known,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL) FROM draws').fetchone()
    for case in spec['cases']:
        for seed in spec['seeds']:
            d=ROOT/'runs/B4'/case/str(seed);r=load(d/'result.json')
            m=load(OLD/'data/physical'/case/'problem.json')
            assert r['status']=='COMPLETED' and r['freeze']==freeze() and r['problem_id']==m['problem_id']
            payload=d/'final-counts.npz';assert sha256(payload.read_bytes())==r['final_counts_SHA256']
            with np.load(payload,allow_pickle=False) as a:
                expected={int(x):int(v) for x,v in zip(a['blocked_ids'],a['counts'])}
            assert sum(expected.values())>=r['budget']/2
            for x in expected:
                assert (x&((1<<m['norb'])-1)).bit_count()==m['nalpha']
                assert (x>>m['norb']).bit_count()==m['nbeta']
            actual={};shots=0;cnotapps=0
            with sqlite3.connect('file:'+str(d/'draws.sqlite')+'?mode=ro',uri=True) as db:
                for event,request,blob,metadata in db.execute('SELECT id,request,counts,metadata FROM draws'):
                    request=json.loads(request);metadata=json.loads(metadata)
                    with np.load(BytesIO(blob),allow_pickle=False) as a:ids,counts=a['ids'],a['counts']
                    total=int(counts.sum());shots+=total
                    assert ledger[(r['run_id'],event)]==(total,sha256(blob))
                    cnotapps+=total*metadata['sampling_resources']['cnot_count']
                    role=request['role']
                    if isinstance(role,list) and role[0]=='independent-final':
                        for x,v in zip(ids,counts):
                            x=interleaved_to_blocked(x,m['norb']);actual[x]=actual.get(x,0)+int(v)
            assert actual==expected and shots==r['budget']==100000
            assert cnotapps==r['cost']['all_model_CNOT_applications']
            for member in r['members']:
                circuit=qpy.load(BytesIO((d/'final-circuits'/(member['id']+'.qpy')).read_bytes()))[0]
                sampling,_=measurement_equivalent_circuit(circuit)
                rr=resources(sampling,len(member['actions']),len(member['actions']))
                assert all(rr[k]==member['sampling_resources'][k] for k in ('cnot_count','cnot_depth','total_depth'))
            for c in r['final_CI']:
                assert c['CI_converged'] and abs(c['spin_square'])<1e-8 and c['residual_norm']<1e-5
                assert c['actual_d']<=c['r_cap']**2 and c['source_counts_SHA256']==r['final_counts_SHA256']
            oo=r['final_singlet_OO']
            assert sha256((d/'optimized-representation.npz').read_bytes())==oo['representation_SHA256']
            assert abs(oo['initial_energy_Ha']-r['final_CI'][-1]['energy_Ha'])<1e-8
            assert all(abs(x['spin_square'])<1e-8 for x in oo['history'])
            assert all(b['energy_Ha']<=a['energy_Ha']+1e-8 for a,b in zip(oo['history'],oo['history'][1:]))
            checks.append({'case':case,'seed':seed,'raw_draws_ledger_final_counts_circuits_and_spin_checked':True})
    assert known==3500000 and unknown==0
    save(ROOT/'CHECKPOINT_B4_AUDIT.json',{'status':'VERIFIED_FOUR_DEVELOPMENT_RUNS_NOT_GOAL_COMPLETION',
        'checks':checks,'new_model_shots':400000,'known_global_production_model_shots':known,
        'unknown_draws':unknown,'goal_complete':False})
    print({'B4_runs_checked':len(checks),'new_model_shots':400000,'unknown_draws':unknown,'goal_complete':False})


if __name__=='__main__':main()
