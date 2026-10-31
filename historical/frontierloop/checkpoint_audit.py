"""Progress evidence audit, explicitly not a completion certificate."""
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import numpy as np
from qiskit import qpy
from benchmark.ansatz import resources,measurement_equivalent_circuit
from paperstudy.ci import interleaved_to_blocked
from raacsnext.io import verify_previous_completion
from .paths import ROOT,V2,OLD
from .io import load,save,sha256,digest


def main():
    with sqlite3.connect('file:'+str(ROOT/'production-shots.sqlite')+'?mode=ro',uri=True)as db:
        ledger={(run,event):(actual,artifact)for run,event,actual,artifact in db.execute('SELECT run,event,actual,artifact_digest FROM draws')}
        known,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL)FROM draws').fetchone()
    checks=[]
    paths=[*(ROOT/'runs/B1').glob('*/*/result.json'),*(ROOT/'runs/B2').glob('*/*/*/result.json'),*(ROOT/'runs/B3').glob('*/*/result.json')]
    for filename in paths:
        r=load(filename);directory=filename.parent;meta=load(OLD/'data/physical'/r['case_id']/'problem.json')
        with np.load(directory/'final-counts.npz',allow_pickle=False)as a:
            expected={int(x):int(n)for x,n in zip(a['blocked_ids'],a['counts'])}
        if sha256((directory/'final-counts.npz').read_bytes())!=r['final_counts_SHA256']:raise RuntimeError('Final count bytes changed')
        actual={};shots=0;cnotapps=0
        with sqlite3.connect('file:'+str(directory/'draws.sqlite')+'?mode=ro',uri=True)as db:
            for event,request,blob,metadata in db.execute('SELECT id,request,counts,metadata FROM draws'):
                request=json.loads(request);metadata=json.loads(metadata)
                with np.load(BytesIO(blob),allow_pickle=False)as a:ids,counts=a['ids'],a['counts']
                total=int(counts.sum());shots+=total
                if ledger[(r['run_id'],event)]!=(total,sha256(blob)):raise RuntimeError('Raw draw/ledger mismatch')
                cnotapps+=total*metadata['sampling_resources']['cnot_count']
                role=request['role']
                if isinstance(role,list)and role[0]=='independent-final':
                    for x,n in zip(ids,counts):
                        x=interleaved_to_blocked(x,meta['norb']);actual[x]=actual.get(x,0)+int(n)
        if actual!=expected or shots!=r['budget']or cnotapps!=r['cost']['all_model_CNOT_applications']:
            raise RuntimeError('Independent final or paid work accounting mismatch')
        if sum(expected.values())<r['budget']/2:raise RuntimeError('Independent final reserve violated')
        for member in r['members']:
            q=qpy.load(BytesIO((directory/'final-circuits'/(member['id']+'.qpy')).read_bytes()))[0]
            sampling,_=measurement_equivalent_circuit(q)
            rr=resources(sampling,len(member['actions']),len(member['actions']))
            if any(rr[k]!=member['sampling_resources'][k]for k in('cnot_count','cnot_depth','total_depth')):
                raise RuntimeError('Final physical circuit recount mismatch')
        for c in r['final_CI']:
            if not c['converged']or c['valid_sector_fraction']!=1 or c['source_counts_SHA256']!=r['final_counts_SHA256']:
                raise RuntimeError('Final CI does not bind qualified counts')
        for name,h in r['code_SHA256'].items():
            if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen run implementation changed')
        checks.append({'case_id':r['case_id'],'cycle':r['cycle'],'seed':r['seed'],'raw_counts_and_resources_checked':True})
    if len(paths)!=28 or known!=3100000 or unknown:raise RuntimeError('Expected completed research checkpoint does not reconcile')
    oo=list((ROOT/'runs/C2-OO').glob('*/*/result.json'))
    if len(oo)!=12:raise RuntimeError('OO comparisons incomplete')
    old=verify_previous_completion()
    volume=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
    if volume>2**30:raise RuntimeError('Development storage cap exceeded')
    save(ROOT/'CHECKPOINT_AUDIT.json',{'goal_complete':False,'status':'VERIFIED_RESEARCH_PROGRESS',
        'sampler_runs_checked':len(paths),'OO_cells_completed':len(oo),'model_shots_known':known,
        'unknown_draws':unknown,'study_bytes':volume,'checks':checks,'prior_study_check':old,
        'all_four_target_passed':False,'source_aligned_strongest_comparison_complete':False,
        'spin_certification_complete':False,'independent_confirmation_complete':False,
        'no_scientific_success_inferred_from_tests':True})
    print({'sampler_runs':len(paths),'OO_cells':len(oo),'model_shots':known,'study_MiB':volume/2**20,'goal_complete':False},flush=True)


if __name__=='__main__':main()
