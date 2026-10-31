"""Independent persisted-evidence reconciliation; no new molecular evaluation."""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from qiskit import qpy
from benchmark.ansatz import resources,measurement_equivalent_circuit
from paperstudy.ci import interleaved_to_blocked
from .paths import ROOT,OLD
from .io import load,save,sha256,digest,write_bytes,verify_previous_completion


def normalize(actions):
    return [{'source':list(a['source']),'target':list(a['target']),'theta':a['theta']}for a in actions]


def weighted_p95(values,weights):
    order=np.argsort(values);target=.95*sum(weights);cumulative=0
    for i in order:
        cumulative+=weights[i]
        if cumulative>=target:return values[i]
    raise RuntimeError('Invalid weighted quantile')


def main():
    completion=load(ROOT/'FINAL_COMPLETION.json');checks=[];aliases=[];tails=[]
    for relative,h in completion['evidence_SHA256'].items():
        if sha256((ROOT/relative).read_bytes())!=h:raise RuntimeError('Completion evidence changed: '+relative)
    with sqlite3.connect('file:'+str(ROOT/'shots.sqlite')+'?mode=ro',uri=True)as db:
        charged,unknown=db.execute('SELECT SUM(actual),SUM(actual IS NULL)FROM draws').fetchone()
        if charged!=5700000 or unknown:raise RuntimeError('Shared ledger totals do not reconcile')
        ledger={(r,e):(n,h)for r,e,n,h in db.execute('SELECT run,event,actual,artifact_digest FROM draws')}
    for filename in sorted((ROOT/'runs/DEVELOPMENT').glob('*/*/*/result.json')):
        r=load(filename);directory=filename.parent;meta=load(OLD/'data/physical'/r['case_id']/'problem.json')
        with np.load(directory/'final-counts.npz',allow_pickle=False)as a:
            final={int(x):int(n)for x,n in zip(a['blocked_ids'],a['counts'])}
        for x in final:
            mask=(1<<meta['norb'])-1
            if((x&mask).bit_count(),(x>>meta['norb']).bit_count())!=(meta['nalpha'],meta['nbeta']):raise RuntimeError('Sector mismatch')
        pooled={};draws=[]
        if r['variant']in('D0','D1','D2','D3'):
            circuit_index={}
            for path in(directory/'circuits').glob('*.json'):
                c=load(path);circuit_index[digest(normalize(c['actions']))]=(path,c)
            for member in r['members']:
                path,c=circuit_index[digest(normalize(member['actions']))]
                # Native D0 includes the redundant n_qubits field in its
                # summary actions; trace/QPY identity omits that field.
                if c['circuit_id']!=member['circuit_id']:
                    if r['variant']!='D0':raise RuntimeError('Unexpected circuit identity mismatch')
                    aliases.append({'run_id':r['run_id'],'native_summary_circuit_id':member['circuit_id'],
                        'actual_trace_QPY_circuit_id':c['circuit_id'],'exact_normalized_actions_equal':True,
                        'reason':'Native summary includes n_qubits in each action; physical trace binds it once globally',
                        'source_result_unchanged':True})
                q=qpy.load(BytesIO(path.with_suffix('.qpy').read_bytes()))[0]
                sampling,_=measurement_equivalent_circuit(q)
                recount=resources(sampling,len(member['actions']),len(member['actions']))
                for field in('cnot_count','cnot_depth','total_depth'):
                    if recount[field]!=member['sampling_resources'][field]:raise RuntimeError('Resource recount mismatch')
            for p in(directory/'draws').glob('*/draw.json'):
                draw=load(p);blob=(p.parent/'counts.npz').read_bytes()
                if sha256(blob)!=draw['counts_SHA256']:raise RuntimeError('Draw artifact mismatch')
                if ledger[(r['run_id'],draw['event_id'])]!=(draw['actual_shots'],draw['counts_SHA256']):raise RuntimeError('Draw/ledger mismatch')
                draws.append(draw)
                role=draw['request']['role']
                if r['variant']!='D0'and isinstance(role,list)and role[0]=='independent-final':
                    with np.load(BytesIO(blob),allow_pickle=False)as a:
                        for x,n in zip(a['ids'],a['counts']):
                            x=interleaved_to_blocked(x,meta['norb']);pooled[x]=pooled.get(x,0)+int(n)
            if r['variant']=='D0':
                from benchmark.local_sampler_primitives import unpack_counts64
                with sqlite3.connect('file:'+str(directory/'native/events.sqlite')+'?mode=ro',uri=True)as db:
                    values=db.execute("SELECT counts64 FROM draws WHERE role='independent_final'").fetchall()
                if len(values)!=1:raise RuntimeError('Native independent final cardinality mismatch')
                ids,counts=unpack_counts64(values[0][0])
                pooled={interleaved_to_blocked(x,meta['norb']):int(n)for x,n in zip(ids,counts)}
            if pooled!=final:raise RuntimeError('Independent final counts reconstruction mismatch')
            if sum(d['actual_shots']for d in draws)!=r['budget']:raise RuntimeError('Raw draw totals mismatch')
            applications=sum(d['actual_shots']*d['metadata']['sampling_resources']['cnot_count']for d in draws)
            if applications!=r['all_model_CNOT_applications']:raise RuntimeError('CNOT-application accounting mismatch')
            for field in('cnot_count','cnot_depth','total_depth'):
                values=[x['sampling_resources'][field]for x in r['members']];weights=[x['shots']for x in r['members']]
                tails.append({'run_id':r['run_id'],'case_id':r['case_id'],'method':r['variant'],'seed':r['seed'],
                    'metric':field,'maximum':max(values),'shot_weighted_p95':weighted_p95(values,weights),
                    'member_count':len(values),'member_weights':weights})
        else:
            draw=load(directory/'draw.json')
            if sha256((directory/'final-counts.npz').read_bytes())!=draw['counts_SHA256']:raise RuntimeError('LUCJ counts mismatch')
            event=digest(draw['request'])
            if ledger[(r['run_id'],event)]!=(100000,draw['counts_SHA256']):raise RuntimeError('LUCJ ledger mismatch')
            q=qpy.load(BytesIO((ROOT/'controls'/r['case_id']/r['variant']/'sampling.qpy').read_bytes()))[0]
            actual={'cnot_count':q.count_ops().get('cx',0),'cnot_depth':q.depth(filter_function=lambda x:x.operation.name=='cx'),
                'total_depth':q.depth()}
            if actual!=r['sampling_resources']:raise RuntimeError('LUCJ direct logical recount mismatch')
            for field,value in actual.items():tails.append({'run_id':r['run_id'],'case_id':r['case_id'],'method':r['variant'],
                'seed':r['seed'],'metric':field,'maximum':value,'shot_weighted_p95':value,'member_count':1,'member_weights':[100000]})
        counts_sha=sha256((directory/'final-counts.npz').read_bytes())
        previous=None
        for c in r['final_CI']:
            if c['source_counts_SHA256']!=counts_sha:raise RuntimeError('CI evidence/actual counts mismatch')
            if previous is not None and c['energy_Ha']>previous+1e-7:raise RuntimeError('Nonmonotone nested Cartesian CI energies')
            previous=c['energy_Ha']
        checks.append({'run_id':r['run_id'],'case':r['case_id'],'method':r['variant'],'seed':r['seed'],
            'raw_counts_ledger_final_CI_binding':True,'resource_recount':True,'sector_all_final_counts':True})
    frozen=load(ROOT/'qualification/scientific-code.json')['code_SHA256']
    for name,h in frozen.items():
        if sha256((ROOT/'raacsnext'/name).read_bytes())!=h:raise RuntimeError('Frozen molecular code changed')
    if len(checks)!=57:raise RuntimeError('Incomplete final audit')
    occupied=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
    free=shutil.disk_usage(ROOT).free
    if occupied>=8*2**30 or free<3*2**30:raise RuntimeError('Final disk limits not satisfied')
    from .report import csv_file
    csv_file('final-member-resource-tails.csv',tails)
    report={'status':'PASS_WITH_DISCLOSED_NONPERFORMANCE_LIMITATIONS','runs_checked':len(checks),'checks':checks,
        'shared_model_shots':charged,'unknown_draws':unknown,'new_molecular_evaluations_by_this_audit':0,
        'resource_recount_new_classical_work_not_quantum_shots':True,'D0_native_summary_trace_aliases':aliases,
        'existing_numerical_results_edited':False,'frozen_scientific_code_unchanged':True,
        'old_study_check':verify_previous_completion(),'study_bytes':occupied,'free_bytes':free,
        'CPU_cap_boundary_overshoots_disclosed':True,'strict_CPU_cap_proof':False,
        'full_paper_superiority_proof':False,'pilot_stop_decision_unchanged':True}
    save(ROOT/'report/final-independent-evidence-audit.json',report)
    print({'runs':len(checks),'D0_aliases':len(aliases),'model_shots':charged,'study_MiB':occupied/2**20},flush=True)


if __name__=='__main__':main()
