"""Complete only four unmeasured N62 draws using fixed exported packets."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import json,os,subprocess,sys,traceback
import numpy as np
from .filter_only_plan import LOCAL,verified
from .io import load,save,write,arrays,sha256,space
from .paths import ROOT
from .filtered_SqDRIFT_source import load_member,cpu_total
from .strong_bank_prepare import prepare
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked
from paperstudy.budget import BudgetLedger

STAGE=LOCAL/'targeted-successor-v1'


def verify(items,folder):
    results={};cpu=0.
    for i,item in enumerate(items):
        request=dict(jobs=[item['request']],tolerance=1e-8)
        r=subprocess.run([sys.executable,'-m','frontierloop.targeted_frame_batch_worker'],input=json.dumps(request),
            capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT)))
        record=dict(index=i,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
            request_SHA256=sha256(json.dumps(request,sort_keys=True).encode()))
        save(folder/('checker-process-'+str(i)+'.json'),record)
        if r.returncode:raise RuntimeError('Targeted worker exit '+str(r.returncode))
        out=json.loads(r.stdout);cpu+=out['CPU_seconds'];results.update(out['results'])
        if any(x['status']!='PASSED' for x in out['results'].values()):raise ValueError('Targeted independent check rejected; full error preserved')
    return dict(results=results,CPU_seconds=cpu,verifier='TARGETED_CLIFFORD_ALL3_PROBABILITY_PLUS_LEAKAGE_AND_DISCARD_BOUND',new_model_shots=0,new_QPU_shots=0)


def run():
    p=verified();space()
    if STAGE.exists():raise RuntimeError('Existing successor; reconcile rather than rerun')
    qualification=load(ROOT/'runs/N62-TARGETED-CLIFFORD-QUALIFICATION-v1/terminal.json')
    if qualification['returncode'] or any(x['status']!='PASSED' for x in qualification['parsed']['results'].values()):raise ValueError('Real failed-member parity required')
    old=load(LOCAL/'source-terminal.json');draws=[d for d in p['draws'] if not (LOCAL/'draws'/d['draw_id']/'result.json').exists()]
    assert old['completed']==8 and len(draws)==4
    ledger=BudgetLedger(LOCAL/'shots.sqlite')
    for d in draws:
        snap=ledger.snapshot(d['draw_id'])
        assert snap['known_actual_shots']==snap['unknown_or_inflight_draws']==0
    save(STAGE/'manifest.json',dict(draws=draws,old_source_terminal_SHA256=sha256((LOCAL/'source-terminal.json').read_bytes()),
        original_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        implementation_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ['filter_only_targeted_successor.py','clifford_targeted_checker.py','stabilizer_target_amplitudes.py','targeted_frame_batch_worker.py']},
        real_qualification_SHA256=sha256((ROOT/'runs/N62-TARGETED-CLIFFORD-QUALIFICATION-v1/terminal.json').read_bytes()),
        old8_not_resampled=True,old_failures_not_overwritten=True,exported_packets_reused_without_regeneration=True,
        source_settings_and_sampling_seeds_unchanged=True,probability_L1_tolerance=1e-8,worker_CPU_cap=30,
        new_source_CPU_cap_seconds=240,new_source_wall_cap_seconds=600,extra_budget_explicit_not_old600cap_increase=True,
        old8_fingerprints={d['draw_id']:sha256((LOCAL/'draws'/d['draw_id']/'result.json').read_bytes()) for d in p['draws'] if d not in draws}))
    cpu,wall=cpu_total(),perf_counter();completed=[];failure=None
    def bound():
        space()
        if cpu_total()-cpu>240 or perf_counter()-wall>600:raise TimeoutError('Finite numerical successor budget')
        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_output_bytes'] or sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())>p['research_root_bytes']:raise OSError('Original storage bounds')
    try:
        for d in draws:
            bound();c=next(c for c in p['cases'] if c['case']==d['case']);meta=load(c['meta']);meta['case_id']=c['case']
            folder=LOCAL/'draws'/d['draw_id'];assert load(folder/'request.json')==d
            items=[];members=[]
            for m in d['members']:
                bound();target=LOCAL/'exports'/c['case']/'members'/m['member_id'];member=load_member(target)
                if member['spec']!=m['spec'] or member['diagonal_terms_retained'] is not False:raise ValueError('Original packet mismatch')
                artifacts={name:dict(path=str(target/name),sha256=sha256((target/name).read_bytes())) for name in ['member.json.gz','circuit.qpy.gz']}
                plan=dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],artifacts=artifacts)
                items.append(prepare(meta,plan,{str(target/name):(target/name).read_bytes() for name in artifacts}));members.append(member)
            checks=verify(items,STAGE/d['draw_id']);save(folder/'compiler-check.json',checks);bound();merged=Counter();records=[]
            for m,member,item in zip(d['members'],members,items):
                state,numerics=member_state(meta,member);target=LOCAL/'exports'/c['case']/'members'/m['member_id']
                req=dict(member_id=m['member_id'],problem_id=d['problem_id'],readout_seed=m['readout_seed'],shots=m['shots'],source_packet_SHA256=sha256((target/'member.json.gz').read_bytes()))
                if ledger.reserve(d['draw_id'],m['member_id'],req,m['shots'])!='NEW_RESERVATION':raise RuntimeError('Unknown/double draw; no retry')
                ids=np.asarray(sorted(state),np.uint64);prob=np.array([abs(state[int(i)])**2 for i in ids]);prob/=prob.sum()
                counts=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],prob);keep=counts>0
                blob=arrays(ids=ids[keep],counts=counts[keep]);write(folder/(m['member_id']+'.npz'),blob);ledger.complete(d['draw_id'],m['member_id'],int(counts.sum()),sha256(blob))
                for x,w in zip(ids[keep],counts[keep]):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
                rec=item['row'];rec.update(request=req,state_numerics=numerics,counts_SHA256=sha256(blob));records.append(rec)
                write(folder/(m['member_id']+'-certificate.json.gz'),item['certificate'])
                for name,b in item['qpy_blobs'].items():write(folder/(m['member_id']+'-'+name+'.qpy.gz'),b)
            ids=np.asarray(sorted(merged),np.uint64);counts=np.array([merged[int(x)] for x in ids],np.uint64);assert int(counts.sum())==1000
            blob=arrays(blocked_ids=ids,counts=counts);write(folder/'counts.npz',blob);fields=['cnot_count','cnot_depth','total_unitary_depth']
            resources={name:{f:sum(m['shots']*r['resources'][name][f] for m,r in zip(d['members'],records))/1000 for f in fields} for name in ['NATIVE','RUSTIQ','PROJECTED']}
            resources['SELECTED']={f:sum(m['shots']*r['policy']['selected_resources'][f] for m,r in zip(d['members'],records))/1000 for f in fields}
            result=dict(status='COMPLETED',**{k:d[k] for k in ['draw_id','case','arm','seed','problem_id']},members=records,resources=resources,
                counts_SHA256=sha256(blob),budget=ledger.snapshot(d['draw_id']),new_model_samples=1000,new_QPU_shots=0,
                coefficient_slots_mean=sum(m['shots']*r['coefficient_slots_original'] for m,r in zip(d['members'],records))/1000,
                independent_parameter_rank=None,VQE_optimized_variables=0,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
                numerical_successor_manifest_SHA256=sha256((STAGE/'manifest.json').read_bytes()))
            save(folder/'result.json',result);completed.append(d['draw_id']);save(STAGE/'progress.json',dict(completed=completed))
            print('N62_TARGETED_DRAW',d['case'],d['seed'],len(completed),flush=True)
    except Exception as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc());save(STAGE/'failure.json',dict(failure,completed=completed));raise
    finally:
        save(STAGE/'final-accounting.json',dict(CPU_seconds_including_children=cpu_total()-cpu,wall_seconds=perf_counter()-wall,
            completed=completed,failure=failure,no_persistent_exporter_created=True,new_model_samples=1000*len(completed)))
    save(STAGE/'terminal.json',dict(status='ALL4_PREVIOUSLY_UNMEASURED_DRAWS_COMPLETED',completed=completed,
        new_model_samples=4000,new_QPU_shots=0,total_N62_model_samples=12000,goal_complete=False))


if __name__=='__main__':run()
