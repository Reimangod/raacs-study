"""Complete only six unmeasured Fe draws; split the SAME frame checks by member."""
from collections import Counter
from pathlib import Path
from time import process_time,perf_counter
import os,sys,json,subprocess,traceback
import numpy as np
from .filtered_SqDRIFT_plan import LOCAL,verified
from .filtered_SqDRIFT_source import Session,load_member,cpu_total
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from .strong_bank_prepare import prepare as prepare_compiler
from .strong_projection import OVERLAY
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked
from paperstudy.budget import BudgetLedger

TAIL=LOCAL/'source-frame-successor-v1'


def frame_verify(items,folder):
    results={};cpu=0.;wall=perf_counter()
    for i,item in enumerate(items):
        request=dict(jobs=[item['request']],tolerance=1e-8)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(OVERLAY),str(ROOT)]))
        r=subprocess.run([sys.executable,'-m','frontierloop.frame_batch_worker'],input=json.dumps(request),
            text=True,capture_output=True,timeout=60,env=env)
        evidence=dict(index=i,returncode=r.returncode,request_SHA256=sha256(json.dumps(request,sort_keys=True).encode()),
            stdout_SHA256=sha256(r.stdout.encode()),stderr=r.stderr[-6000:])
        save(folder/('checker-process-'+str(i)+'.json'),evidence)
        if r.returncode:raise RuntimeError('Member frame check failed; exact returncode '+str(r.returncode))
        out=json.loads(r.stdout);cpu+=out['CPU_seconds'];results.update(out['results'])
        if any(v['status']!='PASSED' for v in out['results'].values()):raise ValueError('Unchanged independent frame check rejected member')
    return dict(results=results,CPU_seconds=cpu,wall_seconds=perf_counter()-wall,new_model_shots=0,new_QPU_shots=0,
        engineering_change='NEW_GATE_LEVEL_FRAME_ENGINE_ALL3_CIRCUITS_SAME_TOLERANCE_30CPU_LIMIT_NOT_OLD_MPS')


def run():
    p=verified();old=load(LOCAL/'source-failure.json');completed=list(old['completed'])
    qual=load(LOCAL/'qualification/Fe-Clifford-frame-v3/terminal.json')
    if not qual['passed']:raise ValueError('Actual Fe gate-level parity required')
    previous_tail=load(LOCAL/'source-tail-v1/failure.json')
    assert not previous_tail['completed']
    assert len(completed)==18 and {x['case'] for x in completed}=={c['case'] for c in p['cases'][:3]}
    if TAIL.exists():raise RuntimeError('Existing tail; reconcile actual process, do not repeat')
    ledger=BudgetLedger(LOCAL/'shots.sqlite')
    with ledger.connect() as db:
        assert db.execute('select count(*),sum(actual) from draws where state="COMPLETED"').fetchone()==(162,18000)
        assert db.execute('select count(*) from draws where state!="COMPLETED"').fetchone()==(0,)
    owner=load(LOCAL/'exports/FE2S2-40-P2/owner.json')
    try:os.kill(owner['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('Prior owner PID exists; inspect rather than resume')
    c=p['cases'][-1];draws=[d for d in p['draws'] if d['case']==c['case']];assert len(draws)==6
    save(TAIL/'manifest.json',dict(original_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        predecessor_failure_SHA256=sha256((LOCAL/'source-failure.json').read_bytes()),draw_ids=[d['draw_id'] for d in draws],
        source_worker_SHA256=sha256(Path(__file__).read_bytes()),source_settings_seeds_model_budgets_unchanged=True,
        checker_SHA256=sha256((ROOT/'frontierloop/clifford_frame_checker_v3.py').read_bytes()),
        batch_worker_SHA256=sha256((ROOT/'frontierloop/frame_batch_worker.py').read_bytes()),
        qualifier_SHA256=sha256((LOCAL/'qualification/Fe-Clifford-frame-v3/terminal.json').read_bytes()),
        verifier_contract='ALL3_ACTUAL_QPY_SAME_1e8_PROBABILITY_WITH_LEAKAGE_AND_DISCARD_BOUNDS_NUMERICAL_FRAME_NOT_OLD_MPS',
        original_MPS_protocol_declared_passed=False,
        completed18_not_repeated=True,probability_tolerance_and_30CPU_per_process_unchanged=True,
        prior_failure_exact_returncode_unknown=True,tail_CPU_cap_seconds=240,
        prior_attempt_administrative_CPU_budget_charge=None,administrative_charge_not_measured_CPU=True,
        total_source_CPU_accounting_will_be_marked_incomplete=True))
    cpu,wall=cpu_total(),perf_counter();session=None;new=[]
    def bound():
        space()
        if cpu_total()-cpu+(session.live_CPU if session else 0)>240:raise TimeoutError('Finite Fe tail CPU budget')
        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_output_bytes']:raise OSError('Original N57 output cap')
        if sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())>p['research_root_bytes']:raise OSError('Original root cap')
    try:
        meta=load(c['meta']);meta['case_id']=c['case'];session=Session(c['base'],TAIL/'exporter')
        for draw in draws:
            folder=LOCAL/'draws'/draw['draw_id']
            if (folder/'request.json').exists():assert load(folder/'request.json')==draw
            else:save(folder/'request.json',draw)
            if (folder/'result.json').exists():raise RuntimeError('Cannot repeat a completed draw')
            ledger.register(draw['draw_id'],1000);items=[];members=[]
            for m in draw['members']:
                bound();target=LOCAL/'exports'/c['case']/'members'/m['member_id']
                if not (target/'member.json.gz').exists():
                    stage=session.directory/'members'/m['member_id'];session.export(draw['arm'],m['spec'],stage)
                    for name in ['member.json.gz','circuit.qpy.gz']:write(target/name,(stage/name).read_bytes())
                member=load_member(target)
                assert member['spec']==m['spec'] and member['diagonal_terms_retained']==(draw['arm']=='FULL_N20')
                plan=dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],
                    artifacts={name:dict(path=str(target/name),sha256=sha256((target/name).read_bytes())) for name in ['member.json.gz','circuit.qpy.gz']})
                blobs={str(target/name):(target/name).read_bytes() for name in ['member.json.gz','circuit.qpy.gz']}
                items.append(prepare_compiler(meta,plan,blobs));members.append(member)
            check=frame_verify(items,TAIL/draw['draw_id']);save(folder/'compiler-check.json',check);merged=Counter();records=[]
            for m,member,item in zip(draw['members'],members,items):
                bound();state,numerics=member_state(meta,member)
                request=dict(member_id=m['member_id'],problem_id=draw['problem_id'],readout_seed=m['readout_seed'],shots=m['shots'],
                    source_packet_SHA256=sha256((LOCAL/'exports'/c['case']/'members'/m['member_id']/'member.json.gz').read_bytes()))
                if ledger.reserve(draw['draw_id'],m['member_id'],request,m['shots'])!='NEW_RESERVATION':raise RuntimeError('Unknown draw outcome; no resample')
                ids=np.array(sorted(state),np.uint64);probs=np.array([abs(state[int(x)])**2 for x in ids]);probs/=probs.sum()
                counts=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],probs);keep=counts>0
                blob=arrays(ids=ids[keep],counts=counts[keep]);write(folder/(m['member_id']+'.npz'),blob)
                ledger.complete(draw['draw_id'],m['member_id'],int(counts.sum()),sha256(blob))
                for x,w in zip(ids[keep],counts[keep]):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
                r=item['row'];r.update(request=request,counts_SHA256=sha256(blob),state_numerics=numerics);records.append(r)
                write(folder/(m['member_id']+'-certificate.json.gz'),item['certificate'])
                for variant,b in item['qpy_blobs'].items():write(folder/(m['member_id']+'-'+variant+'.qpy.gz'),b)
            ids=np.array(sorted(merged),np.uint64);counts=np.array([merged[int(x)] for x in ids],np.uint64);assert int(counts.sum())==1000
            blob=arrays(blocked_ids=ids,counts=counts);write(folder/'counts.npz',blob);fields=['cnot_count','cnot_depth','total_unitary_depth']
            resources={name:{f:sum(m['shots']*r['resources'][name][f] for m,r in zip(draw['members'],records))/1000 for f in fields} for name in ['NATIVE','RUSTIQ','PROJECTED']}
            resources['SELECTED']={f:sum(m['shots']*r['policy']['selected_resources'][f] for m,r in zip(draw['members'],records))/1000 for f in fields}
            result=dict(status='COMPLETED',draw_id=draw['draw_id'],case=draw['case'],arm=draw['arm'],seed=draw['seed'],problem_id=draw['problem_id'],
                counts_SHA256=sha256(blob),budget=ledger.snapshot(draw['draw_id']),members=records,resources=resources,
                coefficient_slots_mean=sum(m['shots']*r['coefficient_slots_original'] for m,r in zip(draw['members'],records))/1000,
                independent_parameter_rank=None,VQE_optimized_variables=0,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
                engineering_successor_SHA256=sha256((TAIL/'manifest.json').read_bytes()),new_model_samples=1000,new_QPU_shots=0)
            save(folder/'result.json',result);new.append({k:result[k] for k in ['draw_id','case','arm','seed','status']})
            save(TAIL/'progress.json',dict(completed=new));print('N57_FE_TAIL',draw['arm'],draw['seed'],len(new),flush=True)
        session.close();session=None;cost=cpu_total()-cpu
        save(TAIL/'terminal.json',dict(status='ALL6_FROZEN_FE_TAIL_DRAWS_COMPLETED',completed=new,new_model_samples=6000,
            CPU_seconds_including_children=cost,wall_seconds=perf_counter()-wall))
        save(TAIL/'combined-source-terminal.json',dict(status='ALL24_FRESH_SOURCE_DRAWS_COMPLETED',completed=completed+new,new_model_samples=24000,new_QPU_shots=0,
            actual_independent_circuit_checks=648,MPS_checks_old18=486,frame_checks_new6=162,CPU_seconds_including_children=None,CPU_accounting_complete=False,
            known_body_CPU_subtotal=sum(x['CPU_seconds_reaped']+x['unreaped_live_exporter_CPU_seconds'] for x in [old,previous_tail])+cost,
            unknown_CPU_scope='Original failed controller final SDK reap/close overhead not recorded; not zero',
            resumed_tail_CPU_seconds=cost,wall_seconds=None,original_failed_attempt_wall_seconds=old['wall_seconds'],tail_wall_seconds=perf_counter()-wall,
            original_failure_preserved=True,old18_not_repeated=True,goal_complete=False))
    except Exception as exc:
        save(TAIL/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=new,CPU_seconds_reaped=cpu_total()-cpu,
            unreaped_live_exporter_CPU_seconds=session.live_CPU if session else 0,wall_seconds=perf_counter()-wall));raise
    finally:
        if session is not None:session.close()


if __name__=='__main__':run()
