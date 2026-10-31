"""N62 source arm; actual-gate probability verification before model draws."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import os,traceback
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space
from .filter_only_plan import LOCAL,verified
from .filtered_SqDRIFT_source import Session,load_member,cpu_total
from .frame_source_successor import frame_verify
from .strong_bank_prepare import prepare
from paperstudy.sqdrift_execution import member_state
from paperstudy.ci import interleaved_to_blocked
from paperstudy.budget import BudgetLedger


def dispatch(arm):
    if arm!='FILTERED_N20':raise ValueError('Only frozen new arm permitted')
    return 'FILTERED_SHORT'


def run():
    p=verified();space()
    if (LOCAL/'source-started.json').exists():raise RuntimeError('Existing attempt; reconcile rather than rerun')
    save(LOCAL/'source-started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=cpu_total(),perf_counter();ledger=BudgetLedger(LOCAL/'shots.sqlite');states=[];session=None;failure=None
    stages=dict(compiler_and_source_validation_parent_CPU=0.,verifier_children_CPU=0.,sampling_parent_CPU=0.,exporter_CPU=0.)
    def bound():
        space()
        if cpu_total()-cpu+(session.live_CPU if session else 0)>p['source_CPU_seconds'] or perf_counter()-wall>p['source_wall_seconds']:raise TimeoutError('Frozen source global budget')
        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_output_bytes'] or sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())>p['research_root_bytes']:raise OSError('Frozen storage budget')
    try:
        for c in p['cases']:
            bound();meta=load(c['meta']);meta['case_id']=c['case'];session=Session(c['base'],LOCAL/'exports'/c['case'])
            if abs(session.ready['filtered_lambda']-c['filtered_lambda'])>1e-8:raise ValueError('Source normalization drift')
            for d in [d for d in p['draws'] if d['case']==c['case']]:
                folder=LOCAL/'draws'/d['draw_id'];save(folder/'request.json',d);ledger.register(d['draw_id'],1000)
                row={k:d[k] for k in ['draw_id','case','arm','seed']};row['status']='STARTED';save(folder/'status.json',row)
                try:
                    items=[];members=[]
                    for m in d['members']:
                        bound();target=session.directory/'members'/m['member_id'];session.export(dispatch(d['arm']),m['spec'],target)
                        member=load_member(target)
                        if member['spec']!=m['spec'] or member['diagonal_terms_retained'] is not False or abs(member['filtered_group_weight_sum']-c['filtered_lambda'])>1e-8:raise ValueError('Wrong operator or spec')
                        artifacts={name:dict(path=str(target/name),sha256=sha256((target/name).read_bytes())) for name in ['member.json.gz','circuit.qpy.gz']}
                        plan=dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],artifacts=artifacts)
                        item=prepare(meta,plan,{str(target/name):(target/name).read_bytes() for name in artifacts})
                        stages['compiler_and_source_validation_parent_CPU']+=item['row']['preparation_CPU_seconds'];items.append(item);members.append(member)
                    checks=frame_verify(items,folder/'verification-processes');stages['verifier_children_CPU']+=checks['CPU_seconds'];save(folder/'compiler-check.json',checks)
                    bound();tick=cpu_total();merged=Counter();records=[]
                    for m,member,item in zip(d['members'],members,items):
                        state,numerics=member_state(meta,member);target=session.directory/'members'/m['member_id']
                        request=dict(member_id=m['member_id'],problem_id=d['problem_id'],readout_seed=m['readout_seed'],shots=m['shots'],source_packet_SHA256=sha256((target/'member.json.gz').read_bytes()))
                        if ledger.reserve(d['draw_id'],m['member_id'],request,m['shots'])!='NEW_RESERVATION':raise RuntimeError('Unknown or duplicate draw; no resampling')
                        ids=np.asarray(sorted(state),np.uint64);prob=np.array([abs(state[int(i)])**2 for i in ids]);prob/=prob.sum()
                        counts=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],prob);keep=counts>0
                        blob=arrays(ids=ids[keep],counts=counts[keep]);write(folder/(m['member_id']+'.npz'),blob);ledger.complete(d['draw_id'],m['member_id'],int(counts.sum()),sha256(blob))
                        for x,w in zip(ids[keep],counts[keep]):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
                        rec=item['row'];rec.update(request=request,counts_SHA256=sha256(blob),state_numerics=numerics);records.append(rec)
                        write(folder/(m['member_id']+'-certificate.json.gz'),item['certificate'])
                        for name,b in item['qpy_blobs'].items():write(folder/(m['member_id']+'-'+name+'.qpy.gz'),b)
                    stages['sampling_parent_CPU']+=cpu_total()-tick
                    ids=np.asarray(sorted(merged),np.uint64);counts=np.array([merged[int(x)] for x in ids],np.uint64)
                    if int(counts.sum())!=1000:raise ValueError('Model sample total mismatch')
                    blob=arrays(blocked_ids=ids,counts=counts);write(folder/'counts.npz',blob);fields=['cnot_count','cnot_depth','total_unitary_depth']
                    resources={name:{f:sum(m['shots']*rec['resources'][name][f] for m,rec in zip(d['members'],records))/1000 for f in fields} for name in ['NATIVE','RUSTIQ','PROJECTED']}
                    resources['SELECTED']={f:sum(m['shots']*rec['policy']['selected_resources'][f] for m,rec in zip(d['members'],records))/1000 for f in fields}
                    result=dict(row,status='COMPLETED',problem_id=d['problem_id'],counts_SHA256=sha256(blob),members=records,resources=resources,
                        coefficient_slots_mean=sum(m['shots']*r['coefficient_slots_original'] for m,r in zip(d['members'],records))/1000,
                        independent_parameter_rank=None,VQE_optimized_variables=0,new_model_samples=1000,new_QPU_shots=0,
                        budget=ledger.snapshot(d['draw_id']),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()))
                    save(folder/'result.json',result);row['status']='COMPLETED'
                except Exception as exc:
                    snapshot=ledger.snapshot(d['draw_id']);row.update(status='FAILED_ENGINEERING_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),budget=snapshot)
                    save(folder/'failure.json',row)
                    if snapshot['unknown_or_inflight_draws'] or snapshot['known_actual_shots']:raise
                    if isinstance(exc,TimeoutError) and 'Exporter observation' in str(exc):raise
                states.append(row);save(folder/'status.json',row);save(LOCAL/'source-progress.json',dict(states=states,CPU_seconds_reaped=cpu_total()-cpu))
                print('N62_SOURCE',c['case'],d['seed'],row['status'],flush=True)
            stages['exporter_CPU']+=session.live_CPU;session.close();session=None
    except Exception as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc());save(LOCAL/'source-failure.json',dict(failure,states=states))
        raise
    finally:
        if session is not None:
            stages['exporter_CPU']+=session.live_CPU;session.close();session=None
        live_owners=[]
        for c in p['cases']:
            owner=LOCAL/'exports'/c['case']/'owner.json'
            if owner.exists():
                pid=load(owner)['pid']
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:live_owners.append(pid)
        save(LOCAL/'final-accounting.json',dict(CPU_seconds_including_reaped_children=cpu_total()-cpu,wall_seconds=perf_counter()-wall,
            stages=stages,CPU_stage_breakdown_complete=False,unattributed_overhead_not_zero=True,
            process_children_reaped=not live_owners,live_exporter_owner_PIDs=live_owners,all12_draws_attempted=len(states)==12,failure=failure))
    save(LOCAL/'source-terminal.json',dict(status='ALL12_FROZEN_SOURCE_STATUSES_RECORDED',states=states,
        completed=sum(r['status']=='COMPLETED' for r in states),failed=sum(r['status']!='COMPLETED' for r in states),
        new_model_samples=1000*sum(r['status']=='COMPLETED' for r in states),new_QPU_shots=0,
        accounting=load(LOCAL/'final-accounting.json'),goal_complete=False))


if __name__=='__main__':run()
