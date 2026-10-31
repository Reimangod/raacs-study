"""N72 rank-one state-activity preflight. No energies enter selection."""
from pathlib import Path
from time import process_time,perf_counter
import gzip,json,os,sys,signal
import numpy as np
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .correlated_source_qualification import LOCAL as N69,masks,orbit_action
from .ensemble_breadth_plan import LOCAL as N65
from .paired_lowshot import paired_sector
from paperstudy.ci import blocked_to_interleaved
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N72-STATE-ACTIVITY-PREFLIGHT-v1'

def flip_halves(x,n):
    return tuple(sum(((int(x)>>(2*n-1-spin-2*i))&1)<<i for i in range(n)) for spin in [0,1])

def score(ids,prob,group,k):
    """Exact nonpair transfer for rank1 offdiagonal groups on paired input."""
    n=len(group[0][0])//2
    if len(group[0][0])!=2*n or n>20:raise ValueError('Even bounded spin register required')
    terms=masks(group);xs={r[0] for r in terms}
    if len(xs)!=1 or 0 in xs:return dict(status='UNSUPPORTED_NOT_SINGLE_NONZERO_X_MASK',probability=None)
    x=next(iter(xs));a,b=flip_halves(x,n)
    if a==b:return dict(status='PAIR_PRESERVING_UNSCORED',probability=None)
    ids=np.asarray(ids,np.uint64);prob=np.asarray(prob,float)
    if ids.ndim!=1 or prob.shape!=ids.shape or len(set(map(int,ids)))!=len(ids) or not np.isfinite(prob).all() or np.any(prob<0) or abs(float(prob.sum())-1)>1e-10:raise ValueError('Normalized unique paired probability required')
    alpha_mask=sum(1<<(2*n-1-2*i) for i in range(n));beta_mask=alpha_mask>>1
    alpha=np.bitwise_count(ids&np.uint64(alpha_mask));beta=np.bitwise_count(ids&np.uint64(beta_mask))
    if np.any(alpha!=k) or np.any(beta!=k) or np.any(((ids&np.uint64(alpha_mask))>>np.uint64(1))!=(ids&np.uint64(beta_mask))):raise ValueError('Wrong paired input sector')
    f=np.zeros(len(ids),complex)
    for _,z,ny,angle in terms:
        parity=np.bitwise_count(ids&np.uint64(z))%2
        f+=angle*(1j**ny)*(1-2*parity.astype(np.int8))
    transferred=prob*np.sin(abs(f))**2;target=ids^np.uint64(x)
    valid=(np.bitwise_count(target&np.uint64(alpha_mask))==k)&(np.bitwise_count(target&np.uint64(beta_mask))==k)
    leakage=float(np.sum(transferred[~valid]))
    if leakage>1e-22:raise ValueError('Group does not preserve requested particle sector')
    return dict(status='SCORED_EXACT_RANK1_NONPAIR_TRANSFER',probability=float(np.sum(transferred[valid])),
        outside_sector_weight=leakage,flip_mask=int(x),source_size=len(ids),in_sector_pruning=False)

def choose(rows):
    valid=[r for r in rows if r['score']['status']=='SCORED_EXACT_RANK1_NONPAIR_TRANSFER']
    return min(valid,key=lambda r:(-r['score']['probability'],r['semantic_id'],r['member_id'])) if valid else None

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; no replace')
    space();p69=load(N69/'protocol.json');old=load(N65/'protocol.json');bindings=dict(p69['bindings']);candidates=[]
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N72-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    for d in old['draws']:
        if d['arm']!='FILTERED_SHORT':continue
        for m in d['members']:
            f=N65/'exports'/d['case']/'members'/m['member_id']/'member.json.gz';blob=f.read_bytes();packet=json.loads(gzip.decompress(blob))
            assert packet['spec']==m['spec'] and packet['diagonal_terms_retained'] is False
            group=[[label,float(angle)*m['spec']['num_terms']] for label,angle in packet['source_mode_group_Pauli_sequence'][0]]
            masks(group);bindings[str(f)]=sha256(blob)
            candidates.append(dict(case=d['case'],time=m['spec']['time'],member_index=m['spec']['member_index'],member_id=m['member_id'],group=group,
                semantic_id=digest(sorted((label,float(angle).hex()) for label,angle in group)),source_packet=str(f)))
    assert len(candidates)==120 and len({(r['case'],r['time']) for r in candidates})==12
    for f in [Path(__file__),N69/'protocol.json',N69/'terminal.json',N65/'protocol.json',tests,
        ROOT/'frontierloop/correlated_source_qualification.py',ROOT/'analysis/N70_DECISION_AND_N72_ACTIVITY_PREFLIGHT_JA.md']:
        bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Research logical quota')
    save(LOCAL/'protocol.json',dict(version='N72_FIXED120_GROUP_STATE_ACTIVITY_PREFLIGHT_NO_ENERGY',bindings=bindings,candidates=candidates,sources=p69['sources'],
        CPU_cap=30,wall_cap=90,new_bytes=4*2**20,root_bytes=2**30,time_shots=[[1,334],[2,333],[3,333]],
        expected_observation_floor=1.,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,selection_uses_energy=False,goal_complete=False))
    print('N72_FROZEN120_NO_ENERGY',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Execution intent exists; inspect process, no repeat')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[];selected=[];cases=[];initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file());base=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def timeout(signum,frame):raise TimeoutError('N72 frozen CPU/wall cap')
    oldalarm=signal.signal(signal.SIGALRM,timeout);oldprof=signal.signal(signal.SIGPROF,timeout)
    signal.setitimer(signal.ITIMER_REAL,p['wall_cap']);signal.setitimer(signal.ITIMER_PROF,p['CPU_cap'])
    def bound():
        space();size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if size>p['new_bytes'] or base+size-initial>p['root_bytes']:raise OSError('N72 storage cap')
    try:
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
            half,v,_=paired_sector(n,k,t2);ids=np.array([blocked_to_interleaved(int(b)|(int(b)<<n),n) for b in half],np.uint64);prob=abs(v)**2
            source={int(b):complex(a) for b,a in zip(ids,v) if a!=0};cache={};case_rows=[];case_selected=[]
            for c in [c for c in p['candidates'] if c['case']==s['case']]:
                bound()
                if c['semantic_id'] not in cache:cache[c['semantic_id']]=score(ids,prob,c['group'],k)
                r=dict(c,score=cache[c['semantic_id']]);rows.append(r);case_rows.append(r)
            for time,shots in p['time_shots']:
                winner=choose([r for r in case_rows if r['time']==time]);baseline=next(r for r in case_rows if r['time']==time and r['member_index']==0)
                r=dict(case=s['case'],time=time,shots=shots,selection=None,original_random_probability=baseline['score'].get('probability'))
                if winner is not None:
                    state,numerical=orbit_action(source,winner['group'],(k,k));actual=0.
                    for b,a in state.items():
                        al,be=flip_halves(b,n)
                        if al!=be:actual+=abs(a)**2
                    error=abs(actual-winner['score']['probability'])
                    if error>1e-10:raise ValueError('Closed form/orbit mismatch')
                    stateids=np.array(sorted(state),np.uint64);amps=np.array([state[int(b)] for b in stateids],complex)
                    file=LOCAL/'states'/f'{s["case"]}-T{int(time)}.npz';blob=arrays(source_mode_ids=stateids,amplitudes=amps);write(file,blob)
                    r.update(selection=winner,orbit=numerical,amplitude_kind=kind,independent_probability_error=error,
                        state_file=str(file),state_SHA256=sha256(blob),large_QPY_gate_state_verified=False)
                selected.append(r);case_selected.append(r)
            observed=sum(r['shots']*r['selection']['score']['probability'] for r in case_selected if r['selection'])
            cases.append(dict(case=s['case'],expected_nonpair_observations_per1000=observed,proceed_to_resource_qualification=all(r['selection'] is not None for r in case_selected) and observed>=p['expected_observation_floor'],
                original_random_expected_observations_per1000=sum(r['shots']*(r['original_random_probability'] or 0) for r in case_selected),unique_semantic_groups=len(cache)))
            save(LOCAL/'progress.json',dict(rows=rows,selected=selected,cases=cases,CPU_seconds=process_time()-cpu));print('N72_CASE',s['case'],observed,flush=True)
        save(LOCAL/'terminal.json',dict(status='ALL120_ACTIVITY_STATUSES_AND12_SELECTIONS_RECORDED',rows=rows,selected=selected,cases=cases,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,GPU_allocations=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),rows=rows,selected=selected,cases=cases,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGALRM,oldalarm);signal.signal(signal.SIGPROF,oldprof)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
