"""N75 final authorized source-exploration loop: chemical proxy times activity."""
from pathlib import Path
from time import process_time,perf_counter
import os,sys,signal
import numpy as np
from pyscf.cc import addons as cc_addons
from pyscf.ci import cisd
import ffsim.variational.ucj_spin_balanced as ucj_module
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,space
from .state_activity_preflight import LOCAL as N72
from .correlated_source_qualification import masks,orbit_action
from .paired_lowshot import paired_sector
from paperstudy.ci import blocked_to_interleaved
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N75-CHEMICAL-ACTIVITY-FINAL-EXPLORATION-v1'
DIRECTIVE=ROOT/'EXPLORATION_CLOSEOUT_DIRECTIVE_2026-10-27.md'

def classify_target(group,n,k):
    terms=masks(group);xs={x for x,_,_,_ in terms}
    if len(xs)!=1 or 0 in xs:return dict(status='UNSUPPORTED_FLIP_CLASS',rank=None)
    x=next(iter(xs));modes=[q for q in range(2*n) if x>>(2*n-1-q)&1]
    holes=[q for q in modes if q//2<k];particles=[q for q in modes if q//2>=k]
    base=dict(flip_modes=modes,occupied_spin_indices=holes,virtual_spin_indices=[q-2*k for q in particles],
        full_generator_semantics='NOT_INFERRED_FROM_FLIP_MASK_OR_Z_CHAIN')
    if len(holes)!=len(particles) or any(sum(q%2==s for q in holes)!=sum(q%2==s for q in particles) for s in [0,1]):
        return dict(base,status='NOT_HF_SECTOR_PRESERVING_TARGET',rank=None)
    if len(holes)==1:return dict(base,status='HF_TARGET_SINGLE',rank=1)
    if len(holes)==2:
        name='HF_TARGET_OPPOSITE_SPIN_DOUBLE' if holes[0]%2!=holes[1]%2 else 'HF_TARGET_SAME_SPIN_DOUBLE'
        return dict(base,status=name,rank=2)
    return dict(base,status='NOT_SINGLE_OR_DOUBLE_HF_TARGET',rank=None)

def component(target,spin1,spin2):
    holes=target.get('occupied_spin_indices',[]);particles=target.get('virtual_spin_indices',[])
    if target['rank']==1:return float(spin1[holes[0],particles[0]])
    if target['rank']==2:return float(spin2[holes[0],holes[1],particles[0],particles[1]])
    return None

def choose(rows):
    good=[r for r in rows if r['priority'] is not None and r['priority']>0]
    return min(good,key=lambda r:(-r['priority'],r['semantic_id'],r['member_id'])) if good else None

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing final loop; no replace or loop reset')
    space();owner=DIRECTIVE.read_text();assert '第3ループは禁止' in owner
    p=load(N72/'protocol.json');t=load(N72/'terminal.json');assert len(t['rows'])==120
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N75-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==5 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    bindings=dict(p['bindings'])
    for f in [Path(__file__),DIRECTIVE,N72/'protocol.json',N72/'terminal.json',tests,
        ROOT/'analysis/N75_FINAL_EXPLORATION_PLAN_JA.md',Path(cc_addons.__file__),Path(cisd.__file__),Path(ucj_module.__file__),
        OLD/'paperstudy/lucj.py',ROOT/'frontierloop/state_activity_preflight.py',ROOT/'frontierloop/correlated_source_qualification.py']:
        bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Research quota')
    save(LOCAL/'protocol.json',dict(version='N75_FINAL_LOOP_CHEMICAL_TIMES_ACTIVITY_ONE_POLICY',bindings=bindings,sources=p['sources'],candidates=t['rows'],
        time_shots=p['time_shots'],component_abs_floor=1e-8,expected_observation_floor=1.,CPU_cap=30,wall_cap=90,new_bytes=4*2**20,root_bytes=2**30,
        exploration_loop=2,maximum_exploration_loops=2,rejection_consumes_loop=True,third_loop_authorized=False,
        selection_uses_energy=False,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,goal_complete=False))
    print('N75_FINAL_LOOP2_FROZEN120_NO_ENERGY',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Final exploration intent already exists; inspect live status, never duplicate')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),exploration_loop=2,rejection_consumes_loop=True))
    cpu,wall=process_time(),perf_counter();rows=[];selected=[];cases=[];initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file());base=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def timeout(signum,frame):raise TimeoutError('N75 final-loop frozen CPU/wall cap')
    a=signal.signal(signal.SIGALRM,timeout);b=signal.signal(signal.SIGPROF,timeout)
    signal.setitimer(signal.ITIMER_REAL,p['wall_cap']);signal.setitimer(signal.ITIMER_PROF,p['CPU_cap'])
    def bound():
        space();size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if size>p['new_bytes'] or base+size-initial>p['root_bytes']:raise OSError('N75 storage cap')
    try:
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];t1,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
            if t1.shape!=(k,n-k) or t2.shape!=(k,k,n-k,n-k) or np.iscomplexobj(t1) or np.iscomplexobj(t2) or not np.isfinite(t1).all() or not np.isfinite(t2).all():raise ValueError('Pinned real initializer shape required')
            spin1=cc_addons.spatial2spin(t1);spin2=cc_addons.spatial2spin(t2)
            half,v,_=paired_sector(n,k,t2);source={blocked_to_interleaved(int(x)|(int(x)<<n),n):complex(a) for x,a in zip(half,v) if a!=0}
            local=[];local_selected=[]
            for c in [c for c in p['candidates'] if c['case']==s['case']]:
                target=classify_target(c['group'],n,k);value=component(target,spin1,spin2);prob=c['score']['probability']
                qualified=value is not None and abs(value)>=p['component_abs_floor'] and prob is not None
                row=dict(c,target=target,amplitude_kind=kind,spin_component=value,priority=float(value*value*prob) if qualified else None,
                    proxy_is_ground_state_importance_certificate=False)
                rows.append(row);local.append(row)
            for time,shots in p['time_shots']:
                winner=choose([r for r in local if r['time']==time]);r=dict(case=s['case'],time=time,shots=shots,selection=winner)
                if winner is not None:
                    state,num=orbit_action(source,winner['group'],(k,k));norm=float(sum(abs(a)**2 for a in state.values()))
                    if abs(norm-1)>1e-10:raise ValueError('Final chemical source norm failure')
                    ids=np.array(sorted(state),np.uint64);amps=np.array([state[int(x)] for x in ids],complex)
                    f=LOCAL/'states'/f'{s["case"]}-T{int(time)}.npz';blob=arrays(source_mode_ids=ids,amplitudes=amps);write(f,blob)
                    r.update(state_file=str(f),state_SHA256=sha256(blob),numerical=num,norm_squared=norm,amplitude_kind=kind,large_QPY_gate_state_verified=False)
                selected.append(r);local_selected.append(r)
            mass=sum(r['shots']*r['selection']['score']['probability'] for r in local_selected if r['selection'])
            cases.append(dict(case=s['case'],amplitude_kind=kind,expected_nonpair_observations_per1000=mass,
                applicable_all3times=all(r['selection'] is not None for r in local_selected),
                proceed_to_one_final_pilot=all(r['selection'] is not None for r in local_selected) and mass>=p['expected_observation_floor']))
            save(LOCAL/'progress.json',dict(rows=rows,selected=selected,cases=cases));bound();print('N75_CASE',s['case'],cases[-1]['proceed_to_one_final_pilot'],mass,flush=True)
        save(LOCAL/'terminal.json',dict(status='FINAL_LOOP2_CHEMICAL_PREFLIGHT_ALL120_RECORDED',rows=rows,selected=selected,cases=cases,
            all4_applicable=all(c['proceed_to_one_final_pilot'] for c in cases),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            exploration_loop=2,rejection_consumes_loop=True,third_loop_authorized=False,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),rows=rows,selected=selected,cases=cases,exploration_loop=2,rejection_consumes_loop=True,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGALRM,a);signal.signal(signal.SIGPROF,b)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
