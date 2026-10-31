"""N69 bounded paired-circuit and independent orbit-exponential qualification."""
from pathlib import Path
from time import process_time,perf_counter
import os,sys,signal,math
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,space
from .pair_group_preflight import LOCAL as N67,group_circuit
from .exact_group_synthesis import LOCAL as N68
from .paired_lowshot import paired_sector,paired_circuit
from paperstudy.lucj import amplitudes_from_saved
from paperstudy.ci import blocked_to_interleaved
from paperstudy.sqdrift_execution import sparse_group

LOCAL=ROOT/'runs/N69-CORRELATED-SOURCE-QUALIFICATION-v1'

def masks(group):
    group_circuit(group)
    n=len(group[0][0]);out=[]
    for label,angle in group:
        x=z=ny=0
        for i,p in enumerate(label):
            b=1<<(n-1-i)
            if p in 'XY':x|=b
            if p in 'ZY':z|=b
            ny+=p=='Y'
        out.append((x,z,ny,float(angle)))
    return out

def flip_basis(group):
    basis={}
    for x,_,_,_ in masks(group):
        for pivot,b in sorted(basis.items(),reverse=True):
            if x>>pivot&1:x^=b
        if x:basis[x.bit_length()-1]=x
    return sorted(basis.items(),reverse=True)

def orbit_action(state,group,nelec=None,max_support=100000):
    """Independent exp(sum Pauli) on disjoint XOR orbits; no amplitude pruning."""
    terms=masks(group);basis=flip_basis(group);n=len(group[0][0])
    if len(basis)>4:raise ValueError('Bounded orbit rank exceeded')
    spans=[0]
    for _,b in basis:spans += [s^b for s in spans]
    index={s:i for i,s in enumerate(spans)};orbits={}
    for b,a in state.items():
        if not 0<=int(b)<1<<n or not np.isfinite(a):raise ValueError('Invalid source')
        rep=int(b)
        for pivot,row in basis:
            if rep>>pivot&1:rep^=row
        if rep not in orbits:orbits[rep]=np.zeros(len(spans),complex)
        orbits[rep][index[int(b)^rep]]=a
    if len(orbits)*len(spans)>4*max_support:raise MemoryError('Orbit temporary support cap')
    output={};cache={};hermitian_error=0.
    for rep,v in orbits.items():
        signature=tuple((rep&z).bit_count()%2 for _,z,_,_ in terms)
        if signature not in cache:
            generator=np.zeros((len(spans),len(spans)),complex)
            for j,s in enumerate(spans):
                for x,z,ny,angle in terms:
                    generator[index[s^x],j]+=angle*(1j**ny)*((-1)**((rep^s)&z).bit_count())
            err=float(np.linalg.norm(generator-generator.conj().T));hermitian_error=max(hermitian_error,err)
            if err>1e-12:raise ValueError('Non-Hermitian orbit generator')
            cache[signature]=expm(-1j*generator)
        for s,a in zip(spans,cache[signature]@v):
            if a!=0:output[rep^s]=complex(a)
    leakage=0.;kept={}
    for b,a in output.items():
        sector=(sum((b>>(n-1-i))&1 for i in range(0,n,2)),sum((b>>(n-1-i))&1 for i in range(1,n,2)))
        if nelec is None or sector==tuple(nelec):kept[b]=a
        else:leakage+=abs(a)**2
    if leakage>1e-22:raise ValueError('Orbit particle-sector leakage')
    if len(kept)>max_support:raise MemoryError('Orbit physical support cap')
    return kept,dict(flip_rank=len(basis),orbit_size=len(spans),orbits=len(orbits),matrix_exponentials=len(cache),
        Hermitian_error=hermitian_error,roundoff_outside_sector_weight=float(leakage),physical_support=len(kept),in_sector_truncation=False)

def l2(left,right):return float(math.sqrt(sum(abs(left.get(b,0)-right.get(b,0))**2 for b in left.keys()|right.keys())))

def paired_check(n,k,t2):
    ids,v,angles=paired_sector(n,k,t2)
    if n>20:raise MemoryError('Paired encoding larger than20qubits')
    actual=Statevector(paired_circuit(n,k,angles,False)).data
    expected=np.zeros(1<<n,complex);expected[ids.astype(np.int64)]=v
    error=float(np.linalg.norm(actual-expected))
    if error>1e-10:raise ValueError('Paired circuit complex-state mismatch')
    source={blocked_to_interleaved(int(b)|(int(b)<<n),n):complex(a) for b,a in zip(ids,v) if a!=0}
    return source,dict(norb=n,nelec=[k,k],paired_encoding_qubits=n,full_register_qubits=2*n,
        paired_complex_L2_error=error,paired_norm_squared=float(np.vdot(v,v).real),pair_support=len(source),
        full_Fock_2n_vector_used=False,paired_dense_bytes=int(actual.nbytes))

def toy():
    group=[['XZXI',.239],['YZYI',.239]]
    t2=np.array([[[[.31]]]])
    source,check=paired_check(2,1,t2);a,leak=sparse_group(source,group,4,(1,1));b,meta=orbit_action(source,group,(1,1))
    if l2(a,b)>1e-12:raise ValueError('Toy orbit parity failed')
    return dict(check=check,orbit=meta,L2=l2(a,b),sequential_leakage=leak)

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; no replace')
    space();toy();old=load(N68/'protocol.json');terminal=load(N68/'terminal.json')
    assert len(terminal['rows'])==12
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N69-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    bindings=dict(old['bindings'])
    for f in [Path(__file__),N68/'protocol.json',N68/'terminal.json',N68/'qualification.json',
        ROOT/'frontierloop/pair_group_preflight.py',ROOT/'analysis/N69_CORRELATED_SOURCE_QUALIFICATION_PLAN_JA.md',tests]:
        bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h
    bounds=[]
    for member in old['members']:
        source=next(s for s in old['sources'] if s['case']==member['case']);meta=load(source['meta']);rank=len(flip_basis(member['group']))
        upper=math.comb(meta['norb'],meta['nalpha'])*(1<<rank)
        if rank>4 or upper>100000:raise MemoryError('Structural support upper bound exceeds frozen domain')
        bounds.append(dict(case=member['case'],time=member['time'],flip_rank=rank,source_support_upper_bound=upper))
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Logical quota; no deletion')
    save(LOCAL/'protocol.json',dict(version='N69_TWO_ROUTE_CORRELATED_SOURCE_QUALIFICATION',bindings=bindings,sources=old['sources'],members=old['members'],
        structural_bounds=bounds,CPU_cap=120,wall_cap=240,new_bytes=4*2**20,root_bytes=2**30,
        L2_tolerance=1e-10,norm_tolerance=1e-10,sector_leakage_cap=1e-22,max_support=100000,
        new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,no_energy_evaluation=True,goal_complete=False))
    print('N69_FROZEN12_QUALIFICATION_ONLY',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Execution intent exists; inspect process, no duplicate')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[];pairs=[];initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
    base=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def timeout(signum,frame):raise TimeoutError('Frozen N69 CPU/wall timer')
    oldalarm=signal.signal(signal.SIGALRM,timeout);oldprof=signal.signal(signal.SIGPROF,timeout)
    signal.setitimer(signal.ITIMER_REAL,p['wall_cap']);signal.setitimer(signal.ITIMER_PROF,p['CPU_cap'])
    def bound():
        space();current=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if current>p['new_bytes'] or base+current-initial>p['root_bytes']:raise OSError('N69 storage cap')
        if process_time()-cpu>p['CPU_cap'] or perf_counter()-wall>p['wall_cap']:raise TimeoutError('N69 budget')
    try:
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];tick=process_time()
            _,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k);source,check=paired_check(n,k,t2)
            pairs.append(dict(case=s['case'],amplitude_kind=kind,CPU_seconds=process_time()-tick,**check))
            for m in [m for m in p['members'] if m['case']==s['case']]:
                bound();tick=process_time();sequential,leak=sparse_group(source,m['group'],2*n,(k,k),p['max_support'])
                orbit,orbitmeta=orbit_action(source,m['group'],(k,k),p['max_support']);error=l2(sequential,orbit)
                norm=float(sum(abs(a)**2 for a in orbit.values()))
                if error>p['L2_tolerance'] or abs(norm-1)>p['norm_tolerance']:raise ValueError('Large correlated group parity/norm failed')
                ids=np.array(sorted(orbit),np.uint64);amplitudes=np.array([orbit[int(b)] for b in ids],complex)
                file=LOCAL/'states'/f'{s["case"]}-T{int(m["time"])}.npz';write(file,arrays(source_mode_ids=ids,amplitudes=amplitudes))
                paired_mass=0.
                for b,a in orbit.items():
                    alpha=sum(((b>>(2*n-1-2*i))&1)<<i for i in range(n));beta=sum(((b>>(2*n-2-2*i))&1)<<i for i in range(n))
                    if alpha==beta:paired_mass+=abs(a)**2
                rows.append(dict(case=s['case'],time=m['time'],amplitude_kind=kind,complex_L2_error=error,norm_squared=norm,
                    sequential_outside_sector_weight=leak,orbit=orbitmeta,nonpaired_probability=max(0.,1-paired_mass),
                    state_file=str(file),state_SHA256=sha256(file.read_bytes()),CPU_seconds=process_time()-tick,
                    actual_large_QPY_gate_level_state_verified=False,energy_Ha=None,Recovery_pct=None))
                bound();save(LOCAL/'progress.json',dict(pairs=pairs,rows=rows,CPU_seconds=process_time()-cpu))
                print('N69_QUALIFIED',s['case'],m['time'],len(rows),flush=True)
        save(LOCAL/'terminal.json',dict(status='ALL12_CORRELATED_SOURCE_TWO_ROUTE_CHECKS_PASSED',pairs=pairs,rows=rows,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,
            large_compiled_QPY_state_verified=False,GPU_allocations=0,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),pairs=pairs,rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0)
        signal.signal(signal.SIGALRM,oldalarm);signal.signal(signal.SIGPROF,oldprof)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
