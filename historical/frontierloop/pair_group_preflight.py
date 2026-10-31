"""N67 paired seed + SDK-sampled single group: phase/resource preflight only."""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import gzip,json,os,sys,math
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli,Statevector
from .paths import ROOT,OLD
from .io import load,save,write,sha256,space
from .filtered_SqDRIFT_plan import verified as verified57
from .ensemble_breadth_plan import LOCAL as N65,verified as verified65
from .paired_lowshot import paired_circuit,paired_sector
from .paired_resource_preflight import schedule
from .slater_prefix_preflight import compile_views
from paperstudy.ci import blocked_to_interleaved
from paperstudy.sqdrift_execution import sparse_group
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N67-PAIR-GROUP-RESOURCE-PREFLIGHT-v1'

def angles_from_t2(n,k,t2):
    t2=np.asarray(t2)
    if t2.shape!=(k,k,n-k,n-k) or np.iscomplexobj(t2) or not np.isfinite(t2).all():raise ValueError('Real finite restricted amplitudes required')
    return [dict(i=i,a=a,theta=float(t2[i,i,a-k,a-k])) for layer in schedule(n,k) for i,a in layer]

def pair_preparation(n,k,angles):
    q=QuantumCircuit(2*n)
    return q.compose(paired_circuit(n,k,angles,True),qubits=[2*i for i in range(n)]+[2*i+1 for i in range(n)])

def group_circuit(group):
    if not group:raise ValueError('Nonempty sampled group required')
    n=len(group[0][0]);q=QuantumCircuit(n);paulis=[]
    for label,angle in group:
        if len(label)!=n or set(label)-set('IXYZ') or not np.isfinite(angle):raise ValueError('Invalid group label/angle')
        p=Pauli(label[::-1])
        if any(not p.commutes(old) for old in paulis):raise ValueError('Noncommuting group; no silent Trotter approximation')
        paulis.append(p);q.append(PauliEvolutionGate(p,time=float(angle)),range(n))
    return q

def reverse_bits(x,n):return sum(((int(x)>>i)&1)<<(n-1-i) for i in range(n))

def fixture(n,k):
    t2=np.zeros((k,k,n-k,n-k))
    for i in range(k):
        for a in range(n-k):t2[i,i,a,a]=(-1)**(i+a)*(.17+.03*(i+a))
    ids,v,angles=paired_sector(n,k,t2);assert angles_from_t2(n,k,t2)==angles
    label=['I']*(2*n)
    for i in range(1,2*k):label[i]='Z'
    xx=label.copy();yy=label.copy();xx[0]=xx[2*k]='X';yy[0]=yy[2*k]='Y'
    group=[[''.join(xx),.239],[''.join(yy),.239]]
    source={blocked_to_interleaved(int(x)|(int(x)<<n),n):complex(a) for x,a in zip(ids,v)}
    target,leak=sparse_group(source,group,2*n,(k,k));expected=np.zeros(1<<(2*n),complex)
    for x,a in target.items():expected[reverse_bits(x,2*n)]=a
    circuit=pair_preparation(n,k,angles).compose(group_circuit(group));actual=Statevector(circuit).data
    overlap=np.vdot(expected,actual);phase=overlap/abs(overlap)
    error=float(np.linalg.norm(actual-phase*expected));outside=0.
    for x,a in enumerate(actual):
        if (sum((x>>(2*i))&1 for i in range(n)),sum((x>>(2*i+1))&1 for i in range(n)))!=(k,k):outside+=abs(a)**2
    if error>1e-10 or outside>1e-20 or leak>1e-22:raise ValueError('Hybrid complex-phase/sector failure')
    compiled=[]
    for level in [1,3]:
        prep,sample,res=compile_views(circuit,level);pv=Statevector(prep).data;sv=Statevector(sample).data
        ph=np.vdot(expected,pv);ph/=abs(ph)
        complex_error=float(np.linalg.norm(pv-ph*expected));prob_error=float(np.sum(abs(abs(sv)**2-abs(expected)**2)))
        if complex_error>1e-10 or prob_error>1e-10:raise ValueError('Compiled preparation phase or sampling probability failure')
        compiled.append(dict(level=level,complex_state_error=complex_error,sampling_probability_L1=prob_error))
    return dict(norb=n,k=k,complex_state_error=error,full_sector_leakage=float(outside),compiled=compiled)

def qualify():
    if (LOCAL/'qualification.json').exists():raise RuntimeError('Existing qualifier; do not repeat')
    cpu,wall=process_time(),perf_counter();rows=[]
    try:
        for n,k in [(2,1),(3,1),(4,2)]:rows.append(fixture(n,k))
        save(LOCAL/'qualification.json',dict(status='ALL3_COMPLEX_STATE_AND_BOTH_COMPILERS_PASSED',rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,code_SHA256=sha256(Path(__file__).read_bytes())))
    except Exception as exc:
        save(LOCAL/'qualification-failed.json',dict(error=repr(exc),rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing protocol; no replace')
    space()
    p65=verified65();verified57();qual=load(LOCAL/'qualification.json')
    assert qual['status']=='ALL3_COMPLEX_STATE_AND_BOTH_COMPILERS_PASSED' and qual['code_SHA256']==sha256(Path(__file__).read_bytes())
    old=load(ROOT/'runs/N25-PAIRED-LOWSHOT-v1/protocol.json');sources=[];members=[];bindings={}
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N67-policy-v2.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==5 and not any(int(s.get(k,0)) for s in suites for k in ['failures','errors'])
    bindings[str(tests)]=sha256(tests.read_bytes())
    for c in p65['cases']:
        source=next(s for s in old['sources'] if s['case']==c['case']);meta=load(c['meta'])
        assert source['definition']['problem_id']==meta['problem_id']
        for field in ['amplitudes','initialization_record_path']:
            file=Path(source[field]);bindings[str(file)]=sha256(file.read_bytes())
        assert bindings[source['amplitudes']]==source['amplitudes_SHA256'] and bindings[source['initialization_record_path']]==source['initialization_record_SHA256']
        sources.append(dict(case=c['case'],meta=c['meta'],amplitudes=source['amplitudes'],initializer=source['initialization_record']))
        d=next(d for d in p65['draws'] if d['case']==c['case'] and d['arm']=='FILTERED_SHORT')
        for m in d['members']:
            if m['spec']['member_index']!=0:continue
            file=N65/'exports'/c['case']/'members'/m['member_id']/'member.json.gz';bindings[str(file)]=sha256(file.read_bytes())
            packet=json.loads(gzip.decompress(file.read_bytes()));assert packet['spec']==m['spec'] and packet['diagonal_terms_retained'] is False
            group=[[label,float(angle)*m['spec']['num_terms']] for label,angle in packet['source_mode_group_Pauli_sequence'][0]]
            group_circuit(group)
            members.append(dict(case=c['case'],time=m['spec']['time'],source_packet=str(file),source_spec=m['spec'],group=group,
                adaptation='FIRST_SDK_SAMPLED_GROUP_RESCALED_TO_N1_IDENTITY_MODE_ORDER_NOT_ORIGINAL_F2Q_CIRCUIT'))
        bindings[c['meta']]=sha256(Path(c['meta']).read_bytes())
    for f in [Path(__file__),ROOT/'frontierloop/paired_lowshot.py',ROOT/'frontierloop/paired_resource_preflight.py',ROOT/'frontierloop/slater_prefix_preflight.py',
        OLD/'paperstudy/sqdrift_execution.py',OLD/'paperstudy/lucj.py',ROOT/'analysis/N66_DECISION_AND_N67_SOURCE_PREFLIGHT_JA.md',ROOT/'analysis/N67_IMPLEMENTATION_CLARIFICATION_JA.md',
        ROOT.parents[1]/'ceo-sampler-benchmark/.runtime-overlays/sqdrift-qiskit25-v1/qiskit_fermions/transpiler/passes/optimization/qdrift.py']:
        bindings[str(f)]=sha256(f.read_bytes())
    base=ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1';records=load(base/'result.json');controls=[];seen=set()
    for r in records['rows']:
        key=(r['case'],r['method'])
        if key in seen:continue
        seen.add(key)
        if r['status']=='MISSING_SAME_PROBLEM_SOURCE':controls.append(dict(case=r['case'],method=r['method'],status='MISSING_SAME_PROBLEM_SOURCE'));continue
        # Use EXACT outputs, never an approximately rounded winner.
        folder=base/r['case']/r['method']/'EXACT';points=[]
        for level in [1,3]:
            f=folder/f'opt{level}-sampling.qpy';bindings[str(f)]=sha256(f.read_bytes());points.append(dict(level=level,file=str(f)))
        controls.append(dict(case=r['case'],method=r['method'],status='AVAILABLE',points=points))
    comparison=ROOT/'analysis/N25-lowshot-pair-summary.json';bindings[str(comparison)]=sha256(comparison.read_bytes())
    archived=load(comparison)['all_original_LUCJ_compiler_points']
    for c in controls:
        if c['status']!='AVAILABLE':continue
        original=next(r for r in archived if r['case']==c['case'] and r['method']==c['method'])
        for point in c['points']:
            q=qpy.load(BytesIO(Path(point['file']).read_bytes()))[0];expected=next(z for z in original['points'] if z['optimization_level']==point['level'])
            assert all(metrics(q)[k]==expected[k] for k in ['CNOT','CNOT_Depth','Total_Depth'])
    assert len(members)==12 and len(sources)==4 and len(controls)==12 and sum(c['status']=='AVAILABLE' for c in controls)==10
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Insufficient research logical quota; no deletion')
    save(LOCAL/'protocol.json',dict(version='N67_PAIR_PLUS_SINGLE_GROUP_COMPLEX_PHASE_RESOURCE_PREFLIGHT_ONLY',sources=sources,members=members,controls=controls,bindings=bindings,
        qualifier_SHA256=sha256((LOCAL/'qualification.json').read_bytes()),CPU_cap=30,wall_cap=90,new_output_bytes=4*2**20,research_root_bytes=2**30,
        initial_research_logical_bytes=used,
        compiler_levels=[1,3],no_original_F2Q_or_HF_projected_certificate_reuse=True,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,goal_complete=False))
    print('N67_FROZEN12',flush=True)

def metrics(q):return dict(CNOT=int(q.count_ops().get('cx',0)),CNOT_Depth=int(q.depth(filter_function=lambda i:i.operation.name=='cx')),Total_Depth=int(q.depth()),
    primitive_coefficient_slots_not_independent_rank=sum(len(i.operation.params) for i in q.data))

def run():
    p=load(LOCAL/'protocol.json')
    space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; no repeat')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[];cache={};controls=[]
    base_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file());initial_local=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
    def bound():
        space()
        if process_time()-cpu>p['CPU_cap'] or perf_counter()-wall>p['wall_cap']:raise TimeoutError('N67 frozen preflight budget')
        current=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if current>p['new_output_bytes'] or base_bytes+current-initial_local>p['research_root_bytes']:raise OSError('N67 output/root cap')
    def compile_record(q,name):
        points=[]
        for level in p['compiler_levels']:
            bound();tick=process_time();prep,sample,res=compile_views(q,level);files={}
            for role,circuit in [('preparation',prep),('sampling',sample)]:
                stream=BytesIO();qpy.dump(circuit,stream);blob=gzip.compress(stream.getvalue(),mtime=0);file=LOCAL/'circuits'/f'{name}-opt{level}-{role}.qpy.gz';write(file,blob)
                restored=qpy.load(BytesIO(gzip.decompress(blob)))[0];assert metrics(restored)==metrics(circuit)
                files[role]=dict(file=str(file),SHA256=sha256(blob))
            points.append(dict(level=level,resources=metrics(sample),preparation_resources=metrics(prep),files=files,CPU_seconds=process_time()-tick))
        return points
    try:
        for control in p['controls']:
            row=dict(control)
            if control['status']=='AVAILABLE':
                pts=[]
                for point in control['points']:
                    q=qpy.load(BytesIO(Path(point['file']).read_bytes()))[0];pts.append(dict(level=point['level'],resources=metrics(q)))
                row['recounted_points']=pts
            controls.append(row)
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];t1,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
            angles=angles_from_t2(n,k,t2);pair=pair_preparation(n,k,angles);pairpoints=compile_record(pair,s['case']+'-PAIR')
            for m in [m for m in p['members'] if m['case']==s['case']]:
                bound();group=group_circuit(m['group']);hf=QuantumCircuit(2*n)
                for i in range(k):hf.x(2*i);hf.x(2*i+1)
                hybrid=pair.compose(group);name=s['case']+'-T'+str(int(m['time']))
                row=dict(case=s['case'],time=m['time'],status='RESOURCE_ONLY_NOT_ENERGY_OR_SAMPLING_QUALIFIED',amplitude_kind=kind,
                    pair_coefficient_slots=len(angles),group_coefficient_slots=len(m['group']),independent_parameter_rank=None,
                    pair=pairpoints,HF_GROUP=compile_record(hf.compose(group),name+'-HF_GROUP'),HYBRID=compile_record(hybrid,name+'-HYBRID'))
                rows.append(row);save(LOCAL/'progress.json',dict(rows=rows,CPU_seconds=process_time()-cpu))
                print('N67_RESOURCE',s['case'],m['time'],len(rows),flush=True)
        save(LOCAL/'terminal.json',dict(status='ALL12_RESOURCE_PREFLIGHT_ROWS_RECORDED',rows=rows,controls=controls,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,GPU_allocations=0,molecular_energy_or_large_state_verified=False,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),rows=rows,controls=controls,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise

if __name__=='__main__':{'qualify':qualify,'prepare':prepare,'run':run}[sys.argv[1]]()
