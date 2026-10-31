"""N68 exact full-register Rustiq group synthesis, no fixed occupations."""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import gzip,os,sys
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Operator,Statevector
from qiskit.synthesis import synth_pauli_network_rustiq
from .paths import ROOT
from .io import load,save,write,sha256,space
from .pair_group_preflight import LOCAL as N67,group_circuit,pair_preparation,angles_from_t2,metrics
from .slater_prefix_preflight import compile_views
from paperstudy.lucj import amplitudes_from_saved
LOCAL=ROOT/'runs/N68-EXACT-GROUP-SYNTHESIS-v1'
FIELDS=['CNOT','CNOT_Depth','Total_Depth']

def rustiq_group(group,optimize_count):
    group_circuit(group)  # Validate finite labels and commuting-group contract.
    n=len(group[0][0]);network=[]
    for label,angle in group:
        support=[i for i,p in enumerate(label) if p!='I']
        if not support:raise ValueError('Identity scalar must be handled explicitly')
        network.append((''.join(label[i] for i in support),support,2*float(angle)))
    return synth_pauli_network_rustiq(n,network,optimize_count=bool(optimize_count),preserve_order=True,upto_clifford=False,upto_phase=False)

def choose(base,points):
    eligible=[base]+[p for p in points if all(p['resources'][k]<=base['resources'][k] for k in FIELDS)]
    return min(eligible,key=lambda p:tuple(p['resources'][k] for k in FIELDS))

def qualify():
    if (LOCAL/'qualification.json').exists():raise RuntimeError('Existing qualifier; no replace')
    groups=[[['XZXI',.239],['YZYI',.239]],
        [['YZZZXI',.123],['XZZZYI',-.123]],
        [['XZZZXIII',.17],['YZZZYIII',.17],['XZZZXIIZ',-.17],['YZZZYIIZ',-.17]]]
    cpu,wall=process_time(),perf_counter();rows=[]
    try:
        for group in groups:
            n=len(group[0][0]);base=group_circuit(group);u=Operator(base).data
            for mode in [True,False]:
                q=rustiq_group(group,mode);error=float(np.linalg.norm(Operator(q).data-u))
                norb=n//2;k=1 if n<8 else 2;t2=np.zeros((k,k,norb-k,norb-k))
                for i in range(k):
                    for a in range(norb-k):t2[i,i,a,a]=.17+(.03*(i+a))
                pair=pair_preparation(norb,k,angles_from_t2(norb,k,t2))
                state_error=float(np.linalg.norm(Statevector(pair.compose(q)).data-Statevector(pair.compose(base)).data))
                if error>1e-10 or state_error>1e-10:raise ValueError('Exact unitary/phase mismatch')
                rows.append(dict(n_qubits=n,optimize_count=mode,whole_unitary_error=error,hybrid_complex_state_error=state_error))
        save(LOCAL/'qualification.json',dict(status='ALL6_FULL_UNITARY_AND_HYBRID_PHASE_CHECKS_PASSED',rows=rows,code_SHA256=sha256(Path(__file__).read_bytes()),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall))
    except Exception as exc:
        save(LOCAL/'qualification-failed.json',dict(rows=rows,error=repr(exc),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; no replace')
    space();old=load(N67/'protocol.json');t=load(N67/'terminal.json');qual=load(LOCAL/'qualification.json')
    assert len(t['rows'])==12 and qual['status']=='ALL6_FULL_UNITARY_AND_HYBRID_PHASE_CHECKS_PASSED'
    assert qual['code_SHA256']==sha256(Path(__file__).read_bytes())
    bindings=dict(old['bindings'])
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N68-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2 and not any(int(s.get(k,0)) for s in suites for k in ['failures','errors'])
    bindings[str(tests)]=sha256(tests.read_bytes())
    for f in [N67/'protocol.json',N67/'terminal.json',Path(__file__),ROOT/'analysis/N68_EXACT_GROUP_SYNTHESIS_PLAN_JA.md']:
        bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Research quota; no deletion')
    save(LOCAL/'protocol.json',dict(version='N68_EXACT_FULL_REGISTER_GROUP_SYNTHESIS_NOT_PROJECTED',bindings=bindings,CPU_cap=30,wall_cap=90,new_bytes=4*2**20,root_bytes=2**30,
        qualifier_SHA256=sha256((LOCAL/'qualification.json').read_bytes()),sources=old['sources'],members=old['members'],compiler_levels=[1,3],
        rustiq_optimize_count=[True,False],no_state_restriction=True,no_angle_pruning=True,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,goal_complete=False))

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution; no repeat')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h
    old=load(N67/'terminal.json');save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[]
    initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file());base_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def bound():
        space();size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if process_time()-cpu>p['CPU_cap'] or perf_counter()-wall>p['wall_cap']:raise TimeoutError('N68 budget')
        if size>p['new_bytes'] or base_bytes+size-initial>p['root_bytes']:raise OSError('N68 storage cap')
    try:
        for s in p['sources']:
            meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
            pair=pair_preparation(n,k,angles_from_t2(n,k,t2))
            for m in [m for m in p['members'] if m['case']==s['case']]:
                original=next(r for r in old['rows'] if r['case']==s['case'] and r['time']==m['time'])
                baseline=dict(min(original['HYBRID'],key=lambda x:tuple(x['resources'][k] for k in FIELDS)),compiler='ORIGINAL_PAULI_GENERIC')
                points=[]
                for mode in p['rustiq_optimize_count']:
                    group=rustiq_group(m['group'],mode);raw=pair.compose(group)
                    for level in p['compiler_levels']:
                        bound();tick=process_time();prep,sample,res=compile_views(raw,level);files={}
                        for role,q in [('preparation',prep),('sampling',sample)]:
                            stream=BytesIO();qpy.dump(q,stream);blob=gzip.compress(stream.getvalue(),mtime=0)
                            f=LOCAL/'circuits'/f'{s["case"]}-T{int(m["time"])}-C{int(mode)}-L{level}-{role}.qpy.gz';write(f,blob)
                            files[role]=dict(file=str(f),SHA256=sha256(blob))
                        points.append(dict(compiler='RUSTIQ_COUNT' if mode else 'RUSTIQ_DEPTH',level=level,resources=metrics(sample),preparation_resources=metrics(prep),files=files,CPU_seconds=process_time()-tick))
                selected=choose(baseline,points)
                rows.append(dict(case=s['case'],time=m['time'],baseline=baseline,points=points,selected=selected,guard_all3_not_increased=all(selected['resources'][k]<=baseline['resources'][k] for k in FIELDS),
                    amplitude_kind=kind,independent_parameter_rank=None,energy_Ha=None,Recovery_pct=None))
                save(LOCAL/'progress.json',dict(rows=rows,CPU_seconds=process_time()-cpu))
        save(LOCAL/'terminal.json',dict(status='ALL12_EXACT_SYNTHESIS_RESOURCE_ROWS_RECORDED',rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,molecular_large_state_verified=False,matched_LUCJ_compiler_opportunity=False,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise

if __name__=='__main__':{'qualify':qualify,'prepare':prepare,'run':run}[sys.argv[1]]()
