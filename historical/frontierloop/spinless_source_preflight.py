"""N34 explicit SDK spinless-L1 variant, independent states and actual resources.

This is NOT an exact reproduction of Yoshida et al.'s unpublished initializer.
No physical energy is evaluated and no molecular optimizer is run here.
"""
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter
import inspect,importlib.metadata,os,resource
import math
import ffsim
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Statevector
from pyscf.fci import cistring
from paperstudy.lucj import amplitudes_from_saved
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,space
from . import prefix_performance_sampling as legacy
from . import analytical_source_optimization as n32
from .slater_prefix_preflight import compile_views

LOCAL=ROOT/'runs/N34-SPINLESS-L1-PREFLIGHT-v1'
METHOD='SDK_DIRECT_RESTRICTED_T2_SPINLESS_L1'


def installed_bindings():
    funcs=dict(factory=ffsim.UCJOpSpinless.from_t_amplitudes,parameters=ffsim.UCJOpSpinless.from_parameters,
        factorization=ffsim.linalg.double_factorized_t2,quantum_gate=ffsim.qiskit.UCJOpSpinlessJW)
    return {name:sha256(inspect.getsource(fun).encode()) for name,fun in funcs.items()}


def pairs(n):return [(i,i+1) for i in range(n-1)]


def make_operator(t1,t2):
    k,v=t1.shape;n=k+v
    if t2.shape!=(k,k,v,v) or np.iscomplexobj(t1) or np.iscomplexobj(t2) or not np.isfinite(t1).all() or not np.isfinite(t2).all():raise ValueError('Explicit real restricted t1/t2 order required')
    return ffsim.UCJOpSpinless.from_t_amplitudes(t2,t1=t1,n_reps=1,interaction_pairs=pairs(n),tol=1e-8,optimize=False)


def logical(op,k):
    q=QuantumCircuit(op.norb)
    q.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(op.norb,k),range(op.norb))
    q.append(ffsim.qiskit.UCJOpSpinlessJW(op),range(op.norb))
    return ffsim.qiskit.PRE_INIT.run(q)


def with_copy(q):
    n=q.num_qubits;full=QuantumCircuit(2*n);full.compose(q,qubits=list(range(n)),inplace=True)
    for i in range(n):full.cx(i,n+i)
    return full


def reference_state(op,k):
    n=op.norb;ids=np.asarray(cistring.make_strings(range(n),k),np.uint64)
    if not np.array_equal(ids,np.sort(ids)) or len(ids)!=math.comb(n,k):raise RuntimeError('Explicit sector address convention mismatch')
    v=ffsim.apply_unitary(ffsim.hartree_fock_state(n,k),op,norb=n,nelec=k)
    if v.shape!=(len(ids),) or abs(np.vdot(v,v).real-1)>1e-10:raise RuntimeError('Actual sector norm/width differs')
    return ids,v


def phase_error(a,b):
    overlap=np.vdot(a,b)
    if abs(overlap)<1e-10:return float('inf')
    return float(np.max(abs(a-b*np.exp(-1j*np.angle(overlap)))))


def primitive(q,k):
    # Fresh boundary condition, independently checked below; no frozen module edit.
    n=q.num_qubits;pad=QuantumCircuit(2*n);pad.compose(q,qubits=list(range(n)),inplace=True)
    module=legacy.qualified_module();cfg,a=module.plan(pad,n,(k,0));v=module.numpy_state(cfg,a)
    return v,cfg,a


def parity(op,k,full_fock=False):
    ids,v=reference_state(op,k);q=logical(op,k);other,cfg,a=primitive(q,k)
    result=dict(ffsim_vs_materialized_state_error=phase_error(v,other),
        ffsim_vs_materialized_probability_L1=float(np.sum(abs(abs(v)**2-abs(other)**2))),
        sector_dimension=len(v),norm_squared=float(np.vdot(v,v).real),CPU_only_vacuum_beta_boundary_qualified=True)
    if full_fock:
        actual=Statevector.from_instruction(q).data;expected=np.zeros(1<<op.norb,complex);expected[ids]=v
        copy=Statevector.from_instruction(with_copy(q)).data;expectedcopy=np.zeros(1<<(2*op.norb),complex);expectedcopy[ids|(ids<<np.uint64(op.norb))]=v
        result.update(ffsim_vs_Qiskit_state_error=phase_error(expected,actual),
            ffsim_vs_Qiskit_probability_L1=float(np.sum(abs(abs(expected)**2-abs(actual)**2))),
            full_copy_isometry_state_error=phase_error(expectedcopy,copy))
    if any(value>1e-8 for key,value in result.items() if key.endswith(('error','L1'))):raise RuntimeError('Independent spinless state/decoder parity failed')
    return ids,v,q,cfg,a,result


def prepare():
    storage=legacy.capacity();n32.verified()
    if LOCAL.exists():raise RuntimeError('Existing N34; preserve/reconcile')
    old=load(ROOT/'runs/N25-PAIRED-LOWSHOT-v1/protocol.json');sources=old['sources']
    if len(sources)!=4:raise RuntimeError('All4 physical conditions required')
    for s in sources:
        for file,key in [('amplitudes','amplitudes_SHA256'),('initialization_record_path','initialization_record_SHA256')]:
            if sha256(Path(s[file]).read_bytes())!=s[key]:raise RuntimeError('Old paid initializer changed')
        if s['definition']['problem_id']!=s['initialization_record']['problem_id']:raise RuntimeError('Initializer/orbital problem mismatch')
    LOCAL.mkdir(parents=True)
    save(LOCAL/'protocol.json',dict(version='N34_EXPLICIT_SDK_DIRECT_RESTRICTED_T2_SPINLESS_L1_PREFLIGHT_V1',method=METHOD,sources=sources,
        code_SHA256=sha256(Path(__file__).read_bytes()),installed_bindings=installed_bindings(),
        helpers={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('prefix_performance_sampling.py','slater_prefix_preflight.py')},
        initializer_helper_SHA256=sha256((OLD/'paperstudy/lucj.py').read_bytes()),
        N32_protocol_SHA256=sha256((n32.LOCAL/'protocol.json').read_bytes()),
        N33_summary_SHA256=sha256((ROOT/'analysis/N33-ALL-summary.json').read_bytes()),
        plan_SHA256=sha256((ROOT/'analysis/N34_SPINLESS_PREFLIGHT_PLAN_JA.md').read_bytes()),
        packages={x:importlib.metadata.version(x) for x in ('numpy','scipy','qiskit','ffsim','pyscf')},
        fixtures=[dict(n=n,k=k,seed=seed) for n,k,seed in [(4,2,34001),(5,2,34002),(6,3,34003)]],
        factory=dict(n_reps=1,tol=1e-8,optimize=False,t1_final_rotation=True,input='UNMODIFIED_REAL_RESTRICTED_TENSOR_ijab',Jastrow='ADJACENT_UPPER_PAIRS'),
        CPU_seconds_cap=600,wall_seconds_cap=600,new_run_cap_bytes=32*2**20,state_and_probability_tolerance=1e-8,
        physical_energy_evaluations=0,new_model_QPU_shots=0,new_molecular_optimizers=0,
        original_author_initializer_exactly_reproduced=False,full40q_statevector_simulated=False,storage_before=storage,goal_complete=False))
    print('N34_FROZEN3FIXTURES_ALL4_SDK_VARIANT_NO_ENERGY_SHOTS',flush=True)


def verified():
    p=load(LOCAL/'protocol.json')
    if p['code_SHA256']!=sha256(Path(__file__).read_bytes()) or p['installed_bindings']!=installed_bindings():raise RuntimeError('Frozen N34 source/API drift')
    if p['initializer_helper_SHA256']!=sha256((OLD/'paperstudy/lucj.py').read_bytes()):raise RuntimeError('Initializer semantics changed')
    for name,h in p['helpers'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Qualified helper drift')
    if {x:importlib.metadata.version(x) for x in p['packages']}!=p['packages']:raise RuntimeError('Installed runtime drift')
    if p['N33_summary_SHA256']!=sha256((ROOT/'analysis/N33-ALL-summary.json').read_bytes()):raise RuntimeError('Bound N33 summary changed')
    return p


def store_qpy(q,path):
    buf=BytesIO();qpy.dump(q,buf);blob=buf.getvalue();write(path,blob);actual=qpy.load(BytesIO(blob))[0]
    if actual.count_ops()!=q.count_ops() or actual.depth()!=q.depth():raise RuntimeError('Actual QPY roundtrip mismatch')
    return dict(path=str(path),SHA256=sha256(blob))


def run():
    p=verified();space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing N34 attempt; inspect actual process/terminal, no blind restart')
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    resource.setrlimit(resource.RLIMIT_CPU,(p['CPU_seconds_cap'],p['CPU_seconds_cap']+1))
    cpu,wall=process_time(),perf_counter();fixtures=[];sources=[]
    def bound():
        space()
        if process_time()-cpu>p['CPU_seconds_cap'] or perf_counter()-wall>p['wall_seconds_cap']:raise TimeoutError('Bounded N34 CPU/wall')
        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_run_cap_bytes']:raise OSError('Bounded new N34 storage')
    try:
        for f in p['fixtures']:
            bound();n,k=f['n'],f['k'];rng=np.random.default_rng(f['seed'])
            count=ffsim.UCJOpSpinless.n_params(n,1,interaction_pairs=pairs(n),with_final_orbital_rotation=True)
            op=ffsim.UCJOpSpinless.from_parameters(rng.normal(scale=.3,size=count),norb=n,n_reps=1,interaction_pairs=pairs(n),with_final_orbital_rotation=True)
            ids,v,q,cfg,a,result=parity(op,k,full_fock=True);fixtures.append(dict(**f,**result,status='PASS'))
            save(LOCAL/'fixture-progress.json',fixtures);print('N34_FIXTURE_PASS',n,k,result,flush=True)
        for s in p['sources']:
            bound();tick,cc=perf_counter(),process_time();d=s['definition'];case=s['case'];n,k=d['norb'],d['nelec'][0]
            for file,key in [('amplitudes','amplitudes_SHA256'),('initialization_record_path','initialization_record_SHA256')]:
                if sha256(Path(s[file]).read_bytes())!=s[key]:raise RuntimeError('Actual initialization provenance changed')
            t1,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k);op=make_operator(t1,t2)
            if op.norb!=n or op.n_reps!=1 or math.comb(n,k)>20000:raise RuntimeError('Registered source dimension changed')
            ids,v,q,cfg,a,result=parity(op,k)
            folder=LOCAL/'sources'/case;folder.mkdir(parents=True)
            operator=folder/'operator.npz';write(operator,arrays(diag_coulomb_mats=op.diag_coulomb_mats,orbital_rotations=op.orbital_rotations,final_orbital_rotation=op.final_orbital_rotation))
            state=folder/'pair-state.npz';write(state,arrays(half_strings=ids,amplitudes=v,probabilities=abs(v)**2))
            theta=op.to_parameters(interaction_pairs=pairs(n));restored=ffsim.UCJOpSpinless.from_parameters(theta,norb=n,n_reps=1,interaction_pairs=pairs(n),with_final_orbital_rotation=True)
            matrix_error=max(float(np.max(abs(getattr(op,name)-getattr(restored,name)))) for name in ('diag_coulomb_mats','orbital_rotations','final_orbital_rotation'))
            if matrix_error>1e-8:raise RuntimeError('Same-mask parameter matrix roundtrip failed')
            views=[]
            for view,circuit in [('PAIR_ENCODING',q),('FULL_SPIN_COPY',with_copy(q))]:
                logicalfile=store_qpy(circuit,folder/view/'logical.qpy');compiled=[]
                for level in (1,3):
                    prep,samp,res=compile_views(circuit,level)
                    compiled.append(dict(resources=res,circuits=dict(preparation=store_qpy(prep,folder/view/f'opt{level}-preparation.qpy'),sampling=store_qpy(samp,folder/view/f'opt{level}-sampling.qpy'))))
                views.append(dict(view=view,n_qubits=circuit.num_qubits,logical=logicalfile,compiler_points=compiled))
            baseline=load(n32.LOCAL/'cases'/case/'result.json')
            record=dict(case=case,method=METHOD,status='RESOURCE_AND_STATE_PREFLIGHT_TERMINAL',definition=d,
                physical_nelec=d['nelec'],sampler_pair_occupations=k,CPU_primitive_padding_nelec=[k,0],
                operator=str(operator),operator_SHA256=sha256(operator.read_bytes()),state=str(state),state_SHA256=sha256(state.read_bytes()),
                source_parity=result,parameter_matrix_roundtrip_error=matrix_error,amplitude_kind=kind,
                initialization_selected_method=s['initialization_record']['selected_method'],
                old_initialization_record=s['initialization_record'],old_initialization_record_SHA256=s['initialization_record_SHA256'],
                optimizer_coordinate_slots=len(theta),known_Slater_chart_declared_parameters=2*k*(n-k)+n*n+n-1,
                physical_Jacobian_rank=None,resources=views,original_C5_L1_resources=baseline['original_resources'],
                CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tick,new_model_QPU_shots=0,new_energy_evaluations=0,
                author_exact_reproduction=False,molecular_quality_unverified=True)
            save(folder/'result.json',record);sources.append(record);save(LOCAL/'progress.json',dict(fixtures=fixtures,sources=sources))
            print('N34_SOURCE_TERMINAL',case,record['CPU_seconds'],[(v['view'],v['compiler_points'][0]['resources']) for v in views],flush=True)
        bound();save(LOCAL/'terminal.json',dict(status='N34_ALL3_FIXTURES_AND4_SOURCE_PREFLIGHTS_PASSED',fixtures=fixtures,sources=sources,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_QPU_shots=0,new_energy_evaluations=0,
            new_molecular_optimizers=0,full40q_statevector_simulated=False,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(error=repr(exc),fixtures=fixtures,sources=sources,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
