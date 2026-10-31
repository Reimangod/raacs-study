"""N23 structurally faithful no-J LUCJ ablation and uniform classical control.

These are known classically sampleable controls, not quantum-advantage methods.
The bounded enumerated one-spin distribution is exponential in orbital number;
it is useful here (<=15504 entries), NOT a general scalable state simulator.
"""
import argparse
from io import BytesIO
from itertools import combinations
from math import comb
from pathlib import Path
from time import perf_counter,process_time
import importlib.metadata
import xml.etree.ElementTree as ET
import ffsim
import numpy as np
from qiskit import QuantumCircuit,qpy
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest
from .prefix_performance_sampling import capacity
from .gaussian_source import checked_counts
from .slater_prefix_preflight import compile_views
from paperstudy.lucj import make_circuit

LOCAL=ROOT/'runs/N23-BACKBONE-UNIFORM-SOURCES-v1'
METHODS=('GAUSSIAN_BACKBONE','UNIFORM_SECTOR')
REPLICATES=(230101,230102,230103)


def backbone(op,n,nelec):
    qc=ffsim.qiskit.PRE_INIT.run(make_circuit(op,n,nelec))
    gate=qc.data[0].operation
    if gate.name!='slater_jw':raise ValueError('Original source lacks Slater reference')
    a,b=gate.orbital_rotation_a.copy(),gate.orbital_rotation_b.copy()
    skipped=0
    for instruction in qc.data[1:]:
        g=instruction.operation
        if g.name=='diag_coulomb_jw':skipped+=1;continue
        if g.name!='orb_rot_jw':raise ValueError('Unrecognized original circuit order')
        a=g.orbital_rotation_a@a;b=g.orbital_rotation_b@b
    if not skipped or not np.allclose(a,op.final_orbital_rotation,atol=1e-10,rtol=0) or not np.array_equal(a,b):
        raise ValueError('No-J product does not match actual shared-spin final rotation')
    occupied=tuple(tuple(map(int,x)) for x in gate.occupied_orbitals)
    if tuple(map(len,occupied))!=tuple(nelec):raise ValueError('Changed particle sector')
    result=QuantumCircuit(2*n)
    result.append(ffsim.qiskit.PrepareSlaterDeterminantJW(n,occupied,(a,b)),range(2*n))
    return result,a,b,occupied,skipped


def one_spin_distribution(n,occupied,u,max_dimension=20000):
    occupied=tuple(occupied);k=len(occupied)
    if len(set(occupied))!=k or any(i<0 or i>=n for i in occupied):raise ValueError('Invalid occupied orbitals')
    if comb(n,k)>max_dimension:raise ValueError('Bounded one-spin enumeration cap; not general large-system simulation')
    u=np.asarray(u)
    if u.shape!=(n,n) or np.linalg.norm(u.conj().T@u-np.eye(n))>1e-10:raise ValueError('Unitary rotation required')
    rows=list(combinations(range(n),k));ids=np.array([sum(1<<i for i in row) for row in rows],dtype=np.uint64)
    probabilities=np.empty(len(rows))
    # Tiny batches avoid retaining every determinant matrix at once.
    for start in range(0,len(rows),1024):
        selected=np.array(rows[start:start+1024],dtype=int).reshape(-1,k)
        minors=u[selected][:,:,occupied]
        probabilities[start:start+len(selected)]=abs(np.linalg.det(minors))**2
    norm=float(probabilities.sum())
    if not np.isfinite(probabilities).all() or abs(norm-1)>1e-10:raise ValueError('Cauchy-Binet normalization failed')
    order=np.argsort(ids)
    return ids[order],probabilities[order]/norm,norm


def draw(ids,pa,pb,n,shots,seed):
    rng=np.random.default_rng(seed)
    alpha=rng.choice(ids,size=shots,p=pa);beta=rng.choice(ids,size=shots,p=pb)
    return alpha | (beta << np.uint64(n))


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N23; preserve and reconcile')
    capacity()
    if load(ROOT/'analysis/N22-support-and-structure-diagnostic.json')['decision']!='STOP_PREFIX_ONLY_NO_SHOT_OR_SEED_EXPANSION':raise RuntimeError('Close N22 first')
    old=load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/protocol.json')
    sources=[s for s in old['sources'] if s['status']=='AVAILABLE' and s['method'].endswith('L1')]
    if len(sources)!=4:raise RuntimeError('All four physical cases required')
    qualification=ROOT/'qualification-N23-source-preexecution-v2-tests.xml'
    if not qualification.exists():raise RuntimeError('Qualify distribution, circuit and sampler before freeze')
    suites=list(ET.parse(qualification).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=4 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise RuntimeError('All four numerical qualification tests must pass')
    LOCAL.mkdir()
    save(LOCAL/'protocol.json',dict(version='N23_ALL_GAUSSIAN_ROTATIONS_NO_J_VS_UNIFORM_SECTOR_V1',sources=sources,
        methods=list(METHODS),replicates=list(REPLICATES),shots_per_draw=100000,
        planned_draws=24,maximum_new_exact_Gaussian_model_equivalent_draws=1200000,maximum_new_classical_uniform_draws=1200000,
        source_rule='REMOVE_ALL_DIAGONAL_J_KEEP_PRODUCT_OF_ALL_REGISTERED_ORBITAL_ROTATIONS_NO_OPTIMIZATION',
        uniform_rule='INDEPENDENT_UNIFORM_ALPHA_BETA_FIXED_PARTICLE_SECTOR_NO_HAMILTONIAN_WEIGHTS',
        no_reference_energy_or_FCI_coefficients=True,no_molecule_specific_tuning=True,
        code_SHA256=sha256(Path(__file__).read_bytes()),qualification_SHA256=sha256(qualification.read_bytes()),
        helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('gaussian_source.py','slater_prefix_preflight.py')},
        packages={name:importlib.metadata.version(name) for name in ('ffsim','qiskit','numpy','scipy')},
        compiler_levels=[1,3],seed_transpiler=8137,approximation_degree=1.,
        one_spin_distribution_dimension_cap=20000,max_case_wall_seconds=120,global_wall_seconds=600,
        circuit_parameters='GENERIC_SHARED_COMPLEX_SLATER_CHART_2_NOCC_NVIRT_NOT_ZERO_OPTIMIZER_VARIABLES_OR_JACOBIAN_RANK',
        uniform_control_has_no_claimed_quantum_circuit=True,original_LUCJ_initialization_costs_retained=True,
        new_QPU_shots=0,GPU_allocations=0,goal_complete=False))
    print('N23_FROZEN_ALL4_CASES24_DRAWS_GAUSSIAN_AND_UNIFORM_SEPARATE',flush=True)


def run():
    p=load(LOCAL/'protocol.json');capacity()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; never blindly restart')
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or any(sha256((ROOT/'frontierloop'/name).read_bytes())!=v for name,v in p['helper_SHA256'].items()):raise RuntimeError('Frozen implementation changed')
    if {name:importlib.metadata.version(name) for name in p['packages']}!=p['packages']:raise RuntimeError('Frozen runtime changed')
    save(LOCAL/'started.json',dict(pid=__import__('os').getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();sources=[];draws=[];attempts=[]
    for s in p['sources']:
        case=s['case'];folder=LOCAL/case;folder.mkdir();cc,tt=process_time(),perf_counter()
        try:
            for field,hfield in [('operator','operator_SHA256'),('construction','construction_SHA256'),('source_result','source_result_SHA256')]:
                if sha256(Path(s[field]).read_bytes())!=s[hfield]:raise RuntimeError('Registered source changed')
            n=s['norb'];nelec=tuple(s['nelec'])
            with np.load(s['operator'],allow_pickle=False) as a:op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
            qc,a,b,occupied,skipped=backbone(op,n,nelec)
            ids,pa,norm=one_spin_distribution(n,occupied[0],a,p['one_spin_distribution_dimension_cap'])
            # Shared spin is an explicit family binding, not accidental equality.
            pb=pa;uniform=np.full(len(ids),1/len(ids))
            distribution=arrays(half_strings=ids,probabilities_a=pa,probabilities_b=pb,rotation_a=a,rotation_b=b)
            write(folder/'source-distribution.npz',distribution)
            outputs=[]
            for level in p['compiler_levels']:
                prep,sampling,resources=compile_views(qc,level);point=dict(resources=resources)
                for role,circuit in [('preparation',prep),('sampling',sampling)]:
                    stream=BytesIO();qpy.dump(circuit,stream);blob=stream.getvalue();path=folder/f'opt{level}-{role}.qpy';write(path,blob)
                    restored=qpy.load(BytesIO(blob))[0]
                    if circuit.count_ops()!=restored.count_ops() or circuit.depth()!=restored.depth():raise RuntimeError('Actual QPY roundtrip differs')
                    point[role]=dict(path=str(path),SHA256=sha256(blob))
                outputs.append(point)
            state=dict(case=case,norb=n,nelec=list(nelec),original_source=s,
                problem_id=load(s['construction'])['problem_id'],source_rule=p['source_rule'],removed_diagonal_blocks=skipped,
                distribution=str(folder/'source-distribution.npz'),distribution_SHA256=sha256(distribution),
                actual_one_spin_sector_dimension=len(ids),Cauchy_Binet_norm=norm,
                declared_parameters=2*nelec[0]*(n-nelec[0]),independent_Jacobian_rank=None,optimizer_variables=0,
                all_compiler_points=outputs,construction_and_distribution_CPU_seconds=process_time()-cc,
                construction_and_distribution_wall_seconds=perf_counter()-tt,original_initialization_cost_not_zero=True)
            save(folder/'source.json',state);sources.append(state)
            for method in p['methods']:
                for replicate in p['replicates']:
                    if perf_counter()-tt>p['max_case_wall_seconds'] or perf_counter()-wall>p['global_wall_seconds']:raise TimeoutError('N23 source resource cap')
                    seed=int(digest(['N23',case,method,replicate,s['operator_SHA256']])[:16],16)%(1<<63)
                    attempt=dict(case=case,method=method,replicate=replicate,status='STARTED',known_generated_samples=None)
                    attempts.append(attempt);save(LOCAL/'draw-attempts.json',attempts)
                    dc,dw=process_time(),perf_counter()
                    samples=draw(ids,pa if method=='GAUSSIAN_BACKBONE' else uniform,pb if method=='GAUSSIAN_BACKBONE' else uniform,n,p['shots_per_draw'],seed)
                    attempt.update(status='GENERATED_PENDING_VALIDATION',known_generated_samples=len(samples));save(LOCAL/'draw-attempts.json',attempts)
                    did=digest([case,method,seed,p['shots_per_draw'],state['problem_id']]);path=folder/(did+'.npz')
                    values,counts=checked_counts(samples,n,nelec,p['shots_per_draw']);payload=arrays(blocked_ids=values,counts=counts);write(path,payload)
                    record=dict(draw_id=did,case=case,method=method,replicate=replicate,status='COMPLETED',seed=seed,
                        payload=str(path),payload_SHA256=sha256(payload),actual_samples=int(counts.sum()),
                        source_record_SHA256=sha256((folder/'source.json').read_bytes()),
                        CPU_seconds=process_time()-dc,wall_seconds=perf_counter()-dw,
                        sample_role='EXACT_CLASSICAL_GAUSSIAN_MODEL_EQUIVALENT' if method=='GAUSSIAN_BACKBONE' else 'CLASSICAL_UNIFORM_RANDOM',actual_QPU_shots=0)
                    save(folder/(did+'.json'),record);draws.append(record);save(LOCAL/'progress.json',dict(sources=sources,draws=draws))
                    attempt.update(status='COMPLETED',draw_id=did,CPU_seconds=record['CPU_seconds']);save(LOCAL/'draw-attempts.json',attempts)
                    print('N23',case,method,replicate,len(values),record['CPU_seconds'],flush=True)
        except Exception as exc:
            if attempts and attempts[-1]['case']==case and attempts[-1]['status']!='COMPLETED':
                attempts[-1].update(status='FAILED_PRESERVED',error=repr(exc));save(LOCAL/'draw-attempts.json',attempts)
            save(folder/'failure-preserved.json',dict(error=repr(exc),traceback=__import__('traceback').format_exc(),known_CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tt))
            for method in p['methods']:
                for replicate in p['replicates']:
                    if not any(d['case']==case and d['method']==method and d['replicate']==replicate for d in draws):
                        draws.append(dict(case=case,method=method,replicate=replicate,status='FAILED_OR_NOT_RUN_PRESERVED',actual_samples=None,error=repr(exc)))
    if len(draws)!=24:raise RuntimeError('All planned source states required')
    out=dict(status='N23_ALL24_DRAW_STATES_TERMINAL',draws=draws,sources=sources,
        exact_Gaussian_model_equivalent_samples=sum(d['actual_samples'] for d in draws if d['status']=='COMPLETED' and d['method']=='GAUSSIAN_BACKBONE'),
        classical_uniform_samples=sum(d['actual_samples'] for d in draws if d['status']=='COMPLETED' and d['method']=='UNIFORM_SECTOR'),
        known_generated_classical_samples_including_failures=sum(a['known_generated_samples'] or 0 for a in attempts),
        attempts_with_unknown_generated_sample_count=sum(a['known_generated_samples'] is None for a in attempts),
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_QPU_shots=0,GPU_allocations=0,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),goal_complete=False)
    save(LOCAL/'terminal.json',out);print(out['status'],out['CPU_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args();globals()[args.action]()
