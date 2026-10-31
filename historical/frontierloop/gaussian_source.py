"""N22 Gaussian-prefix ablation: a known, classically sampleable control.

No energy/reference input, learned parameters, fidelity claim, or quantum
advantage. Extract the *actual* PRE_INIT Slater gate, not a guessed rotation.
"""
from collections import Counter
from io import BytesIO
from pathlib import Path
import argparse
import importlib.metadata
from time import process_time, perf_counter
import ffsim
import numpy as np
from qiskit import QuantumCircuit, qpy
from .paths import ROOT
from .io import load, save, write, arrays, sha256, digest
from .prefix_performance_sampling import capacity
from .slater_prefix_preflight import compile_views
from .slater_prefix_compression import parse_prefix
from paperstudy.lucj import make_circuit

LOCAL = ROOT/'runs/N22-GAUSSIAN-PREFIX-v1'
REPLICATES = (220101, 220102, 220103)


def extract(op, n, nelec):
    circuit = ffsim.qiskit.PRE_INIT.run(make_circuit(op, n, nelec))
    first = circuit.data[0]
    gate = first.operation
    if gate.name != 'slater_jw' or [circuit.find_bit(q).index for q in first.qubits] != list(range(2*n)):
        raise ValueError('Explicit original blocked-register Slater prefix required')
    a, b = np.array(gate.orbital_rotation_a), np.array(gate.orbital_rotation_b)
    occupied = tuple(tuple(map(int, x)) for x in gate.occupied_orbitals)
    if tuple(map(len, occupied)) != tuple(nelec) or not np.array_equal(a, b):
        raise ValueError('Registered shared-spin Slater chart required')
    if max(np.linalg.norm(u.conj().T@u-np.eye(n)) for u in (a, b)) > 1e-10:
        raise ValueError('Nonunitary orbital rotation')
    prefix = QuantumCircuit(2*n)
    prefix.append(gate, range(2*n))
    parsed = parse_prefix(gate.definition, n)
    # The shared alpha/beta chart is the same as the N15 parent chart.
    coordinates = sum(2 if r['name']=='xx_plus_yy' else 1 for r in parsed.records[0])
    return prefix, a, b, occupied, coordinates


def checked_counts(samples, n, nelec, expected):
    ids, counts = np.unique(np.asarray(samples, dtype=np.uint64), return_counts=True)
    mask = (1 << n)-1
    if int(counts.sum()) != expected or any(int(x) >= 1 << (2*n) or (int(x)&mask).bit_count()!=nelec[0] or (int(x)>>n).bit_count()!=nelec[1] for x in ids):
        raise ValueError('Actual draw count or particle sector differs')
    return ids, counts.astype(np.int64)


def probability(x, n, a, b, occupied):
    mask = (1 << n)-1
    sa, sb = int(x)&mask, int(x)>>n
    ia = [i for i in range(n) if sa >> i & 1]
    ib = [i for i in range(n) if sb >> i & 1]
    if (len(ia), len(ib)) != tuple(map(len, occupied)):
        return 0.
    return float(abs(np.linalg.det(a[np.ix_(ia, occupied[0])]))**2 * abs(np.linalg.det(b[np.ix_(ib, occupied[1])]))**2)


def prepare():
    if LOCAL.exists():
        raise RuntimeError('Existing N22: inspect, never overwrite or restart')
    capacity()
    for name in ('N20', 'N21'):
        if load(ROOT/f'analysis/{name}-nested-CI-summary.json')['status'] != name+'_ALL16_POINTS_AND_ALL_ACTUAL_SAVED_ROOTS_AUDITED':
            raise RuntimeError('Close all prior points first')
    prior = load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/protocol.json')
    sources = [s for s in prior['sources'] if s['status']=='AVAILABLE' and s['method'].endswith('L1')]
    if len(sources) != 4 or len({s['case'] for s in sources}) != 4:
        raise RuntimeError('All four physical cases required')
    qualified = ROOT/'qualification-N22-gaussian-preexecution-v2-tests.xml'
    if not qualified.exists():
        raise RuntimeError('Numerical qualification must precede freeze')
    LOCAL.mkdir()
    save(LOCAL/'protocol.json', dict(version='N22_FIXED_GAUSSIAN_PREFIX_CONTROL_V1', sources=sources,
        replicates=list(REPLICATES), shots_per_draw=100000, sampling_chunk=10000,
        seeds='FIRST_16_HEX_SHA256_CASE_OPERATOR_REPLICATE_N22_GAUSSIAN_MOD2POW63',
        max_source_body_wall_seconds=240, global_body_wall_seconds=1200,
        source_policy='FIRST_ACTUAL_PRE_INIT_SLATER_GATE_ONLY_ENTIRE_CORRELATED_SUFFIX_REMOVED',
        compiler_levels=[1,3], approximation_degree=1., seed_transpiler=8137,
        sampling_backend='FFSIM_0p0p84_EXACT_CLASSICAL_PROJECTION_DPP_NO_GPU_OR_QPU',
        classical_efficient_gaussian_control_not_novel_ansatz=True,
        full_LUCJ_comparator='ALL3_N17_ORIGINAL_COMPRESSED_L1_REPLICATES_SAME_READOUT_R128_D16384',
        preliminary_independent_draws_not_confirmatory_test=True,
        no_FCI_energy_or_reference_policy_input=True, no_source_specific_tuning=True,
        no_more_solver_or_cap_rescues_for_Fe_after_N21=True,
        source_initialization_costs_retained_not_free=True,
        reference_integrals_not_rotated=True, new_QPU_shots=0, GPU_allocations=0,
        maximum_new_exact_model_equivalent_samples=1200000,
        qualification_SHA256=sha256(qualified.read_bytes()),
        code_SHA256=sha256(Path(__file__).read_bytes()),
        helper_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ('slater_prefix_preflight.py','slater_prefix_compression.py')},
        packages={name:importlib.metadata.version(name) for name in ('ffsim','qiskit','numpy','scipy')},
        goal_complete=False))
    print('N22_FROZEN_ALL4_CASES12_DRAWS_1p2M_MAX_EXACT_GAUSSIAN_SAMPLES', flush=True)


def run():
    spec = load(LOCAL/'protocol.json')
    if (LOCAL/'started.json').exists():
        raise RuntimeError('Existing N22 execution intent: reconcile actual process; no duplicate')
    if sha256(Path(__file__).read_bytes()) != spec['code_SHA256'] or any(sha256((ROOT/'frontierloop'/name).read_bytes()) != value for name,value in spec['helper_SHA256'].items()):
        raise RuntimeError('Frozen numerical or compiler code changed')
    if {name:importlib.metadata.version(name) for name in spec['packages']} != spec['packages']:
        raise RuntimeError('Frozen runtime changed')
    capacity()
    save(LOCAL/'started.json', dict(protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()), pid=__import__('os').getpid()))
    cpu, tick = process_time(), perf_counter()
    sources, draws = [], []
    generated_samples = 0
    for source in spec['sources']:
        capacity()
        folder=LOCAL/source['case']; folder.mkdir()
        cc, tt = process_time(), perf_counter()
        active_replicate, active_counter, active_generated = None, Counter(), 0
        try:
            for key,hashkey in [('operator','operator_SHA256'),('construction','construction_SHA256'),('source_result','source_result_SHA256')]:
                if sha256(Path(source[key]).read_bytes()) != source[hashkey]:
                    raise RuntimeError('Registered original source changed')
            n, nelec = source['norb'], tuple(source['nelec'])
            with np.load(source['operator'],allow_pickle=False) as z:
                op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
            prefix,a,b,occupied,coordinates = extract(op,n,nelec)
            params=arrays(rotation_a=a,rotation_b=b,occupied_a=np.array(occupied[0]),occupied_b=np.array(occupied[1]))
            write(folder/'gaussian-input.npz', params)
            outputs=[]
            for level in spec['compiler_levels']:
                prep,sampling,resources=compile_views(prefix,level)
                evidence={}
                for role,qc in [('preparation',prep),('sampling',sampling)]:
                    stream=BytesIO();qpy.dump(qc,stream);blob=stream.getvalue();path=folder/f'opt{level}-{role}.qpy';write(path,blob)
                    restored=qpy.load(BytesIO(blob))[0]
                    if restored.count_ops()!=qc.count_ops() or restored.depth()!=qc.depth():
                        raise RuntimeError('Actual circuit archive roundtrip changed resources')
                    evidence[role]=dict(path=str(path),SHA256=sha256(blob))
                outputs.append(dict(resources=resources,**evidence))
            build_cpu,build_wall = process_time()-cc,perf_counter()-tt
            state=dict(case=source['case'],problem_source=source,status='SOURCE_CONSTRUCTED',norb=n,nelec=list(nelec),
                gaussian_input_SHA256=sha256(params),compiler_outputs=outputs,declared_shared_prefix_coordinates=coordinates,
                global_independent_rank=None,optimized_variables_this_study=0,source_initialization_cost_not_included=True,
                construction_compile_CPU_seconds=build_cpu,construction_compile_wall_seconds=build_wall)
            save(folder/'source.json',state);sources.append(state)
            for replicate in spec['replicates']:
                active_replicate, active_counter, active_generated = replicate, Counter(), 0
                if perf_counter()-tick > spec['global_body_wall_seconds']:
                    raise TimeoutError('N22 global sampling cap')
                seed=int(digest([source['case'],source['operator_SHA256'],replicate,'N22_GAUSSIAN'])[:16],16)%(1<<63)
                rng=np.random.default_rng(seed);counter=active_counter;dc,dw=process_time(),perf_counter()
                for _ in range(spec['shots_per_draw']//spec['sampling_chunk']):
                    if perf_counter()-tt > spec['max_source_body_wall_seconds']:
                        raise TimeoutError('N22 source body cap')
                    raw=ffsim.sample_slater(n,occupied,(a,b),shots=spec['sampling_chunk'],bitstring_type=ffsim.BitstringType.INT,seed=rng)
                    generated_samples += len(raw); active_generated += len(raw)
                    ids,counts=checked_counts(raw,n,nelec,spec['sampling_chunk'])
                    counter.update({int(x):int(c) for x,c in zip(ids,counts)})
                ids=np.array(sorted(counter),dtype=np.uint64);counts=np.array([counter[int(x)] for x in ids],dtype=np.int64)
                checked_counts(ids.repeat(counts),n,nelec,spec['shots_per_draw'])
                if any(probability(ids[j],n,a,b,occupied)<=0 for j in np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=int)):
                    raise RuntimeError('Observed determinant has zero analytic Gaussian probability')
                payload=arrays(blocked_ids=ids,counts=counts);path=folder/f'draw-{replicate}.npz';write(path,payload)
                record=dict(case=source['case'],replicate=replicate,seed=seed,payload=str(path),payload_SHA256=sha256(payload),
                    source_SHA256=sha256((folder/'source.json').read_bytes()),actual_samples=int(counts.sum()),status='COMPLETED',
                    sampling_CPU_seconds=process_time()-dc,sampling_wall_seconds=perf_counter()-dw,
                    new_QPU_shots=0,exact_classical_Gaussian_draws=True,not_quantum_advantage=True)
                record['draw_id']=digest([source['case'],source['operator_SHA256'],seed,spec['shots_per_draw'],'GAUSSIAN_PREFIX_V1'])
                save(folder/f'draw-{replicate}.json',record);draws.append(record)
                save(LOCAL/'progress.json',dict(sources=sources,draws=draws))
                print('N22_DRAW',source['case'],replicate,len(ids),record['sampling_wall_seconds'],flush=True)
        except Exception as exc:
            active_completed = any(d['case']==source['case'] and d['replicate']==active_replicate for d in draws)
            partial_file = None
            if active_counter and not active_completed:
                partial_file = folder/'partial-counts-preserved.npz'
                pids=np.array(sorted(active_counter),dtype=np.uint64)
                write(partial_file,arrays(blocked_ids=pids,counts=np.array([active_counter[int(x)] for x in pids],dtype=np.int64)))
            save(folder/'failure-preserved.json',dict(error=repr(exc),traceback=__import__('traceback').format_exc(),CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tt,
                active_replicate=active_replicate,partial_generated_samples=0 if active_completed else active_generated,
                partial_file=str(partial_file) if partial_file else None,
                sampler_call_exception_may_have_unreturned_work=True))
            for replicate in spec['replicates']:
                if not any(d['case']==source['case'] and d['replicate']==replicate for d in draws):
                    draws.append(dict(case=source['case'],replicate=replicate,status='FAILED_OR_NOT_RUN_PRESERVED',error=repr(exc),actual_samples=None))
    if len(draws)!=12:
        raise RuntimeError('All twelve draw states required')
    result=dict(status='N22_ALL12_GAUSSIAN_DRAW_STATES_TERMINAL',sources=sources,draws=draws,
        actual_new_exact_model_equivalent_samples=generated_samples,
        completed_draw_samples=sum(d['actual_samples'] for d in draws if d['status']=='COMPLETED'),
        new_QPU_shots=0,GPU_allocations=0,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-tick,
        code_SHA256=spec['code_SHA256'],protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),goal_complete=False)
    save(LOCAL/'terminal.json',result);print(result['status'],result['CPU_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();globals()[a.action]()
