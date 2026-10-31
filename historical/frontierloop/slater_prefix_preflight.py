"""N15 all-source precision/compiled-resource preflight, no sampling or energy."""
import argparse,importlib.metadata
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import ffsim
import numpy as np
from qiskit import qpy,transpile
from .paths import ROOT,OLD
from .io import load,save,write,sha256,space,digest
from .slater_prefix_compression import parse_prefix
from .slater_prefix_rounding import trajectory,choose,logical_candidate,lower_virtual_swaps
from paperstudy.lucj import make_circuit,BASIS

LOCAL=ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1'
MODULES=('slater_prefix_compression.py','slater_prefix_rounding.py','slater_prefix_preflight.py')


def construction_record(result):
    if isinstance(result.get('construction'),dict):return result['construction']
    if isinstance(result.get('detail'),dict) and isinstance(result['detail'].get('construction'),dict):return result['detail']['construction']
    raise ValueError('Unrecognized historical construction schema')


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N15; do not overwrite')
    space();r2file=ROOT/'runs/R2-COMMON-CI-v1/freeze.json';sources=[x for x in load(r2file)['sources'] if 'LUCJ_COMPRESSED' in x['method_id']]
    if len(sources)!=12 or sum(s['status']=='AVAILABLE' for s in sources)!=10:raise RuntimeError('All12 planned/10available2missing required')
    rows=[]
    for s in sources:
        row=dict(case=s['case'],method=s['method_id'],status=s['status'],problem_id=s['expected_problem_id'])
        if s['status']=='AVAILABLE':
            first=s['method_id'].endswith('L1');name=s['method_id'].removeprefix('C5_')+('-L1' if first else '')
            folder=OLD/('prepared' if first else 'sensitivity-prepared')/s['case']/name
            if sha256(Path(s['source_result']).read_bytes())!=s['source_result_SHA256']:raise RuntimeError('Frozen source record changed')
            record=construction_record(load(s['source_result']));construction=load(folder/'construction.json');operator=folder/'operator.npz'
            if construction!=record or sha256(operator.read_bytes())!=record['operator_sha256'] or construction['problem_id']!=s['expected_problem_id']:raise RuntimeError('Source operator does not match source counts construction')
            physical=OLD/'data/physical'/s['case']/'problem.json';m=load(physical)
            row.update(operator=str(operator),operator_SHA256=sha256(operator.read_bytes()),construction=str(folder/'construction.json'),construction_SHA256=sha256((folder/'construction.json').read_bytes()),
                source_result_SHA256=s['source_result_SHA256'],source_result=s['source_result'],norb=m['norb'],nelec=[m['nalpha'],m['nbeta']],physical_SHA256=sha256(physical.read_bytes()))
        rows.append(row)
    LOCAL.mkdir();spec=dict(version='N15_PREFIX_ROUNDING_RESOURCE_PREFLIGHT_V1',sources=rows,
        policies=['STATE_OVERLAP_ROUND','ANGLE_ONLY_ROUND'],infidelity_tiers=[1e-6,1e-4,.01],
        max_polynomial_infidelity=.01,tiers_are_design_decisions_not_CI_precision_thresholds=True,
        compiler_levels=[1,3],basis=BASIS,seed_transpiler=8137,approximation_degree=1.,
        compiler_control='EXACT_SOURCE_OPT1_OPT3_CNOT_FIRST_PORTFOLIO',
        adoption='ALL3_RESOURCES_NONINCREASE_VS_EXACT_PORTFOLIO_OR_EXACT_FALLBACK',
        code_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in MODULES},
        resource_adapter_SHA256=sha256((OLD/'paperstudy/lucj.py').read_bytes()),R2_source_freeze_SHA256=sha256(r2file.read_bytes()),
        qualification_SHA256=sha256((ROOT/'qualification-slater-prefix-all-sources-v1-tests.xml').read_bytes()),
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','qiskit','ffsim']},
        initial_delete_probe='EXPLORATORY_L1_PROBE_NO_ENERGY_NO_SHOTS_NOT_CONFIRMATORY',
        original_source_suffix_unchanged_logically=True,virtual_swaps_decoded=True,fermionic_CZ_phase_preserved=True,
        classical_overlap_formula='PRODUCT_SPIN_ABS_DET_QTARGET_DAGGER_QCANDIDATE_SQUARED',
        no_FCI_or_energy_policy_inputs=True,new_model_shots=0,new_QPU_shots=0,new_CI_solves=0,GPU_allocations=0,
        independent_parameter_rank_certified=False,novelty_claim=False,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N15_FROZEN12_SOURCES10AVAILABLE2MISSING_72_POLICY_TIER_PLANS',flush=True)


def compile_views(circuit,level):
    preparation=transpile(circuit,basis_gates=BASIS,optimization_level=level,seed_transpiler=8137,approximation_degree=1.)
    if preparation.layout is not None and preparation.layout.final_index_layout()!=list(range(circuit.num_qubits)):raise RuntimeError('Unregistered compiler output permutation')
    blocked=set();keep=[];diagonal={'rz','z','s','sdg','t','tdg','p','cp','cz','rzz','id'}
    for item in reversed(preparation.data):
        wires={preparation.find_bit(q).index for q in item.qubits}
        if item.operation.name in diagonal and not wires&blocked:continue
        keep.append(item);blocked|=wires
    sampling=preparation.copy_empty_like()
    for item in reversed(keep):sampling.append(item.operation,[preparation.find_bit(q).index for q in item.qubits])
    sampling=transpile(sampling,basis_gates=BASIS,optimization_level=level,seed_transpiler=8137,approximation_degree=1.)
    if sampling.layout is not None and sampling.layout.final_index_layout()!=list(range(circuit.num_qubits)):raise RuntimeError('Sampling compiler output permutation')
    resource=dict(CNOT=int(sampling.count_ops().get('cx',0)),CNOT_Depth=int(sampling.depth(filter_function=lambda x:x.operation.name=='cx')),Total_Depth=int(sampling.depth()),
        numeric_primitive_coefficient_slots_not_independent_rank=sum(len(i.operation.params) for i in sampling),optimization_level=level)
    return preparation,sampling,resource


def run():
    spec=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; reconcile without rerun')
    for name,value in spec['code_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=value:raise RuntimeError('Frozen implementation changed')
    if sha256((OLD/'paperstudy/lucj.py').read_bytes())!=spec['resource_adapter_SHA256'] or {n:importlib.metadata.version(n) for n in spec['packages']}!=spec['packages']:raise RuntimeError('Runtime/resource semantics changed')
    save(LOCAL/'started.json',dict(protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_shots=0))
    start,cpu=perf_counter(),process_time();rows=[];sources=[]
    def store(qc,path):
        stream=BytesIO();qpy.dump(qc,stream);blob=stream.getvalue();write(path,blob)
        restored=qpy.load(BytesIO(blob))[0]
        if qc.count_ops()!=restored.count_ops() or qc.depth()!=restored.depth():raise RuntimeError('QPY resource roundtrip mismatch')
        return dict(path=str(path.relative_to(LOCAL)),SHA256=sha256(blob))
    def compiled(qc,folder):
        outputs=[]
        for level in spec['compiler_levels']:
            prep,samp,res=compile_views(qc,level)
            outputs.append(dict(resources=res,preparation=store(prep,folder/f'opt{level}-preparation.qpy'),sampling=store(samp,folder/f'opt{level}-sampling.qpy')))
        return outputs
    key=lambda o:tuple(o['resources'][k] for k in ['CNOT','CNOT_Depth','Total_Depth'])
    for s in spec['sources']:
        case=s['case'];method=s['method'];folder=LOCAL/case/method;folder.mkdir(parents=True)
        if s['status']!='AVAILABLE':
            for policy in spec['policies']:
                for tier in spec['infidelity_tiers']:rows.append(dict(case=case,method=method,policy=policy,tier=tier,status='MISSING_SAME_PROBLEM_SOURCE'))
            continue
        for name,keyname in [('operator','operator_SHA256'),('construction','construction_SHA256'),('source_result','source_result_SHA256')]:
            if sha256(Path(s[name]).read_bytes())!=s[keyname]:raise RuntimeError('Frozen source changed')
        tick,cc=perf_counter(),process_time()
        with np.load(s['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        source=ffsim.qiskit.PRE_INIT.run(make_circuit(op,s['norb'],tuple(s['nelec'])))
        if source.data[0].operation.name!='slater_jw':raise RuntimeError('Source has no qualified prefix')
        prefix=parse_prefix(source.data[0].operation.definition,s['norb']);baseline_outputs=compiled(source,folder/'EXACT');baseline=min(baseline_outputs,key=key)
        store(source,folder/'source-logical.qpy');cache={};traces={}
        for policy in spec['policies']:
            trace=trajectory(prefix,policy,spec['max_polynomial_infidelity']);traces[policy]=trace
            save(folder/(policy+'-trace.json'),trace)
            for tier in spec['infidelity_tiers']:
                point=choose(trace,tier);actions=dict(point['actions']);identity=digest(point['actions'])
                new_compile=identity not in cache
                if new_compile:
                    logical=logical_candidate(source,prefix,actions);actual,mapping=lower_virtual_swaps(logical,len(prefix.circuit),actions)
                    outdir=folder/identity;store(logical,outdir/'logical.qpy');store(actual,outdir/'actual-virtual-swap.qpy')
                    outputs=baseline_outputs if not actions else compiled(actual,outdir)
                    eligible=[o for o in outputs if all(o['resources'][k]<=baseline['resources'][k] for k in ['CNOT','CNOT_Depth','Total_Depth'])]
                    accepted=bool(actions and eligible and min(eligible,key=key)['resources']['CNOT']<baseline['resources']['CNOT'])
                    selected=min(eligible,key=key) if accepted else baseline
                    cache[identity]=dict(decoder_logical_to_physical=mapping,candidate_compiler_outputs=outputs,selected=selected,adopted=accepted,
                        point_identity=identity,original_fallback=not accepted)
                evidence=cache[identity]
                rows.append(dict(case=case,method=method,policy=policy,tier=tier,status='RESOURCE_PREFLIGHT_ONLY',
                    actions=point['actions'],certified_original_input_many_body_infidelity=point['infidelity'],
                    predicted_measurement_TV_upper_bound=float(np.sqrt(point['infidelity'])),
                    nominal_saved_CNOT=point['nominal_saved_CNOT'],exact_control=baseline,
                    adopted_output_many_body_infidelity=point['infidelity'] if evidence['adopted'] else 0.,
                    **evidence,actual_new_candidate_compile=new_compile,new_model_shots=0,new_CI_solves=0))
                save(LOCAL/'progress.json',dict(rows=rows,source_states=sources,terminal_rows=len(rows),planned_rows=72))
        sources.append(dict(case=case,method=method,status='RESOURCE_PREFLIGHT_TERMINAL',source_operator_SHA256=s['operator_SHA256'],
            source_prefix_pairs=len(prefix.pairs),determinant_overlap_evaluations={p:traces[p]['determinant_overlap_evaluations'] for p in traces},
            CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tick))
        print('N15',case,method,'completed',len(cache),'distinct polynomial candidates',flush=True)
    if len(rows)!=72:raise RuntimeError('All72 policy-tier states required')
    result=dict(status='N15_ALL72_RESOURCE_PREFLIGHT_ROWS_TERMINAL_NO_ENERGY_OR_SHOTS',rows=rows,sources=sources,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-start,new_model_shots=0,new_QPU_shots=0,new_CI_solves=0,
        molecular_energy_or_CI_quality_verified=False,independent_parameter_rank_certified=False,full_paper_superiority=False,goal_complete=False,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()))
    save(LOCAL/'result.json',result);print(result['status'],result['CPU_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args();globals()[a.action]()
