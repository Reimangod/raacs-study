"""Additive correction: opaque gates cannot be zero-CNOT resource points."""
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter
import gzip
from qiskit import qpy
from .io import load,save,sha256
from .fixed_confirmation_execution import LOCAL,bound,capacity
from .fixed_confirmation_resources import metrics,KEYS
from .slater_prefix_preflight import compile_views
from paperstudy.lucj import BASIS

def primitive_basis(q):
    unknown=sorted(set(q.count_ops())-set(BASIS))
    return dict(eligible=not unknown,unsupported_operations=unknown,basis=BASIS)

def open_qpy(file,expected):
    blob=Path(file).read_bytes()
    if sha256(blob)!=expected:raise ValueError('Source QPY drift')
    return qpy.load(BytesIO(gzip.decompress(blob) if str(file).endswith('.gz') else blob))[0]

def run():
    science,b=bound();capacity();target=LOCAL/'compiler-qualified-point-selection-v2.json'
    if target.exists():raise RuntimeError('Existing additive qualification; no overwrite')
    oldfile=LOCAL/'compiler-terminal.json';old=load(oldfile);binding=load(LOCAL/'compiler-binding.json')
    for file,h in binding['bindings'].items():
        if sha256(Path(file).read_bytes())!=h:raise RuntimeError('Original compiler inputs drift '+file)
    cpu,wall=process_time(),perf_counter();rows=[];rejected=[];qualified_common=0
    remaining_CPU=science['compiler']['additional_CPU_cap']-old['CPU_seconds']
    remaining_wall=science['compiler']['additional_wall_cap']-old['wall_seconds']
    for r in old['rows']:
        points=[]
        for point in r['points']:
            if process_time()-cpu>remaining_CPU or perf_counter()-wall>remaining_wall:raise TimeoutError('Remaining ORIGINAL compiler budget; preserve partial diagnostic, no extension')
            if point['kind'].startswith('COMMON_QISKIT_OPT'):
                q=open_qpy(point['source_file'],point['source_SHA256']);level=point['resources']['optimization_level']
                prep,sampling,res=compile_views(q,level);qualification=primitive_basis(sampling)
                if res!=point['resources'] or not qualification['eligible']:raise ValueError('Common compiler numerical/basis mismatch')
                qualified_common+=1
            else:
                q=open_qpy(point['file'],point['SHA256']);qualification=primitive_basis(q)
                if qualification['eligible'] and metrics(q)!=point['resources']:raise ValueError('Native primitive recount mismatch')
            p=dict(point,primitive_basis_qualification=qualification)
            if qualification['eligible']:points.append(p)
            else:rejected.append(dict(case=r['case'],method=r['method'],member_id=r['member_id'],point=p,reason='OPAQUE_OR_OUT_OF_REGISTERED_BASIS_NOT_ZERO_CNOT'))
        assert points
        rows.append(dict(r,points=points,selected=min(points,key=lambda p:tuple(p['resources'][k] for k in KEYS))))
    aggregate=[];groups={}
    for r in rows:groups.setdefault((r['case'],r['method']),[]).append(r)
    for (case,method),rr in groups.items():
        total=sum(r['shots_per_replicate'] for r in rr)
        aggregate.append(dict(case=case,method=method,mean={k:sum(r['shots_per_replicate']*r['selected']['resources'][k] for r in rr)/total for k in KEYS},
            maximum={k:max(r['selected']['resources'][k] for r in rr) for k in KEYS},
            coefficient_slots_weighted_mean=sum(r['shots_per_replicate']*r['selected']['resources']['numeric_primitive_coefficient_slots_not_independent_rank'] for r in rr)/total,
            independent_parameter_rank=None,VQE_optimized_variables=0,common_opt_attempts_complete=True))
    elapsed=process_time()-cpu;wall_elapsed=perf_counter()-wall
    assert elapsed+old['CPU_seconds']<=science['compiler']['additional_CPU_cap'] and wall_elapsed+old['wall_seconds']<=science['compiler']['additional_wall_cap']
    save(target,dict(status='REGISTERED_PRIMITIVE_BASIS_RESOURCE_POINTS_QUALIFIED_ADDITIVE_V2',rows=rows,aggregate=aggregate,rejected_points=rejected,
        original_compiler_terminal_SHA256=sha256(oldfile.read_bytes()),qualifier_SHA256=sha256(Path(__file__).read_bytes()),
        repeated_same_existing_opt_checks_not_new_compiler_search=qualified_common,qualification_CPU_seconds=elapsed,qualification_wall_seconds=wall_elapsed,
        CPU_seconds=old['CPU_seconds']+elapsed,wall_seconds=old['wall_seconds']+wall_elapsed,
        sources_readouts_and_CI_unchanged=True,new_model_samples=0,new_CI_roots=0,goal_complete=False))
    print('QUALIFIED_PRIMITIVE_RESOURCE_POINTS',len(rows),'REJECTED',len(rejected),'TOTAL_COMPILER_CPU',old['CPU_seconds']+elapsed,flush=True)

if __name__=='__main__':run()
