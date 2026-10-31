"""Finite common compiler opportunity; existing exact portfolios retained."""
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter
import gzip,os,sys,traceback
from qiskit import qpy
from .io import load,save,sha256,digest
from .paths import ROOT
from .fixed_confirmation_execution import LOCAL,bound,capacity,members,member_id,allocations
from .slater_prefix_preflight import compile_views

KEYS=['CNOT','CNOT_Depth','Total_Depth']

def metrics(q):
    return dict(CNOT=int(q.count_ops().get('cx',0)),CNOT_Depth=int(q.depth(filter_function=lambda x:x.operation.name=='cx')),Total_Depth=int(q.depth()),numeric_primitive_coefficient_slots_not_independent_rank=sum(len(i.operation.params) for i in q))

def prepare():
    science,b=bound();out=LOCAL/'compiler-binding.json'
    if out.exists():raise RuntimeError('Already compiler bound')
    rows=[];bindings={}
    for c in science['cells']:
        if c['status']=='MISSING_SAME_PROBLEM_SOURCE' or c['method']=='CLASSICAL_HF_HCI':continue
        method=c['method'];row=c['source_input']
        oldmembers=None
        if method.startswith('SQDRIFT'):
            base=ROOT/('runs/N65-ENSEMBLE-BREADTH30-v1' if method=='SQDRIFT_N5_FIXED30' else 'runs/N66-FULL20-BREADTH30-v1')
            arm='FULL_N5' if method=='SQDRIFT_N5_FIXED30' else 'FULL_N20'
            draw=next(d for d in load(base/'protocol.json')['draws'] if d['case']==c['case'] and d['arm']==arm)
            folder=base/'draws'/draw['draw_id'];oldmembers=load(folder/'result.json')['members']
        for m,shots in zip(members(c),allocations(c)):
            mid=member_id(method,m);points=[]
            if method=='ACTIVE_GROUP_FIXED_LIBRARY':points=[dict(kind='EXISTING_REGISTERED_NATIVE_EXACT_SELECTED',file=m['QPY']['file'],SHA256=m['QPY']['SHA256'])]
            elif method=='B4_FIXED_SELECTED_ENSEMBLE':points=[dict(kind='EXISTING_AUTHOR_QE_SOURCE',file=m['QPY_file'],SHA256=m['QPY_SHA256'])]
            elif method.startswith('C5_C_LUCJ'):points=[dict(kind='EXISTING_QUALIFIED_ORIGINAL_PRIMITIVE',file=m['original_QPY'],SHA256=m['original_QPY_SHA256'])]
            else:
                old=next(r for r in oldmembers if r['member_id']==mid)
                for kind,h in old['qpy_gz_SHA256'].items():points.append(dict(kind='EXISTING_NATIVE_PORTFOLIO_'+kind,file=str(folder/(mid+'-'+kind+'.qpy.gz')),SHA256=h))
            for p in points:
                assert sha256(Path(p['file']).read_bytes())==p['SHA256'];bindings[p['file']]=p['SHA256']
            rows.append(dict(case=c['case'],method=method,member_id=mid,shots_per_replicate=shots,points=points,
                independent_parameter_rank=None,VQE_optimized_variables=0,registered_input_coordinate_slots=len(m['actions']) if method=='B4_FIXED_SELECTED_ENSEMBLE' else None))
    for f in [Path(__file__),ROOT/'frontierloop/slater_prefix_preflight.py']:bindings[str(f)]=sha256(f.read_bytes())
    save(out,dict(rows=rows,bindings=bindings,common_opt_levels=science['compiler']['common_opt_levels'],CPU_cap=30,wall_cap=90,
        resource_choice='CNOT_THEN_CX_DEPTH_THEN_TOTAL_DEPTH_FROM_SINGLE_POINT_NO_COMPONENTWISE_MIX',compiler_comparison_scope='COMMON_QISKIT_OPT1_OPT3_PLUS_EXISTING_SOURCE_NATIVE_PORTFOLIOS_NOT_EQUAL_NEW_RUSTIQ_FOR_ALL',new_source_search_authorized=False))
    print('COMPILER_BOUND',len(rows),'SOURCES',flush=True)

def run():
    bound();p=load(LOCAL/'compiler-binding.json')
    if (LOCAL/'compiler-started.json').exists():raise RuntimeError('Existing compiler intent; no retry')
    for f,h in p['bindings'].items():
        if sha256(Path(f).read_bytes())!=h:raise RuntimeError('Compiler source drift '+f)
    cpu,wall=process_time(),perf_counter();save(LOCAL/'compiler-started.json',dict(pid=os.getpid(),binding_SHA256=sha256((LOCAL/'compiler-binding.json').read_bytes())))
    rows=[];work=[]
    # Recount every old source first; no energy or count labels in selection.
    for r in p['rows']:
        points=[]
        for v in r['points']:
            blob=Path(v['file']).read_bytes();q=qpy.load(BytesIO(gzip.decompress(blob) if v['file'].endswith('.gz') else blob))[0]
            points.append(dict(v,resources=metrics(q),status='EXISTING_QPY_RECOUNTED'))
        rows.append(dict(r,points=points,common_compiler_attempts=[]))
    # Identical opportunity, fixed member-round-robin across methods/cases.
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r['case'],r['method']),[]).append(i)
    order=[v[k] for k in range(max(map(len,groups.values()))) for v in groups.values() if k<len(v)]
    for index in order:
        r=rows[index];first=r['points'][0];file=Path(first['file']);blob=file.read_bytes()
        for level in p['common_opt_levels']:
            remaining=min(p['CPU_cap']-(process_time()-cpu),p['wall_cap']-(perf_counter()-wall))
            if remaining<=0:
                r['common_compiler_attempts'].append(dict(level=level,status='NOT_RUN_COMMON_COMPILER_CAP'));continue
            tick=process_time()
            try:
                q=qpy.load(BytesIO(gzip.decompress(blob) if file.suffix=='.gz' else blob))[0]
                prep,samp,res=compile_views(q,level)
                point=dict(kind='COMMON_QISKIT_OPT'+str(level),source_file=str(file),source_SHA256=first['SHA256'],resources=res,status='EXACT_TRANSPILER_MEASUREMENT_EQUIVALENT_NOT_NOISE_CHANNEL_EQUIVALENT')
                r['points'].append(point);attempt=dict(level=level,status='COMPLETED',CPU_seconds=process_time()-tick)
            except Exception as exc:attempt=dict(level=level,status='FAILED_PRESERVED',error=repr(exc),CPU_seconds=process_time()-tick)
            r['common_compiler_attempts'].append(attempt);work.append(attempt)
        r['selected']=min(r['points'],key=lambda v:tuple(v['resources'][k] for k in KEYS))
    for r in rows:
        if 'selected' not in r:r['selected']=min(r['points'],key=lambda v:tuple(v['resources'][k] for k in KEYS))
    aggregate=[]
    for key,indices in groups.items():
        rr=[rows[i] for i in indices];total=sum(r['shots_per_replicate'] for r in rr)
        aggregate.append(dict(case=key[0],method=key[1],mean={k:sum(r['shots_per_replicate']*r['selected']['resources'][k] for r in rr)/total for k in KEYS},
            maximum={k:max(r['selected']['resources'][k] for r in rr) for k in KEYS},coefficient_slots_weighted_mean=sum(r['shots_per_replicate']*r['selected']['resources']['numeric_primitive_coefficient_slots_not_independent_rank'] for r in rr)/total,
            independent_parameter_rank=None,VQE_optimized_variables=0,common_opt_attempts_complete=all(len(r['common_compiler_attempts'])==2 and all(a['status']=='COMPLETED' for a in r['common_compiler_attempts']) for r in rr)))
    capacity();save(LOCAL/'compiler-terminal.json',dict(status='BOUNDED_COMMON_COMPILER_OPPORTUNITY_TERMINAL',rows=rows,aggregate=aggregate,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        stronger_cross_framework_compiler_parity_not_fully_qualified=True,new_source_search_authorized=False,goal_complete=False));print('COMPILER_TERMINAL',len(rows),len(work),flush=True)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
