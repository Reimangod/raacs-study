"""N73 selected activity-source exact-synthesis resources, no energy outcomes."""
from pathlib import Path
from time import process_time,perf_counter
from io import BytesIO
import os,sys,signal,gzip
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256,space
from .state_activity_preflight import LOCAL as N72,choose
from .exact_group_synthesis import LOCAL as N68,rustiq_group,FIELDS
from .pair_group_preflight import pair_preparation,angles_from_t2,metrics
from .slater_prefix_preflight import compile_views
from paperstudy.lucj import amplitudes_from_saved

LOCAL=ROOT/'runs/N73-ACTIVITY-RESOURCE-v1'

def choose_resource(points):return min(points,key=lambda r:tuple(r['resources'][k] for k in FIELDS))

def prepare():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; no replace')
    space();p=load(N72/'protocol.json');t=load(N72/'terminal.json');old=load(N68/'protocol.json');bindings=dict(p['bindings'])
    assert len(t['rows'])==120 and len(t['selected'])==12 and all(c['proceed_to_resource_qualification'] for c in t['cases'])
    for r in t['selected']:
        assert r['selection']==choose([x for x in t['rows'] if x['case']==r['case'] and x['time']==r['time']])
        f=Path(r['state_file']);assert sha256(f.read_bytes())==r['state_SHA256'];bindings[str(f)]=r['state_SHA256']
    import xml.etree.ElementTree as ET
    tests=ROOT/'qualification-N73-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors'))
    control=ROOT/'analysis/N67-pair-group-resource-summary.json'
    for f in [Path(__file__),N72/'protocol.json',N72/'terminal.json',N68/'protocol.json',N68/'qualification.json',control,
        ROOT/'frontierloop/exact_group_synthesis.py',ROOT/'analysis/N73_ACTIVITY_RESOURCE_PLAN_JA.md',tests]:bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+4*2**20>2**30:raise OSError('Research quota')
    save(LOCAL/'protocol.json',dict(version='N73_ACTIVITY_SOURCE_RESOURCE_ONLY',bindings=bindings,sources=p['sources'],selected=t['selected'],
        controls=load(control)['controls'],time_shots=p['time_shots'],compiler_levels=old['compiler_levels'],rustiq_optimize_count=old['rustiq_optimize_count'],
        CPU_cap=30,wall_cap=90,new_bytes=4*2**20,root_bytes=2**30,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,goal_complete=False))
    print('N73_FROZEN12_RESOURCE_ONLY',flush=True)

def run():
    p=load(LOCAL/'protocol.json');space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing execution intent; do not duplicate')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Input drift '+f
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[];cases=[];initial=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file());base=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    def timeout(signum,frame):raise TimeoutError('N73 CPU/wall cap')
    a=signal.signal(signal.SIGALRM,timeout);b=signal.signal(signal.SIGPROF,timeout)
    signal.setitimer(signal.ITIMER_REAL,p['wall_cap']);signal.setitimer(signal.ITIMER_PROF,p['CPU_cap'])
    def bound():
        space();size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if size>p['new_bytes'] or base+size-initial>p['root_bytes']:raise OSError('N73 storage cap')
    try:
        for s in p['sources']:
            bound();meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];_,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
            pair=pair_preparation(n,k,angles_from_t2(n,k,t2));case_rows=[]
            for r in [r for r in p['selected'] if r['case']==s['case']]:
                points=[]
                for mode in p['rustiq_optimize_count']:
                    group=rustiq_group(r['selection']['group'],mode)
                    for level in p['compiler_levels']:
                        bound();prep,sampling,_=compile_views(pair.compose(group),level);files={}
                        for role,q in [('preparation',prep),('sampling',sampling)]:
                            stream=BytesIO();qpy.dump(q,stream);blob=gzip.compress(stream.getvalue(),mtime=0)
                            f=LOCAL/'circuits'/f'{s["case"]}-T{int(r["time"])}-C{int(mode)}-L{level}-{role}.qpy.gz';write(f,blob)
                            restored=qpy.load(BytesIO(gzip.decompress(blob)))[0];assert metrics(restored)==metrics(q)
                            files[role]=dict(file=str(f),SHA256=sha256(blob))
                        points.append(dict(compiler='RUSTIQ_COUNT' if mode else 'RUSTIQ_DEPTH',level=level,resources=metrics(sampling),preparation_resources=metrics(prep),files=files))
                row=dict(case=s['case'],time=r['time'],points=points,selected=choose_resource(points),source_semantic_id=r['selection']['semantic_id'],independent_parameter_rank=None)
                rows.append(row);case_rows.append(row)
            mean={key:sum(shots*next(r['selected']['resources'][key] for r in case_rows if r['time']==time) for time,shots in p['time_shots'])/1000 for key in FIELDS}
            maximum={key:max(r['selected']['resources'][key] for r in case_rows) for key in FIELDS}
            controls=[pt['resources'] for c in p['controls'] if c['case']==s['case'] and c['status']=='AVAILABLE' for pt in c['recounted_points']]
            envelope={key:min(c[key] for c in controls) for key in FIELDS}
            cases.append(dict(case=s['case'],weighted_mean=mean,maximum=maximum,LUCJ_componentwise_resource_envelope=envelope,
                proceed_to_finite_screen=all(mean[key]<=envelope[key] for key in FIELDS),matched_LUCJ_compiler_opportunity=False))
            save(LOCAL/'progress.json',dict(rows=rows,cases=cases,CPU_seconds=process_time()-cpu));print('N73_CASE',s['case'],mean,flush=True)
        save(LOCAL/'terminal.json',dict(status='ALL12_ACTIVITY_RESOURCE_POINTS_RECORDED',rows=rows,cases=cases,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,independent_parameter_rank=None,matched_LUCJ_compiler_opportunity=False,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'failed.json',dict(rows=rows,cases=cases,error=repr(exc),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGALRM,a);signal.signal(signal.SIGPROF,b)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
