"""N73B additive resource-constrained compiler selection. Old N73 immutable."""
from pathlib import Path
from io import BytesIO
from time import process_time,perf_counter
import gzip
from qiskit import qpy
from .paths import ROOT
from .io import load,save,sha256
from .activity_resource import LOCAL as N73,choose_resource
from .exact_group_synthesis import FIELDS
from .pair_group_preflight import metrics

LOCAL=ROOT/'runs/N73B-ACTIVITY-RESOURCE-PORTFOLIO-v1'

def choose_admissible(points,envelope):
    eligible=[p for p in points if all(p['resources'][key]<=envelope[key] for key in FIELDS)]
    return choose_resource(eligible) if eligible else None

def run():
    if (LOCAL/'terminal.json').exists():raise RuntimeError('Existing additive portfolio; no replace')
    cpu,wall=process_time(),perf_counter();p=load(N73/'protocol.json');old=load(N73/'terminal.json');rows=[];cases=[];bindings=dict(p['bindings']);recounts=0
    for f in [N73/'protocol.json',N73/'terminal.json',Path(__file__),ROOT/'analysis/N73B_COMPILER_SELECTION_CORRECTION_JA.md']:
        bindings[str(f)]=sha256(f.read_bytes())
    for f,h in bindings.items():assert sha256(Path(f).read_bytes())==h
    for r in old['rows']:
        envelope=next(c for c in old['cases'] if c['case']==r['case'])['LUCJ_componentwise_resource_envelope']
        for point in r['points']:
            for role,key in [('preparation','preparation_resources'),('sampling','resources')]:
                f=Path(point['files'][role]['file']);blob=f.read_bytes();assert sha256(blob)==point['files'][role]['SHA256'];bindings[str(f)]=sha256(blob)
                assert metrics(qpy.load(BytesIO(gzip.decompress(blob)))[0])==point[key];recounts+=1
        selected=choose_admissible(r['points'],envelope)
        rows.append(dict(r,selected=selected,old_CNOT_first_selection=r['selected'],changed=selected!=r['selected'],per_point_resource_envelope=envelope))
    for c in old['cases']:
        rr=[r for r in rows if r['case']==c['case']];available=all(r['selected'] is not None for r in rr)
        mean={key:sum(shots*next(r['selected']['resources'][key] for r in rr if r['time']==time) for time,shots in p['time_shots'])/1000 for key in FIELDS} if available else None
        maximum={key:max(r['selected']['resources'][key] for r in rr) for key in FIELDS} if available else None
        cases.append(dict(case=c['case'],weighted_mean=mean,maximum=maximum,LUCJ_componentwise_resource_envelope=c['LUCJ_componentwise_resource_envelope'],
            proceed_to_finite_screen=available and all(mean[key]<=c['LUCJ_componentwise_resource_envelope'][key] for key in FIELDS),
            old_gate_passed=c['proceed_to_finite_screen'],matched_LUCJ_compiler_opportunity=False))
    save(LOCAL/'terminal.json',dict(status='ADDITIVE_RESOURCE_CONSTRAINED_PORTFOLIO_ALL12_RECORDED',bindings=bindings,rows=rows,cases=cases,QPY_recounts=recounts,
        changed_compiler_points=sum(r['changed'] for r in rows),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        new_QPY_files=0,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,scientific_source_changed=False,gate_relaxed=False,goal_complete=False))
    print('N73B_RECOUNTED',recounts,'CHANGED',sum(r['changed'] for r in rows),flush=True)

if __name__=='__main__':run()
