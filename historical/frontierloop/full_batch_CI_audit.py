"""Read saved N51 roots; reconstruct H/RDM/sector without new eigensolves."""
from pathlib import Path
from time import process_time,perf_counter
import sys
import numpy as np
from pyscf.fci import selected_ci,direct_spin1
from .full_batch_CI_study import LOCAL,verified
from .energy_cache import hole_integrals
from .hole_rdm import electron_rdms
from .portable_penalized_ci import expectation
from .io import load,save,sha256
from .paths import ROOT


def audit_one(p,task,result,folder):
    cpu,wall=process_time(),perf_counter()
    if result['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or result['task_id']!=task['task_id']:raise ValueError('Root protocol/identity differs')
    file=folder/'root.npz'
    source=task['source']
    if sha256(Path(source['payload']).read_bytes())!=source['payload_SHA256'] or sha256(Path(source['source_result']).read_bytes())!=source['source_result_SHA256']:raise ValueError('Original frozen source evidence changed')
    if sha256(file.read_bytes())!=result['root_SHA256']:raise ValueError('Saved root bytes differ')
    with np.load(file,allow_pickle=False) as z:root={k:z[k] for k in z.files}
    strings=np.asarray(task['strings'],np.int64); n=20; ne=(5,5)
    hs=np.sort(((1<<n)-1)^strings)
    if str(root['problem_id'])!=p['problem_id'] or str(root['task_id'])!=task['task_id'] or not np.array_equal(root['electron_strings'],strings) or not np.array_equal(root['solver_strings'],hs) or root['solver_nelec'].tolist()!=list(ne):raise ValueError('Saved physical sector or basis differs')
    if int(root['norb'])!=20 or root['electron_nelec'].tolist()!=[15,15] or not bool(root['particle_hole_used']):raise ValueError('Saved representation label differs')
    if result['method']!=task['method'] or result['row']['common_equation']!=p['equation']:raise ValueError('Scientific method or equation differs')
    v=root['coefficients'].reshape((len(hs),len(hs)))
    if v.size!=task['D'] or abs(float(v.ravel()@v.ravel())-1)>1e-10:raise ValueError('Saved vector normalization or dimension differs')
    with np.load(LOCAL/'integrals.npz',allow_pickle=False) as z:h,g=z['h1'],z['eri']
    hh,gg,core=hole_integrals(h,g,p['core']);solver=selected_ci.SelectedCI();solver.verbose=0
    ci=selected_ci._as_SCIvector(v,(hs,hs));links=selected_ci._all_linkstr_index((hs,hs),n,ne)
    h2=direct_spin1.absorb_h1e(hh,gg,n,ne,.5)
    hv=selected_ci.contract_2e(h2,ci,n,ne,link_index=links)
    sv=solver.contract_ss(ci,n,ne)
    energy=float(np.vdot(v,hv)+core);spin=float(np.vdot(v,sv));obj=energy+.1*spin
    residual=float(np.linalg.norm(hv+.1*sv-(obj-core)*v))
    d1,d2=solver.make_rdm12(ci,n,ne);e1,e2=electron_rdms(d1,d2)
    independent=expectation(h,g,p['core'],e1,e2)
    errors={k:abs(value-result['row'][k]) for k,value in [('energy_Ha',energy),('penalized_energy_Ha',obj),('spin_square',spin)]}
    errors['electron_RDM_energy_Ha']=abs(independent-energy)
    errors['particle_number']=abs(float(np.trace(e1))-30)
    errors['pair_number']=abs(float(np.einsum('pprr->',e2))-870)
    if max(errors.values())>1e-7 or residual>1e-8:raise ValueError('Independent saved-root physical/RDM/residual audit failed')
    return dict(method=task['method'],task_id=task['task_id'],energy_Ha=energy,spin_square=spin,
          residual_norm=residual,errors=errors,R=task['R'],D=task['D'],
          CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
          root_SHA256=result['root_SHA256'])


def run(scope):
    if scope not in ['M4','ALL']:raise ValueError('Explicit scope')
    p=verified();rows=[];coverage=[]
    for i,task in enumerate(p['tasks']):
        if scope=='M4' and i!=0:continue
        folder=LOCAL/('out' if i==0 else 'collected/out')/task['task_id']
        if not (folder/'result.json').exists():
            failure=folder/'process-failure.json'
            coverage.append(dict(method=task['method'],status=load(failure)['status'] if failure.exists() else 'PENDING_NOT_MISSING'));continue
        r=load(folder/'result.json');coverage.append(dict(method=task['method'],status=r['status']))
        if i!=0:
            lane=next(name for name,indices in p['partitions'].items() if name!='M4' and i in indices)
            job=load(LOCAL/(lane+'-job.json'))
            if r.get('job_id')!=job['job_id']:raise ValueError('Actual remote job receipt differs')
            q=load(LOCAL/'collected/out'/('qualification-'+lane+'.json'))
            if q['status']!='QUALIFIED' or q['job_id']!=job['job_id'] or q['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise ValueError('Remote numerical qualification differs')
        if r['status']=='COMPLETED':rows.append(audit_one(p,task,r,folder))
    result=dict(scope=scope,status='N51_SAVED_ROOTS_AUDITED',rows=rows,coverage=coverage,
         new_eigensolves=0,new_model_samples=0,new_QPU_shots=0,goal_complete=False)
    save(ROOT/'analysis'/f'N51-{scope}-root-audit.json',result)
    print('N51_AUDITED',scope,[(r['method'],r['energy_Ha'],r['residual_norm']) for r in rows])


if __name__=='__main__':run(sys.argv[1])
