"""N31 bounded mathematical qualification only, no molecular optimizer/draws."""
from pathlib import Path
from time import process_time,perf_counter
import importlib.metadata
import inspect
import os
import numpy as np
import ffsim
from pyscf.fci import direct_spin1
from .paths import ROOT
from .io import load,save,sha256,space
from . import ucj_backprop_overlay as overlay
from . import finite_source_study as n29
from . import prefix_performance_sampling as legacy

LOCAL=ROOT/'qualification/N31-analytical-UCJ-v1'


def synthetic(n,k,seed,final):
    rng=np.random.default_rng(seed);a=rng.normal(size=(n,n));h=(a+a.T)/2
    x=rng.normal(size=(2,n,n));x=(x+x.transpose(0,2,1))/2;g=np.einsum('Lpq,Lrs->pqrs',x,x)/n
    pairs=([(i,j) for i in range(n) for j in range(i,n)],[(i,j) for i in range(n) for j in range(i,n)])
    count=ffsim.UCJOpSpinBalanced.n_params(n,1,interaction_pairs=pairs,with_final_orbital_rotation=final)
    theta=rng.normal(scale=.25,size=count)
    op=ffsim.UCJOpSpinBalanced.from_parameters(theta,norb=n,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=final)
    return dict(n=n,k=k,seed=seed,final=final,h=h,g=g,core=.431,pairs=pairs,theta=theta,op=op)


def reference_function(h,g,core,n,nelec):
    # PySCF's real contraction applied to real/imaginary parts separately. This
    # does not use the analytical UCJ equations or their differentiated energy.
    effective=direct_spin1.absorb_h1e(h,g,n,nelec,.5)
    def state_energy(v):
        re=np.ascontiguousarray(v.real.reshape(ffsim.dims(n,nelec)))
        im=np.ascontiguousarray(v.imag.reshape(re.shape))
        hr=direct_spin1.contract_2e(effective,re,n,nelec)
        hi=direct_spin1.contract_2e(effective,im,n,nelec)
        return float(np.sum(re*hr)+np.sum(im*hi)+core)
    def energy(op):
        hf=ffsim.hartree_fock_state(n,nelec)
        v=ffsim.apply_unitary(hf,op,norb=n,nelec=nelec)
        return state_energy(v)
    return energy,state_energy


def prepare():
    space();m=overlay.verified_sources()
    if LOCAL.exists():raise RuntimeError('Existing N31 qualifier; do not overwrite/restart')
    p,_=n29.verified();s=next(x for x in load(legacy.N15/'protocol.json')['sources'] if x['case']=='NAPH20-BLOCK-BOYS' and x['method'].endswith('L1'))
    if s['problem_id']!=p['physical'][s['case']]['problem_id']:raise RuntimeError('Same problem required')
    for name,key in [('operator','operator_SHA256'),('construction','construction_SHA256')]:
        if sha256(Path(s[name]).read_bytes())!=s[key]:raise RuntimeError('Original qualification source changed')
    import ffsim.linalg.util as util
    LOCAL.mkdir(parents=True)
    save(LOCAL/'protocol.json',dict(version='N31_PINNED_LICENSED_UCJ_BACKPROP_QUALIFICATION_V1',
        source_commit=m['commit'],vendor_manifest_SHA256=sha256((overlay.VENDOR/'manifest.json').read_bytes()),
        code_SHA256=sha256(Path(__file__).read_bytes()),adapter_SHA256=sha256(Path(overlay.__file__).read_bytes()),
        local_helper_SHA256={name:sha256(inspect.getsource(getattr(util,name)).encode()) for name in ('rotate_one_body_tensor','rotate_two_body_tensor','real_symmetrics_from_parameters_jax','unitary_from_parameters_jax')},
        packages={name:importlib.metadata.version(name) for name in ('ffsim','jax','jaxlib','numpy','scipy','pyscf')},
        fixtures=[dict(n=n,k=k,seed=seed,final=final) for n,k,seed,final in [(2,1,31001,False),(2,1,31002,True),(3,1,31003,True),(4,2,31004,True)]],
        singular_fixture=True,molecular_source=s,molecular_physical=p['physical'][s['case']],
        CPU_seconds_cap=600,wall_seconds_cap=600,new_storage_cap_bytes=128*2**20,energy_tolerance_Ha=1e-8,
        gradient_tolerance=2e-6,FD_step=1e-5,chunk_size=128,new_model_QPU_shots=0,new_optimizer_calls=0,
        original_author_code_unvendored_license_not_stated=True,upstream_ffsim_Apache2_used_instead=True,
        JAX_version_diff_from_author_0p9p2_must_qualify=True,goal_complete=False))
    print('N31_FROZEN_4RANDOM_1SINGULAR_1REAL20Q_NO_OPTIMIZERS_OR_SAMPLES',flush=True)


def verified():
    p=load(LOCAL/'protocol.json');overlay.verified_sources()
    if sha256(Path(__file__).read_bytes())!=p['code_SHA256'] or sha256(Path(overlay.__file__).read_bytes())!=p['adapter_SHA256'] or sha256((overlay.VENDOR/'manifest.json').read_bytes())!=p['vendor_manifest_SHA256']:
        raise RuntimeError('Frozen qualifier/adapter/source drift')
    if {name:importlib.metadata.version(name) for name in p['packages']}!=p['packages']:raise RuntimeError('Runtime drift')
    return p


def run():
    p=verified();space()
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing qualifier attempt; inspect process/terminal, no blind restart')
    save(LOCAL/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    cpu,wall=process_time(),perf_counter();rows=[]
    def bound():
        space()
        if process_time()-cpu>p['CPU_seconds_cap'] or perf_counter()-wall>p['wall_seconds_cap']:raise TimeoutError('Bounded N31 qualifier cap')
        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_storage_cap_bytes']:raise OSError('Bounded new qualification storage')
    def check(f,label):
        bound();tick,cc=perf_counter(),process_time();n,k=f['n'],f['k'];ne=(k,k)
        ref,_=reference_function(f['h'],f['g'],f['core'],n,ne)
        exact=ref(f['op']);value=overlay.energy(f['op'],f['h'],f['g'],f['core'],ne,p['chunk_size'])
        fun=overlay.gradient_function(f['h'],f['g'],f['core'],ne,f['pairs'],f['final'],p['chunk_size'])
        e,gradient=fun(f['theta']);finite=[];step=p['FD_step']
        for i in range(len(f['theta'])):
            bound();x=f['theta'].copy();y=x.copy();x[i]+=step;y[i]-=step
            a=ffsim.UCJOpSpinBalanced.from_parameters(x,norb=n,n_reps=1,interaction_pairs=f['pairs'],with_final_orbital_rotation=f['final'])
            b=ffsim.UCJOpSpinBalanced.from_parameters(y,norb=n,n_reps=1,interaction_pairs=f['pairs'],with_final_orbital_rotation=f['final'])
            finite.append((ref(a)-ref(b))/(2*step))
        error=float(np.max(abs(gradient-finite)));energy_error=max(abs(value-exact),abs(e-exact))
        result=dict(label=label,norb=n,nelec=ne,energy_analytical=value,energy_full_state=exact,
            energy_error_Ha=energy_error,gradient_max_error=error,gradient_components=len(gradient),
            gradient_norm=float(np.linalg.norm(gradient)),CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tick,
            status='PASS' if energy_error<=p['energy_tolerance_Ha'] and error<=p['gradient_tolerance'] else 'PARITY_FAILED_PRESERVED')
        rows.append(result);save(LOCAL/'progress.json',dict(rows=rows));print('N31',label,result,flush=True)
        if result['status']!='PASS':raise RuntimeError('Independent energy/gradient parity failed')
    try:
        for description in p['fixtures']:check(synthetic(**description),f"RANDOM_{description['n']}_{description['seed']}")
        f=synthetic(2,1,31005,True);c=np.sqrt(.5);W=np.array([[c,-c],[c,c]],complex)
        f['op']=ffsim.UCJOpSpinBalanced(np.array([[np.zeros((2,2)),np.diag([np.pi,0.])]]),W[None,:,:],np.eye(2))
        f['theta']=f['op'].to_parameters(interaction_pairs=f['pairs'])
        Q=W.conj().T[:,:1];overlap=complex((Q.conj().T@np.diag([-1.,1.])@Q)[0,0])
        if abs(overlap)>1e-14:raise RuntimeError('Fixture does not reach singular overlap')
        check(f,'SINGULAR_TRANSITION_OVERLAP');rows[-1]['tested_overlap_abs']=abs(overlap)
        bound();tick,cc=perf_counter(),process_time();s=p['molecular_source'];d=p['molecular_physical']
        for name,key in [('operator','operator_SHA256'),('construction','construction_SHA256')]:
            if sha256(Path(s[name]).read_bytes())!=s[key]:raise RuntimeError('Bound real source changed')
        file=Path(d['local_integrals'])
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Bound physical integrals changed')
        with np.load(file,allow_pickle=False) as z:h,g=z['h1'],z['eri']
        with np.load(s['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        ref,state_energy=reference_function(h,g,d['core_energy'],d['norb'],tuple(d['nelec']))
        exact=ref(op);value=overlay.energy(op,h,g,d['core_energy'],tuple(d['nelec']),p['chunk_size'])
        # Separate qualified QPY/primitive path; no state or samples are saved.
        oldspec=load(n29.LUCJ/'protocol.json');circuit=next(c for c in oldspec['circuits'] if c['case']==s['case'])
        cfg=load(n29.LUCJ/'circuits'/circuit['circuit_id']/'config.json')
        with np.load(n29.LUCJ/'circuits'/circuit['circuit_id']/'arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        vector=legacy.qualified_module().numpy_state(cfg,a);primitive=state_energy(vector)
        error=max(abs(value-exact),abs(primitive-exact))
        rows.append(dict(label='ACTUAL20Q_NAPH_FIXED_L1',problem_id=d['problem_id'],norb=d['norb'],nelec=d['nelec'],
            energy_analytical=value,energy_ffsim_reference=exact,energy_independent_primitive=primitive,energy_error_Ha=error,
            CPU_seconds=process_time()-cc,wall_seconds=perf_counter()-tick,status='PASS' if error<=p['energy_tolerance_Ha'] else 'PARITY_FAILED_PRESERVED'))
        save(LOCAL/'progress.json',dict(rows=rows));print('N31_REAL20Q',rows[-1],flush=True)
        if rows[-1]['status']!='PASS':raise RuntimeError('Actual20q physical-H parity failed')
        bound()
        save(LOCAL/'terminal.json',dict(status='N31_6_QUALIFICATION_CASES_PASSED',rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_QPU_shots=0,new_optimizer_calls=0,
            mathematical_function_qualification_not_performance_superiority=True,goal_complete=False))
        print('N31_TERMINAL_PASS6_NO_MOLECULAR_OPTIMIZATION',flush=True)
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='QUALIFICATION_FAILED_PRESERVED',error=repr(exc),rows=rows,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
            new_optimizer_calls=0,new_model_QPU_shots=0));raise


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
