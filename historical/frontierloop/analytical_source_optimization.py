"""N32 paid, bounded same-mask L1 coefficient improvement, all four cases."""
from pathlib import Path
from time import process_time,perf_counter
from io import BytesIO
import importlib.metadata
import os,sys,subprocess,resource,signal
import numpy as np
import ffsim
from scipy.optimize import minimize
from qiskit import qpy
from paperstudy.lucj import interaction_pairs,make_circuit
from . import backprop_qualification_v2 as qualifier
from . import ucj_backprop_overlay_v2 as overlay
from . import finite_source_study as n29
from . import prefix_performance_sampling as legacy
from .slater_prefix_preflight import compile_views
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,space

LOCAL=ROOT/'runs/N32-ANALYTICAL-L1-OPTIMIZATION-v1'


class BudgetReached(RuntimeError):pass


def helper_bindings():
    files=[Path(qualifier.__file__),Path(overlay.__file__),ROOT/'frontierloop/ucj_backprop_overlay.py',
           ROOT/'frontierloop/backprop_qualification.py',ROOT/'frontierloop/slater_prefix_preflight.py',OLD/'paperstudy/lucj.py']
    return {str(p):sha256(p.read_bytes()) for p in files}


def load_source(s):
    for key in ('operator','construction'):
        if sha256(Path(s[key]).read_bytes())!=s[key+'_SHA256']:raise RuntimeError('Original source drift')
    with np.load(s['operator'],allow_pickle=False) as z:
        return ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])


def parameterize(op):
    if op.n_reps!=1 or op.final_orbital_rotation is None:raise ValueError('Actual one-layer/final-rotation scope')
    pairs=interaction_pairs(op.norb);theta=op.to_parameters(interaction_pairs=pairs)
    restored=ffsim.UCJOpSpinBalanced.from_parameters(theta,norb=op.norb,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
    errors={k:float(np.max(abs(getattr(op,k)-getattr(restored,k)))) for k in ('diag_coulomb_mats','orbital_rotations','final_orbital_rotation')}
    if max(errors.values())>1e-8:raise ValueError('Same-mask actual matrix roundtrip failed; no silent mask expansion')
    return pairs,theta,errors


def prepare():
    space();qualifier.verified();t=load(qualifier.LOCAL/'terminal.json')
    if t['status']!='N31_6_QUALIFICATION_CASES_PASSED':raise RuntimeError('Actual N31 results required')
    if LOCAL.exists():raise RuntimeError('N32 exists; preserve/reconcile')
    n29p,_=n29.verified()
    sources=[s for s in load(legacy.N15/'protocol.json')['sources'] if s['method'].endswith('L1')]
    if len(sources)!=4 or {s['case'] for s in sources}!=set(n29p['physical']):raise RuntimeError('All4 fixed sources required')
    records=[]
    for s in sources:
        d=n29p['physical'][s['case']]
        if s['problem_id']!=d['problem_id']:raise RuntimeError('Same physical problem required')
        op=load_source(s);pairs,theta,errors=parameterize(op)
        records.append(dict(source=s,physical=d,pairs=pairs,roundtrip_errors=errors,optimizer_coordinate_slots=len(theta),
            independent_parameter_rank=None,old_initialization_cost=load(s['construction'])['initialization_cost']))
    LOCAL.mkdir(parents=True)
    save(LOCAL/'protocol.json',dict(version='N32_SAME_MASK_L1_ANALYTICAL_LBFGS8_V1',sources=records,
        source_SHA256=sha256(Path(__file__).read_bytes()),helpers=helper_bindings(),
        qualification_terminal_SHA256=sha256((qualifier.LOCAL/'terminal.json').read_bytes()),
        source_protocol_SHA256=sha256((n29.LOCAL/'protocol.json').read_bytes()),
        plan_SHA256=sha256((ROOT/'analysis/N32_BOUNDED_COEFFICIENT_PLAN_JA.md').read_bytes()),
        packages={k:importlib.metadata.version(k) for k in ('ffsim','jax','jaxlib','numpy','scipy','qiskit')},
        CPU_seconds_per_case=600,wall_seconds_per_case=180,case_storage_cap_bytes=16*2**20,new_run_cap_bytes=64*2**20,
        max_energy_gradient_evaluations=8,FD_step=1e-5,FD_tolerance=2e-5,energy_tolerance_Ha=1e-8,
        optimizer=dict(method='L-BFGS-B',maxiter=8,maxls=4,gtol=1e-5,ftol=1e-12),chunk_size=128,
        selection='MIN_PHYSICAL_H_ENERGY_AMONG_INITIAL_AND_OPTIMIZER_EVALUATED_POINTS;FD_POINTS_EXCLUDED',
        new_model_QPU_shots=0,new_CI_solves=0,no_statevector_saved=True,native_paper_reproduction=False,goal_complete=False))
    print('N32_FROZEN_ALL4_SAME_MASK_L1_LBFGS_MAX8_EVAL_EACH',flush=True)


def verified():
    p=load(LOCAL/'protocol.json');qualifier.verified()
    if p['source_SHA256']!=sha256(Path(__file__).read_bytes()) or p['helpers']!=helper_bindings():raise RuntimeError('Frozen N32 code drift')
    if p['qualification_terminal_SHA256']!=sha256((qualifier.LOCAL/'terminal.json').read_bytes()):raise RuntimeError('Qualification drift')
    if p['source_protocol_SHA256']!=sha256((n29.LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Source context drift')
    if {k:importlib.metadata.version(k) for k in p['packages']}!=p['packages']:raise RuntimeError('Runtime drift')
    return p


def compile_resources(op,s,folder,label):
    qc=ffsim.qiskit.PRE_INIT.run(make_circuit(op,s['norb'],tuple(s['nelec'])))
    out=[]
    for level in (1,3):
        prep,samp,res=compile_views(qc,level);paths={}
        for role,c in [('preparation',prep),('sampling',samp)]:
            stream=BytesIO();qpy.dump(c,stream);file=folder/f'{label}-opt{level}-{role}.qpy';write(file,stream.getvalue())
            roundtrip=qpy.load(BytesIO(stream.getvalue()))[0]
            if roundtrip.count_ops()!=c.count_ops() or roundtrip.depth()!=c.depth():raise RuntimeError('QPY roundtrip changed resources')
            paths[role]=dict(file=str(file),SHA256=sha256(file.read_bytes()))
        out.append(dict(resources=res,files=paths))
    return out


def worker(case):
    p=verified();row=next(r for r in p['sources'] if r['source']['case']==case);s=row['source'];d=row['physical']
    folder=LOCAL/'cases'/case
    if folder.exists():raise RuntimeError('Case already attempted; inspect rather than rerun')
    folder.mkdir(parents=True);save(folder/'started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    resource.setrlimit(resource.RLIMIT_CPU,(p['CPU_seconds_per_case'],p['CPU_seconds_per_case']+1))
    resource.setrlimit(resource.RLIMIT_FSIZE,(p['case_storage_cap_bytes'],p['case_storage_cap_bytes']))
    cpu,wall=process_time(),perf_counter();history=[];cache={};optimizer_calls=0;fd_calls=0;direct_calls=0
    def bound():
        space()
        if process_time()-cpu>p['CPU_seconds_per_case'] or perf_counter()-wall>p['wall_seconds_per_case']:raise BudgetReached('Case time cap')
        if sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())>p['case_storage_cap_bytes']:raise BudgetReached('New case storage cap')
    def costs():return dict(CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,energy_gradient_evaluations=len(history),
        direct_energy_evaluations=direct_calls,FD_direct_energy_evaluations=fd_calls,optimizer_calls=optimizer_calls,new_model_QPU_shots=0,new_CI_solves=0)
    try:
        op=load_source(s);pairs,x0,errors=parameterize(op);n=s['norb'];ne=tuple(s['nelec'])
        file=Path(d['local_integrals'])
        if sha256(file.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integral drift')
        with np.load(file,allow_pickle=False) as z:h,g=z['h1'],z['eri']
        write(folder/'initial-parameters.npz',arrays(theta=x0));core=d['core_energy']
        fun=overlay.gradient_function(h,g,core,ne,pairs,True,p['chunk_size'])
        best=[None,None,None]
        def objective(x):
            bound();key=np.asarray(x).tobytes()
            if key in cache:return cache[key]
            if len(history)>=p['max_energy_gradient_evaluations']:raise BudgetReached('Strict eight-evaluation cap')
            c,t=process_time(),perf_counter();e,grad=fun(x)
            record=dict(index=len(history),energy_Ha=e,gradient_norm=float(np.linalg.norm(grad)),gradient_inf=float(np.max(abs(grad))),
                CPU_seconds=process_time()-c,wall_seconds=perf_counter()-t,theta_SHA256=sha256(key))
            history.append(record);cache[key]=(e,grad)
            if best[0] is None or e<best[0]:
                best[:]=[e,np.asarray(x).copy(),grad.copy()]
                write(folder/'best-parameters.npz',arrays(theta=best[1],gradient=best[2]))
            save(folder/'progress.json',dict(history=history,costs=costs()))
            print('N32_EVAL',case,record,flush=True);bound();return e,grad
        e0,g0=objective(x0)
        direct0=overlay.energy(op,h,g,core,ne,p['chunk_size']);direct_calls+=1;bound()
        direction=np.cos(np.arange(len(x0))+1.);direction/=np.linalg.norm(direction);v=[]
        for sign in (-1,1):
            test=ffsim.UCJOpSpinBalanced.from_parameters(x0+sign*p['FD_step']*direction,norb=n,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
            v.append(overlay.energy(test,h,g,core,ne,p['chunk_size']));fd_calls+=1;direct_calls+=1;bound()
        fd=(v[1]-v[0])/(2*p['FD_step']);ad=float(np.dot(g0,direction))
        parity=dict(direct_energy_error_Ha=abs(e0-direct0),directional_FD=fd,directional_AD=ad,directional_error=abs(fd-ad))
        save(folder/'initial-parity.json',parity)
        if parity['direct_energy_error_Ha']>p['energy_tolerance_Ha'] or parity['directional_error']>p['FD_tolerance']:raise RuntimeError('Actual source energy/directional gradient parity failed')
        opts=p['optimizer'];reason='EVALUATION_BUDGET';success=False;iterations=None;optimizer_calls=1
        try:
            sol=minimize(objective,x0,jac=True,method=opts['method'],options={k:v for k,v in opts.items() if k!='method'})
            reason=str(sol.message);success=bool(sol.success);iterations=int(sol.nit)
        except BudgetReached as exc:reason=str(exc)
        bound();outop=ffsim.UCJOpSpinBalanced.from_parameters(best[1],norb=n,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
        last=overlay.energy(outop,h,g,core,ne,p['chunk_size']);direct_calls+=1
        if abs(last-best[0])>p['energy_tolerance_Ha']:raise RuntimeError('Output direct-energy parity failed')
        operator=folder/'optimized-operator.npz';write(operator,arrays(diag_coulomb_mats=outop.diag_coulomb_mats,orbital_rotations=outop.orbital_rotations,final_orbital_rotation=outop.final_orbital_rotation))
        oldresources=compile_resources(op,s,folder,'original');newresources=compile_resources(outop,s,folder,'optimized');bound()
        save(folder/'result.json',dict(status='COMPLETED_BOUNDED_L1_COEFFICIENT_OPTIMIZATION',case=case,problem_id=s['problem_id'],
            initial_energy_Ha=e0,selected_energy_Ha=best[0],delta_energy_Ha=best[0]-e0,selected_gradient_inf=float(np.max(abs(best[2]))),
            optimizer_success=success,optimizer_iterations=iterations,termination_reason=reason,
            optimizer_coordinate_slots=len(x0),independent_parameter_rank=None,roundtrip_errors=errors,initial_parity=parity,
            history=history,costs=costs(),operator=str(operator),operator_SHA256=sha256(operator.read_bytes()),
            source_operator_SHA256=s['operator_SHA256'],old_initialization_cost=row['old_initialization_cost'],
            original_resources=oldresources,optimized_resources=newresources,same_logical_mask=True,
            physical_H_expectation_not_SQD_energy=True,new_finite_sample_SQD_quality_unverified=True,goal_complete=False))
        print('N32_CASE_TERMINAL',case,best[0]-e0,costs(),flush=True)
    except Exception as exc:
        save(folder/'failure.json',dict(status='FAILED_PRESERVED',error=repr(exc),history=history,costs=costs()))
        raise


def run():
    p=verified();space()
    if (LOCAL/'controller-started.json').exists():raise RuntimeError('Existing controller; inspect actual handle, no blind restart')
    save(LOCAL/'controller-started.json',dict(pid=os.getpid()))
    rows=[];cpu=resource.getrusage(resource.RUSAGE_CHILDREN);wall=perf_counter()
    for item in p['sources']:
        case=item['source']['case'];space();folder=LOCAL/'cases'/case
        proc=subprocess.Popen([sys.executable,'-m','frontierloop.analytical_source_optimization','worker',case],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:output,_=proc.communicate(timeout=p['wall_seconds_per_case']+5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:output,_=proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);output,_=proc.communicate()
            write(LOCAL/'logs'/f'{case}.log',output)
            save(LOCAL/'limits'/f'{case}.json',dict(status='WALL_LIMIT_PRESERVED',PID=proc.pid,exitcode=proc.returncode,wall_limit=p['wall_seconds_per_case']))
        else:write(LOCAL/'logs'/f'{case}.log',output)
        if (folder/'result.json').exists():status=load(folder/'result.json')['status']
        elif (LOCAL/'limits'/f'{case}.json').exists():status='WALL_LIMIT_PRESERVED'
        elif (folder/'failure.json').exists():status='FAILED_PRESERVED'
        else:status='CHILD_EXIT_WITHOUT_RESULT_PRESERVED'
        rows.append(dict(case=case,status=status,exitcode=proc.returncode));save(LOCAL/'controller-progress.json',dict(rows=rows))
        print('N32_CONTROLLER',rows[-1],flush=True)
        if sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())>p['new_run_cap_bytes']:raise RuntimeError('New N32 storage cap')
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(LOCAL/'terminal.json',dict(status='N32_ALL4_CASE_STATES_TERMINAL',rows=rows,
        child_CPU_seconds=after.ru_utime+after.ru_stime-cpu.ru_utime-cpu.ru_stime,wall_seconds=perf_counter()-wall,
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),new_model_QPU_shots=0,new_CI_solves=0,goal_complete=False))


if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2])
    else:{'prepare':prepare,'run':run}[sys.argv[1]]()
