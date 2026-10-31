"""N40 fixed nine-point LUCJ J-aa/J-ab pilot with paid search and untouched final seeds."""
from pathlib import Path
from io import BytesIO
import argparse
import importlib.metadata
import itertools
import json
import os
import resource
from time import process_time,perf_counter
import traceback
import numpy as np
import ffsim
from pyscf.fci import cistring
from qiskit import qpy
from qiskit.quantum_info import Statevector
from paperstudy.lucj import naive,make_circuit,sector_to_blocked_full,circuit_views
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,digest,space
from . import prefix_performance_sampling as legacy
from .finite_source_readout import rank_record,basis_key
from .joint_ci_sparse import solve

RUN=ROOT/'runs/N40-DIRECT-SAMPLE-LUCJ-PILOT-v1'
GRID=list(itertools.product((.5,1.,2.),repeat=2))
CASE='NAPH20-BLOCK-BOYS'
REPS=(400101,400102,400103)


def point_id(point):return f'AA{point[0]:g}_AB{point[1]:g}'


def changed(op,point):
    if op.n_reps!=1 or op.diag_coulomb_mats.shape[1]!=2:raise ValueError('Same one-layer balanced source required')
    j=op.diag_coulomb_mats.copy();j[:,0]*=point[0];j[:,1]*=point[1]
    return ffsim.UCJOpSpinBalanced(j,op.orbital_rotations.copy(),op.final_orbital_rotation.copy())


def error_up_to_phase(a,b):
    z=np.vdot(a,b);phase=z/abs(z) if abs(z) else 1
    return float(np.max(abs(b-phase*a)))


def seed(case,rep,role,point=None):return int(digest(['N40_INDEPENDENT_RNG_V1',case,rep,role,point])[:16],16)%(1<<63)


def choose(candidates):
    if len(candidates)!=9 or any(c['status']!='COMPLETED' for c in candidates):
        raise RuntimeError('Incomplete training grid is engineering missingness, not a high energy')
    best=min(c['result']['penalized_energy_Ha'] for c in candidates)
    ties=[c for c in candidates if c['result']['penalized_energy_Ha']<=best+1e-9]
    return min(ties,key=lambda c:(sum((v-1)**2 for v in c['point']),tuple(c['point'])))


def qualify():
    if (RUN/'qualifier.json').exists():raise RuntimeError('No qualifier overwrite')
    cpu,wall=process_time(),perf_counter();rows=[]
    for n,k in ((3,1),(4,2),(5,2)):
        rng=np.random.default_rng(40000+n);v=n-k
        t1=rng.normal(0,.03,(k,v));t2=rng.normal(0,.05,(k,k,v,v));t2=(t2+t2.transpose(1,0,3,2))/2
        op=naive(t1,t2,n,1)
        for point in GRID:
            new=changed(op,point);sector=ffsim.apply_unitary(ffsim.hartree_fock_state(n,(k,k)),new,n,(k,k))
            full=Statevector(make_circuit(new,n,(k,k))).data
            err=error_up_to_phase(sector_to_blocked_full(sector,n,(k,k)),full)
            if err>1e-8:raise ValueError('Independent full-Fock/sector scaling convention mismatch')
            rows.append({'norb':n,'nelec':[k,k],'point':list(point),'state_error':err})
    save(RUN/'qualifier.json',{'status':'ALL27_INDEPENDENT_FULLFOCK_SECTOR_CHECKS_PASSED',
                              'rows':rows,'implementation_SHA256':sha256(Path(__file__).read_bytes()),
                              'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-wall})
    print('N40 all27 small physical state checks passed')


def catalog():
    p=load(ROOT/'runs/N32-ANALYTICAL-L1-OPTIMIZATION-v1/protocol.json')
    result=[]
    for row in p['sources']:
        s,d=row['source'],row['physical']
        for key in ('operator','construction'):
            if sha256(Path(s[key]).read_bytes())!=s[key+'_SHA256']:raise ValueError('Frozen original initialization drift')
        if sha256(Path(d['local_integrals']).read_bytes())!=d['integrals_SHA256'] or s['problem_id']!=d['problem_id']:
            raise ValueError('Same physical integrals/problem required')
        result.append({'source':s,'physical':d,'old_initialization_cost':row['old_initialization_cost'],
                       'optimizer_coordinate_slots_of_original_family':row['optimizer_coordinate_slots']})
    if len(result)!=4:raise ValueError('Preserve all4 source cases')
    return result


def helpers():
    names=['direct_sample_pilot.py','finite_source_readout.py','paired_lowshot.py','joint_ci_sparse.py','joint_ci_sparse.c']
    return {str(ROOT/'frontierloop'/name):sha256((ROOT/'frontierloop'/name).read_bytes()) for name in names}


def prepare():
    space();q=load(RUN/'qualifier.json')
    if q['implementation_SHA256']!=sha256(Path(__file__).read_bytes()) or len(q['rows'])!=27:raise ValueError('Qualifier drift')
    if (RUN/'protocol.json').exists() or (RUN/'sources-started.json').exists():raise RuntimeError('Existing pilot preparation; no overwrite')
    cat=catalog();item=next(s for s in cat if s['source']['case']==CASE);s,d=item['source'],item['physical']
    cpu,wall=process_time(),perf_counter();save(RUN/'sources-started.json',{'case':CASE,'pid':os.getpid()})
    with np.load(s['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
    n,ne=s['norb'],tuple(s['nelec']);module=legacy.qualified_module();points=[]
    for point in GRID:
        if process_time()-cpu>30:raise TimeoutError('Bounded actual-source qualification')
        new=changed(op,point);qc=ffsim.qiskit.PRE_INIT.run(make_circuit(new,n,ne))
        stream=BytesIO();qpy.dump(qc,stream);logical=stream.getvalue()
        restored=qpy.load(BytesIO(logical))[0];cfg,vals=module.plan(restored,n,ne)
        primitive=module.numpy_state(cfg,vals);native=ffsim.apply_unitary(ffsim.hartree_fock_state(n,ne),new,n,ne)
        err=error_up_to_phase(native,primitive);prob=abs(native)**2;norm=float(prob.sum())
        if err>1e-8 or abs(norm-1)>1e-8:raise ValueError('Actual 20q primitive/native source mismatch')
        folder=RUN/'sources'/point_id(point)
        write(folder/'logical.qpy',logical)
        write(folder/'probabilities.npz',arrays(probabilities=prob/norm))
        _,sample,views=circuit_views(qc);stream=BytesIO();qpy.dump(sample,stream);write(folder/'sampling.qpy',stream.getvalue())
        write(folder/'operator.npz',arrays(diag_coulomb_mats=new.diag_coulomb_mats,orbital_rotations=new.orbital_rotations,final_orbital_rotation=new.final_orbital_rotation))
        files={name:{'path':str(folder/name),'SHA256':sha256((folder/name).read_bytes())}
               for name in ('logical.qpy','probabilities.npz','sampling.qpy','operator.npz')}
        points.append({'point':list(point),'point_id':point_id(point),'files':files,'resources':views['Z_sampling_equivalent'],
                       'actual_primitive_state_error_up_to_global_phase':err,'native_norm_squared':norm,
                       'n_qubits':2*n,'sector_dimension':len(prob),'registered_optimized_latent_variables':2,
                       'independent_circuit_parameter_rank':None})
    from . import finite_source_study as n29
    old=load(n29.LOCAL/'protocol.json');library=Path(old['library'])
    if sha256(library.read_bytes())!=old['library_SHA256']:raise ValueError('Qualified sparse solver changed')
    seeds=[seed(CASE,r,'TRAIN',p) for r in REPS for p in GRID]+[seed(CASE,r,role) for r in REPS for role in ('FINAL','BASELINE')]
    if len(seeds)!=len(set(seeds)):raise ValueError('Independent draw seed collision')
    protocol={'version':'N40_FIXED9_FULLSPIN_LUCJ_SQD_OBJECTIVE_PILOT_V1','catalog_all4_definitions':cat,
              'executed_case_scope':[CASE],'other3':'NOT_STARTED_REQUIRES_SAME_SCIENCE_GPU_EXECUTOR_NOT_MISSING_NOT_SCIENTIFIC_FAILURE',
              'grid':[list(p) for p in GRID],'points':points,'replicates':list(REPS),
              'seed_rule':'DOMAIN_SHA256_N40_INDEPENDENT_RNG_V1','train_shots_per_point':500,'final_shots':1000,
              'baseline_total_shots':5500,'total_per_algorithm_per_replicate':5500,'new_model_shot_cap':33000,
              'max_R':32,'max_D':1024,'root_seconds':20,'parent_CPU_seconds':120,'wall_seconds':180,'max_run_bytes':32*2**20,
              'selection':'MIN_H_PLUS_0p1S2_TRAINING_ONLY_TOL1e-9_THEN_DISTANCE1_1_THEN_LEX;ALL9_REQUIRED',
              'library':str(library),'library_SHA256':old['library_SHA256'],'helpers':helpers(),
              'qualifier_SHA256':sha256((RUN/'qualifier.json').read_bytes()),
              'primitive_plan_SHA256':sha256((legacy.QUAL/'primitive_plan.py').read_bytes()),
              'paperstudy_lucj_SHA256':sha256((OLD/'paperstudy/lucj.py').read_bytes()),
              'packages':{k:importlib.metadata.version(k) for k in ('ffsim','numpy','scipy','qiskit','pyscf')},
              'prepare_CPU_seconds':process_time()-cpu,'prepare_wall_seconds':perf_counter()-wall,
              'new_QPU_shots':0,'new_FCI':0,'new_energy_before_freeze':0,'best_paper_superiority_proven':False}
    save(RUN/'protocol.json',protocol);print('N40 fixed9/3rep/33000 maximum model shots frozen; other3 pending')


def verified():
    p=load(RUN/'protocol.json')
    if p['helpers']!=helpers() or p['catalog_all4_definitions']!=catalog():raise ValueError('Pilot code/physical source drift')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256']:raise ValueError('Solver library drift')
    for point in p['points']:
        for v in point['files'].values():
            if sha256(Path(v['path']).read_bytes())!=v['SHA256']:raise ValueError('Frozen source artifact changed')
    if {k:importlib.metadata.version(k) for k in p['packages']}!=p['packages']:raise ValueError('Runtime changed')
    return p


def run():
    p=verified();space()
    if (RUN/'started.json').exists():raise RuntimeError('Do not restart partial/terminal pilot')
    save(RUN/'started.json',{'pid':os.getpid(),'protocol_SHA256':sha256((RUN/'protocol.json').read_bytes())})
    cpu,wall=process_time(),perf_counter();draws=[];attempts=[];states=[];cache={};selections=[]
    item=next(s for s in p['catalog_all4_definitions'] if s['source']['case']==CASE);d=item['physical'];n=d['norb'];ne=tuple(d['nelec'])
    with np.load(d['local_integrals'],allow_pickle=False) as z:h,g=z['h1'],z['eri']
    sa=cistring.make_strings(range(n),ne[0]).astype(np.uint64);sb=cistring.make_strings(range(n),ne[1]).astype(np.uint64)
    blocked=(sa[:,None]|(sb[None,:]<<np.uint64(n))).ravel()
    probabilities={}
    for row in p['points']:
        with np.load(row['files']['probabilities.npz']['path'],allow_pickle=False) as z:probabilities[row['point_id']]=z['probabilities']
    def bound():
        space()
        if process_time()-cpu>120 or perf_counter()-wall>180:raise TimeoutError('N40 finite pilot cost cap')
        if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>p['max_run_bytes']:raise OSError('N40 new storage cap')
    def sample(rep,role,point,shots):
        bound();pid=point_id(point);rngseed=seed(CASE,rep,role,point if role=='TRAIN' else None)
        intent={'case':CASE,'replicate':rep,'role':role,'point':list(point),'seed':rngseed,'shots':shots,
                'problem_id':d['problem_id'],'source_probability_SHA256':next(x for x in p['points'] if x['point_id']==pid)['files']['probabilities.npz']['SHA256']}
        did=digest(intent);folder=RUN/'draws'/did;save(folder/'intent.json',intent)
        values,counts=np.unique(np.random.default_rng(rngseed).choice(blocked,size=shots,p=probabilities[pid]),return_counts=True)
        path=folder/'counts.npz';write(path,arrays(blocked_ids=values,counts=counts.astype(np.uint64)))
        rec=dict(intent,draw_id=did,path=str(path),SHA256=sha256(path.read_bytes()),rank_record=rank_record(values,counts,n,ne[0]))
        save(folder/'result.json',rec);draws.append(rec);save(RUN/'draw-progress.json',{'draws':draws,'actual_model_shots':sum(x['shots'] for x in draws)})
        return rec
    def root(draw,R,arm):
        bound();strings=sorted(draw['rank_record']['rank'][:R]);pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
        key=basis_key(d,pairs,p['library_SHA256']);new=key not in cache
        if new:
            tick=process_time();save(RUN/'roots'/(key+'.started.json'),{'basis_id':key,'pid':os.getpid()})
            try:
                result,v=solve(h,g,d['core_energy'],ne,pairs,p['library'],seconds=20,triangle_cap=8000000)
                f=RUN/'roots'/(key+'.npz');write(f,arrays(joint_pairs=pairs,coefficients=v))
                value={'status':'COMPLETED','basis_id':key,'result':result,'root_path':str(f),'root_SHA256':sha256(f.read_bytes())}
            except Exception as exc:value={'status':'FAILED_PRESERVED','basis_id':key,'result':None,'error':repr(exc)}
            value['attempt_CPU_seconds']=process_time()-tick;cache[key]=value;save(RUN/'roots'/(key+'.json'),value);attempts.append(value)
        value=cache[key]
        row=dict(value,case=CASE,replicate=draw['replicate'],role=draw['role'],arm=arm,point=draw['point'],draw_id=draw['draw_id'],
                 R=R,actual_D=R*R,half_strings=strings,new_CI_attempt=new)
        states.append(row);save(RUN/'CI-progress.json',{'states':states,'attempts':attempts})
        if value['status']!='COMPLETED':raise RuntimeError('Numerical CI failure preserved, not a bad candidate energy')
        return row
    failure=None
    try:
        for rep in REPS:
            training=[root((draw:=sample(rep,'TRAIN',point,500)),draw['rank_record']['R32'],'TRAIN_R32') for point in GRID]
            chosen=choose(training);choice={'replicate':rep,'point':chosen['point'],'training_basis_id':chosen['basis_id'],
                                          'training_penalized_energy_Ha':chosen['result']['penalized_energy_Ha']}
            selections.append(choice);save(RUN/'selections-before-final.json',{'selections':selections})
            final=sample(rep,'FINAL',chosen['point'],1000);baseline=sample(rep,'BASELINE',(1.,1.),5500)
            for arm in ('R32_CAP','COMMON_R'):
                common=min(final['rank_record']['R32'],baseline['rank_record']['R32'])
                for draw in (final,baseline):root(draw,draw['rank_record']['R32'] if arm=='R32_CAP' else common,arm)
            print('N40_REPLICATE_TERMINAL',rep,chosen['point'],flush=True)
        if sum(x['shots'] for x in draws)!=33000 or len(draws)!=33:raise AssertionError('Registered complete model budget mismatch')
        status='NAPH20_ALL3_REPLICATES_27TRAIN_12FINAL_ENDPOINTS_COMPLETE'
    except Exception as exc:failure={'error':repr(exc),'traceback':traceback.format_exc()};status='FAILED_ENGINEERING_PRESERVED'
    terminal={'status':status,'case':CASE,'failure':failure,'draws':draws,'states':states,'selections':selections,
              'attempts':attempts,'actual_new_CI_roots':len(attempts),'actual_model_shots':sum(x['shots'] for x in draws),
              'actual_QPU_shots':0,'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-wall,
              'other3_status':'NOT_STARTED','statistical_superiority_certified':False,'best_paper_superiority_proven':False,
              'protocol_SHA256':sha256((RUN/'protocol.json').read_bytes())}
    save(RUN/'terminal.json',terminal);print(json.dumps({k:v for k,v in terminal.items() if k not in ('draws','states','attempts')}))
    if failure:raise RuntimeError('N40 partial data retained')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['qualify','prepare','run']);args=a.parse_args()
    if args.action=='run':resource.setrlimit(resource.RLIMIT_CPU,(120,120));run()
    elif args.action=='qualify':qualify()
    else:prepare()
