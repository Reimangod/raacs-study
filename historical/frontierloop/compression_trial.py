"""Frozen fixed-circuit D2 screen; no free best-layer selection or remeasurement."""
import argparse
import fcntl
from io import BytesIO
import os
from time import process_time,perf_counter,time
import numpy as np
import ffsim
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,arrays,sha256,digest,space
from .merged_compression import materialize,binding as d1_binding
from .primitive_gpu import build_input
from .primitive_plan import expand_for_execution
from .orbital import optimize as original
from .spin_orbital import optimize as penalized
from .fast_orbital import optimize as singlet
from paperstudy.budget import BudgetLedger
from paperstudy.ci import common_strings
from paperstudy.lucj import circuit_views
from raacsnext.run import physical

DIRECTORY=ROOT/'runs/D2-COMPRESSION-SCREEN'


def binding():
    files=['compression_trial.py','primitive_plan.py','primitive_gpu_worker.py','primitive_gpu.py',
        'fast_orbital.py','singlet.py','hole_rdm.py','orbital.py','spin_orbital.py']
    return {'code_SHA256':{name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in files},
        'protocol_SHA256':sha256((ROOT/'protocol-D2-COMPRESSION-SCREEN.json').read_bytes()),
        'D1_frozen_inputs':d1_binding(),
        'readout_specs_SHA256':{mode:sha256((ROOT/('protocol-'+mode+'.json')).read_bytes()) for mode in load(ROOT/'protocol-D2-COMPRESSION-SCREEN.json')['readouts']},
        'CPU_QPY_qualification_SHA256':sha256((ROOT/'qualification-primitive-QPY-remediation-v2.xml').read_bytes())}


def guard():
    if load(ROOT/'freeze-D2-COMPRESSION-SCREEN.json')!=binding():raise RuntimeError('D2 source/runtime drift')
    return load(ROOT/'protocol-D2-COMPRESSION-SCREEN.json')


def validate_row(case,point):
    manifest=load(DIRECTORY/'manifest.json')
    if manifest['freeze']!=binding():raise RuntimeError('D2 manifest binding changed')
    row=load(DIRECTORY/case/point/'request.json')
    matches=[r for r in manifest['cells'] if r['case']==case and r['point']==point]
    if matches!=[row]:raise RuntimeError('D2 per-cell request differs from frozen manifest')
    return row


def prepare():
    spec=guard();DIRECTORY.mkdir(parents=True,exist_ok=True)
    with (DIRECTORY/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (DIRECTORY/'manifest.json').exists():return load(DIRECTORY/'manifest.json')
        if (DIRECTORY/'prepare-started.json').exists():raise RuntimeError('Partial D2 preparation preserved')
        if load(ROOT/'runs/D1-MERGED-RESOURCE/result.json')['freeze']!=d1_binding():
            raise RuntimeError('D1 source binding failed')
        space();save(DIRECTORY/'prepare-started.json',{'freeze':binding()});rows=[]
        ledger=BudgetLedger(ROOT/'production-shots.sqlite')
        for case in spec['cases']:
            m,_,_=physical(case);ne=m['nalpha'],m['nbeta']
            for point in spec['points']:
                cpu,tick=process_time(),perf_counter();d=DIRECTORY/case/point;d.mkdir(parents=True,exist_ok=True)
                upstream=None
                if point=='EXACT':
                    source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1'
                    with np.load(source/'operator.npz',allow_pickle=False) as a:
                        op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
                    q,_=materialize(op,ne)
                else:
                    cap=int(np.ceil(int(point[1:])/100*m['norb']))
                    source=ROOT/'runs/D1-MERGED-RESOURCE'/case/str(cap)
                    r=load(source/'result.json');p=source/'native-materialized.qpy'
                    if sha256(p.read_bytes())!=r['QPY_SHA256']['native-materialized']:raise RuntimeError('D1 circuit drift')
                    q=qpy.load(BytesIO(p.read_bytes()))[0];upstream=sha256((source/'result.json').read_bytes())
                q=expand_for_execution(q)
                _,_,resources=circuit_views(q)
                identity=digest(['D2-frozen-fixed-sampler',case,point,binding()])
                seed=int(digest([spec['seed_family'],case,point])[:8],16)
                conf=build_input(q,m['norb'],ne,d,None,m['problem_id'],
                    [{'id':identity,'seed':seed,'shots':spec['shots_per_cell']}])
                ledger.register(identity,spec['shots_per_cell'])
                row={'case':case,'point':point,'run_id':identity,'seed':seed,'shots':spec['shots_per_cell'],
                    'problem_id':m['problem_id'],'source_D1_result_SHA256':upstream,
                    'execution_input_id':conf['input_id'],'resources':resources,
                    'exact_execution_QPY_SHA256':conf['source_QPY_SHA256'],
                    'numeric_primitive_coefficient_slots_not_rank':sum(len(i.operation.params) for i in q),
                    'nonredundant_parameter_rank':None,'source_initialization_provenance':
                        str(OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/construction.json'),
                    'input_preparation_CPU_seconds':process_time()-cpu,'input_preparation_wall_seconds':perf_counter()-tick}
                save(d/'request.json',row);rows.append(row)
        result={'cells':rows,'freeze':binding(),'total_model_shots_authorized':sum(r['shots'] for r in rows),
            'all_NOT_STARTED_at_freeze':True,'not_independent_optimizer_trajectories':True}
        save(DIRECTORY/'manifest.json',result);print({'prepared_cells':len(rows),'new_model_shots':0},flush=True);return result


def draw_request(row):
    return {'physical_problem_id':row['problem_id'],'QPY_SHA256':row['exact_execution_QPY_SHA256'],
        'execution_input_id':row['execution_input_id'],'seed':row['seed'],'shots':row['shots'],
        'role':'independent_fixed_sampler_final'}


def local(case,point):
    spec=guard()
    if case!='NAPH20-BLOCK-BOYS':raise ValueError('Only qualified <=20q exact local route')
    if point not in spec['points']:raise ValueError('Unregistered point')
    d=DIRECTORY/case/point;row=validate_row(case,point)
    with (d/'draw-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (d/'draw-result.json').exists():return load(d/'draw-result.json')
        if (d/'draw-started.json').exists():raise RuntimeError('Partial draw preserved; no blind retry')
        save(d/'draw-started.json',{'pid':os.getpid(),'unix':time()});space();cpu,tick=process_time(),perf_counter()
        m,_,_=physical(case);q=qpy.load(BytesIO((d/'source.qpy').read_bytes()))[0]
        if sha256((d/'source.qpy').read_bytes())!=row['exact_execution_QPY_SHA256']:raise RuntimeError('Execution QPY changed')
        vector=ffsim.qiskit.final_state_vector(q,norb=m['norb'],nelec=(m['nalpha'],m['nbeta'])).vec
        norm=float(np.vdot(vector,vector).real)
        if abs(norm-1)>1e-8:raise RuntimeError('Local sector normalization failed')
        build=perf_counter()-tick;ledger=BudgetLedger(ROOT/'production-shots.sqlite')
        reservation=ledger.reserve(row['run_id'],'final',draw_request(row),row['shots'])
        if reservation!='NEW_RESERVATION':raise RuntimeError('Existing draw reservation; reconcile first')
        values=ffsim.sample_state_vector(vector,norb=m['norb'],nelec=(m['nalpha'],m['nbeta']),
            shots=row['shots'],bitstring_type=ffsim.BitstringType.INT,seed=row['seed'])
        ids,counts=np.unique(values,return_counts=True);payload=arrays(blocked_ids=ids.astype(np.uint64),counts=counts.astype(np.uint64))
        write(d/'final-counts.npz',payload);ledger.complete(row['run_id'],'final',int(counts.sum()),sha256(payload))
        result={'status':'EXACT_LOCAL_FIXED_SAMPLER_DRAW_COMPLETED','run_id':row['run_id'],
            'input_id':row['execution_input_id'],'counts_SHA256':sha256(payload),'actual_model_shots':int(counts.sum()),
            'actual_QPU_shots':0,'GPU_used':False,'state_build_wall_seconds':build,'norm_squared':norm,
            'whole_body_CPU_seconds':process_time()-cpu,'whole_body_wall_seconds':perf_counter()-tick,
            'ledger':ledger.snapshot(row['run_id']),'statevector_saved':False}
        save(d/'draw-result.json',result);print('D2 LOCAL',case,point,'shots',int(counts.sum()),flush=True);return result


def readout(case,point,mode):
    spec=guard()
    if case not in spec['cases'] or point not in spec['points'] or mode not in spec['readouts']:
        raise ValueError('Unregistered D2 readout')
    source=DIRECTORY/case/point;row=validate_row(case,point);draw=load(source/'draw-result.json')
    if draw['input_id']!=row['execution_input_id'] or draw['actual_model_shots']!=row['shots']:
        raise RuntimeError('Draw/execution request linkage failed')
    if sha256((source/'final-counts.npz').read_bytes())!=draw['counts_SHA256']:raise RuntimeError('D2 counts binding failed')
    m,h,g=physical(case)
    with np.load(source/'final-counts.npz',allow_pickle=False) as a:ids,counts=a['blocked_ids'],a['counts']
    if int(counts.sum())!=row['shots'] or not np.all(counts>0):raise RuntimeError('Invalid counts')
    mask=(1<<m['norb'])-1
    if any((int(x)&mask).bit_count()!=m['nalpha'] or (int(x)>>m['norb']).bit_count()!=m['nbeta'] for x in ids):
        raise RuntimeError('Counts outside physical particle sector')
    strings,_=common_strings(ids,counts,m['norb'],(m['nalpha'],m['nbeta']),spec['r_cap'])
    d=source/mode;d.mkdir(parents=True,exist_ok=True)
    with (d/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (d/'result.json').exists():return load(d/'result.json')
        if (d/'started.json').exists():raise RuntimeError('Partial OO preserved')
        ledger=BudgetLedger(ROOT/'production-shots.sqlite')
        ledger.record_readout(row['run_id'],mode,'final',{'readout':mode,'r_cap':spec['r_cap'],'counts_SHA256':draw['counts_SHA256']})
        save(d/'started.json',{'unix':time(),'pid':os.getpid()});save(d/'request.json',
            {'source_request_SHA256':sha256((source/'request.json').read_bytes()),'counts_SHA256':draw['counts_SHA256'],
                'CI_strings':strings.tolist(),'mode':mode,'freeze':binding()})
        cpu,tick=process_time(),perf_counter()
        def checkpoint(history,rotations):save(d/'checkpoint.json',{'history':history,'new_model_shots':0})
        ospec=load(ROOT/('protocol-'+mode+'.json'))
        if mode=='C4-SINGLET-OO':r=singlet(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,ospec,mode,checkpoint)
        else:r=(original if mode=='C2-OO' else penalized)(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,ospec,checkpoint)
        rotations,hh,gg=(r.pop(k) for k in ('rotations','final_h1','final_eri'))
        payload=arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
        write(d/'optimized-representation.npz',payload)
        r.update(status='COMPLETED',case=case,point=point,readout=mode,source_run_id=row['run_id'],
            source_counts_SHA256=draw['counts_SHA256'],CI_strings=strings.tolist(),r_cap=spec['r_cap'],
            actual_CI_dimension=len(strings)**2,representation_SHA256=sha256(payload),
            new_model_shots=0,actual_QPU_shots=0,whole_body_CPU_seconds=process_time()-cpu,
            whole_body_wall_seconds=perf_counter()-tick,CI_state_not_sampler_prepared_state=True,goal_complete=False)
        save(d/'result.json',r);print('D2 READOUT',case,point,mode,r['final_energy_Ha'],flush=True);return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','local','readout']);
    p.add_argument('--case');p.add_argument('--point');p.add_argument('--mode');a=p.parse_args()
    if a.action=='freeze':
        import xml.etree.ElementTree as ET
        suites=list(ET.parse(ROOT/'qualification-primitive-QPY-remediation-v2.xml').getroot().iter('testsuite'))
        if sum(int(s.get('tests','0')) for s in suites)!=7 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors')):
            raise RuntimeError('CPU primitive/QPY qualification missing')
        value=binding();path=ROOT/'freeze-D2-COMPRESSION-SCREEN.json'
        if path.exists() and load(path)!=value:raise RuntimeError('Never replace D2 freeze')
        save(path,value)
    elif a.action=='prepare':prepare()
    elif a.action=='local':local(a.case,a.point)
    else:readout(a.case,a.point,a.mode)
