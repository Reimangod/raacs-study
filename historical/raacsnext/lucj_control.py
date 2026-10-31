"""Fresh independent readouts of a source-bound fixed LUCJ comparator.

The qualified operator is reused, not historical measurement counts. All
original construction costs remain visible instead of counting file load as
CCSD or compressed-factorization creation. Large vectors require another lane.
"""
import argparse
import fcntl
from io import BytesIO
from pathlib import Path
import sqlite3
from time import perf_counter,process_time,time
import ffsim
import numpy as np
from qiskit import qpy
from paperstudy.budget import BudgetLedger
from paperstudy.ci import solve
from paperstudy.lucj import make_circuit,circuit_views,interaction_pairs
from .paths import ROOT,OLD
from .io import save,load,write_bytes,array_bytes,digest,sha256,space_check
from .run import physical


def run(case,method='LUCJ_COMPRESSED_L1'):
    protocol=load(ROOT/'protocol.json')
    if case not in protocol['cases'] or method not in('LUCJ_COMPRESSED_L1','LUCJ_NAIVE_L1'):
        raise ValueError('Only registered source-bound L1 controls in this lane')
    m,h,g=physical(case)
    if m['norb']>16:raise MemoryError('40q full number-sector state is not authorized on the local M4')
    source=OLD/'prepared'/case/(('C_'if'COMPRESSED'in method else'B_')+method+'-L1')
    construction=load(source/'construction.json');blob=(source/'operator.npz').read_bytes()
    if construction['problem_id']!=m['problem_id']or sha256(blob)!=construction['operator_sha256']:
        raise RuntimeError('LUCJ operator/physical identity changed')
    root=ROOT/'controls'/case/method;root.mkdir(parents=True,exist_ok=True)
    with(root/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        space_check(4*1024**2)
        with np.load(BytesIO(blob),allow_pickle=False)as a:
            op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
        prep,sampling,views=circuit_views(make_circuit(op,m['norb'],(m['nalpha'],m['nbeta'])))
        if views!=construction['resources']:raise RuntimeError('Independent common resource recount changed')
        slots=len(op.to_parameters(interaction_pairs=interaction_pairs(m['norb'])))
        if slots!=construction['coefficient_slots']:raise RuntimeError('Coefficient-coordinate mismatch')
        tick,cpu=perf_counter(),process_time()
        state=ffsim.apply_unitary(ffsim.hartree_fock_state(m['norb'],(m['nalpha'],m['nbeta'])),
            op,m['norb'],(m['nalpha'],m['nbeta']),copy=False)
        norm=float(np.vdot(state,state).real)
        if abs(norm-1)>1e-8:raise RuntimeError('LUCJ exact number-sector norm failed')
        simulation={'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'state_norm':norm,
            'vector_amplitudes':state.size,'state_saved':False,'algorithm_access_to_vector':False,
            'run_count_sharing_this_simulation':len(protocol['pilot_seeds'])}
        save(root/'shared-simulation.json',simulation)
        save(root/'source.json',{'operator_path':str(source/'operator.npz'),'operator_SHA256':sha256(blob),
            'source_construction':construction,'upstream_initialization':load(OLD/'data/initialization-fresh'/case/'result.json'),
            'code_SHA256':sha256(Path(__file__).read_bytes()),'case_id':case,'problem_id':m['problem_id'],
            'reused_historical_counts':False,'new_CCSD_or_compressed_initialization_this_campaign':False})
        for name,q in(('preparation',prep),('sampling',sampling)):
            out=BytesIO();qpy.dump(q,out);write_bytes(root/(name+'.qpy'),out.getvalue())
        db=sqlite3.connect(ROOT/'queue.sqlite',timeout=30)
        for seed in protocol['pilot_seeds']:
            directory=ROOT/'runs'/'DEVELOPMENT'/case/method/('100000-'+str(seed))
            filename=directory/'result.json'
            if filename.exists():continue
            if time()>load(ROOT/'campaign-start.json')['pilot_deadline_unix']:raise TimeoutError('Pilot deadline')
            config={'schema':'source-bound-LUCJ-control-2','problem_id':m['problem_id'],'case_id':case,
                'variant':method,'stage':'DEVELOPMENT','budget':100000,'seed':seed,
                'operator_SHA256':sha256(blob),'source_construction_SHA256':sha256((source/'construction.json').read_bytes()),
                'code_SHA256':sha256(Path(__file__).read_bytes()),'protocol_SHA256':sha256((ROOT/'protocol.json').read_bytes())}
            identity=digest(config);save(directory/'request.json',config)
            ledger=BudgetLedger(ROOT/'shots.sqlite');ledger.register(identity,100000)
            readout={'seed':int(digest(['LUCJ-practical-v2-independent-final',identity])[:16],16),'shots':100000,
                'operator_SHA256':sha256(blob),'problem_id':m['problem_id']}
            event=digest(readout);counts_path=directory/'final-counts.npz'
            committed=directory/'draw.json'
            status=ledger.reserve(identity,event,readout,100000)
            start,cp=perf_counter(),process_time()
            if committed.exists():
                saved=load(committed)
                if sha256(counts_path.read_bytes())!=saved['counts_SHA256']:raise RuntimeError('Control counts changed')
                with np.load(counts_path,allow_pickle=False)as a:ids,counts=a['blocked_ids'],a['counts']
                ledger.complete(identity,event,int(counts.sum()),saved['counts_SHA256'])
            else:
                if status!='NEW_RESERVATION':raise RuntimeError('Unknown draw; do not measure again')
                samples=ffsim.sample_state_vector(state,norb=m['norb'],nelec=(m['nalpha'],m['nbeta']),
                    shots=100000,bitstring_type=ffsim.BitstringType.INT,seed=readout['seed'])
                ids,counts=np.unique(samples,return_counts=True)
                payload=array_bytes(blocked_ids=ids.astype(np.uint64),counts=counts.astype(np.uint64))
                write_bytes(counts_path,payload)
                save(committed,{'counts_SHA256':sha256(payload),'request':readout,'actual_shots':int(counts.sum()),
                    'CPU_seconds':process_time()-cp,'wall_seconds':perf_counter()-start})
                ledger.complete(identity,event,int(counts.sum()),sha256(payload))
            cis=[]
            for r in(8,16,32,64):
                path=directory/('CI-r'+str(r)+'.json')
                if path.exists():result=load(path)
                else:
                    c,t=process_time(),perf_counter()
                    result=solve(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),ids,counts,r)
                    result.update(CPU_seconds=process_time()-c,wall_seconds=perf_counter()-t,
                        source_counts_SHA256=sha256(counts_path.read_bytes()),new_model_shots=0)
                    save(path,result)
                cis.append(result)
            resources={k:{'cnot_count':v['cnot_count'],'cnot_depth':v['cnot_depth'],'total_depth':v['total_unitary_depth']}
                for k,v in(('sampling_resources',views['Z_sampling_equivalent']),('preparation_resources',views['preparation']))}
            result={**config,**resources,'run_id':identity,'status':'COMPLETED','final_CI':cis,
                'pilot_model_shots':0,'final_model_shots':100000,'actual_QPU_shots':0,'bank_size':1,
                'coefficient_slots_shot_weighted_mean':slots,'coefficient_slots_distinct_member_sum':slots,
                'nonredundant_independent_parameters':None,'VQE_optimized_variables':0,
                'construction_CPU_seconds_this_campaign':0,'historical_construction_cost':construction,
                'shared_simulation':simulation,'attempt_CPU_seconds':process_time()-cp,
                'attempt_wall_seconds':perf_counter()-start,'ledger':ledger.snapshot(identity),
                'all_model_CNOT_applications':100000*resources['sampling_resources']['cnot_count'],
                'performance_claim_authorized':False,'fixed_operator_readout_not_independent_optimizer':True}
            save(filename,result)
            db.execute("UPDATE cells SET status='COMPLETED',result_path=? WHERE case_id=? AND method=? AND seed=?",
                (str(filename.relative_to(ROOT)),case,method,seed));db.commit()
            print(case,method,seed,'E64',cis[-1]['energy_Ha'],flush=True)
        db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--method',default='LUCJ_COMPRESSED_L1')
    a=p.parse_args();run(a.case,a.method)
