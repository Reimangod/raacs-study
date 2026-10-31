import argparse
import fcntl
from io import BytesIO
from time import time,process_time,perf_counter
import numpy as np
import ffsim
from qiskit import qpy
from paperstudy.ci import common_strings,interleaved_to_blocked
from raacsnext.run import physical,summarize_members
from raacsnext.sampler import merge_counts
from .paths import ROOT
from .io import load,save,write,arrays,sha256,digest,space
from .journal import Journal
from .spin_beam import SpinBeamEngine
from .singlet import basis,diagonalize,optimize
from .run_singlet_orbital import freeze as readout_freeze


def code_hashes():
    names=('bank.py','beam.py','journal.py','spin_beam.py','singlet_cache.py','singlet.py','run_b4.py')
    return {name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in names}


def freeze():
    return {'files':code_hashes(),'protocol_SHA256':sha256((ROOT/'protocol-B4.json').read_bytes()),
        'final_OO_protocol_SHA256':sha256((ROOT/'protocol-C4-SINGLET-OO.json').read_bytes()),
        'readout_dependencies':readout_freeze(),
        'qualification_test_SHA256':sha256((ROOT/'test_singlet_cache.py').read_bytes())}


def run(case,seed):
    spec=load(ROOT/'protocol-B4.json');oo_spec=load(ROOT/spec['final_OO_protocol'])
    if case not in spec['cases'] or seed not in spec['seeds']:
        raise ValueError('Unregistered B4 cell')
    if freeze()!=load(ROOT/'freeze-B4.json'):
        raise RuntimeError('B4 code/protocol drift')
    m,h,g=physical(case);budget=spec['total_model_shots_per_run']
    request={'cycle':'B4','variant':'EXACT_SINGLET_BEAM','case_id':case,'problem_id':m['problem_id'],
        'seed':seed,'budget':budget,'freeze':freeze()}
    identity=digest(request);directory=ROOT/'runs/B4'/case/str(seed)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'request.json').exists() and load(directory/'request.json')!=request:
            raise RuntimeError('B4 request mismatch')
        if (directory/'result.json').exists():
            return load(directory/'result.json')
        if (directory/'started.json').exists():
            raise RuntimeError('Partial B4 preserved; no blind retry')
        space();save(directory/'request.json',request)
        save(directory/'started.json',{'pid':__import__('os').getpid(),'unix':time()})
        cpu,tick=process_time(),perf_counter();j=Journal(m,directory,identity,budget)
        try:
            output=SpinBeamEngine(m,h,g,j,directory,budget,spec['construction_CPU_seconds']).run()
            final=output.pop('final_members');observed=merge_counts([x['member']['observed'] for x in final])
            ids=np.array([interleaved_to_blocked(x,m['norb']) for x in sorted(observed)],np.uint64)
            counts=np.array([observed[x] for x in sorted(observed)],np.uint64)
            payload=arrays(blocked_ids=ids,counts=counts)
            write(directory/'final-counts.npz',payload)
            cis=[];ham=ffsim.MolecularHamiltonian(h,g,constant=m['core_energy'])
            for r in spec['final_r_caps']:
                c,t=process_time(),perf_counter()
                strings,_=common_strings(ids,counts,m['norb'],(m['nalpha'],m['nbeta']),r)
                q,info=basis(strings,m['norb'],(m['nalpha'],m['nbeta']))
                value,_=diagonalize(ham,(m['nalpha'],m['nbeta']),strings,q,oo_spec)
                value.update(r_cap=r,actual_d=len(strings)**2,spin_basis=info,
                    CPU_seconds=process_time()-c,wall_seconds=perf_counter()-t,new_model_shots=0,
                    source_counts_SHA256=sha256(payload))
                save(directory/('CI-S0-r'+str(r)+'.json'),value);cis.append(value)
            # Same registered final orbital processing as C4, not part of candidate selection.
            def checkpoint(history,rotations):
                save(directory/'OO-checkpoint.json',{'history':history,'new_quantum_model_shots':0})
                write(directory/'OO-rotations-checkpoint.npz',arrays(rotations=np.array(rotations)))
            oo=optimize(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),strings,oo_spec,checkpoint)
            rotations,hh,gg=(oo.pop(k) for k in ('rotations','final_h1','final_eri'))
            representation=arrays(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=strings)
            write(directory/'optimized-representation.npz',representation)
            oo['representation_SHA256']=sha256(representation)
            if abs(oo['initial_energy_Ha']-cis[-1]['energy_Ha'])>1e-8:
                raise RuntimeError('B4 readout and OO initial state mismatch')
            ledger=j.ledger.snapshot(identity)
            if ledger['known_actual_shots']!=budget or ledger['unknown_or_inflight_draws'] or int(counts.sum())+output['pilot_model_shots']!=budget:
                raise RuntimeError('B4 paid-shot accounting mismatch')
            members=[]
            for x in final:
                stream=BytesIO();qpy.dump(x['member']['circuit'],stream)
                write(directory/'final-circuits'/(x['member']['id']+'.qpy'),stream.getvalue())
                members.append({'id':x['member']['id'],'actions':x['member']['actions'],'shots':x['shots'],
                    'sampling_resources':x['member']['metadata']['sampling_resources'],
                    'preparation_resources':x['member']['metadata']['preparation_resources']})
            result={**request,**output,**summarize_members(final),'run_id':identity,'status':'COMPLETED',
                'members':members,'ledger':ledger,'final_CI':cis,'final_model_shots':int(counts.sum()),
                'final_counts_SHA256':sha256(payload),'cost':j.cost(),'actual_QPU_shots':0,
                'final_singlet_OO':oo,'final_state_is_classical_CI_not_sampler_state':True,
                'run_body_CPU_seconds':process_time()-cpu,'run_body_wall_seconds':perf_counter()-tick,
                'independent_final_not_pilot_counts':True,'performance_claim_authorized':False}
            save(directory/'result.json',result)
            print(case,'B4',seed,'E64',cis[-1]['energy_Ha'],'OO',oo['final_energy_Ha'],
                'depth',output['completed_depth'],'stop',output['construction_stop_reason'],flush=True)
            return result
        except Exception as exc:
            save(directory/'failure.json',{'error':repr(exc),'CPU_seconds':process_time()-cpu,
                'wall_seconds':perf_counter()-tick,'ledger':j.ledger.snapshot(identity)})
            raise
        finally:
            j.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true')
    p.add_argument('--case');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.freeze:
        path=ROOT/'freeze-B4.json';value=freeze()
        if path.exists() and load(path)!=value:
            raise RuntimeError('Never replace B4 freeze')
        save(path,value)
    else:
        run(a.case,a.seed)
