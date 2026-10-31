"""One engineering successor; same bounded P2-inspired objective as v1.

Adds per-trial coefficients, atomic best-state checkpoints, NOMAD MAX_TIME,
and actual failed-call sampling evidence. It does not change the molecular
objective, MPS accuracy, initial operator, angles, seed or final readouts.
"""
import hashlib,json,os,threading
from pathlib import Path
from time import perf_counter,process_time
import ffsim,numpy as np,PyNomad
from qiskit import QuantumCircuit
import quimb.tensor as qtn
from qiskit_quimb import quimb_circuit
from paperstudy.ci import solve
from tn_checkpoint import save,arrays,publish_best,load_best,atomic,sha


def operator_arrays(op):return {'diag_coulomb_mats':op.diag_coulomb_mats,'orbital_rotations':op.orbital_rotations,
                               'final_orbital_rotation':op.final_orbital_rotation}


def run(filename):
    root=Path(filename).parent;config=json.loads(Path(filename).read_text())
    if(root/'optimization-result.json').exists():return
    for name,expected in config['input_sha256'].items():
        if sha((root/name).read_bytes())!=expected:raise RuntimeError('TN successor input changed')
    n=config['norb'];ne=tuple(config['nelec']);pairs=tuple([tuple(p)for p in v]for v in config['pairs'])
    with np.load(root/'integrals.npz',allow_pickle=False)as a:h,g=a['h1'].copy(),a['eri'].copy()
    with np.load(root/'initial-operator.npz',allow_pickle=False)as a:initial=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
    theta=initial.to_parameters(interaction_pairs=pairs);start,cpu=perf_counter(),process_time()
    lock=threading.Lock();records=[];best=[None];rejected=[0];draws=[0]
    directory=root/'surrogate-evaluations';directory.mkdir(exist_ok=True)
    def objective(point):
        x=np.array([point.get_coord(i)for i in range(point.size())])
        with lock:
            if len(records)>=500 or perf_counter()-start>3000:rejected[0]+=1;return 0
            i=len(records);records.append({'index':i,'status':'RUNNING'})
            theta_sha=sha(x.astype('<f8').tobytes());arrays(directory/(str(i)+'-theta.npz'),{'theta':x})
            save(directory/(str(i)+'-started.json'),{'index':i,'theta_sha256':theta_sha,'config_id':config['config_id'],
                'started_wall_seconds':perf_counter()-start})
            save(root/'phase.json',{'started':len(records),'completed':sum(r['status']=='COMPLETED'for r in records),'elapsed_wall_seconds':perf_counter()-start})
        tick=perf_counter();observed_count=None
        try:
            op=ffsim.UCJOpSpinBalanced.from_parameters(x,norb=n,n_reps=1,interaction_pairs=pairs,with_final_orbital_rotation=True)
            q=QuantumCircuit(2*n);q.append(ffsim.qiskit.PrepareHartreeFockJW(n,ne),range(2*n));q.append(ffsim.qiskit.UCJOpSpinBalancedJW(op),range(2*n))
            circuit=quimb_circuit(q.decompose(reps=2),quimb_circuit_class=qtn.CircuitMPS,max_bond=50,cutoff=1e-10,progbar=False)
            observed=list(circuit.sample(10000,seed=config['surrogate_seed']))
            ids,counts=np.unique(np.array([int(s[::-1],2)for s in observed],np.uint64),return_counts=True)
            counts_sha=arrays(directory/(str(i)+'-counts.npz'),{'blocked_ids':ids,'counts':counts.astype(np.uint64)})
            observed_count=int(counts.sum())
            with lock:draws[0]+=observed_count
            ci=solve(h,g,config['core_energy'],ne,ids,counts.astype(np.uint64),32,seconds=120)
            value=ci['energy_Ha'];record={'index':i,'status':'COMPLETED','theta_sha256':theta_sha,'CI':ci,
                'surrogate_samples':observed_count,'counts_sha256':counts_sha,'wall_seconds':perf_counter()-tick,'actual_QPU_shots':0}
            with lock:
                records[i]=record;save(directory/(str(i)+'.json'),record)
                if best[0]is None or value<best[0]:
                    publish_best(root,config,record,x,operator_arrays(op));best[0]=value
                save(root/'progress.json',{'started':len(records),'completed':sum(r['status']=='COMPLETED'for r in records),
                    'best_surrogate_energy_Ha':best[0],'actual_classical_surrogate_samples':draws[0],
                    'elapsed_wall_seconds':perf_counter()-start,'CPU_seconds':process_time()-cpu,'not_independent_final_result':True})
            point.setBBO(str(value).encode());return 1
        except Exception as exc:
            with lock:
                records[i]={'index':i,'status':'FAILED_PRESERVED','theta_sha256':theta_sha,'error':repr(exc),
                    'surrogate_samples_known':observed_count,'unknown_is_not_zero':True,'wall_seconds':perf_counter()-tick}
                save(directory/(str(i)+'.json'),records[i])
            return 0
    options=['BB_OUTPUT_TYPE OBJ','MAX_BB_EVAL 500','MAX_TIME 3000','DISPLAY_DEGREE 0','PSD_MADS_OPTIMIZATION true',
        'PSD_MADS_NB_VAR_IN_SUBPROBLEM 20','PSD_MADS_SUBPROBLEM_MAX_BB_EVAL 20','PSD_MADS_NB_SUBPROBLEM 3',
        'NB_THREADS_PARALLEL_EVAL 4','SEED '+str(config['optimizer_seed']),'CACHE_FILE '+str(root/'nomad-cache.txt')]
    save(root/'optimizer-started.json',{'configuration':config,'options':options,'initial_coefficients':theta.tolist(),'pid':os.getpid()})
    try:result=PyNomad.optimize(objective,theta.tolist(),[],[],options);optimizer_error=None
    except Exception as exc:result=None;optimizer_error=repr(exc)
    pointer=load_best(root,config) # Never substitute the initial state if no completed objective exists.
    atomic(root/'optimized-operator.npz',(root/pointer['operator']).read_bytes())
    save(root/'optimization-result.json',{'status':'TN_COMMON_READOUT_SURROGATE_RETURNED','configuration':config,'options':options,
        'problem_id':config['problem_id'],'optimizer_result':result,'optimizer_error':optimizer_error,
        'optimizer_convergence_not_claimed':True,'selected_from_completed_bounded_objectives':True,
        'source_initial_coefficients':theta.tolist(),'selected_coefficients':pointer['theta'],
        'operator_sha256':pointer['operator_sha256'],'checkpoint':pointer,'objective_records':records,
        'best_surrogate_energy_Ha':pointer['surrogate_energy_Ha'],'surrogate_samples':draws[0],
        'incomplete_MPS_draws_not_persisted_are_unknown':True,'actual_QPU_shots':0,'final_exact_sampling_still_required':True,
        'primitive_boundary_rejections':rejected[0],'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-start,
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'paper_native_Dice_objective_reproduced':False,
        'failed_original_attempt_cost_not_erased':True})


if __name__=='__main__':
    import sys;run(sys.argv[1])
