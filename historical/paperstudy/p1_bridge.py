"""Bounded P1-settings bridge on the documented naphthalene reconstruction.

Source shot allocation and lengths/repetitions are exercised, but the native
SDK grouping/current compiler is not the unavailable original author driver.
Do not call these numerical P1 reproductions or primary matched-budget cells.
"""
import json
from pathlib import Path
from time import perf_counter,process_time
import numpy as np,ffsim
from .common_execution import load_physical,final_readouts
from .hardware_postprocess_worker import projected
from .input_adapter import array_bytes
from .io import ROOT,save_json,atomic_bytes,digest,sha256
from .lucj import naive,amplitudes_from_saved,interaction_pairs,make_circuit,circuit_views
from .sensitivity_sqdrift import run as sqdrift,allocations

CASE='NAPH20-P1';B=768000;SEED=320001


def full_readout(source,out):
    filename=out/'full-Cartesian-CI.json'
    if filename.exists():return
    m,h,g=load_physical(CASE);n=m['norb']
    with np.load(source,allow_pickle=False)as z:ids,counts=z['blocked_ids'],z['counts']
    bits=((ids[:,None]>>np.arange(2*n-1,-1,-1,dtype=np.uint64))&1).astype(bool)
    start,cpu=perf_counter(),process_time()
    try:
        result,_=projected(h,g,m['core_energy'],(m['nalpha'],m['nbeta']),bits,start+600,63504)
        status='ALL_OBSERVED_SPIN_STRINGS_CURRENT_CARTESIAN_CI_COMPLETED'
    except(TimeoutError,MemoryError)as exc:result={'failure':repr(exc)};status='RESOURCE_CAPPED_PREFIX_PRESERVED'
    save_json(filename,{'status':status,'CI':result,'source_counts_sha256':sha256(source.read_bytes()),
        'new_model_shots':0,'source_P1_Dice_postprocessing_reproduced':False,
        'postprocessing':'All observed spin strings, spin-symmetrized Cartesian closure, current PySCF',
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-start})


def lucj(L):
    out=ROOT/'paper-anchors/P1-bridge-v1'/('LUCJ-L'+str(L));out.mkdir(parents=True,exist_ok=True)
    resultfile=out/'result.json'
    if not resultfile.exists():
        m,h,g=load_physical(CASE);n=m['norb'];ne=(m['nalpha'],m['nbeta']);tick,cpu=perf_counter(),process_time()
        t1,t2,kind=amplitudes_from_saved(ROOT/'data/initialization-fresh'/CASE/'amplitudes.npz',n,ne[0])
        op=naive(t1,t2,n,L);_,_,views=circuit_views(make_circuit(op,n,ne))
        atomic_bytes(out/'operator.npz',array_bytes({'diag_coulomb_mats':op.diag_coulomb_mats,
            'orbital_rotations':op.orbital_rotations,'final_orbital_rotation':op.final_orbital_rotation}))
        save_json(out/'request.json',{'case':CASE,'problem_id':m['problem_id'],'n_reps':L,'model_shots':B,'seed':SEED,
            'source':'P1 v3 Fig6; CCSD-initialized, unoptimized LUCJ L=1/7/22',
            'original_ffsim_0p0p57_reproduced':False,'current_ffsim':'0.0.84','original_author_integrals_byte_match':False})
        state=ffsim.apply_unitary(ffsim.hartree_fock_state(n,ne),op,n,ne,copy=False)
        samples=ffsim.sample_state_vector(state,norb=n,nelec=ne,shots=B,bitstring_type=ffsim.BitstringType.INT,seed=SEED)
        ids,counts=np.unique(samples,return_counts=True)
        atomic_bytes(out/'final-counts.npz',array_bytes({'blocked_ids':ids.astype(np.uint64),'counts':counts.astype(np.uint64)}))
        readouts=final_readouts(out,m,h,g,ids,counts)
        save_json(resultfile,{'status':'P1_LUCJ_SETTINGS_CURRENT_RUNTIME_BRIDGE_COMPLETED','case_id':CASE,'n_reps':L,
            'problem_id':m['problem_id'],'model_shots':B,'actual_QPU_shots':0,'final_CI':readouts,'resource_views':views,
            'coefficient_slots':len(op.to_parameters(interaction_pairs=interaction_pairs(n))),
            'physical_nonredundant_parameter_dimension':None,'initialization':kind,'VQE_optimized_variables':0,
            'published_curve_match_verified':False,'whole_run_replicates':1,'CPU_seconds':process_time()-cpu,
            'wall_seconds':perf_counter()-tick,'model_CNOT_applications':B*views['Z_sampling_equivalent']['cnot_count']})
    full_readout(out/'final-counts.npz',out);print('P1_LUCJ_BRIDGE',L,flush=True)


def main():
    import fcntl
    with(ROOT/'P1-bridge-controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec={'case':CASE,'B':B,'seed':SEED,'SqDRIFT_lengths':[5,20,50],'k':[1,2,3],
            'randomizations_per_group':500,'shots_per_member':512,'LUCJ_repetitions':[1,7,22],
            'readout':'Common small r for crosswalk, separately full observed Cartesian CI, zero additional shots',
            'P1_author_pipeline_reproduced':False,'original_grouping_driver_unavailable':True,
            'code_sha256':sha256(Path(__file__).read_bytes())}
        path=ROOT/'paper-anchors/P1-bridge-v1/spec.json'
        if path.exists()and json.loads(path.read_text())!=spec:raise RuntimeError('P1 bridge spec changed')
        save_json(path,spec)
        for L in(1,7,22):lucj(L)
        for length in(5,20,50):
            allocation=allocations(B,SEED,length,512)
            if len(allocation)!=1500 or any(M!=512 for _,M in allocation):raise RuntimeError('Not the source shot allocation')
            result=sqdrift(CASE,B,SEED,length,512,lane='P1_SETTINGS_BRIDGE_NOT_REPRODUCTION')
            source=ROOT/'common-runs/P1_SETTINGS_BRIDGE_NOT_REPRODUCTION'/CASE/result['method_id']/(str(B)+'-'+str(SEED))
            out=ROOT/'paper-anchors/P1-bridge-v1'/('SqDRIFT-N'+str(length));out.mkdir(parents=True,exist_ok=True)
            save_json(out/'source-result.json',{'result':str((source/'result.json').relative_to(ROOT)),
                'source_result_sha256':sha256((source/'result.json').read_bytes()),'original_author_grouping_verified':False})
            full_readout(source/'final-counts.npz',out);print('P1_SQDRIFT_BRIDGE',length,flush=True)
        save_json(ROOT/'paper-anchors/P1-bridge-v1/terminal.json',{'settings_points_completed':6,
            'original_paper_reproduction_claimed':False,'main_comparison_complete':False})


if __name__=='__main__':main()
