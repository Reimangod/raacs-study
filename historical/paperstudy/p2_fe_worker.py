"""Fe P2 source-amplitude anchor: exact sampling and original batch settings.

Current SDK/PySCF replaces unavailable historical Dice. The difference remains
explicit; no smaller common-CI substitute or reuse of fresh C-lane amplitudes.
"""
import hashlib,json,os,gc
from pathlib import Path
from time import perf_counter,process_time
from unittest.mock import patch
import numpy as np
import ffsim
from pyscf.fci import selected_ci
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian,solve_sci_batch


def save(path,value):
    p=path.with_suffix('.pending');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)


class TimedCI(selected_ci.SelectedCI):
    def __init__(self,deadline):super().__init__();self.deadline=deadline
    def contract_2e(self,*args,**kwargs):
        if perf_counter()>self.deadline:raise TimeoutError('Paper anchor CI matvec boundary')
        return super().contract_2e(*args,**kwargs)


def run(filename):
    root=Path(filename).parent;config=json.loads(Path(filename).read_text())
    if(root/'result.json').exists():return
    for name,expected in config['input_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('P2 Fe input mismatch')
    n,ne=config['norb'],tuple(config['nelec']);start,cpu=perf_counter(),process_time();deadline=start+3000
    with np.load(root/'integrals.npz',allow_pickle=False)as z:h,g=z['h1'].copy(),z['eri'].copy()
    with np.load(root/'operator.npz',allow_pickle=False)as z:
        op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
    tick=perf_counter();vec=ffsim.apply_unitary(ffsim.hartree_fock_state(n,ne),op,n,ne,copy=False)
    simulation=perf_counter()-tick
    samples=ffsim.sample_state_vector(vec,norb=n,nelec=ne,shots=100000,bitstring_type=ffsim.BitstringType.INT,seed=0)
    ids,counts=np.unique(samples,return_counts=True)
    with(root/'raw-counts.npz').open('wb')as f:np.savez_compressed(f,blocked_ids=ids.astype(np.uint64),counts=counts.astype(np.uint64))
    save(root/'sampling.json',{'problem_id':config['problem_id'],'model_shots':100000,'actual_QPU_shots':0,
        'state_norm_squared':float(np.vdot(vec,vec).real),'exact_simulation_seconds':simulation,
        'counts_sha256':hashlib.sha256((root/'raw-counts.npz').read_bytes()).hexdigest(),'observed_unique':len(ids)})
    del vec;gc.collect()
    pool=BitArray.from_samples(samples,num_bits=2*n);del samples
    history=[];checks=[]
    def bounded(strings,one,two,norb,nelec):
        dimensions=[len(a)*len(b)for a,b in strings];r=max(max(len(a),len(b))for a,b in strings)
        estimate=max(dimensions)*8*32+r*norb*norb*max(nelec)*8*20+norb**4*8*16
        checks.append({'actual_batch_dimensions':dimensions,'working_bytes_estimate':estimate})
        save(root/'subspace-checks.json',{'checks':checks})
        if estimate>8192*2**20:raise MemoryError('Original P2 batch spaces exceed allocated workspace; no dimension reduction')
        with patch.object(selected_ci,'SelectedCI',lambda:TimedCI(deadline)):
            return solve_sci_batch(strings,one,two,norb,nelec,spin_sq=None,max_cycle=200,tol=1e-10,
                max_memory=8192,verbose=0,davidson_only=True)
    def callback(results):
        history.append({'batch_energies_Ha':[float(r.energy)+config['core_energy']for r in results],
                        'actual_batch_dimensions':[int(r.sci_state.amplitudes.size)for r in results]})
        save(root/'history.json',{'iterations':history})
    try:
        result=diagonalize_fermionic_hamiltonian(h,g,pool,samples_per_batch=4000,norb=n,nelec=ne,num_batches=10,
            max_iterations=1,energy_tol=1e-5,occupancies_tol=1e-3,sci_solver=bounded,symmetrize_spin=True,
            carryover_threshold=1e-3,max_dim=4000,callback=callback,seed=0)
        energies=history[-1]['batch_energies_Ha']
        status='P2_FE_SOURCE_SETTINGS_CURRENT_RUNTIME_COMPLETED';failure=None
    except(TimeoutError,MemoryError)as exc:
        status='P2_FE_NATIVE_BATCH_RESOURCE_CAPPED_PRESERVED';failure=repr(exc);energies=[]
    save(root/'result.json',{'status':status,'case_id':config['case_id'],'kind':config['kind'],
        'configuration':config,'problem_id':config['problem_id'],'model_shots':100000,'actual_QPU_shots':0,
        'history':history,'failure':failure,'mean_energy_Ha':float(np.mean(energies))if energies else None,
        'minimum_energy_Ha':min(energies)if energies else None,'maximum_energy_Ha':max(energies)if energies else None,
        'batch_range_is_not_confidence_interval':True,'source_solver_deviation':'Current PySCF instead of author Dice',
        'published_numeric_match_verified':False,'full_historical_runtime_reproduced':False,
        'wall_seconds':perf_counter()-start,'CPU_seconds':process_time()-cpu,'slurm_job_id':os.environ.get('SLURM_JOB_ID')})


if __name__=='__main__':
    import sys;run(sys.argv[1])
