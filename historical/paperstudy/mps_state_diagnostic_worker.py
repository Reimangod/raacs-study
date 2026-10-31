"""Bounded <=20q fixed-state MPS diagnostic; never used for optimization.

No shots, CI, new coefficients, FCI reference, or saved full statevectors.
Large-system TN training is not certified by this small-system diagnostic.
"""
import hashlib,json,os,signal
from pathlib import Path
from time import perf_counter,process_time
import numpy as np


def little_endian(vector,nqubits):
    if nqubits>20:raise MemoryError('Full-state diagnostic restricted to20q')
    return np.asarray(vector).reshape([2]*nqubits).transpose(tuple(reversed(range(nqubits)))).reshape(-1)


def compare(exact,approx):
    exact=np.asarray(exact).reshape(-1);approx=np.asarray(approx).reshape(-1)
    en=float(np.vdot(exact,exact).real);an=float(np.vdot(approx,approx).real)
    if en<=0 or an<=0:raise ValueError('Nonzero states required')
    e,a=exact/np.sqrt(en),approx/np.sqrt(an);overlap=np.vdot(e,a)
    aligned=a/(overlap/abs(overlap))if abs(overlap)>0 else a
    return {'exact_norm_squared':en,'MPS_norm_squared':an,
        'normalized_state_infidelity':float(max(0.,1-abs(overlap)**2)),
        'normalized_Z_probability_L1':float(np.sum(abs(abs(e)**2-abs(a)**2))),
        'global_phase_aligned_L2':float(np.linalg.norm(e-aligned))}


def save(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def run(filename):
    import ffsim,quimb.tensor as qtn
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_quimb import quimb_circuit
    from importlib.metadata import version
    root=Path(filename).parent;request=json.loads(Path(filename).read_text());n=request['norb'];ne=tuple(request['nelec'])
    if 2*n>20:raise MemoryError('Never allocate full40q state')
    if hashlib.sha256((root/'operator.npz').read_bytes()).hexdigest()!=request['operator_sha256']:
        raise RuntimeError('Fixed selected operator changed')
    start,cpu=perf_counter(),process_time()
    # Independent circuit reference checks bit order, including entanglement
    # and complex amplitudes; not just an HF basis-state round trip.
    toy=QuantumCircuit(4);toy.x(0);toy.h(1);toy.cx(1,3);toy.ry(.37,2);toy.rz(.29,1)
    test=quimb_circuit(toy,quimb_circuit_class=qtn.CircuitMPS,max_bond=8,cutoff=1e-14,progbar=False)
    parity=compare(Statevector.from_instruction(toy).data,little_endian(test.to_dense(),4))
    if parity['global_phase_aligned_L2']>1e-10:raise RuntimeError('Independent MPS endian/state parity failed')
    with np.load(root/'operator.npz',allow_pickle=False)as a:
        op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
    exact_sector=ffsim.apply_unitary(ffsim.hartree_fock_state(n,ne),op,n,ne,copy=False)
    exact=ffsim.qiskit.ffsim_vec_to_qiskit_vec(exact_sector,n,ne)
    circuit=QuantumCircuit(2*n);circuit.append(ffsim.qiskit.PrepareHartreeFockJW(n,ne),range(2*n))
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(op),range(2*n));circuit=circuit.decompose(reps=2)
    def timeout(*_):raise TimeoutError('Per-chi diagnostic resource cap')
    signal.signal(signal.SIGALRM,timeout);rows=[]
    for chi in request['bond_grid']:
        tick,old_cpu=perf_counter(),process_time();signal.alarm(request['per_chi_seconds'])
        try:
            mps=quimb_circuit(circuit,quimb_circuit_class=qtn.CircuitMPS,max_bond=chi,
                cutoff=request['cutoff'],progbar=False)
            dense=little_endian(mps.to_dense(),2*n);values=compare(exact,dense)
            sector=ffsim.qiskit.qiskit_vec_to_ffsim_vec(dense,n,ne)
            values['valid_Nalpha_Nbeta_probability']=float(np.vdot(sector,sector).real/np.vdot(dense,dense).real)
            row={'chi':chi,'status':'COMPLETED',**values}
        except (TimeoutError,MemoryError)as e:row={'chi':chi,'status':'RESOURCE_CAPPED','failure':repr(e)}
        finally:signal.alarm(0)
        row.update(CPU_seconds=process_time()-old_cpu,wall_seconds=perf_counter()-tick);rows.append(row)
        save(root/('chi'+str(chi)+'.json'),row)
        if row['status']!='COMPLETED':break
    save(root/'result.json',{'status':'BOUNDED_FIXED_20Q_MPS_DIAGNOSTIC_TERMINAL','request':request,
        'rows':rows,'independent_bit_order_parity':parity,'no_new_model_shots':True,'actual_QPU_shots':0,
        'new_optimizer_calls':0,'statevectors_saved':False,'other_cases_certified':False,
        'runtime':{p:version(p)for p in('ffsim','qiskit','quimb','qiskit-quimb')},
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-start,'Slurm_job_id':os.environ.get('SLURM_JOB_ID')})


if __name__=='__main__':
    import sys;run(sys.argv[1])
