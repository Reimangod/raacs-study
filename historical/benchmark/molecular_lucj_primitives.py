"""Native ffsim balanced/unbalanced LUCJ from the exact saved active Hamiltonian.

PySCF custom-integral HF objects are fixed references, not new orbital SCF.
There is no FCI solver, sector-state simulation or full-Fock allocation here.
"""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import ffsim
import numpy as np
from pyscf import ao2mo,cc,gto,mp,scf
from qiskit import QuantumCircuit
from .common import digest,file_digest,save_json,save_npz
from .ansatz import compile_logical,measurement_equivalent_circuit
from .study.core import rss_peak


def fixed_integral_reference(backend,info):
    """Embed the same active h1/eri/core and HF determinant in PySCF.

    For open shell this is the shared ROHF spatial gauge embedded in UHF form;
    no claim that this fixed UHF reference is independently UHF-stationary.
    """
    n=backend.norb;na,nb=info['nalpha'],info['nbeta']
    mol=gto.M(verbose=0);mol.nelectron=na+nb;mol.spin=na-nb;mol.incore_anyway=True
    mol.nao_nr=lambda *args:n
    mf=scf.RHF(mol) if na==nb else scf.UHF(mol)
    mf.max_memory=768;mf.get_hcore=lambda *args:backend.h1;mf.get_ovlp=lambda *args:np.eye(n)
    mf.energy_nuc=lambda *args:backend.core;mf._eri=ao2mo.restore(8,backend.eri,n)
    if na==nb:
        mf.mo_coeff=np.eye(n);mf.mo_occ=np.array([2.]*na+[0.]*(n-na))
    else:
        mf.mo_coeff=np.stack([np.eye(n),np.eye(n)])
        mf.mo_occ=np.array([[1.]*na+[0.]*(n-na),[1.]*nb+[0.]*(n-nb)])
    dm=mf.make_rdm1();fock=mf.get_fock(dm=dm)
    mf.mo_energy=np.diag(fock) if na==nb else np.stack([np.diag(x) for x in fock])
    actual=float(mf.energy_tot(dm=dm));mf.e_tot=actual
    if abs(actual-info['HF_energy_Ha'])>1e-8:raise RuntimeError('Fixed active-space HF embedding mismatch')
    ov=(float(np.linalg.norm(fock[:na,na:])) if na==nb else
        max(float(np.linalg.norm(fock[0,:na,na:])),float(np.linalg.norm(fock[1,:nb,nb:]))))
    mf.converged=ov<1e-6
    if na==nb and not mf.converged:raise RuntimeError('Saved closed-shell reference is not stationary in its active MO gauge')
    return mf,{'fixed_HF_embedding_energy_Ha':actual,'HF_energy_parity_error_Ha':abs(actual-info['HF_energy_Ha']),
        'occupied_virtual_Fock_norm':ov,'UHF_stationary':bool(mf.converged) if na!=nb else None,
        'no_new_SCF_orbital_optimization':True,'reference_scope':'closed RHF' if na==nb else 'fixed ROHF gauge, UHF-form post-HF embedding'}


def initial_amplitudes(backend,info,folder,kind,cycles=150,seconds=180.):
    if kind not in ('mp2','ccsd'):raise ValueError('MP2 or CCSD')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    binding={'problem':info['problem_id'],'kind':kind,'max_cycles':cycles,'convergence_tolerance':1e-8,
             'wall_boundary_seconds':seconds,'implementation_sha256':file_digest(__file__)}
    path=folder/(kind+'.json');archive=folder/(kind+'.npz')
    if path.exists():
        out=json.loads(path.read_text())
        if out['binding']!=binding:raise RuntimeError('Amplitude dependency drift')
        if out['status']!='COMPLETED':raise RuntimeError('Preserved amplitude initialization failure: '+str(path))
        if file_digest(archive)!=out['archive_sha256']:raise RuntimeError('Amplitude archive mismatch')
        with np.load(archive) as z:
            if info['nalpha']==info['nbeta']:t1,t2=z['t1'],z['t2']
            else:t1=(z['t1a'],z['t1b']);t2=(z['t2aa'],z['t2ab'],z['t2bb'])
        return t1,t2,out
    start=perf_counter();out={'binding':binding,'quantum_shots':0,'FCI_used':False,'hardware_shots':0}
    try:
        mf,qualification=fixed_integral_reference(backend,info);balanced=info['nalpha']==info['nbeta']
        if kind=='mp2':
            solver=mp.MP2(mf);solver.max_memory=768;solver.kernel();t2=solver.t2
            if balanced:t1=np.zeros((info['nalpha'],backend.norb-info['nalpha']))
            else:t1=tuple(np.zeros((k,backend.norb-k)) for k in (info['nalpha'],info['nbeta']))
            converged=True
        else:
            solver=cc.CCSD(mf);solver.max_memory=768;solver.max_cycle=cycles;solver.conv_tol=1e-8
            def callback(env):
                if perf_counter()-start>seconds or rss_peak()>2*2**30:raise RuntimeError('CCSD cycle-boundary wall/RSS cap')
            solver.callback=callback;solver.kernel();converged=bool(solver.converged);t1,t2=solver.t1,solver.t2
        if not balanced:
            # PySCF UMP2 currently returns a list, UCCSD a tuple. Normalize the
            # public aa/ab/bb contract without altering any numerical array.
            t1,t2=tuple(t1),tuple(t2)
        arrays=([t1,t2] if balanced else [*t1,*t2])
        if not all(np.isfinite(x).all() for x in arrays):raise RuntimeError('Nonfinite correlation amplitudes')
        if not converged:raise RuntimeError('CCSD did not converge; do not initialize LUCJ from failed amplitudes')
        if balanced:save_npz(archive,t1=t1,t2=t2)
        else:save_npz(archive,t1a=t1[0],t1b=t1[1],t2aa=t2[0],t2ab=t2[1],t2bb=t2[2])
        out.update(status='COMPLETED',archive_sha256=file_digest(archive),qualification=qualification,
                   correlation_method=type(solver).__name__,total_energy_Ha=float(solver.e_tot),
                   singles_policy='zero singles for MP2 double-amplitude factorization' if kind=='mp2' else 'converged CCSD singles',
                   amp_shapes=[list(a.shape) for a in arrays],converged=converged)
    except Exception as exc:
        out.update(status='INITIALIZATION_FAILURE_PRESERVED',error=repr(exc),partial_classical_work_preserved=True)
        out['wall_seconds']=perf_counter()-start;out['peak_RSS']=rss_peak();save_json(path,out);raise
    out['wall_seconds']=perf_counter()-start;out['peak_RSS']=rss_peak();save_json(path,out)
    return t1,t2,out


def ucj_pairs(norb,balanced):
    same=[(i,i+1) for i in range(norb-1)];opposite=[(i,i) for i in range(norb)]
    return (same,opposite) if balanced else (same,opposite,same)


def ucj_from_amplitudes(t1,t2,norb,balanced,layers=2):
    pairs=ucj_pairs(norb,balanced);cls=ffsim.UCJOpSpinBalanced if balanced else ffsim.UCJOpSpinUnbalanced
    op=cls.from_t_amplitudes(t2,t1=t1,n_reps=layers,interaction_pairs=pairs,optimize=False)
    if op.final_orbital_rotation is None:
        final=np.eye(norb) if balanced else np.stack([np.eye(norb),np.eye(norb)])
        op=cls(op.diag_coulomb_mats,op.orbital_rotations,final)
    return op,np.asarray(op.to_parameters(interaction_pairs=pairs),float),pairs


def ucj_from_parameters(parameters,norb,balanced,layers=2):
    cls=ffsim.UCJOpSpinBalanced if balanced else ffsim.UCJOpSpinUnbalanced
    return cls.from_parameters(np.asarray(parameters),norb=norb,n_reps=layers,interaction_pairs=ucj_pairs(norb,balanced),with_final_orbital_rotation=True)


def ucj_circuits(op,norb,nelec):
    balanced=isinstance(op,ffsim.UCJOpSpinBalanced)
    gate=ffsim.qiskit.UCJOpSpinBalancedJW(op) if balanced else ffsim.qiskit.UCJOpSpinUnbalancedJW(op)
    blocked=QuantumCircuit(2*norb);blocked.append(ffsim.qiskit.PrepareHartreeFockJW(norb,nelec),range(2*norb));blocked.append(gate,range(2*norb))
    wires=[2*norb-1-2*i for i in range(norb)]+[2*norb-2-2*i for i in range(norb)]
    prep=QuantumCircuit(2*norb);prep.compose(blocked,qubits=wires,inplace=True)
    for a in range(norb):
        for b in range(a):prep.cz(2*norb-1-2*a,2*norb-2-2*b)
    sampling,removed=measurement_equivalent_circuit(prep)
    return compile_logical(prep),sampling,removed
