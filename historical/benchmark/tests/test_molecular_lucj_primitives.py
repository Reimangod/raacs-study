import json
import ffsim
import numpy as np
from qiskit.quantum_info import Statevector
from benchmark.common import ROOT
from benchmark.resource_adaptive_sampler import integral_problem
from benchmark.extended_molecular_problems import ExtendedMolecularSpec,build_extended_problem
from benchmark.molecular_lucj_primitives import fixed_integral_reference,initial_amplitudes,ucj_from_amplitudes,ucj_circuits
from benchmark.molecular_mps_scaling import no_full_state_guards
from benchmark.chemistry import determinant_ids


def physical_parity(backend,info,t1,t2):
    op,params,_=ucj_from_amplitudes(t1,t2,backend.norb,info['nalpha']==info['nbeta'])
    prep,sample,_=ucj_circuits(op,backend.norb,(info['nalpha'],info['nbeta']))
    vec=ffsim.apply_unitary(ffsim.hartree_fock_state(backend.norb,(info['nalpha'],info['nbeta'])),op,
        norb=backend.norb,nelec=(info['nalpha'],info['nbeta']))
    ids,signs,_,_=determinant_ids(backend.norb,info['nalpha'],info['nbeta'])
    state=np.zeros(1<<info['n_qubits'],complex);state[ids]=vec*signs
    actual=Statevector.from_instruction(prep).data;phase=np.vdot(actual,state);phase/=abs(phase)
    assert np.linalg.norm(state-phase*actual)<1e-8
    assert np.max(abs(abs(state)**2-abs(Statevector.from_instruction(sample).data)**2))<1e-9
    assert len(params)>0


def test_same_active_integral_CCSD_matches_saved_native_small_reference(tmp_path):
    h,info=integral_problem(ROOT/'results/standard-v1/references/H4_R1.1')
    with no_full_state_guards():
        mf,qualification=fixed_integral_reference(h,info)
        assert qualification['HF_energy_parity_error_Ha']<1e-9
        t1,t2,record=initial_amplitudes(h,info,tmp_path,'ccsd')
    with np.load(ROOT/'results/standard-v1/references/H4_R1.1/ccsd.npz') as old:
        assert np.max(abs(t2-old['t2']))<1e-6
        assert np.max(abs(t1-old['t1']))<1e-6
    physical_parity(h,info,t1,t2)
    assert record['status']=='COMPLETED' and record['quantum_shots']==0


def test_open_shell_UCCSD_UMP2_and_native_unbalanced_UCJ_not_balanced_proxy(tmp_path):
    problem=tmp_path/'physical'
    with no_full_state_guards():
        build_extended_problem(ExtendedMolecularSpec('CH2',1.1,'valence',spin=2),problem)
        h,info=integral_problem(problem)
        for kind in ('mp2','ccsd'):
            t1,t2,record=initial_amplitudes(h,info,tmp_path/'amplitudes',kind)
            assert isinstance(t2,tuple) and len(t2)==3
            assert record['qualification']['HF_energy_parity_error_Ha']<1e-9
    physical_parity(h,info,t1,t2)
    assert record['correlation_method']=='UCCSD'
