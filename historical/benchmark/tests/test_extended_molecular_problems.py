import numpy as np
from benchmark.extended_molecular_problems import ExtendedMolecularSpec,build_extended_problem,extended_queue
from benchmark.molecular_mps_scaling import no_full_state_guards
from benchmark.resource_adaptive_sampler import integral_problem


def test_geometry_and_declared_space_queue():
    assert len(extended_queue('quick'))==4 and len(extended_queue('standard'))==10
    s=ExtendedMolecularSpec('C6H6',1.4,'pi');xyz=s.geometry()
    assert len(xyz)==12 and all(abs(z[2])<1e-12 for _,z in xyz)
    assert np.isclose(np.linalg.norm(np.array(xyz[0][1])-np.array(xyz[1][1])),1.4)
    rectangle=ExtendedMolecularSpec('C4H4',1.34,'pi',1.56).geometry()
    assert np.isclose(np.linalg.norm(np.array(rectangle[0][1])-rectangle[1][1]),1.34)


def test_real_pi_projection_active_electrons_and_no_FCI(tmp_path):
    with no_full_state_guards():
        for spec in (ExtendedMolecularSpec('C4H4',1.34,'pi',1.56),ExtendedMolecularSpec('C6H6',1.4,'pi'),
                     ExtendedMolecularSpec('C6H6',1.4,'pi_sigma12')):
            p=tmp_path/(spec.molecule+'-'+spec.space);m=build_extended_problem(spec,p)
            expected={'C4H4-pi':(4,4),'C6H6-pi':(6,6),'C6H6-pi_sigma12':(12,12)}[spec.molecule+'-'+spec.space]
            assert (m['n_electrons'],m['n_orbitals'])==expected
            assert abs(m['active_orbital_selection']['pi_projector_trace']-(4 if spec.molecule=='C4H4' else 6))<1e-7
            assert not m['FCI_computed'] and not m['full_Fock_H_constructed'] and not m['full_sector_enumerated']
            h,info=integral_problem(p)
            assert abs(h.matrix([info['hf_id']],[info['hf_id']])[0,0]-m['hf_energy_Ha'])<1e-9
            assert build_extended_problem(spec,p)==m


def test_triplet_CH2_occupations_not_assumed_spin_balanced(tmp_path):
    with no_full_state_guards():
        m=build_extended_problem(ExtendedMolecularSpec('CH2',1.1,'valence',spin=2),tmp_path/'triplet')
    assert (m['nalpha'],m['nbeta'],m['n_orbitals'])==(4,2,6)
    assert m['independent_HF_energy_error_Ha']<1e-9
    assert m['spin_scope'].startswith('fixed Nalpha,Nbeta')
