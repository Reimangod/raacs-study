"""New Level-B integrals: open-shell CH2 and validated planar carbon pi spaces.

Independent successor to the immutable canonical-orbital molecular builder.
No SCF orbital relaxation after selection, CASSCF, FCI, ground-state reference,
full determinant enumeration or full-Fock Hamiltonian is used.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json
from math import comb
from pathlib import Path
from time import perf_counter
import numpy as np
from pyscf import ao2mo,gto,lib,mcscf,scf
from .common import digest,file_digest,save_json,save_npz
from .integral_hamiltonian import IntegralHamiltonian


@dataclass(frozen=True)
class ExtendedMolecularSpec:
    molecule:str
    distance_A:float
    space:str
    secondary_distance_A:float|None=None
    basis:str='sto-3g'
    charge:int=0
    spin:int=0
    angle_degrees:float=120.
    CH_distance_A:float=1.08

    def geometry(self):
        r=self.distance_A
        if not np.isfinite(r) or r<=0:raise ValueError('Positive finite geometry required')
        if self.molecule=='CH2':
            half=np.deg2rad(self.angle_degrees)/2
            return [('C',(0.,0.,0.)),('H',(r*np.sin(half),r*np.cos(half),0.)),('H',(-r*np.sin(half),r*np.cos(half),0.))]
        if self.molecule=='C4H4':
            other=r if self.secondary_distance_A is None else self.secondary_distance_A
            if not np.isfinite(other) or other<=0:raise ValueError('Positive rectangle side')
            carbon=[np.array([x*r/2,y*other/2,0.]) for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))]
        elif self.molecule=='C6H6':
            carbon=[np.array([r*np.cos(k*np.pi/3),r*np.sin(k*np.pi/3),0.]) for k in range(6)]
        else:raise ValueError('Registered Level-B molecule only')
        hydrogen=[c+self.CH_distance_A*c/np.linalg.norm(c) for c in carbon]
        return [('C',tuple(map(float,c))) for c in carbon]+[('H',tuple(map(float,c))) for c in hydrogen]


def choose_active_orbitals(mf,spec):
    """Fixed physical projector / occupation rules, never energy-outcome choice."""
    nmo=mf.mo_coeff.shape[1];occ=np.asarray(mf.mo_occ);energy=np.asarray(mf.mo_energy)
    occupied=[i for i in range(nmo) if occ[i]>.5]
    paired=[i for i in range(nmo) if occ[i]>1.5]
    diagnostic={'selection_rule':spec.space,'canonical_MO_occupations':occ.tolist(),
                'canonical_MO_energies_Ha':energy.tolist()}
    if spec.space=='valence':
        ncore=sum(1 for atom in mf.mol.atom_charges() if atom==6)
        inactive=sorted(paired,key=lambda i:(energy[i],i))[:ncore]
        if len(inactive)!=ncore:raise ValueError('Enough doubly occupied carbon core orbitals required')
        active=[i for i in range(nmo) if i not in inactive]
    elif spec.space in ('pi','pi_sigma12'):
        if spec.spin!=0:raise ValueError('This pi selection is qualified for restricted closed shell only')
        # AO pz orbitals span the out-of-molecular-plane subspace in STO-3G.
        # Use its S-metric orthogonal projector, NOT raw AO coefficient squares.
        labels=mf.mol.ao_labels(fmt=False)
        pz=[i for i,(atom,symbol,nl,component) in enumerate(labels) if symbol=='C' and component=='z' and 'p' in nl]
        carbon=sum(1 for z in mf.mol.atom_charges() if z==6)
        if len(pz)!=carbon or spec.basis.lower()!='sto-3g':raise ValueError('Qualified minimal-basis carbon-pz subspace only')
        overlap=mf.get_ovlp();szz=overlap[np.ix_(pz,pz)]
        projection=mf.mo_coeff.T@overlap[:,pz]
        weights=np.einsum('mi,ij,mj->m',projection,np.linalg.solve(szz,np.eye(len(pz))),projection)
        pi=[i for i,w in enumerate(weights) if w>1-1e-7]
        if len(pi)!=carbon or max(abs(weights[pi]-1),default=0)>1e-7:raise RuntimeError('Canonical MOs did not resolve the planar pi subspace')
        active=pi.copy()
        if spec.space=='pi_sigma12':
            if carbon!=6:raise ValueError('Registered pi+sigma12 is benzene only')
            sigma_occ=sorted((i for i in paired if i not in pi),key=lambda i:(-energy[i],i))[:3]
            sigma_virtual=sorted((i for i in range(nmo) if occ[i]<.5 and i not in pi),key=lambda i:(energy[i],i))[:3]
            if len(sigma_occ)!=3 or len(sigma_virtual)!=3:raise ValueError('Required sigma buffer unavailable')
            active+=sigma_occ+sigma_virtual
            diagnostic['sigma_buffer_original_MO_indices']={'occupied':sigma_occ,'virtual':sigma_virtual}
        inactive=[i for i in paired if i not in active]
        diagnostic.update(pi_AO_indices=pz,pi_projection_weights=weights.tolist(),pi_original_MO_indices=pi,
                          pi_projector_trace=float(weights.sum()),pi_projector_expected_rank=carbon)
    else:raise ValueError('Registered valence,pi,pi_sigma12 active space only')
    if any(i not in active and i not in inactive for i in occupied):raise ValueError('Singly occupied orbital accidentally frozen/discarded')
    active=sorted(active,key=lambda i:(-occ[i],energy[i],i))
    inactive=sorted(inactive,key=lambda i:(energy[i],i))
    external=[i for i in range(nmo) if i not in active and i not in inactive]
    order=inactive+active+external
    if len(set(order))!=nmo:raise RuntimeError('Orbital partition not bijective')
    diagnostic.update(inactive_original_MO_indices=inactive,active_original_MO_indices=active,
                      external_original_MO_indices=external,reordered_full_MO_indices=order,
                      selection_does_not_use_reference_or_sampler_energy=True)
    return order,len(inactive),len(active),diagnostic


def build_extended_problem(spec,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cfg={'schema':'Level-B-physical-integrals-v1','spec':asdict(spec),'geometry':json.loads(json.dumps(spec.geometry())),
         'pyscf':__import__('pyscf').__version__,'implementation_sha256':file_digest(__file__),
         'FCI_solve':False,'full_Fock_matrix':False,'full_sector_enumeration':False,
         'integral_cap_bytes':256*2**20,'max_qubits':62,'mapping':'JW interleaved alpha,beta; big-endian IDs'}
    binding=digest(cfg);path=directory/'problem.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['config_digest']!=binding:raise RuntimeError('Frozen new physical-input identity drift')
        for name,sha in old['artifact_sha256'].items():
            if file_digest(directory/name)!=sha:raise RuntimeError('New physical artifact mismatch')
        return old
    start=perf_counter();lib.num_threads(1)
    mol=gto.M(atom=spec.geometry(),unit='Angstrom',basis=spec.basis,charge=spec.charge,spin=spec.spin,verbose=0)
    mf=scf.RHF(mol) if spec.spin==0 else scf.ROHF(mol)
    mf.max_memory=1024;mf.conv_tol=1e-11;mf.max_cycle=150;mf.kernel();solver=type(mf).__name__
    if not mf.converged:
        mf=mf.newton();mf.max_memory=1024;mf.conv_tol=1e-11;mf.max_cycle=150;mf.kernel();solver+='+Newton'
    if not mf.converged:raise RuntimeError('SCF nonconvergence: retain as physical-input failure')
    order,ncore,norb,selection=choose_active_orbitals(mf,spec)
    ne=mol.nelectron-2*ncore;na=(ne+spec.spin)//2;nb=ne-na;n=2*norb
    if n>62 or 8*norb**4>cfg['integral_cap_bytes']:raise RuntimeError('Registered integral/register bound')
    coeff=mf.mo_coeff[:,order]
    if np.max(abs(coeff.T@mf.get_ovlp()@coeff-np.eye(len(order))))>1e-8:raise RuntimeError('MO orthonormality')
    # ROHF.get_veff returns two spin blocks, while the generic RHF CAS helper
    # contracts a single closed-core potential. Build the doubly occupied
    # inactive-core contribution explicitly, valid for RHF and ROHF alike.
    core_coeff=coeff[:,:ncore];active_coeff=coeff[:,ncore:ncore+norb]
    core_dm=2*core_coeff@core_coeff.T
    vj,vk=mf.get_jk(mol,core_dm,hermi=1)
    core_potential=vj-.5*vk;one_ao=mf.get_hcore()
    h1=active_coeff.T@(one_ao+core_potential)@active_coeff
    core=mol.energy_nuc()+np.einsum('ij,ji',core_dm,one_ao)+.5*np.einsum('ij,ji',core_dm,core_potential)
    eri=ao2mo.restore(1,ao2mo.kernel(mol,coeff[:,ncore:ncore+norb]),norb)
    hf=sum(1<<(n-1-2*i) for i in range(na))+sum(1<<(n-2-2*i) for i in range(nb))
    backend=IntegralHamiltonian(h1,eri,float(core));independent=float(backend.matrix([hf],[hf])[0,0])
    if abs(independent-mf.e_tot)>1e-8:raise RuntimeError('Active-space frozen-core HF energy/occupations mismatch')
    save_npz(directory/'integrals.npz',h1=h1,eri=eri,ecore=np.asarray(core),mo_coeff=coeff,
             mo_occ=np.asarray(mf.mo_occ)[order],mo_energy=np.asarray(mf.mo_energy)[order])
    meta={'config':cfg,'config_digest':binding,'problem_id':digest([binding,file_digest(directory/'integrals.npz')]),
          'n_orbitals':norb,'n_qubits':n,'n_electrons':ne,'nalpha':na,'nbeta':nb,'hf_id':hf,
          'hf_energy_Ha':float(mf.e_tot),'core_constant_Ha':float(core),'ncore':ncore,
          'sector_dimension':comb(norb,na)*comb(norb,nb),'SCF_solver':solver,'SCF_converged':True,
          'independent_HF_energy_error_Ha':abs(independent-mf.e_tot),'active_orbital_selection':selection,
          'construction_wall_seconds':perf_counter()-start,'FCI_computed':False,'full_Fock_H_constructed':False,
          'full_sector_enumerated':False,'ground_state_reference_energy_Ha':None,
          'chemical_accuracy':'UNKNOWN_NO_GROUND_STATE_REFERENCE','hardware_shots':0,
          'spin_scope':'fixed Nalpha,Nbeta; no S-squared certificate for subsequent sampler',
          'artifact_sha256':{'integrals.npz':file_digest(directory/'integrals.npz')}}
    save_json(path,meta);return meta


def extended_queue(mode):
    if mode=='quick':
        return [ExtendedMolecularSpec('CH2',1.10,'valence',spin=2),ExtendedMolecularSpec('C4H4',1.34,'pi',1.56),
                ExtendedMolecularSpec('C6H6',1.40,'pi'),ExtendedMolecularSpec('C6H6',1.40,'pi_sigma12')]
    if mode!='standard':raise ValueError('Quick or standard')
    out=[ExtendedMolecularSpec('CH2',r,'valence',spin=2) for r in (1.10,1.50)]
    out += [ExtendedMolecularSpec('C4H4',a,space,b) for a,b in ((1.34,1.56),(1.45,1.45)) for space in ('pi','valence')]
    out += [ExtendedMolecularSpec('C6H6',r,space) for r in (1.40,1.80) for space in ('pi','pi_sigma12')]
    return out
