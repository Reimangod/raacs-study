"""Occupied/virtual-block Boys stratum: exact same physical HF reference.

Full active-space Boys mixing occupied/virtual orbitals would require a
non-computational Slater reference. This registered block-preserving variant
does not silently change HF and does not claim P1's unspecified LO identity.
"""
import copy,json
from time import perf_counter,process_time
import numpy as np
from pyscf import gto,lo
from .common_execution import load_physical
from .input_adapter import array_bytes,hf_energy
from .io import ROOT,atomic_bytes,save_json,digest,sha256,assert_safe_physical_input

SOURCE='NAPH20-P1';CASE='NAPH20-BLOCK-BOYS'


def rotate(h,g,t1,t2,U):
    no=len(t1);o,v=U[:no,:no],U[no:,no:]
    if np.max(abs(U[:no,no:]))>1e-12 or np.max(abs(U[no:,:no]))>1e-12:raise ValueError('Occupied/virtual mixing changes the computational HF reference')
    return U.T@h@U,np.einsum('ip,jq,kr,ls,ijkl->pqrs',U,U,U,U,g,optimize=True),o.T@t1@v,np.einsum('ip,jq,ar,bs,ijab->pqrs',o,o,v,v,t2,optimize=True)


def cc_energy(h,g,t1,t2):
    no=len(t1);f=h.copy()
    for k in range(no):f+=2*g[:,:,k,k]-g[:,k,k,:]
    tau=t2+np.einsum('ia,jb->ijab',t1,t1);ovvo=g[:no,no:,no:,:no]
    return float(2*np.einsum('ia,ia',f[:no,no:],t1)+2*np.einsum('ijab,iabj',tau,ovvo)-np.einsum('jiab,iabj',tau,ovvo))


def prepare():
    folder=ROOT/'data/physical'/CASE
    if(folder/'problem.json').exists():return json.loads((folder/'problem.json').read_text())
    m,h,g=load_physical(SOURCE);start,cpu=perf_counter(),process_time()
    source_init=ROOT/'data/initialization-fresh'/SOURCE
    init=json.loads((source_init/'result.json').read_text())
    if init['selected_method']!='CCSD' or sha256((source_init/'amplitudes.npz').read_bytes())!=init['amplitudes_sha256']:raise RuntimeError('Converged source CCSD binding required')
    with np.load(source_init/'amplitudes.npz',allow_pickle=False)as a:t1,t2=a['ccsd_t1'].copy(),a['ccsd_t2'].copy()
    with np.load(ROOT/'data/physical'/SOURCE/'orbital-basis.npz',allow_pickle=False)as a:mo,S=a['AVAS_MO'].copy(),a['AO_overlap'].copy()
    mol=gto.M(atom=m['geometry'],basis=m['basis'],unit='Angstrom',spin=0,charge=0,verbose=0,max_memory=512)
    if np.max(abs(mol.intor_symmetric('int1e_ovlp')-S))>1e-8:raise RuntimeError('AO order/runtime does not match saved active orbitals')
    n=m['norb'];no=m['nalpha'];ncore=(mol.nelectron-2*no)//2;active=mo[:,ncore:ncore+n]
    if np.max(abs(active.T@S@active-np.eye(n)))>1e-8:raise RuntimeError('Saved active orbitals are not orthonormal')
    U=np.zeros((n,n));localization=[]
    for lo_,hi in((0,no),(no,n)):
        C=active[:,lo_:hi];localizer=lo.Boys(mol,C);localizer.init_guess='atomic';localizer.max_cycle=1000;localizer.conv_tol=1e-8
        transformed=localizer.kernel();block=C.T@S@transformed
        U[lo_:hi,lo_:hi]=block;localization.append({'group':[lo_,hi],'max_cycle':1000,'conv_tol':1e-8,'initial_guess':'atomic deterministic current PySCF'})
    if np.max(abs(U.T@U-np.eye(n)))>1e-8:raise RuntimeError('Boys transform is not orthogonal')
    hp,gp,t1p,t2p=rotate(h,g,t1,t2,U);back=rotate(hp,gp,t1p,t2p,U.T)
    errors={name:float(np.max(abs(a-b)))for name,a,b in zip(('h1','eri','t1','t2'),back,(h,g,t1,t2))}
    hf=hf_energy(hp,gp,m['core_energy'],no,no);correlation=cc_energy(h,g,t1,t2);corrp=cc_energy(hp,gp,t1p,t2p)
    if max(errors.values())>1e-8 or abs(hf-m['hf_energy_Ha'])>1e-8 or abs(correlation-corrp)>1e-8:raise RuntimeError('Physical HF/tensor/amplitude covariance check failed')
    source_cc=next(a['total_energy_Ha']for a in init['attempts']if a['status']=='COMPLETED')
    if abs(m['hf_energy_Ha']+correlation-source_cc)>1e-7:raise RuntimeError('Source CCSD energy formula/index convention mismatch')
    # Same occupied subspace, hence the same physical HF determinant up to
    # global phase. No FCI energy or coefficients are used in this construction.
    occ_projector_error=float(np.max(abs(active[:,:no]@active[:,:no].T-(active@U)[:,:no]@(active@U)[:,:no].T)))
    hashes={k:sha256(a.astype('<f8').tobytes())for k,a in(('h1',hp),('eri',gp))}
    physical={k:m[k]for k in('norb','nalpha','nbeta','hf_id','core_energy','mapping','orbital_order')}
    physical['physical_arrays_sha256']=hashes
    meta={'schema':'paper-physical-input-1',**physical,'problem_id':digest(physical),'case_id':CASE,'n_qubits':2*n,
        'hf_energy_Ha':hf,'geometry':m['geometry'],'basis':m['basis'],
        'source':{'kind':'REGISTERED_OCC_VIR_BLOCK_BOYS_STRATUM','parent_case':SOURCE,'parent_problem_id':m['problem_id'],
            'Boys_full_occ_vir_mixing_performed':False,'original_P1_LO_byte_identity_verified':False,'localization':localization},
        'qualification':{'HF_absolute_difference_Ha':abs(hf-m['hf_energy_Ha']),'occupied_subspace_projector_max_difference':occ_projector_error,
            'orthogonality_max_error':float(np.max(abs(U.T@U-np.eye(n)))),'inverse_tensor_transform_errors':errors,
            'CCSD_correlation_energy_covariance_difference_Ha':abs(correlation-corrp),'same_physical_reference_determinant':True}}
    assert_safe_physical_input(meta)
    atomic_bytes(folder/'integrals.npz',array_bytes({'h1':hp,'eri':gp}));atomic_bytes(folder/'orbital-basis.npz',array_bytes({'canonical_to_localized':U,'localized_active_MO':active@U}))
    new_init=ROOT/'data/initialization-fresh'/CASE
    blob=array_bytes({'ccsd_t1':t1p,'ccsd_t2':t2p});atomic_bytes(new_init/'amplitudes.npz',blob)
    rotation_cpu=process_time()-cpu;rotation_wall=perf_counter()-start
    save_json(new_init/'result.json',{'status':'FRESH_INITIALIZATION_COMPLETED','status_is_adapter_compatibility_not_new_CCSD_solve':True,
        'selected_method':'COVARIANT_ROTATION_OF_CONVERGED_CCSD','case_id':CASE,'problem_id':meta['problem_id'],
        'creation_kind':'BASIS_TRANSFORMATION_NOT_NEW_CCSD_OPTIMIZATION','amplitudes_sha256':sha256(blob),
        'source_converged_CCSD':str((source_init/'result.json').relative_to(ROOT)),'source_amplitudes_sha256':init['amplitudes_sha256'],
        'upstream_CCSD_CPU_seconds':init['CPU_process_seconds'],'CPU_process_seconds':rotation_cpu,'total_wall_seconds':rotation_wall,
        'rotation_and_localization_cost_shared_with_physical_preparation':True,'standalone_cost_requires_upstream_CCSD':True,
        'new_CCSD_solves':0,'original_saved_author_amplitudes_used':False,'FCI_or_SCI_reference_used':False,'actual_QPU_shots':0})
    save_json(folder/'preparation-cost.json',{'CPU_seconds':rotation_cpu,'wall_seconds':rotation_wall,
        'includes_localization_integral_amplitude_transform_and_checks':True,'initialization_compatibility_record_is_same_work_not_additive':True})
    save_json(folder/'problem.json',meta)
    return meta


def evaluation_crosswalk():
    # Called by the report/controller, not by a selection/sampling worker.
    m=json.loads((ROOT/'data/physical'/CASE/'problem.json').read_text())
    reference=json.loads((ROOT/'data/evaluation-only'/(SOURCE+'.json')).read_text())
    reference['problem_id']=m['problem_id'];reference['same_spectrum_under_registered_orthogonal_active_rotation']=True
    reference['source_reference_case']=SOURCE
    save_json(ROOT/'data/evaluation-only'/(CASE+'.json'),reference)


if __name__=='__main__':print(json.dumps(prepare()),flush=True)
