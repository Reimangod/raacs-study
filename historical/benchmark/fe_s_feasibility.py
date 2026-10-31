"""Bounded public-integral Fe2S2 qualification, not a ground-state benchmark.

No source repository code is executed. Old producer modules/results are unchanged.
All exact matrices here are small selected-determinant matrices, never full Fock.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from time import perf_counter
from unittest.mock import patch
import urllib.request
import zlib
import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_binary, save_json, save_npz
from .integral_hamiltonian import IntegralHamiltonian
from .local_sampler_primitives import valid_sector, unpack_counts64
from .molecular_mps_scaling import no_full_state_guards
from .mps_layout_qualification import permute_sampling, logical_id
from .mps_sampling import construct_mps
from . import resource_adaptive_sampler as native
from .circuit_sampling_runtime import SamplingSpec
from .spin_blocked_sampling import SpinBlockedCountSampler
from .ansatz import measurement_equivalent_circuit, resources
from .study.core import rss_peak

COMMIT = '865bd762d3f96ee79bb362eb3dd46951b48b426e'
REPOSITORY = 'https://github.com/jrm874/sqd_data_repository'
FE2_PATH = 'integrals/2Fe-2S/fcidump_Fe2S2_MO.txt'
FE4_PATH = 'integrals/4Fe-4S/fcidump_Fe4S4_MO.txt'
FE2_SIZE = 959069
FE2_BLOB = '55e0dbab07d4d1754e042e38f98b34b566921f31'
SPEC = ROOT/'study-v1/FE_S_BOUNDED_FEASIBILITY.md'


def source_url(path):
    return f'https://raw.githubusercontent.com/jrm874/sqd_data_repository/{COMMIT}/{path}'


def git_blob(raw):
    return hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()


def header(raw):
    text = raw.decode('ascii').upper()
    result = {}
    for field in ('NORB', 'NELEC', 'MS2'):
        match = re.search(r'\b'+field+r'\s*=\s*([+-]?\d+)', text)
        if not match:
            raise ValueError('Missing FCIDUMP '+field)
        result[field] = int(match.group(1))
    n, ne, ms = result['NORB'], result['NELEC'], result['MS2']
    if n < 1 or not 0 <= ne <= 2*n or (ne+ms)%2 or abs(ms)>ne:
        raise ValueError('Invalid particle/spin header')
    result.update(n_qubits=2*n, nalpha=(ne+ms)//2, nbeta=(ne-ms)//2)
    if not 0 <= result['nalpha'] <= n or not 0 <= result['nbeta'] <= n:
        raise ValueError('Invalid spin occupations')
    return result


def verify_download(raw):
    if len(raw) != FE2_SIZE or git_blob(raw) != FE2_BLOB:
        raise ValueError('Pinned Fe2 data size/blob mismatch')
    return hashlib.sha256(raw).hexdigest()


def get_inputs(root):
    folder = root/'public-inputs'; folder.mkdir(parents=True, exist_ok=True)
    target = folder/'fcidump_Fe2S2_MO.txt'
    if target.exists():
        raw = target.read_bytes()
    else:
        with urllib.request.urlopen(source_url(FE2_PATH), timeout=30) as response:
            raw = response.read(2*2**20+1)
        verify_download(raw)
        save_binary(target, raw)
    sha = verify_download(raw)
    h2 = header(raw[:4096])
    if h2 != {'NORB':20, 'NELEC':30, 'MS2':0, 'n_qubits':40, 'nalpha':15, 'nbeta':15}:
        raise ValueError('Unexpected pinned Fe2 physical header')
    hp = folder/'Fe4S4-header.txt'
    if hp.exists():
        raw4 = hp.read_bytes()
    else:
        with urllib.request.urlopen(source_url(FE4_PATH), timeout=30) as response:
            raw4 = response.read(4096)
        header(raw4); save_binary(hp, raw4)
    h4 = header(raw4)
    if h4['n_qubits'] != 72:
        raise ValueError('Unexpected Fe4 header')
    evidence = {'repository':REPOSITORY, 'commit':COMMIT,
        'Fe2':{'url':source_url(FE2_PATH), 'bytes':len(raw), 'git_blob_sha1':git_blob(raw),
               'sha256':sha, 'header':h2},
        'Fe4':{'url':source_url(FE4_PATH), 'header_bytes_read':len(raw4),
               'header_sha256':file_digest(hp), 'header':h4,
               'status':'NOT_EXECUTED_72_QUBITS_EXCEED_QUALIFIED_62_BIT_BACKEND',
               'full_file_downloaded':False, 'active_space_truncated':False},
        'redistribution_license':'NOT_ESTABLISHED_NO_PUBLICATION_OR_UPLOAD',
        'source_code_downloaded_or_executed':False}
    save_json(folder/'provenance.json', evidence)
    return target, evidence


def occupation_reference(norb, na, nb):
    return sum(1 << (2*norb-1-(2*p+s)) for s, occ in ((0,na),(1,nb)) for p in range(occ))


def independent_diagonal(h1, eri, core, na, nb):
    occ = [(p,s) for s,count in ((0,na),(1,nb)) for p in range(count)]
    return float(core + sum(h1[p,p] for p,_ in occ) + .5*sum(
        eri[p,p,q,q]-(eri[p,q,q,p] if s==t else 0.) for p,s in occ for q,t in occ))


def apply_fermions(raw, operations, n):
    sign = 1
    for create, q in operations:
        bit = 1 << (n-1-q)
        if bool(raw & bit) == create:
            return None
        if (raw >> (n-q)).bit_count()%2:
            sign = -sign
        raw ^= bit
    return raw, sign


def explicit_element(bra, ket, h1, eri, core):
    """Independent direct second-quantized sum, without Slater–Condon formulas."""
    n = 2*len(h1); value = float(core) if bra==ket else 0.
    occupied = [i for i in range(n) if ket >> (n-1-i) & 1]
    for i in occupied:
        base = ket ^ (1 << (n-1-i))
        missing = [p for p in range(n) if (bra ^ base) >> (n-1-p) & 1]
        if not (base & ~bra) and len(missing)==1 and missing[0]%2==i%2:
            p = missing[0]; result = apply_fermions(ket, [(False,i),(True,p)], n)
            if result is not None and result[0]==bra: value += h1[p//2,i//2]*result[1]
        for j in occupied:
            if i==j: continue
            base2 = base ^ (1 << (n-1-j))
            if base2 & ~bra: continue
            missing2 = [p for p in range(n) if (bra ^ base2) >> (n-1-p) & 1]
            if len(missing2)!=2: continue
            for p,r in (missing2, missing2[::-1]):
                if p%2!=i%2 or r%2!=j%2: continue
                result = apply_fermions(ket, [(False,i),(False,j),(True,r),(True,p)], n)
                if result is not None and result[0]==bra:
                    value += .5*eri[p//2,i//2,r//2,j//2]*result[1]
    return float(value)


def explicit_matrix(ids, backend):
    return np.array([[explicit_element(int(x),int(y),backend.h1,backend.eri,backend.core)
                      for y in ids] for x in ids])


def build_problem(source, folder):
    from pyscf.tools import fcidump
    from pyscf import ao2mo
    parsed = fcidump.read(str(source), verbose=False)
    n, ne, ms = parsed['NORB'], parsed['NELEC'], parsed['MS2']
    if 2*n>62 or (ne+ms)%2: raise ValueError('Unqualified external representation')
    na, nb = (ne+ms)//2, (ne-ms)//2
    h1 = np.asarray(parsed['H1']); eri = ao2mo.restore(1,parsed['H2'],n)
    core = float(parsed['ECORE'])
    if not all(np.isfinite(a).all() for a in (h1,eri,np.array(core))): raise ValueError('Nonfinite physical input')
    symmetry = {'h1':float(np.max(abs(h1-h1.T))),
        'eri_ij':float(np.max(abs(eri-eri.transpose(1,0,2,3)))),
        'eri_kl':float(np.max(abs(eri-eri.transpose(0,1,3,2)))),
        'eri_pair':float(np.max(abs(eri-eri.transpose(2,3,0,1))))}
    if max(symmetry.values())>1e-10: raise ValueError('Non-Hermitian spatial integrals')
    hf = occupation_reference(n,na,nb); ref = independent_diagonal(h1,eri,core,na,nb)
    backend = IntegralHamiltonian(h1,eri,core)
    ids = [hf]
    for source_q, target_q in ((0,2*na),(1,2*nb+1),(2,2*na+2)):
        if target_q<2*n:
            ids.append(hf ^ (1<<(2*n-1-source_q)) ^ (1<<(2*n-1-target_q)))
    if 0<na<n and 0<nb<n:
        ids.append(hf ^ (1<<(2*n-1)) ^ (1<<(2*n-2)) ^ (1<<(2*n-1-2*na)) ^ (1<<(2*n-2-2*nb)))
    ids = sorted(set(ids))
    if not all(valid_sector(x,2*n,na,nb) for x in ids): raise ValueError('Qualification IDs outside sector')
    direct = explicit_matrix(ids,backend); actual = backend.matrix(ids,ids)
    err = float(np.max(abs(direct-actual)))
    diag_err = abs(float(backend.matrix([hf],[hf])[0,0])-ref)
    if max(err,diag_err)>1e-8: raise ValueError('Independent physical matrix mismatch')
    folder.mkdir(parents=True, exist_ok=True)
    meta = {'schema':'external-FCIDUMP-occupation-reference-v1', 'n_qubits':2*n,
        'nalpha':na, 'nbeta':nb, 'hf_id':hf, 'hf_energy_Ha':ref, 'core_energy_Ha':core,
        'reference_semantics':'FIRST_ORBITALS_OCCUPATION_REFERENCE_NOT_VERIFIED_RHF',
        'hf_fields':'LEGACY_COMPATIBILITY_ALIASES_ONLY', 'S_squared_certified':False,
        'ground_state_reference':None, 'source_sha256':file_digest(source),
        'FCI_computed':False, 'full_Fock_H_constructed':False,
        'validation':{'symmetry_errors':symmetry, 'named_ids':ids, 'matrix_entries':len(ids)**2,
            'independent_explicit_fermion_max_error_Ha':err,
            'independent_reference_diagonal_error_Ha':diag_err}}
    if (folder/'problem.json').exists():
        if json.loads((folder/'problem.json').read_text())!=meta:
            raise RuntimeError('Existing physical adapter drift')
        with np.load(folder/'integrals.npz') as z:
            if not np.array_equal(z['h1'],h1) or not np.array_equal(z['eri'],eri):
                raise RuntimeError('Existing physical arrays drift')
    else:
        save_npz(folder/'integrals.npz',h1=h1,eri=eri)
        save_json(folder/'problem.json',meta)
    return meta


def boundary(root):
    if shutil.disk_usage(root).free<2.75*2**30: raise RuntimeError('FREE_DISK_FLOOR_PRESERVE_PARTIAL')
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>64*2**20:
        raise RuntimeError('OUTPUT_BOUNDARY_PRESERVE_PARTIAL')


def audit_cell(problem, cell):
    # No production helper for selected-space ranking or matrix construction.
    backend,info = native.integral_problem(problem)
    result = json.loads((cell/'result.json').read_text())
    if file_digest(cell/'events.sqlite')!=result['event_database_sha256']: raise RuntimeError('Event digest mismatch')
    db = sqlite3.connect(f'file:{cell/"events.sqlite"}?mode=ro',uri=True)
    paid=0; final={}; checks=0; maxerr=0.; norms=[]; valid=0; rawfinal=0
    for role,raw,metadata,requested,recorded in db.execute('SELECT role,counts64,metadata,requested,raw_shots FROM draws'):
        ids,counts=unpack_counts64(raw); meta=json.loads(metadata)
        if int(counts.sum())!=recorded or recorded!=meta['actual_raw_model_shots'] or recorded>requested:
            raise RuntimeError('Raw sample accounting mismatch')
        layout=meta['layout_evidence']; mapped={logical_id(x,layout['logical_to_simulator_qubit']):int(c)
            for x,c in zip(layout['backend_ids'],layout['backend_counts'])}
        if mapped!=dict(zip(map(int,ids),map(int,counts))): raise RuntimeError('Logical mapping mismatch')
        norms.append(meta['original_build_metadata']['retained_norm_squared']); paid+=recorded
        if role=='independent_final':
            for x,c in zip(ids,counts): final[int(x)]=final.get(int(x),0)+int(c)
            rawfinal+=recorded
            valid+=sum(int(c) for x,c in zip(ids,counts) if valid_sector(int(x),info['n_qubits'],info['nalpha'],info['nbeta']))
    chosen=sorted([info['hf_id']]+[x for x,c in sorted(final.items(),key=lambda a:(-a[1],a[0]))
        if x!=info['hf_id'] and valid_sector(x,info['n_qubits'],info['nalpha'],info['nbeta'])][:result['algorithm']['CI_cap']-1])
    if chosen!=result['final_selected_ids']: raise RuntimeError('Observed-only final selection mismatch')
    matrix=explicit_matrix(chosen,backend); eig=float(np.linalg.eigvalsh(matrix)[0]);checks+=1
    maxerr=max(maxerr,abs(eig-result['final_QSCI_energy_Ha']))
    if maxerr>1e-8 or paid!=result['actual_total_model_shots']: raise RuntimeError('Independent final root/accounting mismatch')
    if min(norms)<SpinBlockedCountSampler.minimum_retained_norm_squared: raise RuntimeError('Unqualified approximation')
    record={'status':'COUNTS_LAYOUT_SECTOR_AND_INDEPENDENT_FINAL_CI_CHECKED',
        'event_database_sha256':file_digest(cell/'events.sqlite'), 'actual_model_samples':paid,
        'independent_final_model_samples':rawfinal, 'final_sector_fraction':valid/max(1,rawfinal),
        'minimum_draw_MPS_retained_norm_squared':min(norms), 'independent_final_Ritz_solves':checks,
        'independent_matrix_entries':len(chosen)**2, 'maximum_energy_difference_Ha':maxerr,
        'S_squared_certified':False, 'ground_state_error_Ha':None, 'hardware_shots':0}
    db.close();save_json(cell/'independent-validation.json',record);return record


def bond_checks(cell, folder):
    result=json.loads((cell/'result.json').read_text()); allocations=result['final_circuit_allocations']
    if len(allocations)!=1: raise ValueError('Registered last-only final circuit')
    db=sqlite3.connect(f'file:{cell/"events.sqlite"}?mode=ro',uri=True)
    raw,actions=db.execute('SELECT preparation_qpy,actions FROM circuits WHERE id=?',(allocations[0]['circuit_id'],)).fetchone();db.close()
    prep=qpy.load(BytesIO(zlib.decompress(raw)))[0];sampling,_=measurement_equivalent_circuit(prep)
    blocked,mapping=permute_sampling(sampling);states=[];rows=[]
    folder.mkdir(parents=True,exist_ok=True)
    for bond in (64,128,256):
        start=perf_counter()
        try:
            state,meta=construct_mps(blocked,bond,1e-12,20,2*2**30)
            psi=state.psi.copy();psi.normalize();states.append((bond,psi))
            row={'bond':bond,'status':'COMPLETED','metadata':meta}
        except Exception as exc:row={'bond':bond,'status':'RESOURCE_OR_ENGINEERING_FAILURE_PRESERVED','error':repr(exc)}
        row['wall_seconds']=perf_counter()-start;rows.append(row);save_json(folder/'progress.json',rows)
    if states:
        reference_bond,reference=states[-1]
        for bond,psi in states:
            next(r for r in rows if r['bond']==bond).update(reference_bond=reference_bond,
                normalized_overlap_squared_to_highest_completed_bond=float(abs(psi.H@reference)**2))
    out={'rows':rows,'same_physical_circuit':True,'logical_to_simulator_qubit':mapping,
        'sampling_resources':resources(sampling,len(json.loads(actions)),len(json.loads(actions))),
        'new_model_samples':0,'hardware_shots':0,'not_global_error_certificate':True}
    save_json(folder/'result.json',out);return out


def run(root,mode):
    import fcntl
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    # Preserve distinct completed versions and partial attempts; never retry for free.
    with (root/'run.lock').open('a') as lock, threadpool_limits(limits=1), no_full_state_guards():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);boundary(root)
        files=('fe_s_feasibility.py','resource_adaptive_sampler.py','spin_blocked_sampling.py',
            'mps_layout_qualification.py','circuit_sampling_runtime.py','mps_sampling.py','local_sampler_primitives.py',
            'integral_hamiltonian.py','integral_connections.py','ansatz.py')
        cfg={'schema':'FeS-bounded-feasibility-v1','mode':mode,'source_commit':COMMIT,
            'spec_sha256':file_digest(SPEC),'implementation':{p:file_digest(ROOT/'benchmark'/p) for p in files},
            'FCI_allowed':False,'full_Fock_vector_allowed':False,'hardware_shots':0,
            'reference_energy_for_accuracy':None,'whole_goal_complete':False}
        if (root/'manifest.json').exists():
            if json.loads((root/'manifest.json').read_text())!=cfg:raise RuntimeError('Frozen Fe feasibility identity drift')
        else:save_json(root/'manifest.json',cfg)
        if (root/'completion.json').exists():return json.loads((root/'completion.json').read_text())
        start=perf_counter()
        if mode=='quick':
            problem=ROOT/'results/standard-v1/references/H4_R1.1';source=None
        else:
            qp=ROOT/'results/FeS-feasibility-quick-v1/completion.json'
            if not qp.exists() or json.loads(qp.read_text())['status']!='QUALIFICATION_EXECUTED_AND_VALIDATED':
                raise RuntimeError('Actual quick qualification required')
            path,source=get_inputs(root);problem=root/'physical-problem';build_problem(path,problem)
        spec=native.RAACSSpec(score='full_cost',total_model_shots=512 if mode=='quick' else 2000,
            max_steps=2 if mode=='quick' else 4,CI_cap=32 if mode=='quick' else 64,
            parent_connection_cap=64,target_cap=64,candidate_bank_cap=16,candidates_measured_per_step=2,seed=101)
        sampling=SamplingSpec(backend='mps',max_bond=128,construction_seconds_cap=20,sampling_seconds_cap=15)
        save_json(root/'execution-binding.json',{'algorithm':asdict(spec),'sampling':asdict(sampling),
            'reference_semantics':'SAVED_H4_RHF' if mode=='quick' else 'OCCUPATION_REFERENCE_NOT_VERIFIED_RHF',
            'source':source,'runtime_provider':'SpinBlockedCountSampler'})
        cell=root/'RAACS-full-cost'
        original=native.integral_problem
        def adapter(path):
            backend,info=original(path)
            if mode!='quick':
                info['initial_reference_semantics']='FIRST_ORBITALS_OCCUPATION_REFERENCE_NOT_VERIFIED_RHF'
                info['HF_energy_Ha_is_legacy_alias']=True
            return backend,info
        try:
            with patch.object(native,'CircuitCountSampler',SpinBlockedCountSampler),patch.object(native,'integral_problem',adapter):
                result=native.run_sampler(problem,cell,spec,sampling)
            boundary(root);audit=audit_cell(problem,cell);bonds=bond_checks(cell,root/'same-circuit-bonds')
            completion={'status':'QUALIFICATION_EXECUTED_AND_VALIDATED','mode':mode,
                'result_sha256':file_digest(cell/'result.json'),'validation_sha256':file_digest(cell/'independent-validation.json'),
                'bond_result_sha256':file_digest(root/'same-circuit-bonds/result.json'),
                'accepted_steps':result['accepted_steps'],'final_QSCI_energy_Ha':result['final_QSCI_energy_Ha'],
                'actual_CI_dimension':result['actual_final_CI_dimension'],'actual_model_samples':result['actual_total_model_shots'],
                'sampling_resources':bonds['sampling_resources'],'actual_wall_seconds':perf_counter()-start,
                'process_RSS_hwm_bytes':rss_peak(),'ground_state_error_Ha':None,'hardware_shots':0,
                'not_general_fe_ground_state_or_quantum_advantage_result':True,'whole_goal_complete':False}
            save_json(root/'completion.json',completion);return completion
        except Exception as exc:
            save_json(root/'failure-preserved.json',{'error':repr(exc),'elapsed_seconds':perf_counter()-start,
                'partial_ledger_preserved':(cell/'events.sqlite').exists(),'actual_new_model_samples':'READ_COMMITTED_LEDGER_NOT_ASSUMED_ZERO',
                'scientific_performance_failure':False,'whole_goal_complete':False})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','pilot'],default='quick');p.add_argument('--output')
    a=p.parse_args();print(json.dumps(run(a.output or ROOT/f'results/FeS-feasibility-{a.mode}-v1',a.mode)))
