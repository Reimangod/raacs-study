"""Bounded independent fixed-N/Sz ffsim audit of every standard NOCI space.

Uses no FCI amplitudes. Saved small sparse H is evaluation-only. A128-vector
LRU limits sector-vector storage; no all-configurations vector cache is kept.
The count-only experiment itself remains integral-only and cannot access this.
"""
from __future__ import annotations
import argparse
import csv
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import ffsim
import numpy as np
from scipy import sparse
from scipy.linalg import svd,eigh
from qiskit import qpy
from qiskit.quantum_info import Statevector
from .common import ROOT,file_digest,save_json,rng_for
from .chemistry import determinant_ids
from .multibasis_study import unpack,valid,uniform_counts
from .ansatz import measurement_equivalent_circuit,resources
from .study.core import rss_peak


def sector_vector(rotation,identifier,nalpha,nbeta):
    n=len(rotation);ids,signs,_,_=determinant_ids(n,nalpha,nbeta)
    positions=np.flatnonzero(ids==int(identifier))
    if len(positions)!=1:raise ValueError('Selected vector outside registered sector')
    initial=np.zeros(len(ids),complex);initial[positions[0]]=signs[positions[0]]
    return ffsim.apply_orbital_rotation(initial,np.asarray(rotation),n,(nalpha,nbeta))*signs


def rotate_full_audit_only(state,rotation):
    n=len(rotation)
    if n>8:raise ValueError('Small evaluation-only full-state parity')
    sectors=set()
    for idx in np.flatnonzero(abs(state)>1e-14):
        sectors.add(tuple(sum(int(idx)>>(2*n-1-2*p-s)&1 for p in range(n)) for s in (0,1)))
    out=np.zeros_like(state)
    for na,nb in sectors:
        ids,signs,_,_=determinant_ids(n,na,nb)
        out[ids]=ffsim.apply_orbital_rotation(state[ids]*signs,rotation,n,(na,nb))*signs
    return out


def audit(root):
    root=Path(root);cfg=json.loads((root/'manifest.json').read_text());start=perf_counter()
    for p,h in cfg['implementation'].items():
        if file_digest(ROOT/'benchmark'/p)!=h:raise RuntimeError('Frozen code drift: '+p)
    for name,key in [('MULTIBASIS_NOCI_V1.md','spec_sha256'),('MULTIBASIS_ACCOUNTING_V2.md','accounting_spec_sha256')]:
        if file_digest(ROOT/'study-v1'/name)!=cfg[key]:raise RuntimeError('Frozen spec drift')
    db=sqlite3.connect(f"file:{root/'events.sqlite'}?mode=ro",uri=True)
    statuses=dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())
    planned=len(cfg['cases'])*len(cfg['methods'])*len(cfg['arms'])*len(cfg['M'])*len(cfg['seeds'])
    if sum(statuses.values())!=planned:raise RuntimeError('Full standard task ledger incomplete')
    spaces=circuits=draws=replays=0;maximum_energy=maximum_rotation=0.;unique_quantum=inherited_quantum=uniform_samples=0
    max_vector_bytes=0;source_missing=statuses.get('MISSING_SOURCE_METHOD',0);rank_rows=[]
    rank_diagnostic_seconds=0.
    if any(k not in ('COMPLETED','MISSING_SOURCE_METHOD') for k in statuses):raise RuntimeError('Preserve failure evidence; standard is not all validated')
    for case in cfg['cases']:
        if rss_peak()>2*2**30:raise RuntimeError('Standard audit2GiB RSS boundary')
        b=unpack(db.execute('SELECT payload FROM bases WHERE case_id=?',(case,)).fetchone()[0]);info=b['problem']
        na,nb=info['nalpha'],info['nbeta'];n=info['n_qubits']//2;rotations=[np.asarray(x) for x in b['rotations']]
        ids,_,_,_=determinant_ids(n,na,nb)
        folder=Path(cfg['source'])/'references'/case
        meta=json.loads((folder/'reference.json').read_text());full_h=sparse.load_npz(folder/'hamiltonian.npz')
        h=full_h[ids][:,ids].tocsr();del full_h
        cache=OrderedDict();space_cache={};probabilities={}
        def vec(node):
            node=tuple(node)
            if node in cache:
                val=cache.pop(node);cache[node]=val;return val
            val=sector_vector(rotations[node[0]],node[1],na,nb);cache[node]=val
            while len(cache)>128:cache.popitem(last=False)
            return val
        for method,basis,raw,blob in db.execute('SELECT method,basis,qpy,payload FROM circuits WHERE case_id=?',(case,)).fetchall():
            record=unpack(blob);qc=qpy.load(BytesIO(bytes(raw)))[0]
            from .multibasis_primitives import source_preparation
            source,_=source_preparation(case,method);source_vector=Statevector.from_instruction(source).data
            expected=rotate_full_audit_only(source_vector,rotations[basis].conj().T)
            actual=Statevector.from_instruction(qc).data;error=float(np.max(abs(actual-expected)))
            maximum_rotation=max(maximum_rotation,error)
            if error>1e-8:raise RuntimeError('Independent fixed-sector rotation direction/phase')
            sample,_=measurement_equivalent_circuit(qc);p=abs(Statevector.from_instruction(sample).data)**2
            if np.max(abs(p-abs(actual)**2))>1e-9:raise RuntimeError('Post-basis probability compiler')
            rr=resources(sample,0,0)
            if any(rr[k]!=record['resources'][k] for k in ('cnot_count','cnot_depth','total_depth')):raise RuntimeError('Native logical resources')
            if record['resources']['parameter_count'] is not None:raise RuntimeError('Unknown parameter count became zero')
            probabilities[(method,basis)]=p/p.sum();circuits+=1
        for aid,method,strategy,M,rep,status in db.execute('SELECT id,method,strategy,M,rep,status FROM acquisitions WHERE case_id=?',(case,)).fetchall():
            if status!='COMPLETED':raise RuntimeError('Unresolved physical acquisition')
            total=0
            for did,basis,ds,blob in db.execute('SELECT id,basis,status,payload FROM draws WHERE acquisition=? ORDER BY basis',(aid,)).fetchall():
                if ds!='COMPLETED':raise RuntimeError('Unresolved physical draw')
                x=unpack(blob);rng,seed=rng_for('multiple-orbital-measurement-NOCI-v1','physical-draw',aid,basis)
                if str(seed)!=x['seed'] or sum(x['counts'])!=x['requested'] or len(x['ids'])!=len(set(x['ids'])):raise RuntimeError('Seed/count binding')
                if sum(c for i,c in zip(x['ids'],x['counts']) if valid(i,info))!=x['valid_sector_raw']:raise RuntimeError('Valid-sector accounting')
                if any(c<=0 for c in x['counts']):raise RuntimeError('Nonpositive count')
                total+=x['actual_raw'];shots=x['metadata']['raw_simulated_quantum_shots'];unique_quantum+=shots
                inherited_quantum+=shots if 'inherited_quick_raw_draw_id' in x else 0
                uniform_samples+=x['metadata']['classical_uniform_samples']
                # Predetermined low-M/first-seed physical/acquisition replay.
                # Other draws receive full integer/identity checks, not a claim
                # that all random draws were independently regenerated.
                if M==min(cfg['M']) and rep==cfg['seeds'][0]:
                    if strategy=='uniform_multibasis':
                        ii,cc=uniform_counts(n,info,x['requested'],rng)
                    else:
                        counts=rng.multinomial(x['requested'],probabilities[(method,basis)])
                        ii=np.flatnonzero(counts).tolist();cc=counts[ii].tolist()
                    if ii!=x['ids'] or cc!=x['counts']:raise RuntimeError('Predetermined acquisition replay')
                    replays+=1
                draws+=1
            if total!=M:raise RuntimeError('Raw allocation total')
        controls={}
        for tid,taskblob,method,arm,M,rep in db.execute("SELECT id,payload,method,arm,M,rep FROM tasks WHERE case_id=? AND status='COMPLETED'",(case,)).fetchall():
            task=unpack(taskblob)
            if db.execute('SELECT status FROM acquisitions WHERE id=?',(task['acquisition_id'],)).fetchone()!=('COMPLETED',):raise RuntimeError('Readout/acquisition linkage')
            readouts=db.execute('SELECT R,payload FROM readouts WHERE task=? ORDER BY R',(tid,)).fetchall()
            if [r for r,_ in readouts]!=cfg['R']:raise RuntimeError('Missing R')
            for R,blob in readouts:
                out=unpack(blob);nodes=tuple(tuple(x) for x in out['selected_nodes'])
                if len(nodes)>R or len(set(nodes))!=len(nodes):raise RuntimeError('R cap / basis-aware dedup')
                if nodes in space_cache:
                    energy,rank,spectrum=space_cache[nodes]
                else:
                    v=np.column_stack([vec(node) for node in nodes]);u,singular,_=svd(v,full_matrices=False)
                    cutoff=cfg['rank_relative_cutoff']*singular[0]**2;keep=singular**2>cutoff
                    q=u[:,keep];hp=q.conj().T@(h@q);energy=float(eigh((hp+hp.conj().T)/2,eigvals_only=True)[0]);rank=int(keep.sum())
                    spectrum=eigh(v.conj().T@v,eigvals_only=True)
                    space_cache[nodes]=(energy,rank,spectrum)
                    if len(space_cache)>10000:space_cache.pop(next(iter(space_cache)))
                error=abs(energy-out['energy_Ha']);maximum_energy=max(maximum_energy,error)
                if rank!=out['rank'] or error>1e-7 or np.max(abs(spectrum-out['overlap_eigenvalues']))>1e-8:raise RuntimeError('Independent sector projection/SVD')
                if out['energy_Ha']<meta['fci_energy']-1e-7:raise RuntimeError('Post-hoc variational evaluation')
                if arm!='canonical':controls.setdefault((method,M,rep,R),{})[arm]=(out,nodes)
                spaces+=1
        rank_start=perf_counter()
        for (method,M,rep,R),group in controls.items():
            if len(group)!=3:continue
            target=min(x[0]['rank'] for x in group.values())
            for arm,(primary,nodes) in group.items():
                v=np.column_stack([vec(node) for node in nodes]);gram=v.conj().T@v;found=None
                for prefix in range(3,len(nodes)+1):
                    w=eigh(gram[:prefix,:prefix],eigvals_only=True)
                    if int(np.count_nonzero(w>cfg['rank_relative_cutoff']*w[-1]))!=target:continue
                    u,singular,_=svd(v[:,:prefix],full_matrices=False)
                    keep=singular**2>cfg['rank_relative_cutoff']*singular[0]**2
                    q=u[:,keep];hp=q.conj().T@(h@q)
                    energy=float(eigh((hp+hp.conj().T)/2,eigvals_only=True)[0]);found=(prefix,energy);break
                rank_rows.append({'case_id':case,'method':method,'M':M,'rep':rep,'R':R,'arm':arm,'target_rank':target,
                    'status':'MATCHED' if found else 'NOT_MATCHABLE_WITH_ALL_ANCHORS',
                    'prefix_vectors':found[0] if found else None,'energy_Ha':found[1] if found else None,
                    'error_mHa':1000*(found[1]-meta['fci_energy']) if found else None,
                    'scope':'POST_HOC_SECONDARY_FIRST_RANK_MATCHED_FREQUENCY_PREFIX; all3anchors retained; no energy selection'})
        rank_diagnostic_seconds+=perf_counter()-rank_start
        max_vector_bytes=max(max_vector_bytes,sum(v.nbytes for v in cache.values()))
        print(case,'spaces',spaces,'circuits',circuits,flush=True)
    db.close()
    with (root/'actual-d-secondary-diagnostic.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rank_rows[0]) if rank_rows else ['status'])
        writer.writeheader();writer.writerows(rank_rows)
    result={'status':'ALL_STANDARD_MULTIBASIS_SPACES_INDEPENDENTLY_VALIDATED','planned_tasks':planned,
            'tasks_by_status':statuses,'preserved_missing_source_tasks':source_missing,'all_projected_readouts_checked':spaces,
            'physical_measurement_circuit_checks':circuits,'all_raw_draw_records_checked':draws,'predetermined_draw_replays':replays,
            'max_energy_error_Ha':maximum_energy,'max_rotation_state_amplitude_error':maximum_rotation,
            'linked_unique_quantum_model_shots':unique_quantum,'inherited_quick_quantum_model_shots':inherited_quantum,
            'new_standard_quantum_model_shots':unique_quantum-inherited_quantum,'classical_uniform_samples':uniform_samples,
            'maximum_sector_vector_cache_bytes':max_vector_bytes,'vector_cache_entry_cap':128,'peak_RSS':rss_peak(),
            'actual_d_diagnostic_rows':len(rank_rows),'actual_d_matches':sum(x['status']=='MATCHED' for x in rank_rows),
            'actual_d_diagnostic_seconds':rank_diagnostic_seconds,
            'actual_d_diagnostic_sha256':file_digest(root/'actual-d-secondary-diagnostic.csv'),
            'hardware_shots':0,'new_quantum_acquisitions_in_audit':0,
            'FCI_scope':'only saved energy as evaluation lower bound; no FCI amplitudes loaded',
            'audit_wall_seconds':perf_counter()-start,'audit_code_sha256':file_digest(__file__),
            'database_sha256':file_digest(root/'events.sqlite'),'whole_goal_complete':False}
    save_json(root/'validation.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/multibasis-NOCI-standard-v2'))
    print(json.dumps(audit(p.parse_args().output)))
