"""Bounded classical NOCI bases and actual interleaved-JW measurement rotations.

This is an independently specified baseline, NOT a reproduction of the optimized
CDF/selection/global-refinement algorithm of arXiv:2608.11569v2.
"""
from __future__ import annotations
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import zlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from qiskit import QuantumCircuit,qpy
import ffsim
from .common import ROOT,file_digest,resolve_artifact,rng_for
from .nonorthogonal_ci import occupied_columns,generalized_ground

METHODS = ("B1:source_final_ceo_ovp","B1:source_final_ceo_mvp","B1:lucj_ccsd_2_vqe",
           "RAACS:full_cost-qsci_local_grid-last_only-s101-exact_small")


def rotation_from_angles(angles,norb,noccupied):
    pairs=[(i,a) for i in range(noccupied) for a in range(noccupied,norb)]
    if len(angles)!=len(pairs):raise ValueError("Occupied/virtual rotation parameters")
    k=np.zeros((norb,norb))
    for angle,(i,a) in zip(angles,pairs):k[a,i]=angle;k[i,a]=-angle
    return expm(k)


def construct_bases(backend,info,evaluation_cap=48):
    """Two sequential restricted real orbital optimizations, HF retained.

    Energy/coefficients are computed only in the <=3 rotated-HF Slater space.
    Bounded Powell is not claimed stationary. Best actually evaluated rotations
    are selected, including probes. No sampler state, reference/FCI label, or
    measurements enter basis construction.
    """
    if info['nalpha']!=info['nbeta']:raise ValueError("Closed-shell basis protocol")
    start=perf_counter();n=backend.norb;occ=info['nalpha'];hf=info['hf_id']
    rotations=[np.eye(n)];fixed=[occupied_columns(rotations[0],hf)];trace=[]
    p=occ*(n-occ)
    for stage in range(2):
        rng,seed=rng_for('bounded-NOCI-measurement-bases-v1',info['problem_id'],stage)
        initial=rng.normal(0,.25,p);best=None;stage_trace=[]
        def objective(angles):
            nonlocal best
            u=rotation_from_angles(angles,n,occ)
            columns=fixed+[occupied_columns(u,hf)]
            h,s=backend.matrices(columns);out,_=generalized_ground(h,s)
            row={'stage':stage,'evaluation':len(stage_trace),'angles':np.asarray(angles).tolist(),
                 'energy_Ha':out['energy_Ha'],'rank':out['rank']}
            stage_trace.append(row)
            if best is None or out['energy_Ha']<best[0]:best=(out['energy_Ha'],u.copy(),row)
            return out['energy_Ha']
        opt=minimize(objective,initial,method='Powell',options={'maxfev':evaluation_cap,'maxiter':evaluation_cap,
                   'xtol':1e-4,'ftol':1e-7})
        rotations.append(best[1]);fixed.append(occupied_columns(best[1],hf))
        trace.append({'seed':str(seed),'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message),
                      'selection':'lowest actually evaluated NOCI energy','best':best[2],'trace':stage_trace})
    h,s=backend.matrices(fixed);ground,_=generalized_ground(h,s)
    return rotations,{'basis_protocol':'bounded-greedy-NOCI-real-restricted-v1','bases':3,'basis_objective_evaluations':sum(len(x['trace']) for x in trace),
            'basis_construction_seconds':perf_counter()-start,'stages':trace,'classical_anchor_ground':ground,
            'classical_anchor_H_real':h.real.tolist(),'classical_anchor_S_real':s.real.tolist(),
            'basis_construction_quantum_shots':0,'FCI_used':False,'sampler_state_used':False,
            'not_original_CDF_NOCI_reproduction':True,'orbital_parameters_per_nonidentity_basis':p}


def measurement_rotation(preparation,rotation):
    """Apply Gamma(U†) to full preparation BEFORE any Z-only phase deletion.

    ffsim uses blocked spin creators; the benchmark uses interleaved creators.
    The fermionic permutation phases must appear on BOTH sides for a generic
    input state, not only after an HF preparation.
    """
    n=len(rotation)
    if preparation.num_qubits!=2*n or preparation.num_clbits:raise ValueError("Unmeasured interleaved-JW input")
    result=preparation.copy()
    if np.max(abs(rotation-np.eye(n)))<1e-14:return result
    wires=[2*n-1-2*i for i in range(n)]+[2*n-2-2*i for i in range(n)]
    def phase():
        for a in range(n):
            for b in range(a):result.cz(2*n-1-2*a,2*n-2-2*b)
    phase()
    blocked=QuantumCircuit(2*n);blocked.append(ffsim.qiskit.OrbitalRotationJW(n,np.asarray(rotation).conj().T),range(2*n))
    result.compose(blocked,qubits=wires,inplace=True)
    phase()
    return result


def source_preparation(case,method):
    folder=ROOT/'results/RAACS-independent-comparison-standard-sharded-v1/readouts'/case
    db=sqlite3.connect(f"file:{folder/'readouts.sqlite'}?mode=ro",uri=True)
    row=db.execute('SELECT metadata FROM cells WHERE method=?',(method,)).fetchone();db.close()
    if row is None:return None,{'status':'MISSING_SOURCE_METHOD','case':case,'method':method}
    metadata=json.loads(row[0]);path=Path(metadata['result_path'])
    if file_digest(path)!=metadata['result_sha256']:raise RuntimeError('Source result hash')
    record=json.loads(path.read_text());origin={'source_result':str(path),'source_result_sha256':metadata['result_sha256'],
        'historical_construction_record':metadata,'historical_quantum_objective_shots':None,
        'state_scope':'full preparation, not phase-stripped Z sampler'}
    if method.startswith('RAACS:'):
        ep=path.parent/'events.sqlite'
        if file_digest(ep)!=record['event_database_sha256']:raise RuntimeError('Source events hash')
        allocations=record['final_circuit_allocations']
        if len(allocations)!=1:raise ValueError('Only registered single final sampler')
        cid=allocations[0]['circuit_id'];db=sqlite3.connect(f'file:{ep}?mode=ro',uri=True)
        raw=zlib.decompress(db.execute('SELECT preparation_qpy FROM circuits WHERE id=?',(cid,)).fetchone()[0]);db.close()
        origin.update(source_events_sha256=record['event_database_sha256'],source_circuit_id=cid)
    else:
        p=resolve_artifact(ROOT/'results/standard-v1',record['circuit_qpy']) if 'circuit_qpy' in record else path.parent/'preparation.qpy'
        raw=p.read_bytes();origin.update(preparation_qpy=str(p),preparation_sha256=file_digest(p))
        if 'circuit_sha256' in record and origin['preparation_sha256']!=record['circuit_sha256']:raise RuntimeError('Preparation hash')
    qc=qpy.load(BytesIO(raw))[0]
    if qc.num_qubits>16 or qc.num_clbits:raise ValueError('Registered small unmeasured preparation only')
    return qc,origin
