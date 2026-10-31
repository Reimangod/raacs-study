"""Same-circuit spin-blocked tensor layout diagnostic; no physical circuit edit.

Literal qubit permutation inside the simulator, with all measured bits mapped
back. This is not an orbital transformation, fermion-sign approximation, new
ansatz, or fewer physical gates. No dense/sector physical state is constructed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from qiskit import QuantumCircuit
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,save_json
from .molecular_strong_analysis import final_evidence
from .molecular_mps_scaling import no_full_state_guards
from .mps_sampling import construct_mps
from .molecular_strong_controls import boundary


def spin_blocked_mapping(nqubits):
    if nqubits%2:raise ValueError('Even interleaved spin register required')
    n=nqubits//2;mapping=[None]*nqubits
    for i in range(n):mapping[2*n-1-2*i]=i;mapping[2*n-2-2*i]=n+i
    assert sorted(mapping)==list(range(nqubits))
    return mapping


def permute_sampling(qc):
    if qc.num_clbits:raise ValueError('Unmeasured sampler required')
    mapping=spin_blocked_mapping(qc.num_qubits);out=QuantumCircuit(qc.num_qubits,name='spin-blocked-simulator-only')
    out.compose(qc,qubits=mapping,inplace=True)
    assert dict(out.count_ops())==dict(qc.count_ops())
    return out,mapping


def logical_id(backend_id,mapping):
    return sum(((int(backend_id)>>int(b))&1)<<a for a,b in enumerate(mapping))


def run(root):
    source=ROOT/'results/molecular-strong-controls-quick-v1';root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source/'manifest.json').read_text())
    selected=[(x,m) for x in cfg['queue'] if x['id'].startswith(('H10_','C6H6_pi_sigma12_'))
              for m in ('LUCJ-MP2-2-fixed','LUCJ-CCSD-2-fixed')]
    manifest={'source_manifest_sha256':file_digest(source/'manifest.json'),
        'source_validation_sha256':file_digest(source/'validation.json'),'implementation_sha256':file_digest(__file__),
        'mps_engine_sha256':file_digest(ROOT/'benchmark/mps_sampling.py'),
        'analysis_source_sha256':file_digest(ROOT/'benchmark/molecular_strong_analysis.py'),
        'tasks':[(x['id'],m) for x,m in selected],'bonds':[64,128,256],
        'cutoff':1e-12,'seconds_per_build':120,'RSS_cap_bytes':2*2**30,
        'new_quantum_model_shots':0,'no_new_orbital_or_circuit_optimization':True,'hardware_shots':0}
    path=root/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(manifest))
    else:save_json(path,manifest)
    records=[]
    with threadpool_limits(limits=1),no_full_state_guards():
        for item,method in selected:
            key=digest([item['id'],method]);p=root/(key+'.json')
            if p.exists():records.append(json.loads(p.read_text()));continue
            if why:=boundary(root,cfg):save_json(root/'execution-status.json',{'status':why,'records':len(records)});return
            r,counts,paid,qc,physical=final_evidence(source,item,method)
            blocked,mapping=permute_sampling(qc);states=[];rows=[]
            for bond in manifest['bonds']:
                tick=perf_counter()
                try:
                    state,metadata=construct_mps(blocked,bond,manifest['cutoff'],manifest['seconds_per_build'],manifest['RSS_cap_bytes'])
                    psi=state.psi.copy();psi.normalize();states.append((bond,psi))
                    row={'bond':bond,'status':'COMPLETED','metadata':metadata}
                except Exception as exc:row={'bond':bond,'status':'RESOURCE_OR_ENGINEERING_FAILURE_PRESERVED','error':repr(exc)}
                row['wall_seconds']=perf_counter()-tick;rows.append(row)
                save_json(root/'active-progress.json',{'case':item['id'],'method':method,'rows':rows})
            if states:
                refbond,ref=states[-1]
                for row in rows:
                    psi=next((s for b,s in states if b==row['bond']),None)
                    if psi is not None:
                        overlap=float(abs(psi.H@ref)**2);assert -1e-8<=overlap<=1+1e-8
                        row.update(normalized_overlap_squared_to_highest_completed_bond=overlap,reference_bond=refbond)
            oldnorm=min(x['metadata']['original_build_metadata']['retained_norm_squared'] for x in paid if x['metadata'])
            record={'case':item['id'],'method':method,'logical_to_simulator_qubit':mapping,'original_interleaved_bond64_minimum_norm_squared':oldnorm,
                'rows':rows,'source_database_sha256':physical['source_database_sha256'],
                'physical_gates_unchanged':True,'new_quantum_model_shots':0,'hardware_shots':0,
                'not_exact_error_certificate':True,'not_sampler_advantage_evidence':True}
            save_json(p,record);records.append(record);print(item['id'],method,[(x['bond'],x['status'],x.get('metadata',{}).get('retained_norm_squared')) for x in rows],flush=True)
        save_json(root/'execution-status.json',{'status':'SAME_CIRCUIT_LAYOUT_DIAGNOSTICS_COMPLETE','records':len(records),'planned':len(selected),
            'results':records,'new_quantum_model_shots':0,'hardware_shots':0,'whole_goal_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/molecular-MPS-layout-qualification-v1'))
    a=p.parse_args();run(a.output)
