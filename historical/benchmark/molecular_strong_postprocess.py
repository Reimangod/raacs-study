"""All-cell same-circuit MPS/native-resource analysis; no extra acquisitions."""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
from time import perf_counter
import zlib
import numpy as np
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,save_json,save_binary
from .ansatz import circuit_bytes
from .molecular_strong_analysis import final_evidence
from .mps_layout_qualification import permute_sampling
from .molecular_mps_scaling import no_full_state_guards
from .mps_sampling import construct_mps
from .gate_noise_runtime import compile_sampler
from .study.core import rss_peak


def cfg(source):
    source=Path(source).resolve();m=json.loads((source/'manifest.json').read_text())
    assert m['schema']=='molecular-strong-controls-v2-spin-blocked'
    layout=source/'physical-layout-validation-v2.json'
    assert layout.exists(), 'Actual terminal source audit required'
    return {'schema':'molecular-strong-postprocess-v1','source':str(source),
        'source_manifest_sha256':file_digest(source/'manifest.json'),
        'source_validation_sha256':file_digest(source/'validation.json'),
        'source_physical_layout_validation_sha256':file_digest(layout),
        'queue':m['queue'],'methods':m['methods'],'bonds':[128,256],'cutoff':1e-12,
        'construction_seconds_cap':120.,'RSS_cap_bytes':2*2**30,
        'free_floor_bytes':int(2.75*2**30),'output_cap_bytes':128*2**20,
        'targets':['all_to_all_cx','line_cx'],'hardware_shots':0,'new_model_shots':0,
        'full_Fock_vector_allowed':False,'FCI_allowed':False,
        'implementation':{p:file_digest(ROOT/'benchmark'/p) for p in (
            'molecular_strong_postprocess.py','molecular_strong_analysis.py','mps_layout_qualification.py',
            'molecular_mps_scaling.py','mps_sampling.py','gate_noise_runtime.py','ansatz.py','common.py')},
        'spec_sha256':file_digest(ROOT/'study-v1/MOLECULAR_STRONG_POSTPROCESS_V1.md'),
        'packages':{p:importlib.metadata.version(p) for p in ('qiskit','quimb','numpy','scipy')}}


def boundary(root,c):
    if shutil.disk_usage(root).free<c['free_floor_bytes']:return 'PENDING_CAPACITY_BOUNDARY'
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>c['output_cap_bytes']:return 'PENDING_OUTPUT_BOUNDARY'


def structural_native_check(compiled,resource,nqubits):
    mapping=resource['measurement_physical_to_logical_clbit']
    # Runtime JSON map is a list of [physical, classical] pairs.
    pairs=list(mapping.items()) if isinstance(mapping,dict) else [tuple(p) for p in mapping]
    assert len(pairs)==nqubits and sorted(int(v) for _,v in pairs)==list(range(nqubits))
    assert sorted(int(k) for k,_ in pairs)==list(range(nqubits))
    actual=[(compiled.find_bit(g.qubits[0]).index,compiled.find_bit(g.clbits[0]).index)
            for g in compiled.data if g.operation.name=='measure']
    assert sorted(actual)==sorted(pairs)
    assert int(compiled.count_ops().get('cx',0))==resource['native_2q_count']
    if resource['target']=='line_cx':
        for gate in compiled.data:
            if gate.operation.num_qubits==2:
                a,b=[compiled.find_bit(q).index for q in gate.qubits];assert abs(a-b)==1
    return True


def diagnostic(sample,c):
    blocked,mapping=permute_sampling(sample);rows=[];states=[]
    for bond in c['bonds']:
        start=perf_counter()
        try:
            state,meta=construct_mps(blocked,bond,c['cutoff'],c['construction_seconds_cap'],c['RSS_cap_bytes'])
            psi=state.psi.copy();psi.normalize();states.append((bond,psi))
            row={'bond':bond,'status':'COMPLETED','metadata':meta}
        except Exception as exc:row={'bond':bond,'status':'RESOURCE_OR_ENGINEERING_FAILURE_PRESERVED','error':repr(exc)}
        row['wall_seconds']=perf_counter()-start;rows.append(row)
    if len(states)==len(c['bonds']):
        overlap=float(abs(states[0][1].H@states[-1][1])**2)
        assert -1e-8<=overlap<=1+1e-8
    else:overlap=None
    return {'logical_to_simulator_qubit':mapping,'rows':rows,'normalized_overlap_squared':overlap,
        'same_physical_circuit':True,'new_model_shots':0,'not_exact_error_certificate':True}


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def report(root,c,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'report-v1';folder.mkdir(exist_ok=True);status=[];native=[];bonds=[]
    for r in records:
        status.append({k:r.get(k) for k in ('case','method','status','source_status','n_qubits','wall_seconds','peak_process_RSS_hwm')})
        for n in r.get('native',[]):
            native.append({'case':r['case'],'method':r['method'],'n_qubits':r['n_qubits'],**n})
        for b in r.get('MPS',{}).get('rows',[]):
            bonds.append({'case':r['case'],'method':r['method'],'n_qubits':r['n_qubits'],'bond':b['bond'],
                'status':b['status'],'retained_norm_squared':b.get('metadata',{}).get('retained_norm_squared'),
                'normalized_overlap_squared_128_256':r['MPS']['normalized_overlap_squared'],
                'wall_seconds':b['wall_seconds']})
    write_csv(folder/'all-planned-status.csv',status);write_csv(folder/'native-resources.csv',native);write_csv(folder/'bond-convergence.csv',bonds)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in c['methods']:
        nn=[r for r in native if r['method']==method and r.get('target')=='line_cx' and r.get('native_2q_count') is not None]
        if nn:
            axes[0].scatter([r['n_qubits'] for r in nn],[r['native_2q_count'] for r in nn],s=20,label=method)
        bb=[r for r in bonds if r['method']==method and r['bond']==128 and r['normalized_overlap_squared_128_256'] is not None]
        if bb:axes[1].scatter([r['n_qubits'] for r in bb],[max(0,1-r['normalized_overlap_squared_128_256']) for r in bb],s=20,label=method)
    axes[0].set(xlabel='Qubits',ylabel='Native line-CX count',yscale='symlog')
    axes[1].set(xlabel='Qubits',ylabel='1 − normalized overlap² (bond128 vs256)',yscale='symlog')
    for ax in axes:ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7)
    fig.suptitle('Same final circuits | diagnostic only | missing/capped retained | hardware0')
    fig.tight_layout(rect=(0,.20,1,.94))
    for ext in ('png','svg','pdf'):fig.savefig(folder/('native-resources-and-bond-convergence.'+ext),dpi=180)
    plt.close(fig)
    (folder/'REPORT.md').write_text(f'''# Identical-circuit native-resource and MPS diagnostics

{len(records)} planned records. Every source cell remains, including failed or
missing circuits. Classical SCI is not represented as a zero-gate quantum state.
All native targets are SYNTHETIC, not calibrated hardware. Routing includes
actual SWAPs before CX decomposition; logical readout maps are structural checks.
No exact full-state verification of large compiled distributions is claimed.

Bond128/256 compare identical physical final circuits, not independently trained
ansätze. Norm and normalized overlap are empirical convergence diagnostics, not
exact distribution error certificates. Failures/caps remain, not algorithm losses.
No new model or hardware shots, no FCI/full-Fock/sector vector. All work here is
additional classical analysis, not free policy construction. Values below zero
from roundoff in1-overlap are shown as0; underlying overlaps remain unchanged.

Source: {c['source']}. Main source CI/bitmapping/cost audits remain separate.
This does not complete the six-phase project or establish hardware superiority.
''')
    files={str(p.relative_to(folder)):file_digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='manifest.json'}
    save_json(folder/'manifest.json',{'source_config_sha256':file_digest(root/'manifest.json'),'files':files,'whole_goal_complete':False})


def run(source,output):
    source=Path(source).resolve();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);c=cfg(source)
    with (root/'run.lock').open('a') as lock,threadpool_limits(limits=1),no_full_state_guards():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=root/'manifest.json'
        if path.exists():assert json.loads(path.read_text())==c
        else:save_json(path,c)
        completion=root/'execution-status.json'
        if completion.exists() and json.loads(completion.read_text())['status']=='MOLECULAR_NATIVE_AND_BOND_ANALYSIS_EXECUTED_REPORTED':
            report_manifest=json.loads((root/'report-v1/manifest.json').read_text())
            assert report_manifest['source_config_sha256']==file_digest(root/'manifest.json')
            for p,sha in report_manifest['files'].items():assert file_digest(root/'report-v1'/p)==sha
            return
        records=[]
        for item in c['queue']:
            for method in c['methods']:
                key=digest([item['id'],method]);folder=root/'cells'/key;folder.mkdir(parents=True,exist_ok=True);path=folder/'result.json'
                if path.exists():records.append(json.loads(path.read_text()));continue
                if why:=boundary(root,c):save_json(root/'execution-status.json',{'status':why,'terminal':len(records)});return
                start=perf_counter();record={'case':item['id'],'method':method,'hardware_shots':0,'new_model_shots':0}
                try:
                    source_result=source/'cases'/item['id']/method/'result.json'
                    if not source_result.exists():
                        record.update(status='SOURCE_PHYSICAL_INPUT_MISSING',source_status=None)
                    elif method=='classical-SCI64':
                        record.update(status='CLASSICAL_CONTROL_NO_QUANTUM_CIRCUIT',source_status=json.loads(source_result.read_text())['status'])
                    else:
                        r,counts,paid,qc,physical=final_evidence(source,item,method)
                        record['source_result_sha256']=file_digest(source_result);record['source_status']=r['status']
                        if qc is None:record['status']='NO_FINAL_CIRCUIT_PRESERVED'
                        else:
                            record.update(n_qubits=qc.num_qubits,status='DIAGNOSTICS_EXECUTED',
                                source_database_sha256=physical['source_database_sha256'],final_circuit_id=physical['final_circuit_id'])
                            native=[]
                            for target in c['targets']:
                                tick=perf_counter()
                                try:
                                    compiled,resource=compile_sampler(qc,target)
                                    structural_native_check(compiled,resource,qc.num_qubits)
                                    archive=folder/(target+'.qpy.z');save_binary(archive,zlib.compress(circuit_bytes(compiled)))
                                    native.append({'status':'COMPLETED','wall_seconds':perf_counter()-tick,
                                        'compiled_qpy_sha256':file_digest(archive),**resource})
                                except Exception as exc:native.append({'target':target,'status':'NATIVE_COMPILE_FAILURE_PRESERVED','error':repr(exc)})
                                save_json(folder/'progress.json',{'native':native})
                            record['native']=native;record['MPS']=diagnostic(qc,c)
                except Exception as exc:record.update(status='ENGINEERING_FAILURE_PRESERVED',error=repr(exc))
                record['wall_seconds']=perf_counter()-start;record['peak_process_RSS_hwm']=rss_peak()
                save_json(path,record);records.append(record)
                save_json(root/'execution-status.json',{'status':'RUNNING','terminal':len(records),'planned':len(c['queue'])*len(c['methods'])})
                print(item['id'],method,record['status'],flush=True)
        report(root,c,records)
        save_json(root/'execution-status.json',{'status':'MOLECULAR_NATIVE_AND_BOND_ANALYSIS_EXECUTED_REPORTED',
            'terminal':len(records),'planned':len(c['queue'])*len(c['methods']),'hardware_shots':0,'new_model_shots':0,'whole_goal_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
