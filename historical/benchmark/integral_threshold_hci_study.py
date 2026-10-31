"""Versioned classical HCI variational comparisons on exact saved integrals."""
from dataclasses import asdict
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
from time import perf_counter
import traceback
from threadpoolctl import threadpool_limits
from .common import ROOT,digest,file_digest,save_json
from .resource_adaptive_sampler import integral_problem
from .molecular_mps_scaling import no_full_state_guards
from .threshold_hci import HCISpec,HeatBathConnections,variational


def queue(mode):
    source=ROOT/'results/standard-v1/references'
    small=(['H2_R0.7','H4_R1.1'] if mode=='quick' else sorted(p.name for p in source.iterdir() if p.is_dir()))
    items=[{'id':'original-'+case,'problem':str(source/case),'source_cohort':'original30'} for case in small]
    e=ROOT/'results/molecular-strong-controls-standard-v2'
    if (e/'manifest.json').exists():
        cfg=json.loads((e/'manifest.json').read_text())
        for item in cfg['queue']:
            if mode=='quick' and not item['id'].startswith('H10_R1.25_'):continue
            p=item['problem'] if item['kind']=='inherited' else str(e/'cases'/item['id']/'physical-problem')
            items.append({'id':'E-'+item['id'],'problem':p,'source_cohort':'E26',
                'E_case_path':str(e/'cases'/item['id'])})
    else:raise RuntimeError('Frozen E physical queue required; do not invent inputs')
    return items


def config(mode):
    return {'schema':'integral-threshold-HCI-variational-v1','mode':mode,'queue':queue(mode),
        'thresholds':[1e-2,3e-3,1e-3],'base_spec':asdict(HCISpec()),
        'free_floor_bytes':int(2.75*2**30),'output_cap_bytes':128*2**20,
        'E_source_manifest_sha256':file_digest(ROOT/'results/molecular-strong-controls-standard-v2/manifest.json'),
        'implementation':{p:file_digest(ROOT/'benchmark'/p) for p in ('integral_threshold_hci_study.py',
            'threshold_hci.py','integral_hamiltonian.py','resource_adaptive_sampler.py','molecular_mps_scaling.py','common.py')},
        'spec_sha256':file_digest(ROOT/'study-v1/INTEGRAL_THRESHOLD_HCI_V1.md'),
        'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','numba')},
        'quantum_shots':0,'hardware_shots':0,'FCI_used':False,'PT2_implemented':False}


def run(output,mode):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);cfg=config(mode)
    with (root/'run.lock').open('a') as lock,threadpool_limits(limits=1),no_full_state_guards():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=root/'manifest.json'
        if path.exists():assert json.loads(path.read_text())==cfg
        else:save_json(path,cfg)
        if mode=='standard':
            v=ROOT/'results/integral-threshold-HCI-quick-v1/validation.json'
            assert v.exists() and json.loads(v.read_text())['status']=='THRESHOLD_HCI_QUICK_INDEPENDENTLY_VALIDATED'
        rows=[]
        for item in cfg['queue']:
            case=root/'cases'/item['id'];case.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(root).free<cfg['free_floor_bytes'] or sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>cfg['output_cap_bytes']:
                save_json(root/'execution-status.json',{'status':'PENDING_CAPACITY_OR_OUTPUT_BOUNDARY','terminal':len(rows)});return
            problem=Path(item['problem']);metadata=problem/('problem.json' if (problem/'problem.json').exists() else 'reference.json')
            archive=problem/('integrals.npz' if (problem/'problem.json').exists() else 'reference.npz')
            if not metadata.exists() or not archive.exists():
                failure=Path(item.get('E_case_path',''))/'physical-failure.json'
                if not failure.exists():
                    save_json(root/'execution-status.json',{'status':'PENDING_PHYSICAL_INPUT','case':item['id'],'terminal':len(rows)});return
                for epsilon in cfg['thresholds']:
                    folder=case/('epsilon-'+str(epsilon));folder.mkdir(exist_ok=True)
                    r={'status':'SOURCE_PHYSICAL_INPUT_FAILURE_PRESERVED','source_failure_sha256':file_digest(failure)}
                    save_json(folder/'result.json',r);rows.append({'case':item['id'],'epsilon1':epsilon,**r})
                continue
            backend,info=integral_problem(problem)
            source={'metadata_sha256':file_digest(metadata),'archive_sha256':file_digest(archive),'problem':info,'path':str(problem)}
            if (case/'source.json').exists():assert json.loads((case/'source.json').read_text())==source
            else:save_json(case/'source.json',source)
            connections=None
            for epsilon in cfg['thresholds']:
                folder=case/('epsilon-'+str(epsilon));folder.mkdir(exist_ok=True);path=folder/'result.json'
                if path.exists():r=json.loads(path.read_text())
                elif (folder/'progress.json').exists() or (folder/'started.json').exists():
                    raise RuntimeError('Partial classical task preserved; no uncosted replay under same task')
                else:
                    save_json(folder/'started.json',{'case':item['id'],'epsilon1_Ha':epsilon,
                        'problem_id':info['problem_id'],'implementation_sha256':cfg['implementation']['threshold_hci.py'],
                        'cost_if_interrupted':'UNKNOWN_CLASSICAL_WORK; no silent replay','quantum_shots':0})
                    try:
                        if connections is None:
                            connections=HeatBathConnections(backend)
                            # A resumed case may rebuild the table. Each build is
                            # a new charged event; never overwrite the first one.
                            builds=case/'table-builds';builds.mkdir(exist_ok=True)
                            save_json(builds/(str(len(list(builds.glob('*.json'))))+'.json'),connections.setup)
                        spec=HCISpec(**{**cfg['base_spec'],'epsilon1_Ha':epsilon})
                        r=variational(backend,info['hf_id'],spec,connections,lambda h:save_json(folder/'progress.json',h))
                    except Exception as exc:r={'status':'ENGINEERING_FAILURE_PRESERVED','error':repr(exc),'traceback':traceback.format_exc()}
                    r.update(case=item['id'],problem_id=info['problem_id'],quantum_shots=0,hardware_shots=0)
                    save_json(path,r)
                rows.append({'case':item['id'],'epsilon1':epsilon,'status':r['status']})
                save_json(root/'execution-status.json',{'status':'RUNNING','terminal':len(rows),'planned':len(cfg['queue'])*3})
                print(item['id'],epsilon,r['status'],flush=True)
        save_json(root/'execution-status.json',{'status':'THRESHOLD_HCI_VARIATIONAL_COHORT_EXECUTED_PENDING_AUDIT',
            'terminal':len(rows),'planned':len(cfg['queue'])*3,'rows':rows,'quantum_shots':0,'hardware_shots':0,'whole_goal_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','standard'],default='quick');p.add_argument('--output')
    a=p.parse_args();run(a.output or ROOT/f'results/integral-threshold-HCI-{a.mode}-v1',a.mode)
