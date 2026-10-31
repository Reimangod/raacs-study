"""N37: fixed 4x4 SqDRIFT-N20 prefix, same TKET opportunities for all inputs."""
import argparse
import base64
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
from time import process_time, perf_counter
import traceback
from qiskit import qpy
from .paths import ROOT
from .io import load, save, sha256, space
from .tket_qualification import compile_tket, MODES, binding as qualifier_binding
from .tket_runtime_preflight import RUN as QUALIFY
from .strong_projection import expected_addresses, source_plans, OVERLAY, FIELDS
from .frozen_projection_screen import read_selected
from paperstudy.lucj import circuit_views
from paperstudy.sqdrift_execution import member_state

RUN = ROOT/'runs/N37-TKET-FIXED-PREFIX-v1'
PARENT = ROOT/'runs/G4-STRONG-PROJECTION-v1/compiled-members.sqlite'
CASES = ['NAPH20-BLOCK-BOYS','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2']
INPUTS = ['NATIVE','RUSTIQ','PROJECTED']


def no_increase(a,b):
    return all(a[k] <= b[k] for k in FIELDS)


def select(resources):
    native=resources['NATIVE']
    controls = [n for n,r in resources.items() if not n.startswith('PROJECTED') and no_increase(r,native)]
    key=lambda n: tuple(resources[n][k] for k in FIELDS)+(n,)
    base=min(controls,key=key)
    proposals=[n for n,r in resources.items() if n.startswith('PROJECTED') and no_increase(r,resources[base])]
    selected=min([base]+proposals,key=key)
    return {'control':base,'selected':selected,'control_resources':resources[base],
            'selected_resources':resources[selected],
            'no_increase_vs_strong_generic':no_increase(resources[selected],resources[base]),
            'no_increase_vs_native':no_increase(resources[selected],native),
            'selection_uses_energy_or_counts':False}


def binding():
    if qualifier_binding()!=load(QUALIFY/'qualification-freeze.json'):
        raise RuntimeError('Qualified adapter drift')
    q=load(QUALIFY/'qualification-terminal.json')
    if not q['status'].startswith('PASSED_16') or len(q['rows'])!=16 or q['failure']:
        raise RuntimeError('N36 qualifier not passed')
    plans=source_plans({'cases':CASES,'members_per_case':4})
    d=sqlite3.connect('file:'+str(PARENT)+'?mode=ro',uri=True)
    identities=[]
    for source in plans:
        for member in source['members']:
            row=d.execute('select record,native_gz,rustiq_gz,projected_gz,certificate_gz from members where case_id=? and idx=?',
                          (source['case'],member['index'])).fetchone()
            rec=json.loads(row[0])
            if rec['status']!='CERTIFIED_RESOURCE_POINT' or rec['member_id']!=member['member_id']:
                raise ValueError('Frozen source prefix mismatch')
            identities.append({'case':source['case'],'index':member['index'],'member_id':member['member_id'],
                               'record_SHA256':sha256(row[0].encode()),
                               'artifact_SHA256':[sha256(x) for x in row[1:]]})
    d.close()
    return {'implementation_SHA256':sha256(Path(__file__).read_bytes()),
            'adapter_qualification_SHA256':sha256((QUALIFY/'qualification-terminal.json').read_bytes()),
            'adapter_freeze_SHA256':sha256((QUALIFY/'qualification-freeze.json').read_bytes()),
            'parent_db_SHA256':sha256(PARENT.read_bytes()),'sources':plans,'identities':identities,
            'inputs':INPUTS,'compiler_modes':list(MODES),'postcompile':'UNCHANGED_circuit_views_LOGICAL_OPT1_SEED8137',
            'policy':'MIN_CNOT_THEN_CXDEPTH_THEN_TOTAL_NAME_SAFE_VS_NATIVE;PROJECTED_MUST_NOT_INCREASE_ANY_CONTROL_METRIC',
            'max_parent_CPU_seconds':300,'max_wall_seconds':600,'max_verifier_wall_seconds_per_member':45,
            'max_new_run_bytes':16*2**20,'probability_L1_tolerance':1e-8,
            'new_model_shots':0,'new_energy_CI_optimizer':0,
            'scope':'FIXED_FIRST4_OF783_EACH4CASE_NOT_FULLBANK_NOT_PCOAST_MQT_REPRODUCTION_NOT_HARDWARE'}


def blob(qc):
    s=BytesIO();qpy.dump(qc,s);return gzip.compress(s.getvalue(),mtime=0)


def run():
    spec=load(RUN/'freeze.json')
    if binding()!=spec:raise RuntimeError('N37 drift')
    if (RUN/'started.json').exists():raise RuntimeError('No duplicate/restart of frozen N37')
    space();cpu,wall=process_time(),perf_counter()
    save(RUN/'started.json',{'freeze_SHA256':sha256((RUN/'freeze.json').read_bytes())})
    old=sqlite3.connect('file:'+str(PARENT)+'?mode=ro',uri=True)
    new=sqlite3.connect(RUN/'compiled.sqlite')
    new.execute('PRAGMA synchronous=FULL')
    new.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,record TEXT,PRIMARY KEY(case_id,idx))')
    new.execute('CREATE TABLE circuits(case_id TEXT,idx INTEGER,variant TEXT,qpy_gz BLOB,PRIMARY KEY(case_id,idx,variant))')
    rows=[]; failure=None
    try:
        for source in spec['sources']:
            artifacts=read_selected([e for m in source['members'] for e in m['artifacts'].values()])
            meta=load(source['meta_path'])
            for member in source['members']:
                space()
                if process_time()-cpu>spec['max_parent_CPU_seconds'] or perf_counter()-wall>spec['max_wall_seconds']:
                    raise TimeoutError('N37 fixed global cap')
                if sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())>spec['max_new_run_bytes']:
                    raise OSError('N37 run budget')
                raw=old.execute('select record,certificate_gz,native_gz,rustiq_gz,projected_gz from members where case_id=? and idx=?',
                                (source['case'],member['index'])).fetchone()
                rec=json.loads(raw[0]); certificate=json.loads(gzip.decompress(raw[1]))
                original=json.loads(gzip.decompress(artifacts[member['artifacts']['member.json.gz']['path']]))
                expected,_=member_state(meta,original)
                resources={}; evidence={}; compiled={}; requests={}; times={}
                for name,b in zip(INPUTS,raw[2:]):
                    if sha256(b)!=rec['qpy_gz_SHA256'][name]:raise ValueError('Old QPY hash mismatch')
                    qc=qpy.load(BytesIO(gzip.decompress(b)))[0]
                    probabilities=expected_addresses(expected,int(meta['n_qubits']),original['permutation'],certificate,name=='PROJECTED')
                    resources[name]=rec['resources'][name];compiled[name]=b
                    requests[name]={'qpy_gz_base64':base64.b64encode(b).decode(),'expected_probabilities':probabilities}
                    for mode in MODES:
                        label=name+'_'+mode;tc,tw=process_time(),perf_counter()
                        output,info=compile_tket(qc,mode)
                        _,sample,views=circuit_views(output)
                        times[label]={'CPU_seconds':process_time()-tc,'wall_seconds':perf_counter()-tw}
                        resources[label]=views['Z_sampling_equivalent'];evidence[label]=info;compiled[label]=blob(sample)
                        requests[label]={'qpy_gz_base64':base64.b64encode(compiled[label]).decode(),'expected_probabilities':probabilities}
                env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(OVERLAY),str(ROOT)]))
                child=subprocess.run([sys.executable,'-m','frontierloop.mps_distribution_check'],
                                     input=json.dumps({'circuits':requests,'tolerance':spec['probability_L1_tolerance']}),
                                     text=True,capture_output=True,timeout=spec['max_verifier_wall_seconds_per_member'],env=env)
                if child.returncode:raise RuntimeError('Independent MPS verifier: '+child.stderr[-4000:])
                check=json.loads(child.stdout)
                policy=select(resources)
                if not policy['no_increase_vs_strong_generic'] or not policy['no_increase_vs_native']:
                    raise AssertionError('Resource safety guard')
                row={'case':source['case'],'index':member['index'],'member_id':member['member_id'],
                     'status':'VERIFIED_ALL9_VARIANTS','resources':resources,'policy':policy,'compiler_evidence':evidence,
                     'compile_cost':times,'independent_MPS_check':check,'problem_id':rec['problem_id'],
                     'parent_record_SHA256':sha256(raw[0].encode()),'qpy_gz_SHA256':{k:sha256(v) for k,v in compiled.items()},
                     'parameters_independent_rank':'UNKNOWN_NO_PARAMETER_REDUCTION_CLAIM',
                     'sampling_probability_only_not_general_noise_or_unitary_equivalence':True,
                     'new_model_shots':0,'new_energy_CI_optimizer':0}
                new.execute('INSERT INTO members VALUES(?,?,?)',(source['case'],member['index'],json.dumps(row,sort_keys=True,allow_nan=False)))
                for name,b in compiled.items():
                    new.execute('INSERT INTO circuits VALUES(?,?,?,?)',(source['case'],member['index'],name,b))
                new.commit();rows.append(row)
                print(source['case'],member['index'],policy['control'],policy['selected'],
                      [policy['control_resources'][k] for k in FIELDS],'->',
                      [policy['selected_resources'][k] for k in FIELDS],flush=True)
        status='COMPLETED_ALL16_MEMBERS_144_PROBABILITY_CHECKS'
    except Exception as exc:
        failure={'error':repr(exc),'traceback':traceback.format_exc()};status='FAILED_ENGINEERING_PRESERVED'
    finally:
        old.close();new.close()
    terminal={'status':status,'completed_members':len(rows),'failure':failure,'CPU_seconds_parent':process_time()-cpu,
              'wall_seconds':perf_counter()-wall,'verifier_CPU_seconds':sum(r['independent_MPS_check']['CPU_seconds'] for r in rows),
              'new_model_QPU_shots':[0,0],'new_energy_CI_optimizer':[0,0,0]}
    save(RUN/'terminal.json',terminal);print(json.dumps(terminal))
    if failure:raise RuntimeError('N37 failed; preserve partial evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    if a.action=='freeze':
        if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'freeze.json',binding());print('N37 fixed prefix frozen, no outcomes observed')
    else:
        resource.setrlimit(resource.RLIMIT_CPU,(300,300));run()
