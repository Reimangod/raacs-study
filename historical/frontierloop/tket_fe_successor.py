"""N38 additive completion of only four unverified Fe members after N37 grammar failure."""
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
from time import process_time,perf_counter
import traceback
from qiskit import qpy
from .paths import ROOT
from .io import load,save,sha256,space
from .tket_swap_successor import compile_tket,binding as qualifier_binding,RUN as QUALIFY
from .tket_prefix_study import INPUTS,MODES,FIELDS,PARENT,OVERLAY,blob,select,expected_addresses,read_selected,member_state,circuit_views

RUN=ROOT/'runs/N38-TKET-FE-SUCCESSOR-v1'
PREVIOUS=ROOT/'runs/N37-TKET-FIXED-PREFIX-v1'


def binding():
    terminal=load(PREVIOUS/'terminal.json');q=load(QUALIFY/'terminal.json')
    if terminal['completed_members']!=12 or 'swap' not in terminal['failure']['error']:
        raise RuntimeError('Unexpected incident state')
    if qualifier_binding()!=load(QUALIFY/'freeze.json') or len(q['rows'])!=18 or not q['unknown_gate_rejected']:
        raise RuntimeError('Exact SWAP repair not qualified')
    previous=load(PREVIOUS/'freeze.json')
    if sha256(PARENT.read_bytes())!=previous['parent_db_SHA256']:raise ValueError('Original G4 data changed')
    d=sqlite3.connect('file:'+str(PREVIOUS/'compiled.sqlite')+'?mode=ro',uri=True)
    keys=set(d.execute('select case_id,idx from members'));d.close()
    planned={(s['case'],m['index']) for s in previous['sources'] for m in s['members']}
    remain=planned-keys
    if remain!={('FE2S2-40-P2',i) for i in range(4)}:raise RuntimeError('Would rerun completed work or change scope')
    return {'implementation_SHA256':sha256(Path(__file__).read_bytes()),
            'old_N37_freeze_SHA256':sha256((PREVIOUS/'freeze.json').read_bytes()),
            'old_N37_terminal_SHA256':sha256((PREVIOUS/'terminal.json').read_bytes()),
            'old_N37_db_SHA256':sha256((PREVIOUS/'compiled.sqlite').read_bytes()),
            'new_qualifier_SHA256':sha256((QUALIFY/'terminal.json').read_bytes()),
            'new_qualifier_freeze_SHA256':sha256((QUALIFY/'freeze.json').read_bytes()),
            'sources':[s for s in previous['sources'] if s['case']=='FE2S2-40-P2'],
            'remaining':sorted([list(k) for k in remain]),'original_source_db_SHA256':sha256(PARENT.read_bytes()),
            'max_new_parent_CPU_seconds':300,'max_new_wall_seconds':330,
            'previous_paid_parent_CPU_seconds':terminal['CPU_seconds_parent'],
            'combined_parent_CPU_limit_less_than_600_seconds':True,
            'max_new_run_bytes':4*2**20,'probability_L1_tolerance':1e-8,
            'change':'ONLY_EXACT_SWAP_QASM_GRAMMAR;SAME_2_PASSES_PER3INPUTS;SAME_SELECTION_AND_QISKIT_OPT1',
            'new_model_QPU_shots':[0,0],'new_energy_CI_optimizer':[0,0,0]}


def process_member(source,member,artifacts,old,spec):
    cpu,wall=process_time(),perf_counter()
    meta=load(source['meta_path'])
    raw=old.execute('select record,certificate_gz,native_gz,rustiq_gz,projected_gz from members where case_id=? and idx=?',
                    (source['case'],member['index'])).fetchone()
    rec=json.loads(raw[0]);cert=json.loads(gzip.decompress(raw[1]))
    original=json.loads(gzip.decompress(artifacts[member['artifacts']['member.json.gz']['path']]))
    expected,_=member_state(meta,original)
    resources={};evidence={};compiled={};requests={};times={}
    for name,b in zip(INPUTS,raw[2:]):
        if sha256(b)!=rec['qpy_gz_SHA256'][name]:raise ValueError('Old QPY changed')
        qc=qpy.load(BytesIO(gzip.decompress(b)))[0]
        probs=expected_addresses(expected,int(meta['n_qubits']),original['permutation'],cert,name=='PROJECTED')
        resources[name]=rec['resources'][name];compiled[name]=b
        requests[name]={'qpy_gz_base64':base64.b64encode(b).decode(),'expected_probabilities':probs}
        for mode in MODES:
            label=name+'_'+mode;c,w=process_time(),perf_counter()
            output,info=compile_tket(qc,mode)
            _,sample,views=circuit_views(output)
            resources[label]=views['Z_sampling_equivalent'];evidence[label]=info;compiled[label]=blob(sample)
            times[label]={'CPU_seconds':process_time()-c,'wall_seconds':perf_counter()-w}
            requests[label]={'qpy_gz_base64':base64.b64encode(compiled[label]).decode(),'expected_probabilities':probs}
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(OVERLAY),str(ROOT)]))
    child=subprocess.run([sys.executable,'-m','frontierloop.mps_distribution_check'],
                         input=json.dumps({'circuits':requests,'tolerance':spec['probability_L1_tolerance']}),
                         text=True,capture_output=True,timeout=45,env=env)
    if child.returncode:raise RuntimeError('Independent MPS: '+child.stderr[-4000:])
    check=json.loads(child.stdout);policy=select(resources)
    row={'case':source['case'],'index':member['index'],'member_id':member['member_id'],'status':'VERIFIED_ALL9_VARIANTS',
         'resources':resources,'policy':policy,'compiler_evidence':evidence,'compile_cost':times,
         'independent_MPS_check':check,'problem_id':rec['problem_id'],'parent_record_SHA256':sha256(raw[0].encode()),
         'qpy_gz_SHA256':{k:sha256(v) for k,v in compiled.items()},
         'parameters_independent_rank':'UNKNOWN_NO_PARAMETER_REDUCTION_CLAIM',
         'member_CPU_seconds_parent':process_time()-cpu,'member_wall_seconds':perf_counter()-wall,
         'new_model_shots':0,'new_energy_CI_optimizer':0}
    return row,compiled


def run():
    spec=load(RUN/'freeze.json')
    if binding()!=spec:raise RuntimeError('N38 drift')
    if (RUN/'started.json').exists():raise RuntimeError('No duplicate N38')
    space();cpu,wall=process_time(),perf_counter();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    child_before=children.ru_utime+children.ru_stime
    save(RUN/'started.json',{'freeze_SHA256':sha256((RUN/'freeze.json').read_bytes())})
    old=sqlite3.connect('file:'+str(PARENT)+'?mode=ro',uri=True);new=sqlite3.connect(RUN/'compiled.sqlite')
    new.execute('PRAGMA synchronous=FULL')
    new.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,record TEXT,PRIMARY KEY(case_id,idx))')
    new.execute('CREATE TABLE circuits(case_id TEXT,idx INTEGER,variant TEXT,qpy_gz BLOB,PRIMARY KEY(case_id,idx,variant))')
    rows=[];failure=None
    try:
        for source in spec['sources']:
            artifacts=read_selected([e for m in source['members'] for e in m['artifacts'].values()])
            for member in source['members']:
                space()
                if process_time()-cpu>spec['max_new_parent_CPU_seconds'] or perf_counter()-wall>spec['max_new_wall_seconds']:
                    raise TimeoutError('New N38 cap')
                row,compiled=process_member(source,member,artifacts,old,spec)
                new.execute('INSERT INTO members VALUES(?,?,?)',(source['case'],member['index'],json.dumps(row,sort_keys=True,allow_nan=False)))
                for name,b in compiled.items():new.execute('INSERT INTO circuits VALUES(?,?,?,?)',(source['case'],member['index'],name,b))
                new.commit();rows.append(row)
                if (RUN/'compiled.sqlite').stat().st_size>spec['max_new_run_bytes']:raise OSError('New N38 storage cap')
                p=row['policy'];print(source['case'],member['index'],p['control'],p['selected'],
                                      [p['control_resources'][k] for k in FIELDS],'->',[p['selected_resources'][k] for k in FIELDS],flush=True)
        status='COMPLETED_4_PREVIOUSLY_UNVERIFIED_FE_MEMBERS_36_PROBABILITY_CHECKS'
    except Exception as exc:
        failure={'error':repr(exc),'traceback':traceback.format_exc()};status='FAILED_ENGINEERING_PRESERVED'
    finally:old.close();new.close()
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    terminal={'status':status,'completed_members':len(rows),'failure':failure,'CPU_seconds_parent':process_time()-cpu,
              'CPU_seconds_children_including_startup':children.ru_utime+children.ru_stime-child_before,
              'verifier_CPU_seconds':sum(r['independent_MPS_check']['CPU_seconds'] for r in rows),
              'wall_seconds':perf_counter()-wall,'old_paid_cost_never_deleted':spec['previous_paid_parent_CPU_seconds'],
              'new_model_QPU_shots':[0,0],'new_energy_CI_optimizer':[0,0,0]}
    save(RUN/'terminal.json',terminal);print(json.dumps(terminal))
    if failure:raise RuntimeError('N38 partial evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    if a.action=='freeze':
        if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'freeze.json',binding());print('N38 exact4 remaining frozen; old12 not rerun')
    else:resource.setrlimit(resource.RLIMIT_CPU,(300,300));run()
