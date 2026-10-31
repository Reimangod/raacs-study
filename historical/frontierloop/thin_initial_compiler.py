"""N39: single existing initial-state pass, no expensive peephole synthesis."""
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
from qiskit import qpy,qasm2
from qiskit.quantum_info import Statevector
from pytket.passes import SimplifyInitial
from pytket.qasm import circuit_from_qasm_str,circuit_to_qasm_str
from .paths import ROOT
from .io import load,save,sha256,space
from .tket_swap_successor import parse_export,binding as adapter_binding,RUN as ADAPTER
from .tket_qualification import fixtures,state_error
from .tket_prefix_study import INPUTS,FIELDS,PARENT,OVERLAY,blob,select,expected_addresses,read_selected,member_state,circuit_views

RUN=ROOT/'runs/N39-THIN-INITIAL-COMPILER-v1'
PREV=[ROOT/'runs/N37-TKET-FIXED-PREFIX-v1',ROOT/'runs/N38-TKET-FE-SUCCESSOR-v1']


def compile_thin(qc):
    if qc.num_clbits or any(x.operation.name in ('reset','measure','initialize') for x in qc.data):
        raise ValueError('Only unitary circuit with known zero input')
    tk=circuit_from_qasm_str(qasm2.dumps(qc))
    p=SimplifyInitial(allow_classical=False,create_all_qubits=True)
    p.apply(tk)
    implicit=tk.implicit_qubit_permutation();tk.replace_implicit_wire_swaps()
    text=circuit_to_qasm_str(tk);out=parse_export(text)
    if out.num_qubits!=qc.num_qubits or out.num_clbits or any(x.operation.name in ('reset','measure','initialize') for x in out.data):
        raise ValueError('Export width/nonunitary mismatch')
    return out,{'passes':[p.to_dict()],'pass_count':1,'peephole_resynthesis':False,
                'implicit_permutation_materialized':[[str(k),str(v)] for k,v in implicit.items()],
                'output_qasm_SHA256':sha256(text.encode()),'zero_state_equivalence_only':True,
                'full_unitary_equivalence_claimed':False}


def qualify():
    if (RUN/'qualifier.json').exists():raise RuntimeError('No qualifier overwrite')
    if adapter_binding()!=load(ADAPTER/'freeze.json'):raise ValueError('Qualified parser/runtime drift')
    cpu,wall=process_time(),perf_counter();rows=[]
    for name,qc in fixtures():
        out,info=compile_thin(qc);a,b=Statevector(qc).data,Statevector(out).data
        error=state_error(a,b);prob=float(sum(abs(abs(a)**2-abs(b)**2)))
        if error>1e-8 or prob>1e-8:raise ValueError('Thin initial-state qualifier failed')
        rows.append({'fixture':name,'state_error':error,'probability_L1':prob,'evidence':info})
    save(RUN/'qualifier.json',{'status':'ALL8_SMALL_ZERO_INPUT_CHECKS_PASSED','rows':rows,
                              'implementation_SHA256':sha256(Path(__file__).read_bytes()),
                              'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-wall})
    print('N39 all8 small initial-state checks passed')


def binding():
    q=load(RUN/'qualifier.json');implementation=sha256(Path(__file__).read_bytes())
    if q['implementation_SHA256']!=implementation or len(q['rows'])!=8 or not q['status'].startswith('ALL8'):
        raise RuntimeError('N39 qualifier drift/missing')
    if adapter_binding()!=load(ADAPTER/'freeze.json'):raise ValueError('Private adapter/runtime changed')
    prior=load(PREV[0]/'freeze.json')
    if sha256(PARENT.read_bytes())!=prior['parent_db_SHA256']:raise ValueError('Parent original source changed')
    for source in prior['sources']:
        for name in ('meta','source_result'):
            path=source['meta_path'] if name=='meta' else source['source_result']
            if sha256(Path(path).read_bytes())!=source['meta_SHA256' if name=='meta' else 'source_result_SHA256']:
                raise ValueError('Physical source/problem drift')
    return {'implementation_SHA256':implementation,'qualifier_SHA256':sha256((RUN/'qualifier.json').read_bytes()),
            'adapter_freeze_SHA256':sha256((ADAPTER/'freeze.json').read_bytes()),
            'old_db_SHA256':{str(p/'compiled.sqlite'):sha256((p/'compiled.sqlite').read_bytes()) for p in PREV},
            'source_db_SHA256':sha256(PARENT.read_bytes()),'sources':prior['sources'],
            'fixed_policy':'N37_SELECT_SAME3RESOURCES_SAFETY_AND_RESOURCE_ONLY_TIEBREAK',
            'initial_state_passes':1,'peephole_passes':0,
            'max_parent_CPU_seconds':30,'max_wall_seconds':90,'max_run_bytes':4*2**20,
            'probability_L1_tolerance':1e-8,'new_model_QPU_shots':[0,0],'new_energy_CI_optimizer':[0,0,0],
            'scope':'SAME_FIRST4_OF783_EACH4CASES_LOGICAL_OPT1_NOT_WHOLEBANK_OR_NOVEL_COMPILER'}


def member(source,plan,artifacts,parent,old):
    meta=load(source['meta_path']);raw=parent.execute(
        'select record,certificate_gz,native_gz,rustiq_gz,projected_gz from members where case_id=? and idx=?',
        (source['case'],plan['index'])).fetchone()
    rec=json.loads(raw[0]);cert=json.loads(gzip.decompress(raw[1]))
    if rec['member_id']!=plan['member_id']:raise ValueError('Source identity changed')
    original=json.loads(gzip.decompress(artifacts[plan['artifacts']['member.json.gz']['path']]))
    expected,_=member_state(meta,original)
    resources={};compiled={};requests={};cost={};evidence={}
    for name,b in zip(INPUTS,raw[2:]):
        if sha256(b)!=rec['qpy_gz_SHA256'][name]:raise ValueError('Original QPY changed')
        qc=qpy.load(BytesIO(gzip.decompress(b)))[0]
        probs=expected_addresses(expected,int(meta['n_qubits']),original['permutation'],cert,name=='PROJECTED')
        resources[name]=rec['resources'][name];compiled[name]=b
        requests[name]={'qpy_gz_base64':base64.b64encode(b).decode(),'expected_probabilities':probs}
        c,w=process_time(),perf_counter();output,info=compile_thin(qc)
        _,sample,views=circuit_views(output)
        label=name+'_THIN';resources[label]=views['Z_sampling_equivalent'];compiled[label]=blob(sample)
        cost[label]={'CPU_seconds':process_time()-c,'wall_seconds':perf_counter()-w};evidence[label]=info
        requests[label]={'qpy_gz_base64':base64.b64encode(compiled[label]).decode(),'expected_probabilities':probs}
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(OVERLAY),str(ROOT)]))
    child=subprocess.run([sys.executable,'-m','frontierloop.mps_distribution_check'],
                         input=json.dumps({'circuits':requests,'tolerance':1e-8}),text=True,capture_output=True,timeout=15,env=env)
    if child.returncode:raise RuntimeError('MPS verifier: '+child.stderr[-3000:])
    previous=next((db.execute('select record from members where case_id=? and idx=?',(source['case'],plan['index'])).fetchone()
                   for db in old if db.execute('select count(*) from members where case_id=? and idx=?',(source['case'],plan['index'])).fetchone()[0]),None)
    if previous is None:raise ValueError('Missing same-member heavy comparator')
    previous=json.loads(previous[0])
    return {'case':source['case'],'index':plan['index'],'member_id':plan['member_id'],'problem_id':rec['problem_id'],
            'status':'ALL6_VARIANTS_PROBABILITY_VERIFIED','resources':resources,'policy':select(resources),
            'heavy_policy':previous['policy'],'heavy_compile_CPU_seconds':sum(x['CPU_seconds'] for x in previous['compile_cost'].values()),
            'compile_cost':cost,'compiler_evidence':evidence,'independent_MPS_check':json.loads(child.stdout),
            'qpy_gz_SHA256':{k:sha256(v) for k,v in compiled.items()},
            'parameter_rank':'UNKNOWN_NO_PARAMETER_REDUCTION_CLAIM','new_model_QPU_shots':[0,0],
            'new_energy_CI_optimizer':[0,0,0]},compiled


def run():
    spec=load(RUN/'freeze.json')
    if spec!=binding():raise ValueError('N39 freeze drift')
    if (RUN/'started.json').exists():raise RuntimeError('No duplicate frozen run')
    space();cpu,wall=process_time(),perf_counter();r=resource.getrusage(resource.RUSAGE_CHILDREN);before=r.ru_utime+r.ru_stime
    save(RUN/'started.json',{'freeze_SHA256':sha256((RUN/'freeze.json').read_bytes())})
    parent=sqlite3.connect('file:'+str(PARENT)+'?mode=ro',uri=True)
    old=[sqlite3.connect('file:'+str(p/'compiled.sqlite')+'?mode=ro',uri=True) for p in PREV]
    d=sqlite3.connect(RUN/'compiled.sqlite');d.execute('PRAGMA synchronous=FULL')
    d.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,record TEXT,PRIMARY KEY(case_id,idx))')
    d.execute('CREATE TABLE circuits(case_id TEXT,idx INTEGER,variant TEXT,qpy_gz BLOB,PRIMARY KEY(case_id,idx,variant))')
    rows=[];failure=None
    try:
        for source in spec['sources']:
            artifacts=read_selected([e for m in source['members'] for e in m['artifacts'].values()])
            for plan in source['members']:
                space()
                if process_time()-cpu>30 or perf_counter()-wall>90:raise TimeoutError('N39 bounded pilot cap')
                row,compiled=member(source,plan,artifacts,parent,old)
                d.execute('INSERT INTO members VALUES(?,?,?)',(source['case'],plan['index'],json.dumps(row,sort_keys=True,allow_nan=False)))
                for name,b in compiled.items():d.execute('INSERT INTO circuits VALUES(?,?,?,?)',(source['case'],plan['index'],name,b))
                d.commit();rows.append(row)
                if (RUN/'compiled.sqlite').stat().st_size>4*2**20:raise OSError('N39 compact storage cap')
                p=row['policy'];print(source['case'],plan['index'],p['control'],p['selected'],flush=True)
        status='ALL16_THIN_MEMBERS_96_ACTUAL_CIRCUITS_VERIFIED'
    except Exception as exc:failure={'error':repr(exc),'traceback':traceback.format_exc()};status='FAILED_ENGINEERING_PRESERVED'
    finally:
        d.close();parent.close()
        for db in old:db.close()
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    terminal={'status':status,'completed_members':len(rows),'failure':failure,'parent_CPU_seconds':process_time()-cpu,
              'wall_seconds':perf_counter()-wall,'children_CPU_seconds_including_startup':r.ru_utime+r.ru_stime-before,
              'verifier_internal_CPU_seconds':sum(r['independent_MPS_check']['CPU_seconds'] for r in rows),
              'new_model_QPU_shots':[0,0],'new_energy_CI_optimizer':[0,0,0]}
    save(RUN/'terminal.json',terminal);print(json.dumps(terminal))
    if failure:raise RuntimeError('N39 partial failure preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualify','freeze','run']);a=p.parse_args()
    if a.action=='qualify':qualify()
    elif a.action=='freeze':
        if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
        save(RUN/'freeze.json',binding());print('N39 single-pass thin compiler protocol frozen')
    else:resource.setrlimit(resource.RLIMIT_CPU,(30,30));run()
