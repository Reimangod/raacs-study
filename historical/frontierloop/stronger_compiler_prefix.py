"""G6: common exact opt3 chance, output permutation explicitly decoded."""
import argparse,base64,gzip,json,sqlite3
from io import BytesIO
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET
from qiskit import qpy,transpile
from .paths import ROOT
from .io import load,save,sha256,space
from .strong_projection import FREEZE as G4_FREEZE,RUN as G4_RUN,binding as g4_binding,FIELDS,expected_addresses,total_cpu
from .frozen_projection_screen import read_selected
from .strong_bank_prepare import verify_batch
from paperstudy.sqdrift_execution import member_state

PROTOCOL=ROOT/'protocol-G6-OPT3-PREFIX.json'
FREEZE=ROOT/'freeze-G6-OPT3-PREFIX.json'
RUN=ROOT/'runs/G6-OPT3-PREFIX-v1'
BASIS=['rz','ry','rx','h','s','sdg','x','cx']


def move_probabilities(probs,permutation):
    if sorted(permutation)!=list(range(len(permutation))): raise ValueError('Nonbijective output layout')
    result={}
    for x,p in probs.items():
        if not 0<=x<1<<len(permutation):raise ValueError('Input address outside register')
        y=sum(((x>>q)&1)<<dest for q,dest in enumerate(permutation))
        result[y]=p
    return result


def upgrade(q):
    compiled=transpile(q,basis_gates=BASIS,optimization_level=3,seed_transpiler=8137,
        approximation_degree=1.0,num_processes=1,qubits_initially_zero=True)
    perm=compiled.layout.final_index_layout() if compiled.layout else list(range(q.num_qubits))
    if compiled.num_qubits!=q.num_qubits or sorted(perm)!=list(range(q.num_qubits)):
        raise ValueError('Changed width/ancilla semantics not in this prefix scope')
    # Same Z-measurement normalization as G4, without a second transpiler pass.
    diagonal={'rz','z','s','sdg','t','tdg','p','cp','cz','rzz','id'}
    blocked=set();keep=[]
    for item in reversed(compiled.data):
        wires={compiled.find_bit(b).index for b in item.qubits}
        if item.operation.name in diagonal and not wires&blocked:continue
        keep.append(item);blocked|=wires
    sample=compiled.copy_empty_like()
    for item in reversed(keep):sample.append(item.operation,[sample.qubits[compiled.find_bit(b).index] for b in item.qubits])
    if any(x.operation.name not in BASIS for x in sample.data):raise ValueError('Unexpected final basis')
    resources=dict(cnot_count=int(sample.count_ops().get('cx',0)),
        cnot_depth=int(sample.depth(lambda x:x.operation.num_qubits==2) or 0),
        total_unitary_depth=int(sample.depth() or 0))
    return sample,perm,resources


def choose(resources):
    anchor=resources['NATIVE_OPT1']
    eligible=lambda r,b:all(r[k]<=b[k] for k in FIELDS)
    score=lambda name:tuple(resources[name][k] for k in FIELDS)+(name,)
    controls=[name for name in resources if not name.startswith('PROJECTED') and eligible(resources[name],anchor)]
    base=min(controls,key=score)
    candidates=[name for name in resources if name.startswith('PROJECTED') and eligible(resources[name],resources[base])
        and any(resources[name][k]<resources[base][k] for k in FIELDS)]
    selected=min(candidates,key=score) if candidates else base
    return dict(baseline=base,selected=selected,baseline_resources=resources[base],selected_resources=resources[selected],
        no_native_metric_increase=eligible(resources[selected],anchor),uses_energy=False)


def binding():
    if g4_binding()!=load(G4_FREEZE):raise RuntimeError('G4 source/qualified kernel drift')
    xml=ROOT/'qualification-opt3-prefix-v1-tests.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
    if sum(int(x.get('tests',0)) for x in suites)!=3 or any(int(x.get(k,0)) for x in suites for k in ['errors','failures','skipped']):
        raise RuntimeError('Opt3 permutation/guard qualification required')
    return dict(protocol_SHA256=sha256(PROTOCOL.read_bytes()),G4_freeze_SHA256=sha256(G4_FREEZE.read_bytes()),
        G4_database_SHA256=load(G4_RUN/'terminal.json')['database_SHA256'],qualifier_SHA256=sha256(xml.read_bytes()),
        implementation_SHA256={str(p):sha256(p.read_bytes()) for p in [Path(__file__),ROOT/'test_opt3_prefix.py',
            ROOT/'frontierloop/strong_bank_prepare.py',ROOT/'frontierloop/mps_batch_worker.py']})


def run():
    if binding()!=load(FREEZE):raise RuntimeError('G6 freeze drift')
    space();RUN.mkdir(exist_ok=False)
    cpu,wall=total_cpu(),perf_counter();rows=[]
    save(RUN/'started.json',dict(freeze_SHA256=sha256(FREEZE.read_bytes()),new_model_shots=0))
    oldpath=G4_RUN/'compiled-members.sqlite'
    if sha256(oldpath.read_bytes())!=load(FREEZE)['G4_database_SHA256']:raise RuntimeError('G4 evidence drift')
    old=sqlite3.connect('file:'+str(oldpath)+'?mode=ro',uri=True)
    db=sqlite3.connect(RUN/'compiled-members.sqlite');db.execute('CREATE TABLE members(case_id TEXT,idx INTEGER,record TEXT,native_opt3 BLOB,rustiq_opt3 BLOB,projected_opt3 BLOB,PRIMARY KEY(case_id,idx))')
    try:
        for source in load(G4_FREEZE)['sources']:
            space();meta=load(source['meta_path']);batch=[];prepared=[]
            blobs=read_selected([e for m in source['members'] for e in m['artifacts'].values()])
            for plan in source['members']:
                space()
                if sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())>128*2**20:
                    raise OSError('G6 compact output cap')
                if total_cpu()-cpu>600 or perf_counter()-wall>600:raise TimeoutError('G6 frozen work cap')
                data=old.execute('SELECT record,certificate_gz,native_gz,rustiq_gz,projected_gz FROM members WHERE case_id=? AND idx=?',
                    (source['case'],plan['index'])).fetchone()
                previous=json.loads(data[0]);cert=json.loads(gzip.decompress(data[1]))
                member=json.loads(gzip.decompress(blobs[plan['artifacts']['member.json.gz']['path']]))
                expected,_=member_state(meta,member);resources={};compiled={};requests={};perms={}
                for name,blob in zip(['NATIVE','RUSTIQ','PROJECTED'],data[2:]):
                    if sha256(blob)!=previous['qpy_gz_SHA256'][name]:raise RuntimeError('G4 QPY hash')
                    q=qpy.load(BytesIO(gzip.decompress(blob)))[0];q3,perm,r=upgrade(q)
                    resources[name+'_OPT1']={k:previous['resources'][name][k] for k in FIELDS}
                    resources[name+'_OPT3']=r;perms[name]=perm
                    stream=BytesIO();qpy.dump(q3,stream);compiled[name]=gzip.compress(stream.getvalue(),mtime=0)
                    oldprobs=expected_addresses(expected,meta['n_qubits'],member['permutation'],cert,name=='PROJECTED')
                    requests[name]=dict(qpy_gz_base64=base64.b64encode(compiled[name]).decode(),
                        expected_probabilities=move_probabilities(oldprobs,perm))
                key=source['case']+':'+str(plan['index'])+':'+plan['member_id']
                batch.append(dict(request=dict(key=key,circuits=requests)))
                row=dict(case=source['case'],index=plan['index'],member_id=plan['member_id'],
                    original_shot_weight=plan['historical_member_shots'],resources=resources,policy=choose(resources),
                    input_to_output_permutations=perms,new_model_shots=0,new_QPU_shots=0,new_energy_CI_evaluations=0,
                    QPY_gz_SHA256={k:sha256(v) for k,v in compiled.items()})
                prepared.append((row,compiled,key))
            result=verify_batch(batch)
            for row,compiled,key in prepared:
                check=result['results'][key]
                row.update(status=check['status'],independent_opt3_check=check)
                db.execute('INSERT INTO members VALUES(?,?,?,?,?,?)',(row['case'],row['index'],json.dumps(row,sort_keys=True),
                    compiled['NATIVE'],compiled['RUSTIQ'],compiled['PROJECTED']));rows.append(row)
            db.commit()
            save(RUN/'progress.json',dict(rows=rows,CPU_seconds=total_cpu()-cpu))
            if any(r['status']!='PASSED' for r,_,_ in prepared):raise RuntimeError('G6 failed actual-circuit parity, preserved')
            print(source['case'],'ALL16_OPT3_PARITIES_PASSED',flush=True)
    finally:
        old.close();db.close()
    save(RUN/'terminal.json',dict(status='FIXED64_OPT3_PREFIX_TERMINAL_NOT_WHOLE_BANK',rows=rows,
        CPU_seconds_including_children=total_cpu()-cpu,wall_seconds=perf_counter()-wall,
        database_SHA256=sha256((RUN/'compiled-members.sqlite').read_bytes()),
        new_model_shots=0,new_QPU_shots=0,new_energy_CI_evaluations=0,no_whole_bank_energy_join=True,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:
        b=binding()
        if FREEZE.exists() and load(FREEZE)!=b:raise RuntimeError('Never replace G6 freeze')
        save(FREEZE,b);print('G6_FIXED64_OPT3_FROZEN')
    else:run()
