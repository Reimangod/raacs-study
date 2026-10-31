"""Read-only N33 collector successor; same raw results, fewer SSH connections."""
import base64,json,shlex
from pathlib import PurePosixPath
from benchmark.gpu_remote import ssh
from . import analytical_source_sampling as s
from .io import load,save,write,sha256


def run():
    s.verified();m=s.configured();sub=load(s.SAMPLING/'protocol.json');m.capacity()
    for file in sorted(s.SAMPLING.glob('lane*-job.json')):
        job=load(file);lane=job['lane'];marker=s.SAMPLING/f'lane{lane}-collection.json'
        if marker.exists():continue
        circuits=[next(c for c in sub['circuits'] if c['circuit_id']==cid) for cid in job['circuit_ids']]
        names=[f'lane{lane}.out',f'lane{lane}.err']
        for circuit in circuits:
            base='circuits/'+circuit['circuit_id']+'/output/'
            names.extend(base+n for n in ('started.json','result.json'))
            for r in circuit['sampling_requests']:names.extend(base+r['id']+suffix for suffix in ('.npz','-draw-started.json','-draw-completed.json'))
        if len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):raise RuntimeError('Explicit safe member set required')
        code='''import pathlib,json,subprocess,base64,hashlib
r=pathlib.Path(REMOTE);out={};total=0
account=subprocess.check_output(['sacct','-X','-n','-j',JOB,'--format=JobID,State,Elapsed,ExitCode,AllocTRES','-P'],text=True)
rows=[x.split('|') for x in account.splitlines() if x.split('|')[0]==JOB]
assert len(rows)==1 and rows[0][1]=='COMPLETED' and rows[0][3]=='0:0'
for name in NAMES:
 p=r/name; assert p.is_file() and not p.is_symlink();size=p.stat().st_size;total+=size
 assert total<=2097152
 b=p.read_bytes();assert len(b)==size
 out[name]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
print(json.dumps(dict(accounting=account,members=out)))
'''
        code='REMOTE='+repr(s.REMOTE)+'\nJOB='+repr(job['job_id'])+'\nNAMES='+repr(names)+'\n'+code
        # No sbatch and no scientific command is issued by this read-only call.
        blob=ssh(shlex.join(['python3','-c',code]),timeout=45)
        if len(blob.encode())>4*2**20:raise RuntimeError('Bounded transfer response')
        response=json.loads(blob);members=response['members']
        if set(members)!=set(names):raise RuntimeError('Exact evidence membership mismatch')
        manifest={};total=0
        for name,row in members.items():
            b=base64.b64decode(row['base64'],validate=True)
            if len(b)!=row['bytes'] or sha256(b)!=row['SHA256']:raise RuntimeError('Actual raw evidence byte/hash mismatch')
            total+=len(b);manifest[name]={k:row[k] for k in ('bytes','SHA256')};path=s.SAMPLING/'collected'/name
            if path.exists() and path.read_bytes()!=b:raise RuntimeError('Preserve conflicting local evidence; do not overwrite')
            if not path.exists():write(path,b)
        for c in circuits:
            out=s.SAMPLING/'collected/circuits'/c['circuit_id']/'output';started=load(out/'started.json');result=load(out/'result.json')
            if started['job_id']!=job['job_id'] or result['job_id']!=job['job_id'] or started['input_id']!=c['input_id'] or result['cupy']!='13.6.0':raise RuntimeError('Job/executor binding mismatch')
            if result['primitive_semantics']!='HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP':raise RuntimeError('Wrong qualified primitive')
            for request in c['sampling_requests']:
                if load(out/(request['id']+'-draw-started.json'))!=request or load(out/(request['id']+'-draw-completed.json')) not in result['sampling_results']:raise RuntimeError('Actual draw event mismatch')
                with m.db() as db:status=db.execute('SELECT status FROM draws WHERE draw_id=?',(request['id'],)).fetchone()[0]
                if status=='REGISTERED':m.ledger_transition(request['id'],'STARTED',sha256((out/(request['id']+'-draw-started.json')).read_bytes()))
            m.accept_result(c,out)
        save(marker,dict(job=job,accounting=response['accounting'],manifest=manifest,total_selected_bytes=total,
            bundle_response_SHA256=sha256(blob.encode()),collector='ADDITIVE_SINGLE_READONLY_SSH_SESSION_V1',
            remote_evidence_deleted=False,actual_QPU_shots=0))
        print('N33_SINGLE_SESSION_COLLECTED',lane,len(circuits),total,'bytes',flush=True)
    with m.db() as db:
        cells=dict(db.execute('SELECT status,count(*) FROM cells GROUP BY status').fetchall());draws=dict(db.execute('SELECT status,count(*) FROM draws GROUP BY status').fetchall());shots=db.execute('SELECT sum(actual_shots) FROM draws').fetchone()[0] or 0
    save(s.SAMPLING/'sampling-progress-v2.json',dict(logical_cells=cells,unique_draws=draws,actual_new_model_shots=shots,actual_QPU_shots=0,new_CI_roots=0,goal_complete=False))
    print('N33_SAMPLING_ACTUAL',draws,shots,flush=True)


if __name__=='__main__':run()
