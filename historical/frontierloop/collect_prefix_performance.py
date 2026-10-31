"""N17 terminal lane collection. Copies bounded counts, never dense states."""
import json
from pathlib import PurePosixPath
import shlex
import subprocess
import tarfile
from .paths import ROOT
from .io import load,save,write,sha256
from .prefix_performance_sampling import (
    LOCAL,REMOTE,verified_protocol,capacity,accept_result,db,ledger_transition,
)
from benchmark.gpu_remote import ssh,OPTIONS,HOST


def safe(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path)!=name:
        raise ValueError('Unsafe evidence name')


def collect():
    spec=verified_protocol();capacity();jobs=[load(p) for p in sorted(LOCAL.glob('lane*-job.json'))]
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T"',timeout=30)
    live={s.split('|')[0] for s in queue.splitlines()}
    for job in jobs:
        lane=job['lane'];marker=LOCAL/f'lane{lane}-collection.json'
        if marker.exists():
            print('N17_ALREADY_COLLECTED_LANE',lane,flush=True);continue
        if job['job_id'] in live:
            print('N17_LANE_STILL_LIVE_NO_RESTART',lane,job['job_id'],flush=True);continue
        account=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30).strip()
        lines=[l.split('|') for l in account.splitlines() if l.split('|')[0]==job['job_id']]
        if len(lines)!=1 or lines[0][1]!='COMPLETED' or lines[0][3]!='0:0':
            save(LOCAL/f'lane{lane}-noncomplete-accounting.json',dict(accounting=account,job=job))
            print('N17_LANE_NONCOMPLETE_PRESERVED',lane,account,flush=True);continue
        circuits=[next(c for c in spec['circuits'] if c['circuit_id']==cid) for cid in job['circuit_ids']]
        names=[f'lane{lane}.out',f'lane{lane}.err']
        for circuit in circuits:
            base='circuits/'+circuit['circuit_id']+'/output/'
            names.extend(base+n for n in ['started.json','result.json'])
            for request in circuit['sampling_requests']:
                names.extend(base+request['id']+suffix for suffix in ['.npz','-draw-started.json','-draw-completed.json'])
        for name in names:safe(name)
        script="""import pathlib,hashlib,json
r=pathlib.Path(ROOT);names=NAMES;out={}
for name in names:
 p=r/name
 if not p.is_file() or p.is_symlink():raise RuntimeError('Missing or invalid expected evidence '+name)
 out[name]={'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(out))
""".replace('ROOT',repr(REMOTE)).replace('NAMES',repr(names))
        manifest=json.loads(ssh(shlex.join(['python3','-c',script]),timeout=30))
        total=sum(v['bytes'] for v in manifest.values())
        if set(manifest)!=set(names) or total>128*2**20:raise RuntimeError('Oversized or incomplete lane evidence')
        archive=LOCAL/f'lane{lane}-evidence.tar.gz'
        if archive.exists():raise RuntimeError('Partial collection archive exists; reconcile rather than overwrite')
        save(LOCAL/f'lane{lane}-collection-intent.json',dict(manifest=manifest,accounting=account))
        with archive.open('xb') as output:
            command=['rtk','proxy','ssh','-p','2222',*OPTIONS,HOST,'tar -czf - -C '+shlex.quote(REMOTE)+' -T -']
            subprocess.run(command,input=('\n'.join(names)+'\n').encode(),stdout=output,check=True,timeout=55)
        collected=LOCAL/'collected';seen=set()
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                safe(member.name)
                if not member.isfile() or member.name not in manifest or member.name in seen:
                    raise RuntimeError('Unexpected evidence member')
                blob=tar.extractfile(member).read();record=manifest[member.name]
                if len(blob)!=record['bytes'] or sha256(blob)!=record['SHA256']:raise RuntimeError('Count evidence hash mismatch')
                path=collected/member.name
                if path.exists() and path.read_bytes()!=blob:raise RuntimeError('Local evidence collision')
                if not path.exists():write(path,blob)
                seen.add(member.name)
        if seen!=set(names):raise RuntimeError('Missing lane evidence')
        for circuit in circuits:
            output=collected/'circuits'/circuit['circuit_id']/'output'
            started=load(output/'started.json');result=load(output/'result.json')
            if started['job_id']!=job['job_id'] or result['job_id']!=job['job_id'] or started['input_id']!=circuit['input_id']:
                raise RuntimeError('Actual Slurm owner/input mismatch')
            if result['primitive_semantics']!='HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP':
                raise RuntimeError('Unqualified GPU primitive semantics')
            for request in circuit['sampling_requests']:
                if load(output/(request['id']+'-draw-started.json'))!=request:
                    raise RuntimeError('Actual raw draw request differs')
                completed=load(output/(request['id']+'-draw-completed.json'))
                if completed not in result['sampling_results']:raise RuntimeError('Unbound draw-completion evidence')
                with db() as c:state=c.execute('SELECT status FROM draws WHERE draw_id=?',(request['id'],)).fetchone()[0]
                if state=='REGISTERED':ledger_transition(request['id'],'STARTED',sha256((output/(request['id']+'-draw-started.json')).read_bytes()))
            accept_result(circuit,output)
        save(marker,dict(job=job,accounting=account,manifest=manifest,total_selected_bytes=total,
            archive_SHA256=sha256(archive.read_bytes()),remote_evidence_deleted=False,actual_QPU_shots=0))
        print('N17_LANE_COLLECTED',lane,len(circuits),'unique circuits',total,'bytes',flush=True)
    with db() as c:
        states=dict(c.execute('SELECT status,count(*) FROM cells GROUP BY status').fetchall())
        draws=dict(c.execute('SELECT status,count(*) FROM draws GROUP BY status').fetchall())
        shots=c.execute('SELECT sum(actual_shots) FROM draws').fetchone()[0] or 0
    save(LOCAL/'sampling-progress.json',dict(logical_cells=states,unique_draws=draws,
        actual_new_model_shots=shots,actual_QPU_shots=0,new_CI_roots=0,goal_complete=False))
    print('N17_SAMPLING_PROGRESS',states,draws,shots,flush=True)


if __name__=='__main__':collect()
