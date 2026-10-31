"""Bounded, hash-checked terminal N9 evidence stream; never copy source caches."""
import json,os,shlex,signal,subprocess,tarfile
from pathlib import PurePosixPath,Path
from .paths import ROOT
from .io import load,save,write,sha256,space
from .paired_dice_repair import LOCAL,REMOTE
from benchmark.gpu_remote import ssh,OPTIONS,HOST

ALLOWED={'terminal.json','started.json','build.json','build.log','progress.json',
         'staging.json','slurm.out','slurm.err','source-diff.patch','outputs'}


def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or p.parts[0] not in ALLOWED or '\n' in name:
        raise ValueError('Unsafe or non-evidence archive member')
    return p


def collect():
    job=load(LOCAL/'job.json')['job_id'];queue=ssh('squeue -h -u reiwatanabe -o "%i|%T"',timeout=35)
    if any(x.split('|')[0]==job for x in queue.splitlines()):raise RuntimeError('Exact worker still live; no collection')
    acc=ssh('sacct -X -n -j '+job+' --format=JobID,State,ExitCode -P',timeout=35).strip()
    if not any('|'+state+'|' in acc for state in ['COMPLETED','FAILED']):raise RuntimeError('Authoritative terminal accounting required')
    script='''import pathlib,json,hashlib
r=pathlib.Path(ROOT);allowed=set(ALLOWED)
if not (r/'terminal.json').is_file():raise RuntimeError('Terminal result absent')
files={}
for p in sorted(r.rglob('*')):
 name=p.relative_to(r).as_posix()
 if p.relative_to(r).parts[0] not in allowed:continue
 if p.is_symlink():raise RuntimeError('Symlink not evidence')
 if not p.is_file() or p.name.endswith('.tmp'):continue
 b=p.read_bytes();files[name]={'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest()}
print(json.dumps({'files':files,'protocol_SHA256':hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest()}))
'''.replace('ROOT',repr(REMOTE)).replace('ALLOWED',repr(sorted(ALLOWED)))
    manifest=json.loads(ssh(shlex.join(['python3','-c',script]),timeout=35))
    if manifest['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Producer protocol differs')
    total=sum(x['bytes'] for x in manifest['files'].values())
    if total>32*1024**2 or len(manifest['files'])>500:raise RuntimeError('Compact evidence cap; never copy full source tree')
    for name in manifest['files']:safe_member(name)
    space();archive=LOCAL/'evidence-bundle-v1.tar.gz';marker=LOCAL/'bundle-collection-v1.json'
    if archive.exists() or marker.exists():raise RuntimeError('Prior evidence transfer exists; preserve/reconcile, never overwrite')
    cmd=['rtk','proxy','ssh','-p','2222',*OPTIONS,HOST,'tar -czf - -C '+shlex.quote(REMOTE)+' -T -']
    with archive.open('xb') as out:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=subprocess.PIPE,start_new_session=True)
        try:_,err=p.communicate(('\n'.join(manifest['files'])+'\n').encode(),timeout=45)
        except subprocess.TimeoutExpired:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if p.returncode:raise RuntimeError('Evidence stream failed: '+err.decode(errors='replace'))
    target=LOCAL/'collected';target.mkdir(exist_ok=True);seen=set()
    with tarfile.open(archive,'r:gz') as t:
        for m in t:
            safe_member(m.name)
            if not m.isfile() or m.name not in manifest['files'] or m.name in seen:raise RuntimeError('Unexpected archive member')
            row=manifest['files'][m.name];b=t.extractfile(m).read()
            if len(b)!=row['bytes'] or sha256(b)!=row['SHA256']:raise RuntimeError('Evidence digest mismatch')
            f=target/m.name
            if f.exists() and sha256(f.read_bytes())!=row['SHA256']:raise RuntimeError('Conflicting preserved evidence')
            if not f.exists():write(f,b)
            seen.add(m.name)
    if seen!=set(manifest['files']):raise RuntimeError('Missing evidence member')
    save(marker,dict(**manifest,selected_bytes=total,archive_SHA256=sha256(archive.read_bytes()),
                    archive_bytes=archive.stat().st_size,accounting=acc,collector_SHA256=sha256(Path(__file__).read_bytes())))
    print('N9_TERMINAL_EVIDENCE_BUNDLE_VERIFIED',len(seen),total,flush=True)


if __name__=='__main__':collect()
