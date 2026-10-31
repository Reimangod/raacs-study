"""N10 terminal evidence only, bounded64MiB, native raw retained remotely."""
import gzip,json,os,shlex,signal,subprocess,tarfile
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256,space
from .paired_pool_study import LOCAL,REMOTE
from .collect_small_pool_bundle import safe_member
from benchmark.gpu_remote import ssh,OPTIONS,HOST


def collect():
    job=load(LOCAL/'job.json')['job_id'];queue=ssh('squeue -h -u reiwatanabe -o "%i|%T"',timeout=35)
    if any(x.split('|')[0]==job for x in queue.splitlines()):raise RuntimeError('Exact worker still live')
    acc=ssh('sacct -X -n -j '+job+' --format=JobID,State,ExitCode -P',timeout=35).strip()
    if '|COMPLETED|' not in acc:raise RuntimeError('Need authoritative completed job evidence')
    script='''import pathlib,json,hashlib
r=pathlib.Path(ROOT_PATH);files={};raw={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if not (r/'terminal.json').exists():raise RuntimeError('No terminal result')
for p in sorted(r.rglob('*')):
 if p.is_symlink():raise RuntimeError('Symlink not evidence')
 if not p.is_file():continue
 name=p.relative_to(r).as_posix();parts=p.relative_to(r).parts
 if p.name=='dets.bin':raw[name]=sha(p);continue
 if parts[0] not in ['cells','roots','qualification','started.json','progress.json','terminal.json','observations.json','ci-attempts.json','slurm.out','slurm.err']:continue
 if p.name.endswith('.tmp'):continue
 files[name]={'bytes':p.stat().st_size,'SHA256':sha(p)}
print(json.dumps({'files':files,'native_raw_SHA256':raw,'protocol_SHA256':sha(r/'protocol.json')}))
'''.replace('ROOT_PATH',repr(REMOTE))
    manifest=json.loads(ssh(shlex.join(['python3','-c',script]),timeout=45));total=sum(r['bytes'] for r in manifest['files'].values())
    if manifest['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or total>64*1024**2:raise RuntimeError('Wrong or oversized bounded evidence')
    for name in manifest['files']:safe_member(name)
    space();archive=LOCAL/'terminal-evidence-bundle-v1.tar.gz';marker=LOCAL/'bundle-collection-v1.json'
    if archive.exists() or marker.exists():raise RuntimeError('Preserve/reconcile existing transfer; no overwrite')
    command=['rtk','proxy','ssh','-p','2222',*OPTIONS,HOST,'tar -czf - -C '+shlex.quote(REMOTE)+' -T -']
    with archive.open('xb') as out:
        p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=subprocess.PIPE,start_new_session=True)
        try:_,err=p.communicate(('\n'.join(manifest['files'])+'\n').encode(),timeout=45)
        except subprocess.TimeoutExpired:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    if p.returncode:raise RuntimeError('Evidence stream failed: '+err.decode(errors='replace'))
    target=LOCAL/'collected';target.mkdir(exist_ok=True);seen=set()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            safe_member(member.name)
            if not member.isfile() or member.name not in manifest['files'] or member.name in seen:raise RuntimeError('Unexpected archive member')
            data=tf.extractfile(member).read();record=manifest['files'][member.name]
            if len(data)!=record['bytes'] or sha256(data)!=record['SHA256']:raise RuntimeError('Evidence hash mismatch')
            dest=target/member.name
            if dest.exists() and sha256(dest.read_bytes())!=record['SHA256']:raise RuntimeError('Conflicting local evidence')
            if not dest.exists():write(dest,data)
            seen.add(member.name)
    if seen!=set(manifest['files']):raise RuntimeError('Missing evidence member')
    for name,value in manifest['native_raw_SHA256'].items():
        if sha256(gzip.decompress((target/(name+'.gz')).read_bytes()))!=value:raise RuntimeError('Lossless native raw mismatch')
    save(marker,dict(**manifest,accounting=acc,total_uncompressed_selected_bytes=total,archive_bytes=archive.stat().st_size,
        archive_SHA256=sha256(archive.read_bytes()),collector_SHA256=sha256(Path(__file__).read_bytes()),raw_remote_deleted=False))
    print('N10_BOUNDED_EVIDENCE_COLLECTED',len(seen),total,flush=True)


if __name__=='__main__':collect()
