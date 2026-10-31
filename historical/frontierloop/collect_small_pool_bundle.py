"""Recover only a terminal N7 evidence transfer, not scientific execution.

Single tar stream avoids the per-file SFTP latency that exceeded the original
50s directory-copy bound. Raw native dets.bin remains remote; its lossless gzip
and raw digest suffice. Existing incomplete local transfers are preserved.
"""
import gzip,hashlib,json,shlex,subprocess,tarfile
from pathlib import Path,PurePosixPath
from .paths import ROOT
from .io import load,save,sha256,space,write
from .small_initial_pool import LOCAL,REMOTE
from benchmark.gpu_remote import ssh,OPTIONS,HOST


def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in ['cells','roots','qualification',
        'started.json','progress.json','terminal.json','observations.json','ci-attempts.json','slurm.out','slurm.err']:
        raise ValueError('Non-evidence archive path')
    return p


def collect():
    job=load(LOCAL/'job.json')['job_id'];queue=ssh('squeue -h -u reiwatanabe -o "%i|%T"',timeout=35)
    if any(line.split('|')[0]==job for line in queue.splitlines()):raise RuntimeError('No transfer while exact worker live')
    accounting=ssh('sacct -X -n -j '+job+' --format=JobID,State,ExitCode -P',timeout=35).strip()
    if '|COMPLETED|' not in accounting:raise RuntimeError('Need authoritative completed job evidence')
    script='''import pathlib,json,hashlib
r=pathlib.Path(ROOT_PATH)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
if not (r/'terminal.json').exists():raise RuntimeError('No terminal artifact')
files={};raw={}
for p in sorted(r.rglob('*')):
    if not p.is_file():continue
    name=p.relative_to(r).as_posix();parts=p.relative_to(r).parts
    if p.is_symlink():raise RuntimeError('Symlink not allowed')
    if p.name=='dets.bin':raw[name]=sha(p);continue
    if parts[0] not in ['cells','roots','qualification','started.json','progress.json','terminal.json','observations.json','ci-attempts.json','slurm.out','slurm.err']:continue
    if p.name.endswith('.tmp'):continue
    files[name]={'bytes':p.stat().st_size,'SHA256':sha(p)}
print(json.dumps({'files':files,'native_raw_SHA256':raw,'protocol_SHA256':sha(r/'protocol.json')}))
'''.replace('ROOT_PATH',repr(REMOTE))
    manifest=json.loads(ssh(shlex.join(['python3','-c',script]),timeout=45))
    if manifest['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Producer protocol mismatch')
    total=sum(f['bytes'] for f in manifest['files'].values())
    if total>32*1024**2:raise RuntimeError('Bounded compact evidence still too large')
    for name in manifest['files']:safe_member(name)
    space();archive=LOCAL/'terminal-evidence-bundle-v1.tar.gz';marker=LOCAL/'bundle-collection-v1.json'
    if marker.exists():raise RuntimeError('Bundle already collected; inspect existing result rather than transfer again')
    if archive.exists():raise RuntimeError('Prior bundle transfer may be partial; preserve and reconcile explicitly')
    cmd=['rtk','proxy','ssh','-p','2222',*OPTIONS,HOST,'tar -czf - -C '+shlex.quote(REMOTE)+' -T -']
    with archive.open('wb') as output:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=output,stderr=subprocess.PIPE)
        _,err=p.communicate(('\n'.join(manifest['files'])+'\n').encode(),timeout=60)
    if p.returncode:raise RuntimeError('Evidence-only stream failed: '+err.decode(errors='replace'))
    target=LOCAL/'collected';target.mkdir(exist_ok=True);quarantined=[];seen=set()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            safe_member(member.name)
            if not member.isfile() or member.name not in manifest['files'] or member.name in seen:raise RuntimeError('Unexpected archive member')
            data=tf.extractfile(member).read();record=manifest['files'][member.name]
            if len(data)!=record['bytes'] or sha256(data)!=record['SHA256']:raise RuntimeError('Evidence stream digest mismatch')
            dest=target/member.name
            if dest.exists() and sha256(dest.read_bytes())!=record['SHA256']:
                backup=dest.with_name(dest.name+'.incomplete-SFTP-v1')
                if backup.exists():raise RuntimeError('Partial transfer backup already exists')
                dest.rename(backup);quarantined.append(str(backup.relative_to(LOCAL)))
            if not dest.exists():write(dest,data)
            seen.add(member.name)
    if seen!=set(manifest['files']):raise RuntimeError('Missing evidence bundle member')
    for name,digest in manifest['native_raw_SHA256'].items():
        gz=target/(name+'.gz')
        if gz.exists() and sha256(gzip.decompress(gz.read_bytes()))!=digest:raise RuntimeError('Native archive raw digest mismatch')
    save(marker,dict(**manifest,accounting=accounting,total_uncompressed_selected_bytes=total,
        archive_SHA256=sha256(archive.read_bytes()),archive_bytes=archive.stat().st_size,
        incomplete_transfers_preserved=quarantined,raw_remote_deleted=False,
        collector_SHA256=sha256(Path(__file__).read_bytes())))
    print('N7_EVIDENCE_ONLY_BUNDLE_COLLECTED',len(seen),total,archive.stat().st_size,flush=True)


if __name__=='__main__':collect()
