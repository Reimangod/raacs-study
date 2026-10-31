"""Bounded, read-only collection of N4 evidence; never dispatches computation."""
import argparse,json,shlex,subprocess
from pathlib import Path
from .paths import ROOT
from .io import load,save,space,sha256
from .native_hci_bootstrap import LOCAL,REMOTE
from benchmark.gpu_remote import ssh,OPTIONS,HOST


def collect():
    job=load(LOCAL/'job.json')['job_id']
    if ssh('squeue -h -j '+job+' -o %T',timeout=35).strip():
        raise RuntimeError('Collect terminal evidence only; job still live')
    code='''import json,pathlib,hashlib,gzip
r=pathlib.Path(ROOT_PATH)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
spec=json.loads((r/'protocol.json').read_text())
if not (r/'terminal.json').exists():raise RuntimeError('Missing terminal; retain partials, do not fabricate completion')
rows=[]
for p in spec['plans']:
    folder=r/'cells'/p['task_id'];files={}
    for name in ['result.json','input.dat','native.log','shci.e','dets.bin','dets.bin.gz','AlphaDets.bin','BetaDets.bin']:
        f=folder/name
        if f.exists():files[name]={'bytes':f.stat().st_size,'SHA256':sha(f)}
    rows.append({'task_id':p['task_id'],'files':files,'result':json.loads((folder/'result.json').read_text())})
out={'remote_root':str(r),'protocol_SHA256':sha(r/'protocol.json'),'cells':rows,'top':{}}
for name in ['started.json','progress.json','terminal.json','common-ci-observations.json','common-ci-attempts.json']:
    p=r/name;out['top'][name]={'SHA256':sha(p),'value':json.loads(p.read_text())}
print(json.dumps(out))
'''.replace('ROOT_PATH',repr(REMOTE))
    bundle=json.loads(ssh(shlex.join(['python3','-c',code]),timeout=50))
    if bundle['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):
        raise RuntimeError('Remote/local frozen protocol mismatch')
    target=LOCAL/'collected';target.mkdir(exist_ok=True)
    # Preserve exact bytes. One transport for all top records avoids5 handshakes.
    subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,
        *[HOST+':'+REMOTE+'/'+name for name in bundle['top']],str(LOCAL)],check=True,timeout=45)
    for name,record in bundle['top'].items():
        if sha256((LOCAL/name).read_bytes())!=record['SHA256']:raise RuntimeError('Top evidence transport mismatch')
    files=[]
    for row in bundle['cells']:
        for name,record in row['files'].items():
            if name=='dets.bin':continue  # Lossless gzip copied instead, raw remote retained.
            files.append((row['task_id'],name,record))
    total=sum(v['bytes'] for _,_,v in files)
    if total>32*1024**2:raise RuntimeError('Evidence exceeds bounded32MiB collection; raw remote retained')
    space()
    for row in bundle['cells']:
        task=row['task_id'];folder=target/task;folder.mkdir(exist_ok=True);missing=[]
        for _,name,record in [x for x in files if x[0]==task]:
            dest=folder/name
            if dest.exists():
                if sha256(dest.read_bytes())!=record['SHA256']:raise RuntimeError('Existing collected evidence differs')
            else:missing.append(name)
        if missing:
            subprocess.run(['rtk','proxy','scp','-P','2222',*OPTIONS,
                *[HOST+':'+REMOTE+'/cells/'+task+'/'+name for name in missing],str(folder)],check=True,timeout=45)
        for _,name,record in [x for x in files if x[0]==task]:
            if sha256((folder/name).read_bytes())!=record['SHA256']:raise RuntimeError('Cell evidence transport mismatch')
    save(LOCAL/'evidence-collection.json',dict(**bundle,downloaded_cell_bytes=total,
        raw_remote_deleted=False,collector_SHA256=sha256(Path(__file__).read_bytes())))
    print('N4_BOUNDED_EVIDENCE_COLLECTED',len(bundle['cells']),total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect']);p.parse_args();collect()
