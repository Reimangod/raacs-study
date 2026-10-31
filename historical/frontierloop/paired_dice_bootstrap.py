"""N9 additive engineering repair: exact base executor, cached minimal archive.

The old failed attempt and its full-archive cache remain evidence. No molecular
outcome occurred. Only source staging changes; input adapter/toys are identical.
"""
import hashlib,importlib.util,json,os,subprocess,sys,tarfile
from pathlib import Path


def run(path):
    path=Path(path).resolve();root=path.parent;spec=json.loads(path.read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(__file__)!=spec['bootstrap_SHA256']:raise RuntimeError('Frozen staging bootstrap changed')
    prior=Path(spec['prior_failed_root'])
    for name,value in spec['prior_failed_evidence_SHA256'].items():
        if sha(prior/name)!=value:raise RuntimeError('Prior failed attempt evidence changed')
    original=Path(spec['original_source']);real_run=subprocess.run
    def staged_run(argv,*args,**kw):
        if argv[:2]!=['git','archive']:return real_run(argv,*args,**kw)
        if argv!=['git','archive',spec['original_commit']] or Path(kw.get('cwd','')).resolve()!=original.resolve():
            raise RuntimeError('Unexpected archive request')
        env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
        listed=real_run(['git','ls-tree','--name-only',spec['original_commit']],cwd=original,
                        env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=20)
        roots=listed.stdout.decode().splitlines()
        paths=[x for x in roots if x in ['SHCI','eigen','Makefile','LICENSE','COPYING','LICENSE.txt']
               or x.endswith(('.cpp','.h','.hpp','.c','.cc'))]
        if 'SHCI' not in paths or 'eigen' not in paths or 'Makefile' not in paths:raise RuntimeError('Minimal source paths unavailable')
        if any('/' in x or x.startswith('-') for x in paths):raise RuntimeError('Unexpected source path')
        kw['env']=env
        result=real_run(['git','archive',spec['original_commit'],'--',*paths],*args,**kw)
        if result.returncode:return result
        output=kw['stdout'];output.flush();archive=Path(output.name)
        if archive.stat().st_size>32*1024**2:raise RuntimeError('Minimal archive exceeds32MiB')
        with tarfile.open(archive) as t:
            members=t.getmembers()
            if sum(m.size for m in members)>64*1024**2:raise RuntimeError('Expanded source exceeds64MiB')
            if any(m.issym() or m.islnk() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
                raise RuntimeError('Nonregular source archive')
        (root/'staging.json').write_text(json.dumps(dict(mode='MINIMAL_CACHED_SOURCE_ONLY_NO_LAZY_FETCH',paths=paths,
            archive_bytes=archive.stat().st_size,expanded_bytes=sum(m.size for m in members)),sort_keys=True,indent=2)+'\n')
        return result
    # Module-local subprocess object is shared: intercept only the one exact
    # archive invocation, forwarding every compile/native call unchanged.
    subprocess.run=staged_run
    module_spec=importlib.util.spec_from_file_location('base_worker',root/'base_worker.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.run(path)


if __name__=='__main__':run(sys.argv[1])
