"""Distinct H1 freeze and bounded CPU dispatch; immutable F2/D2 are not modified."""
import argparse
import fcntl
from pathlib import Path
import shlex
import subprocess
import xml.etree.ElementTree as ET
from .paths import ROOT
from .io import load,save,write,sha256,digest,space
from . import marginal_remote as f2
from paperstudy.budget import BudgetLedger
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.gpu_qualification import PYTHON,REMOTE as OLD_REMOTE

DIRECTORY=ROOT/'runs/H1-HALF-REMAP-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/H1-half-remap-v1'


def inputs():
    out=f2.input_paths();out['worker.py']=ROOT/'frontierloop/half_remap_worker.py'
    out['frontierloop/half_remap.py']=ROOT/'frontierloop/half_remap.py'
    out['frontierloop/marginal_remote_worker.py']=ROOT/'frontierloop/marginal_remote_worker.py'
    return out


def binding():
    return {'input_SHA256':{name:sha256(p.read_bytes()) for name,p in inputs().items()},
        'protocol_SHA256':sha256((ROOT/'protocol-H1-HALF-REMAP.json').read_bytes()),
        'qualification_SHA256':sha256((ROOT/'qualification-half-remap-tests.xml').read_bytes()),
        'readout_SHA256':{mode:sha256((ROOT/('protocol-'+mode+'.json')).read_bytes()) for mode in load(ROOT/'protocol-H1-HALF-REMAP.json')['readouts']},
        'dispatcher_SHA256':sha256(Path(__file__).read_bytes())}


def prepare():
    value=binding();DIRECTORY.mkdir(exist_ok=True)
    if (DIRECTORY/'manifest.json').exists():
        m=load(DIRECTORY/'manifest.json')
        if m['freeze']!=value:raise RuntimeError('Immutable H1 freeze changed')
        return m
    tests=list(ET.parse(ROOT/'qualification-half-remap-tests.xml').getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in tests)!=2 or any(int(s.get(k,'0')) for s in tests for k in ['failures','errors']):
        raise RuntimeError('Half-remapping qualification missing')
    space();source=load(f2.DIRECTORY/'manifest.json');spec=load(ROOT/'protocol-H1-HALF-REMAP.json')
    if spec['cases']!=source['protocol']['cases']:raise RuntimeError('H1 fixed cases changed')
    spec['readout_specs']={mode:load(ROOT/('protocol-'+mode+'.json')) for mode in spec['readouts']}
    cases={case:{**source['cases'][case],'run_id':digest(['H1-half-remap-v1',source['cases'][case]['physical']['problem_id'],value])} for case in spec['cases']}
    manifest={'freeze':value,'protocol':spec,'cases':cases,'input_SHA256':value['input_SHA256'],
        'expected_runtime':f2.RUNTIME,'F2_or_D2_molecular_outcomes_not_provided':True,
        'source_F2_manifest_SHA256':sha256((f2.DIRECTORY/'manifest.json').read_bytes()),'GPU_requested':False,'goal_complete':False}
    save(DIRECTORY/'manifest.json',manifest);return manifest


def submit():
    manifest=prepare()
    with (DIRECTORY/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (DIRECTORY/'job.json').exists():return load(DIRECTORY/'job.json')
        if (DIRECTORY/'submission-intent.json').exists():raise RuntimeError('Uncertain H1 job, do not submit twice')
        free_text=ssh('df -Pk /share/reiwatanabe',timeout=45);free=int(free_text.strip().splitlines()[-1].split()[3])*1024
        if free<manifest['protocol']['remote_free_floor_bytes']+manifest['protocol']['new_namespace_bytes_cap']:raise OSError('Remote capacity reserve')
        save(DIRECTORY/'capacity.json',{'available_bytes':free,'df':free_text,'local_free_bytes':space()})
        parent=load(f2.DIRECTORY/'manifest.json');copy=[];uploads=[]
        for name,p in inputs().items():
            old='worker.py' if name=='frontierloop/marginal_remote_worker.py' else name
            if parent['input_SHA256'].get(old)==manifest['input_SHA256'][name]:
                # Copies immutable compact data/code inside /share, not environments or outcomes.
                src=f2.REMOTE+'/'+old;dst=REMOTE+'/'+name
                copy.append('mkdir -p '+shlex.quote(str(Path(dst).parent)))
                copy.append('test "$(sha256sum '+shlex.quote(src)+" | cut -d ' ' -f 1)\" = "+shlex.quote(manifest['input_SHA256'][name]))
                copy.append('cp '+shlex.quote(src)+' '+shlex.quote(dst))
            else:uploads.append((name,p))
        ssh('set -eu\n'+'\n'.join(copy),timeout=90)
        for name,p in uploads:
            ssh('mkdir -p '+shlex.quote(str(Path(REMOTE+'/'+name).parent)));send(p,REMOTE+'/'+name)
        send(DIRECTORY/'manifest.json',REMOTE+'/manifest.json')
        script='\n'.join(['#!/bin/bash','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 JAX_PLATFORMS=cpu',
            'export PYTHONPATH='+REMOTE+':'+OLD_REMOTE+'/cpu-runtime-v1/packages',
            shlex.join([PYTHON,'-u',REMOTE+'/worker.py',REMOTE])])+'\n'
        write(DIRECTORY/'job.sbatch',script.encode());send(DIRECTORY/'job.sbatch',REMOTE+'/job.sbatch')
        ledger=BudgetLedger(ROOT/'production-shots.sqlite')
        for case,row in manifest['cases'].items():
            cap=manifest['protocol']['model_shots_per_case'];ledger.register(row['run_id'],cap)
            request={'kind':'H1-HSQD-spin-remap-common-setup','problem_id':row['physical']['problem_id'],'freeze_digest':digest(manifest['freeze']),'half_register_shots':cap}
            if ledger.reserve(row['run_id'],'final',request,cap)!='NEW_RESERVATION':raise RuntimeError('Existing H1 reservation')
        command=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=8G','--time=00:30:00',
            '--job-name=frontier-H1-HSQD-CPU-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',REMOTE+'/job.sbatch'])
        intent={'remote':REMOTE,'command':command,'manifest_SHA256':sha256((DIRECTORY/'manifest.json').read_bytes()),'GPU_requested':False,'model_shots_reserved':400000}
        save(DIRECTORY/'submission-intent.json',intent);job_id=ssh(command).strip().split(';')[0]
        if not job_id.isdigit():raise RuntimeError('Uncertain H1 submission reply')
        job={**intent,'job_id':job_id};save(DIRECTORY/'job.json',job);print('H1_CPU_SUBMITTED',job_id,flush=True);return job


def collect():
    manifest=load(DIRECTORY/'manifest.json');job=load(DIRECTORY/'job.json')
    if manifest['freeze']!=binding():raise RuntimeError('H1 source/freeze drift')
    if (DIRECTORY/'collected.json').exists():return load(DIRECTORY/'collected.json')
    live=ssh('squeue -h -j '+job['job_id']+' -o "%i %T %j %R"',timeout=45)
    accounting=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Start,End,Elapsed,ExitCode,AllocTRES -P',timeout=45)
    save(DIRECTORY/'observed.json',{'live':live,'accounting':accounting})
    if live.strip():print('H1_CPU_LIVE',live.strip(),flush=True);return None
    if not any(s in accounting for s in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED']):return None
    size=int(ssh('du -sk '+REMOTE,timeout=45).split()[0])*1024
    if size>manifest['protocol']['new_namespace_bytes_cap']:raise RuntimeError('H1 namespace exceeds download cap')
    space();out=DIRECTORY/'remote-output';out.mkdir(exist_ok=True)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/.',str(out)],check=True,timeout=180)
    ledger=BudgetLedger(ROOT/'production-shots.sqlite');checks=[]
    for case,row in manifest['cases'].items():
        d=out/'outputs'/case
        if not (d/'draw-result.json').exists():checks.append({'case':case,'draw_status':'MISSING_PRESERVED'});continue
        r=load(d/'draw-result.json');payload=(d/'half-counts.npz').read_bytes()
        if r['run_id']!=row['run_id'] or sha256(payload)!=r['counts_SHA256'] or r['known_model_shots']!=100000:raise RuntimeError('H1 draw/request binding mismatch')
        ledger.complete(row['run_id'],'final',r['known_model_shots'],sha256(payload))
        checks.append({'case':case,'draw_status':'COMPLETED','readouts':{mode:'COMPLETED' if (d/mode/'result.json').exists() else 'NOT_COMPLETED' for mode in manifest['protocol']['readouts']}})
    result={'status':'H1_COMMON_SETUP_PREFIX_COLLECTED_NOT_NATIVE_PAPER_REPRODUCTION','checks':checks,'accounting':accounting,'goal_complete':False}
    save(DIRECTORY/'collected.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','collect']);args=p.parse_args();globals()[args.action]()
