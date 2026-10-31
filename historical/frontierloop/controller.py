import fcntl
from importlib.metadata import version
import os
import resource
import sqlite3
import subprocess
import sys
from time import time
import xml.etree.ElementTree as ET
from .paths import ROOT,V2,PROJECT
from .io import load,save,sha256,space
from .run import code_hashes


def freeze(cycle):
    path=ROOT/('freeze-'+cycle+'.json')
    spec=load(ROOT/('protocol-'+cycle+'.json'))
    suite=ET.parse(ROOT/'qualification-tests.xml').getroot().find('testsuite')
    if int(suite.get('failures',0))+int(suite.get('errors',0))or int(suite.get('tests'))<6:
        raise RuntimeError('Focused qualification not green')
    old=load(V2/'qualification/scientific-code.json')['code_SHA256']
    for name,h in old.items():
        if sha256((V2/'raacsnext'/name).read_bytes())!=h:raise RuntimeError('Old scientific code changed')
    current={'protocol_SHA256':sha256((ROOT/('protocol-'+cycle+'.json')).read_bytes()),
        'new_code_SHA256':code_hashes(),'old_code_SHA256':old,'tests_passed':int(suite.get('tests')),
        'runtime':{p:version(p)for p in('numpy','scipy','pyscf','qiskit','ffsim')},
        'prior_tests_ledger':'shots.sqlite contains synthetic qualification only; production-shots.sqlite is separate'}
    if path.exists()and load(path)!=current:raise RuntimeError('Cycle freeze drift')
    if not path.exists():save(path,current)
    return spec


def main(cycle='B1'):
    with(ROOT/('controller-'+cycle+'.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);spec=freeze(cycle)
        save(ROOT/('owner-'+cycle+'.json'),{'pid':os.getpid(),'started_unix':time()})
        logroot=ROOT/'dispatch'/cycle
        for seed in spec['seeds']:
            for case in spec['cases']:
                result=ROOT/'runs'/cycle/case/str(seed)/'result.json'
                if result.exists():continue
                space();folder=logroot/case/str(seed);folder.mkdir(parents=True,exist_ok=True)
                if(folder/'started.json').exists():raise RuntimeError('Prior dispatch must be reconciled, not resubmitted')
                cpu0=resource.getrusage(resource.RUSAGE_CHILDREN);started=time()
                command=['rtk','proxy',sys.executable,'-u','-m','frontierloop.run','--case',case,'--seed',str(seed),'--cycle',cycle]
                env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
                with(folder/'worker.log').open('a')as log:
                    child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    save(folder/'started.json',{'pid':child.pid,'command':command,'started_unix':started})
                    code=child.wait()
                cpu1=resource.getrusage(resource.RUSAGE_CHILDREN)
                save(folder/'terminal.json',{'exit_code':code,'wall_seconds':time()-started,
                    'CPU_seconds':cpu1.ru_utime+cpu1.ru_stime-cpu0.ru_utime-cpu0.ru_stime,'result_exists':result.exists()})
                if code or not result.exists():raise RuntimeError('Run failed, evidence preserved at '+str(folder))
                r=load(result)
                print(cycle,case,seed,r['final_CI'][-1]['energy_Ha'],r['bank_size'],r['construction_stop_reason'],flush=True)
                save(ROOT/('progress-'+cycle+'.json'),{'completed':len(list((ROOT/'runs'/cycle).glob('*/*/result.json'))),
                    'planned':spec['planned_runs'],'last_case':case,'last_seed':seed,'goal_achieved':False})
        save(ROOT/('cycle-'+cycle+'-runs-finished.json'),{'completed':spec['planned_runs'],'goal_achieved':False,'finished_unix':time()})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cycle',default='B1');a=p.parse_args();main(a.cycle)
