import fcntl
import os
import resource
import subprocess
import sys
from time import time
import xml.etree.ElementTree as ET
from .paths import ROOT,PROJECT
from .io import load,save,sha256,space
from .run_b3 import code_hashes


def main():
    with(ROOT/'controller-B3.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);spec=load(ROOT/'protocol-B3.json')
        if not(ROOT/'cycle-B2-runs-finished.json').exists():raise RuntimeError('Do not mix the research cycles')
        suite=ET.parse(ROOT/'qualification-B3-tests.xml').getroot().find('testsuite')
        if int(suite.get('failures',0))+int(suite.get('errors',0))or int(suite.get('tests'))<11:raise RuntimeError('B3 tests not green')
        binding={'protocol_SHA256':sha256((ROOT/'protocol-B3.json').read_bytes()),'code_SHA256':code_hashes(),
            'tests_passed':int(suite.get('tests')),'exploratory':True}
        freeze=ROOT/'freeze-B3.json'
        if freeze.exists()and load(freeze)!=binding:raise RuntimeError('B3 freeze drift')
        if not freeze.exists():save(freeze,binding)
        save(ROOT/'owner-B3.json',{'pid':os.getpid(),'started_unix':time()})
        for case in spec['cases']:
            seed=spec['seeds'][0];path=ROOT/'runs/B3'/case/str(seed)/'result.json'
            if path.exists():continue
            space();folder=ROOT/'dispatch/B3'/case;folder.mkdir(parents=True,exist_ok=True)
            if(folder/'started.json').exists():raise RuntimeError('Prior dispatch must be reconciled, not repeated')
            command=['rtk','proxy',sys.executable,'-u','-m','frontierloop.run_b3','--case',case,'--seed',str(seed)]
            before=resource.getrusage(resource.RUSAGE_CHILDREN);t=time();env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
            with(folder/'worker.log').open('a')as log:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                save(folder/'started.json',{'pid':child.pid,'started_unix':t,'command':command});code=child.wait()
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            save(folder/'terminal.json',{'exit_code':code,'wall_seconds':time()-t,
                'CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'result_exists':path.exists()})
            if code or not path.exists():raise RuntimeError('B3 failed; retain evidence')
            r=load(path);print('B3',case,r['final_CI'][-1]['energy_Ha'],r['bank_size'],r['completed_depth'],flush=True)
            save(ROOT/'progress-B3.json',{'completed':len(list((ROOT/'runs/B3').glob('*/*/result.json'))),'planned':4,
                'last_case':case,'goal_achieved':False})
        save(ROOT/'cycle-B3-runs-finished.json',{'completed':4,'goal_achieved':False,'finished_unix':time()})


if __name__=='__main__':main()
