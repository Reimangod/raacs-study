import fcntl
import os
import resource
import subprocess
import sys
from time import time
import xml.etree.ElementTree as ET
from .paths import ROOT,PROJECT
from .io import load,save,sha256,space
from .run_orbital import code_hashes


def main():
    with(ROOT/'controller-C2-OO.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);spec=load(ROOT/'protocol-C2-OO.json')
        suite=ET.parse(ROOT/'qualification-orbital-tests.xml').getroot().find('testsuite')
        if int(suite.get('failures',0))+int(suite.get('errors',0))or int(suite.get('tests'))!=1:raise RuntimeError('OO qualification not green')
        freeze={'protocol_SHA256':sha256((ROOT/'protocol-C2-OO.json').read_bytes()),'code_SHA256':code_hashes(),
            'independent_small_FCI_RDM_and_rotation_test_passed':True,'exploratory':True}
        path=ROOT/'freeze-C2-OO.json'
        if path.exists()and load(path)!=freeze:raise RuntimeError('OO freeze drift')
        if not path.exists():save(path,freeze)
        save(ROOT/'owner-C2-OO.json',{'pid':os.getpid(),'started_unix':time()})
        for case in spec['cases']:
            for method in spec['methods']:
                output=ROOT/'runs/C2-OO'/case/method/'result.json'
                if output.exists():continue
                space();folder=ROOT/'dispatch/C2-OO'/case/method;folder.mkdir(parents=True,exist_ok=True)
                if(folder/'started.json').exists():raise RuntimeError('Prior OO launch requires reconciliation')
                command=['rtk','proxy',sys.executable,'-u','-m','frontierloop.run_orbital','--case',case,'--method',method]
                t=time();before=resource.getrusage(resource.RUSAGE_CHILDREN);env=dict(os.environ);env['PYTHONPATH']=str(ROOT);env['JAX_PLATFORMS']='cpu'
                with(folder/'worker.log').open('a')as log:
                    child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    save(folder/'started.json',{'pid':child.pid,'command':command,'started_unix':t});code=child.wait()
                after=resource.getrusage(resource.RUSAGE_CHILDREN)
                save(folder/'terminal.json',{'exit_code':code,'wall_seconds':time()-t,
                    'CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'new_quantum_model_shots':0})
                if code or not output.exists():raise RuntimeError('OO worker failed; retain all costs and checkpoints')
                r=load(output);print('C2',case,method,r['initial_energy_Ha'],r['final_energy_Ha'],flush=True)
                save(ROOT/'progress-C2-OO.json',{'completed':len(list((ROOT/'runs/C2-OO').glob('*/*/result.json'))),
                    'planned':12,'last_case':case,'last_method':method,'goal_achieved':False})
        save(ROOT/'cycle-C2-OO-runs-finished.json',{'completed':12,'goal_achieved':False,'new_quantum_model_shots':0,'finished_unix':time()})


if __name__=='__main__':main()
