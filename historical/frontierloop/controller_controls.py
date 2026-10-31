"""Single-worker C5 dispatcher with durable per-cell terminal observations."""
import fcntl
import subprocess
import sys
from time import time
from .paths import ROOT
from .io import load,save,space
from .run_controls import freeze


def main():
    with (ROOT/'controller-C5-CONTROLS.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=load(ROOT/'protocol-C5-CONTROLS.json')
        if freeze()!=load(ROOT/'freeze-C5-CONTROLS.json'):
            raise RuntimeError('C5 frozen code changed')
        started=time();rows=[]
        for case in spec['cases']:
            for method in spec['methods']:
                for readout in spec['readouts']:
                    space()
                    directory=ROOT/'runs/C5-CONTROLS'/case/method/readout
                    result=directory/'result.json';failure=directory/'failure.json'
                    if result.exists():
                        status=load(result)['status'];returncode=None
                    elif failure.exists():
                        status=load(failure)['status'];returncode=None
                    else:
                        try:
                            task=subprocess.run([sys.executable,'-u','-m','frontierloop.run_controls',
                                '--case',case,'--method',method,'--readout',readout],
                                timeout=spec['max_worker_wall_seconds'])
                            returncode=task.returncode
                            status=load(result)['status'] if result.exists() else 'FAILED_ENGINEERING_PRESERVED'
                        except subprocess.TimeoutExpired:
                            returncode=None;status='WORKER_WALL_CAP_EVIDENCE_PRESERVED'
                    rows.append({'case':case,'method':method,'readout':readout,
                        'status':status,'returncode':returncode})
                    save(ROOT/'progress-C5-CONTROLS.json',{'started_unix':started,
                        'updated_unix':time(),'rows':rows,'new_model_shots':0,'goal_complete':False})
        save(ROOT/'cycle-C5-CONTROLS-runs-finished.json',{'started_unix':started,
            'finished_unix':time(),'rows':rows,'new_model_shots':0,'goal_complete':False})
        print('C5 terminal observations',len(rows),'completed',sum(r['status']=='COMPLETED' for r in rows),flush=True)


if __name__=='__main__':
    main()
