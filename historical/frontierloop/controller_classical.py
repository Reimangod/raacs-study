"""One-worker fresh classical control after terminal C5/B4 execution."""
import fcntl
import resource
import subprocess
import sys
from time import time
from .paths import ROOT
from .io import load,save,space
from .run_classical_control import guard


def main():
    with (ROOT/'controller-C6-CLASSICAL-CISD.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c5=load(ROOT/'cycle-C5-CONTROLS-runs-finished.json')
        b4=load(ROOT/'cycle-B4-runs-finished.json')
        if len(c5['rows'])!=120 or len(b4['rows'])!=4 or any(x['returncode'] for x in b4['rows']):
            raise RuntimeError('Previous production phases are not terminal')
        spec=guard();rows=[];started=time()
        def worker(arguments):
            space();before=resource.getrusage(resource.RUSAGE_CHILDREN);tick=time()
            p=subprocess.run([sys.executable,'-u','-m','frontierloop.run_classical_control',*arguments],
                timeout=spec['max_worker_wall_seconds'])
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            return {'returncode':p.returncode,
                'worker_whole_process_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                'worker_whole_process_wall_seconds':time()-tick}
        for case in spec['cases']:
            result=worker(['--prepare','--case',case])
            rows.append({'phase':'FRESH_CISD_SOURCE','case':case,**result})
            save(ROOT/'progress-C6-CLASSICAL-CISD.json',{'started_unix':started,'updated_unix':time(),
                'rows':rows,'new_quantum_model_shots':0,'goal_complete':False})
            if result['returncode']:
                # Preserve the cause, without silently replacing it with a
                # historical coefficient vector or a zero-cost HF success.
                continue
            for variant in spec['variants']:
                for readout in spec['readouts']:
                    result=worker(['--case',case,'--variant',variant,'--readout',readout])
                    rows.append({'phase':'MATCHED_CI_OO_READOUT','case':case,
                        'variant':variant,'readout':readout,**result})
                    save(ROOT/'progress-C6-CLASSICAL-CISD.json',{'started_unix':started,'updated_unix':time(),
                        'rows':rows,'new_quantum_model_shots':0,'goal_complete':False})
        save(ROOT/'cycle-C6-CLASSICAL-CISD-runs-finished.json',{'started_unix':started,
            'finished_unix':time(),'rows':rows,'new_quantum_model_shots':0,'goal_complete':False})


if __name__=='__main__':
    main()
