"""Same all-three-circuit probability requirement, different numerical engine."""
import json,sys,traceback
from time import process_time,perf_counter
from .mps_batch_worker import validate
from .clifford_targeted_checker import check


def run(request):
    jobs=validate(request);cpu,wall=process_time(),perf_counter();results={}
    for job in jobs:
        try:
            value=check(dict(circuits=job['circuits'],tolerance=request['tolerance']))
            results[job['key']]=dict(status='PASSED',check=value)
        except Exception as exc:
            results[job['key']]=dict(status='FAILED_ENGINEERING_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
    return dict(results=results,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        verifier='TARGETED_CLIFFORD_AMPLITUDES_GATE_LEVEL_NUMERICAL_NOT_MPS',new_model_shots=0,new_QPU_shots=0)


if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    print(json.dumps(run(json.load(sys.stdin)),allow_nan=False))
