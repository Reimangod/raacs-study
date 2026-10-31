"""Bounded batching of the unchanged G4 verifier, not a new simulator."""
import json
import sys
from time import process_time, perf_counter
import traceback
from .mps_distribution_check import check_request


def validate(request):
    jobs = request['jobs']
    if not 1 <= len(jobs) <= 16 or request['tolerance'] != 1e-8:
        raise ValueError('Frozen batch size/tolerance')
    keys = [x['key'] for x in jobs]
    if len(set(keys)) != len(keys) or any(not isinstance(x, str) or not x for x in keys):
        raise ValueError('Duplicate/invalid member namespace')
    if any(set(x['circuits']) != {'NATIVE','RUSTIQ','PROJECTED'} for x in jobs):
        raise ValueError('All three actual circuit controls required')
    return jobs


def run(request):
    jobs = validate(request)
    cpu, wall = process_time(), perf_counter()
    results = {}
    for job in jobs:
        try:
            value = check_request({'circuits': job['circuits'], 'tolerance': request['tolerance']})
            results[job['key']] = {'status': 'PASSED', 'check': value}
        except Exception as exc:
            results[job['key']] = {'status': 'FAILED_ENGINEERING_PRESERVED', 'error': repr(exc),
                'traceback': traceback.format_exc()}
    return {'results': results, 'CPU_seconds': process_time()-cpu,
        'wall_seconds': perf_counter()-wall, 'new_model_shots': 0, 'new_QPU_shots': 0}


if __name__ == '__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    print(json.dumps(run(json.load(sys.stdin)), allow_nan=False))
