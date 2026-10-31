"""Cold M4 subprocess timing for the exact same numeric qualification fixtures."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT,save_json,file_digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--fixture",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
        env[key]="1"
    command=[sys.executable,str(ROOT/"benchmark/gpu_profile_worker.py"),"--fixture",a.fixture,
             "--backend","cpu","--output",str(out)]
    started=time.perf_counter()
    result=subprocess.run(command,env=env)
    save_json(out/"launch.json",{"process_wall_seconds":time.perf_counter()-started,
                "returncode":result.returncode,"fixture_sha256":file_digest(a.fixture),
                "worker_code_sha256":file_digest(ROOT/"benchmark/gpu_profile_worker.py"),
                "platform":"Mac CPU; no CUDA was executed"})
    raise SystemExit(result.returncode)


if __name__=="__main__":main()
