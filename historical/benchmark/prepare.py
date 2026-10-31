"""Chemistry-only cache preparation; never starts ADAPT, sampling or GPU work."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import traceback
from .config import cases_for
from .common import save_json


def prepare_case(case,root):
    import fcntl
    from .chemistry import build_reference
    lock_path=Path(root)/"locks"/f"{case.id}.lock";lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            r=build_reference(case,root,threads=1)
            return {"case_id":case.id,"status":"REFERENCE_READY","problem":r["meta"]["request_digest"]}
        except Exception as e:
            return {"case_id":case.id,"status":"FAILED_REFERENCE","error":repr(e),"traceback":traceback.format_exc()}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--molecules",nargs="+",required=True)
    p.add_argument("--workers",type=int,default=2);a=p.parse_args()
    cases=[c for c in cases_for("full") if c.molecule in a.molecules]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=[pool.submit(prepare_case,c,a.output) for c in cases]
        for task in as_completed(tasks):
            result=task.result();results.append(result);print(result["case_id"],result["status"],flush=True)
    save_json(Path(a.output)/"preparation-status.json",results)
    if any(r["status"]!="REFERENCE_READY" for r in results):raise SystemExit(1)


if __name__=="__main__":main()
