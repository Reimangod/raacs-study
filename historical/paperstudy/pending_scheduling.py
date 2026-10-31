"""Move only this study's unstarted jobs to the general-use Slurm partition.

No priority manipulation, GPU/CPU/memory increase, time extension, duplicate
submission, or interaction with another chat's job. Original job IDs and
scientific requests survive. Both partitions are allowed by AIC's manual.
"""
import json
from time import sleep,time
from benchmark.gpu_remote import ssh
from .gpu_qualification import REMOTE
from .io import ROOT,save_json


def fields(text):
    return dict(piece.split("=",1)for piece in text.split()if "="in piece)


def validate_unchanged(before,after):
    a,b=fields(before),fields(after)
    # NumCPUs is an allocation field: Slurm may round one requested logical
    # thread to a whole physical core when PENDING becomes RUNNING. Preserve
    # and compare the request fields, while retaining that actual allocation
    # in the full accounting rather than miscalling it a changed request.
    for key in ("JobId","UserId","Command","TimeLimit","CPUs/Task","MinMemoryNode","ReqTRES","TresPerNode","TresPerTask"):
        if a.get(key)!=b.get(key):raise RuntimeError("Unexpected scheduling resource/request change: "+key)
    if b["Partition"]!="gpu-standard":raise RuntimeError("Partition change was not applied")


def sweep():
    rows=ssh("squeue -u reiwatanabe -h -o '%i|%j|%T|%P'").splitlines()
    for row in rows:
        job,name,state,partition=row.split("|")
        if state!="PENDING"or partition!="gpu-short"or not name.startswith("paper-")or not job.isdigit():continue
        before=ssh("scontrol show job "+job);f=fields(before)
        if f.get("JobState")!="PENDING"or not f.get("Command","").startswith(REMOTE+"/"):continue
        record=ROOT/"scheduling-addendum-v2"/(job+".json")
        if record.exists():continue
        value={"job_id":job,"before":before,"recorded_at_unix":time(),
            "change":"Only pending partition: gpu-short -> gpu-standard; normal QOS, same job ID/CPU/RAM/GPU/time/scientific request",
            "source":"https://docs.keioaic.dev/slurm_user_manual","scientific_settings_unchanged":True}
        save_json(record,value)
        reply=ssh("scontrol update JobId="+job+" Partition=gpu-standard")
        after=ssh("scontrol show job "+job);value.update(reply=reply,after=after)
        validate_unchanged(before,after);value["resource_invariance_verified"]=True
        save_json(record,value);print("PENDING_PARTITION_ONLY",job,flush=True)


if __name__=="__main__":
    import argparse,fcntl
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");args=p.parse_args()
    with (ROOT/"pending-scheduling.lock").open("a")as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            sweep()
            if not args.watch:break
            sleep(60)
