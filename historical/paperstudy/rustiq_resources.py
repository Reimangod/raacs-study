"""Equivalent compiler sensitivity from the very same SqDRIFT members.

Readouts, random sequences and primary code remain immutable. Only the exact
order-preserving synthesis changes, in a separately labelled resource column.
"""
from io import BytesIO
import gzip
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import perf_counter,process_time
import numpy as np
from qiskit import qpy
from .sqdrift_execution import Session
from .lucj import circuit_views
from .io import ROOT,save_json,digest,sha256
from .analysis_products import percentile

PROJECT=ROOT.parents[2]/"ceo-sampler-benchmark"


class RustiqSession(Session):
    def __init__(self,base,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH=str(PROJECT/".runtime-overlays/sqdrift-qiskit25-v1")+os.pathsep+str(ROOT),
                 OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        self.log=(self.directory/"worker-stderr.log").open("a")
        self.process=subprocess.Popen(["rtk","proxy",sys.executable,"-u","-m","paperstudy.rustiq_native_worker",
            "--base",str(base),"--directory",str(directory)],cwd=PROJECT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1)
        if not self.read(180).get("ready"):raise RuntimeError("Compiler session unavailable")


def run():
    root=ROOT/"rustiq-control-v1"
    qualification=json.loads((root/"qualification-v2.json").read_text())
    if qualification["status"]!="INDEPENDENT_NATIVE_RUSTIQ_STATE_PARITY_PASSED":raise RuntimeError("Unqualified compiler")
    db=sqlite3.connect(ROOT/"run-queue.sqlite")
    sources=db.execute("SELECT case_id,budget,seed,result_path FROM planned_cells WHERE method_id='D_SQDRIFT_N20' AND status='COMPLETED' ORDER BY budget,case_id,seed").fetchall();db.close()
    sessions={};completed=[]
    try:
        for case,B,seed,path in sources:
            output=root/"common-readout-resource-joins"/case/(str(B)+"-"+str(seed)+".json")
            if output.exists():completed.append(json.loads(output.read_text()));continue
            result=json.loads((ROOT/path).read_text())
            entries=[];tick,cpu=perf_counter(),process_time()
            for member in result["member_records"]:
                member_id=digest(member["spec"])
                directory=root/"members"/case/member_id
                record_path=directory/"resource.json"
                if record_path.exists():record=json.loads(record_path.read_text())
                else:
                    if case not in sessions:sessions[case]=RustiqSession(ROOT/"native-SqDRIFT"/case/"base.json",root/"members"/case)
                    reply=sessions[case].export(member["spec"],directory)
                    circuit=qpy.load(BytesIO(gzip.decompress((directory/"circuit.qpy.gz").read_bytes())))[0]
                    _,_,views=circuit_views(circuit)
                    record={"member_id":member_id,"spec":member["spec"],"resources":views,
                            "worker_sha256":sha256(Path(__file__).with_name("rustiq_native_worker.py").read_bytes()),
                            "export_cost":reply,"new_model_shots":0,"source_member_order_preserved":True}
                    save_json(record_path,record)
                entries.append((member["shots"],record))
            aggregate=[]
            for view in ("preparation","Z_sampling_equivalent"):
                for metric in ("cnot_count","cnot_depth","total_unitary_depth"):
                    values=[r["resources"][view][metric] for _,r in entries];weights=[w for w,_ in entries]
                    aggregate.append({"view":view,"metric":metric,"shot_weighted_mean":float(np.average(values,weights=weights)),
                                      "maximum":max(values),"p95":percentile(values,weights),
                                      "model_shot_weighted_applications":int(sum(w*v for w,v in zip(weights,values)))})
            record={"case_id":case,"budget":B,"seed":seed,"method":"D_SQDRIFT_N20", "lane":"S_RESOURCE_ONLY_COMPILER_CONTROL",
                    "source_result":path,"source_result_sha256":sha256((ROOT/path).read_bytes()),"source_counts_unchanged":True,
                    "resources":aggregate,"new_model_shots":0,"actual_QPU_shots":0,
                    "compiler":"Qiskit2.5 bundled Rustiq; preserve_order=true, upto_clifford=false, upto_phase=false; common postcompiler opt1",
                    "paper_Rustiq_version_0p0p8_reproduced":False,
                    "local_CPU_seconds":process_time()-cpu,"wall_seconds":perf_counter()-tick,
                    "native_exporter_CPU_separately_in_member_records":True}
            save_json(output,record);completed.append(record)
            print("RUSTIQ_RESOURCE_JOIN",case,B,seed,flush=True)
    finally:
        for session in sessions.values():session.close()
    save_json(root/"summary.json",{"observed_primary_SqDRIFT_runs_recompiled":len(completed),"planned_primary_SqDRIFT_runs":120,
              "new_model_shots":0,"primary_outcomes_mutated":False,"whole_experiment_complete":False})


def main(watch=False):
    import fcntl
    from time import sleep
    with(ROOT/"rustiq-controller.lock").open("a")as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            run()
            if not watch:return
            db=sqlite3.connect(ROOT/"run-queue.sqlite")
            pending=db.execute("SELECT COUNT(*) FROM planned_cells WHERE method_id='D_SQDRIFT_N20' AND status IN ('NOT_STARTED','RUNNING','WAITING_REMOTE_EXECUTOR')").fetchone()[0]
            db.close()
            if not pending:return
            sleep(60)


if __name__=="__main__":
    import argparse;p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");main(p.parse_args().watch)
