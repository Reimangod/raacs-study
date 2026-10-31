"""Numerics-only full ADAPT trajectory. No FCI data or chemistry dependency.

This worker is shared by NumPy/CUDA qualification and molecular execution.
Circuit reconstruction and independent author/Qiskit validation happen on Mac.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from .cuda_backend import ExcitationBackend
from .common import digest,save_json,save_npz,file_digest


def execute(bundle,output,backend,max_steps=None):
    bundle=Path(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request=json.loads((bundle/"request.json").read_text())
    for name,sha in request["files"].items():
        if file_digest(bundle/name)!=sha:raise RuntimeError("Input checksum mismatch: "+name)
    catalog=json.loads((bundle/"pool.json").read_text())
    cfg=request["config"];nsteps=cfg["max_steps"] if max_steps is None else max_steps
    init=perf_counter();engine=ExcitationBackend(sparse.load_npz(bundle/"hamiltonian.npz"),catalog,backend)
    initialization_seconds=perf_counter()-init
    indices=[];theta=np.array([]);next_step=0;previous=0.
    identity=digest({"request":request,"backend":backend,"steps":nsteps})
    paths=sorted(output.glob("step-*.json"))
    if paths:
        last=json.loads(paths[-1].read_text())
        if last["execution_id"]!=identity:raise RuntimeError("Cannot resume a different execution")
        indices=last["selected_operators"];theta=np.array(last["optimized_parameters"])
        next_step=last["step"]+1;previous=last["trajectory_seconds"]
    started=perf_counter();terminal="MAX_STEPS"
    for step in range(next_step,nsteps+1):
        opt=None;selection_start=perf_counter()
        state=engine.prepare(indices,theta,request["hf_id"])
        gradients=engine.selection_gradients(state)
        selection_seconds=perf_counter()-selection_start
        if step and np.linalg.norm(gradients)<=cfg["gradient_tolerance"]:
            terminal="CONVERGED";break
        selected=None;additions=[];opt_seconds=0.
        if step:
            selected=int(np.argmax(np.abs(gradients)))
            if request["method"]=="ceo_mvp":
                support=catalog["operators"][selected]["support"]
                additions=[i for i,op in enumerate(catalog["operators"]) if op["support"]==support and abs(gradients[i])>1e-10]
            else:additions=[selected]
            indices=indices+additions;theta=np.r_[theta,np.zeros(len(additions))]
        initial_theta=theta.tolist()
        if step:
            opt_start=perf_counter()
            opt=minimize(lambda x:engine.energy_and_gradient(indices,x,request["hf_id"]),theta,
                         method="BFGS",jac=True,options={"gtol":cfg["optimizer_gtol"],"maxiter":cfg["optimizer_maxiter"]})
            theta=opt.x;opt_seconds=perf_counter()-opt_start
        state=engine.prepare(indices,theta,request["hf_id"])
        energy=float(engine.host(engine.xp.vdot(state,engine.h@state).real))
        state_path=output/f"step-{step:04d}.npz"
        save_npz(state_path,state=engine.host(state))
        cp={"execution_id":identity,"step":step,"selected_operators":indices,
            "optimized_parameters":theta.tolist(),"initial_parameters":initial_theta,
            "selected_this_step":selected,"selected_group":additions,
            "pool_gradients_before_append":gradients.tolist(),"energy":energy,
            "selection_seconds":selection_seconds,"optimization_seconds":opt_seconds,
            "objective_evaluations":int(opt.nfev) if opt else 0,"gradient_evaluations":int(opt.njev) if opt else 0,
            "optimizer_iterations":int(opt.nit) if opt else 0,"optimizer_success":bool(opt.success) if opt else None,
            "optimizer_message":str(opt.message) if opt else "HF_NOT_OPTIMIZED",
            "final_parameter_gradient_norm":float(np.linalg.norm(opt.jac)) if opt else None,
            "trajectory_seconds":previous+perf_counter()-started,"state_sha256":file_digest(state_path)}
        save_json(output/f"step-{step:04d}.json",cp)
        print(request["case_id"],request["method"],backend,step,energy,flush=True)
        if cp["trajectory_seconds"]>cfg["task_seconds"]:
            terminal="TIME_CENSORED";break
    status={"status":terminal,"execution_id":identity,"request_digest":digest(request),
            "backend":backend,"device":getattr(engine,"device_name","CPU"),"slurm_job":os.environ.get("SLURM_JOB_ID"),
            "initialization_seconds":initialization_seconds,"trajectory_seconds":previous+perf_counter()-started,
            "checkpoints":len(list(output.glob("step-*.json"))),"dense_expm":0,
            "numpy":np.__version__,"scipy":__import__("scipy").__version__}
    save_json(output/"terminal.json",status);return status


def main():
    p=argparse.ArgumentParser();p.add_argument("--bundle",required=True);p.add_argument("--output",required=True)
    p.add_argument("--backend",choices=["numpy","cupy"],required=True);p.add_argument("--max-steps",type=int)
    a=p.parse_args();execute(a.bundle,a.output,a.backend,a.max_steps)


if __name__=="__main__":main()
