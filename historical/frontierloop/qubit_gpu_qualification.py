"""N16 prerequisite: one-A100, zero-shot qubit-primitive successor parity."""
import argparse,shlex,subprocess
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from pyscf.fci import cistring
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest,arrays,space
from .qubit_primitive_successor import generated_plan,generated_worker,plan_module,fixtures
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST
from paperstudy.gpu_qualification import PYTHON,ENV

LOCAL=ROOT/'qualification/N16-qubit-primitive-GPU-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N16-qubit-primitive-GPU-v1'
BASE=OLD/'paperstudy/gpu_lucj_engine.py'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N16 qualification; reconcile without rerun')
    space();LOCAL.mkdir();cpu,tick=process_time(),perf_counter()
    write(LOCAL/'worker.py',generated_worker().encode());write(LOCAL/'primitive_plan.py',generated_plan().encode())
    module=plan_module();items=[]
    for name,q,n,ne in fixtures():
        d=LOCAL/name;d.mkdir();stream=BytesIO();qpy.dump(q,stream);write(d/'source.qpy',stream.getvalue())
        restored=qpy.load(BytesIO(stream.getvalue()))[0];config,values=module.plan(restored,n,ne)
        full=Statevector.from_instruction(restored).data;a=cistring.make_strings(range(n),ne[0]);b=cistring.make_strings(range(n),ne[1])
        values['expected_sector_vector']=full[(a[:,None]|(b[None,:]<<n)).reshape(-1)]
        payload=arrays(**values);write(d/'arrays.npz',payload)
        config.update(arrays_SHA256=sha256(payload),worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),base_kernel_SHA256=sha256(BASE.read_bytes()),
            source_QPY_SHA256=sha256((d/'source.qpy').read_bytes()),physical_problem_id='SYNTHETIC_NONADJACENT_QUALIFICATION',sampling_requests=[])
        config['input_id']=digest(config);save(d/'config.json',config)
        items.append(dict(name=name,input_id=config['input_id'],config_SHA256=sha256((d/'config.json').read_bytes()),arrays_SHA256=config['arrays_SHA256']))
    value=dict(version='N16_NONADJACENT_QUBIT_CUDA_QUALIFICATION_V1',inputs=items,
        worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),base_kernel_SHA256=sha256(BASE.read_bytes()),
        generated_plan_SHA256=sha256((LOCAL/'primitive_plan.py').read_bytes()),generator_SHA256=sha256((ROOT/'frontierloop/qubit_primitive_successor.py').read_bytes()),
        qualification_code_SHA256=sha256(Path(__file__).read_bytes()),CPU_tests_SHA256=sha256((ROOT/'qualification-qubit-primitive-successor-v1-tests.xml').read_bytes()),
        reference='INDEPENDENT_QISKIT_FULL_FOCK_NOT_FFSIM_NONLOCAL_FERMIONIC_HOP',
        CPU_reference_seconds=process_time()-cpu,CPU_reference_wall_seconds=perf_counter()-tick,
        new_molecular_energy_evaluations=0,new_model_shots=0,new_QPU_shots=0,GPU_allocations_requested=1,
        production_execution_authorized=False,goal_complete=False)
    save(LOCAL/'prepared.json',value);print('N16_THREE_TINY_INPUTS_PREPARED_NO_SHOTS',flush=True)


def submit():
    if (LOCAL/'submission-intent.json').exists():raise RuntimeError('Existing exact submission intent; no duplicate')
    p=load(LOCAL/'prepared.json')
    if sha256((LOCAL/'worker.py').read_bytes())!=p['worker_SHA256'] or sha256(BASE.read_bytes())!=p['base_kernel_SHA256']:raise RuntimeError('Frozen worker/base changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=30).strip()!='NEW':raise RuntimeError('Remote exists; reconcile exact job')
    queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%b|%R"',timeout=30)
    save(LOCAL/'pre-dispatch-queue.json',dict(raw=queue,other_jobs_not_modified=True))
    ssh('mkdir -p '+shlex.quote(REMOTE),timeout=30);send(LOCAL/'worker.py',REMOTE+'/worker.py');send(BASE,REMOTE+'/base_kernel.py')
    commands=[]
    for item in p['inputs']:
        d=LOCAL/item['name'];remote=REMOTE+'/'+item['name'];ssh('mkdir '+shlex.quote(remote),timeout=30)
        for name in ['config.json','arrays.npz']:send(d/name,remote+'/'+name)
        commands.append(shlex.join([PYTHON,'-u',REMOTE+'/worker.py',remote+'/config.json',remote+'/arrays.npz',remote+'/output']))
    script='\n'.join(['#!/bin/bash','set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1',
        'export CUDA_PATH='+ENV+'/lib/python3.12/site-packages/nvidia/cuda_runtime',
        'export LD_LIBRARY_PATH="$CUDA_PATH/lib:'+ENV+'/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"',
        'export CUPY_ACCELERATORS=""',*commands])+'\n'
    write(LOCAL/'job.sbatch',script.encode());send(LOCAL/'job.sbatch',REMOTE+'/job.sbatch')
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-short','--gres=gpu:a100:1','--cpus-per-task=1','--mem=4G','--time=00:05:00',
        '--job-name=frontier-N16-qubit-primitive-qual-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err',REMOTE+'/job.sbatch'])
    save(LOCAL/'submission-intent.json',dict(command=cmd,prepared_SHA256=sha256((LOCAL/'prepared.json').read_bytes())))
    job=ssh(cmd,timeout=30).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown submit response; no blind retry')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=1));print('N16_SUBMITTED_ONE_A100_ZERO_SHOTS',job,flush=True)


def status():
    job=load(LOCAL/'job.json')['job_id'];q=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
    live='\n'.join(x for x in q.splitlines() if x.split('|')[0]==job)
    accounting=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30).strip()
    save(LOCAL/'status.json',dict(job_id=job,live=live,accounting=accounting));print(live or accounting or 'UNRESOLVED_NO_RESTART',flush=True)


def collect():
    if (LOCAL/'collected.json').exists():raise RuntimeError('Already collected; inspect existing evidence')
    status();s=load(LOCAL/'status.json')
    if s['live'] or '|COMPLETED|' not in s['accounting']:raise RuntimeError('Need actual COMPLETED qualification')
    p=load(LOCAL/'prepared.json');records=[]
    for item in p['inputs']:
        name=item['name'];d=LOCAL/name/'output';d.mkdir(exist_ok=True)
        # Tiny JSON only; the raw references/arrays remain bound local inputs.
        blob=ssh('cat '+shlex.quote(REMOTE+'/'+name+'/output/result.json'),timeout=30).encode();write(d/'result.json',blob)
        r=load(d/'result.json');parity=r.get('independent_state_reference_parity')
        if r['input_id']!=item['input_id'] or r['status']!='EXACT_MATERIALIZED_CIRCUIT_GPU_COMPLETED' or r['sampling_results'] or not parity or max(parity.values())>1e-8 or r['primitive_semantics']!='HAMMING_SECTOR_QUBIT_XXYY_NOT_NONLOCAL_FERMIONIC_HOP':raise RuntimeError('Actual GPU qualification failed')
        records.append(dict(name=name,parity=parity,result_SHA256=sha256(blob),result=r))
    for name in ['slurm.out','slurm.err']:write(LOCAL/name,ssh('cat '+shlex.quote(REMOTE+'/'+name),timeout=30).encode())
    result=dict(status='N16_ALL3_NONADJACENT_QUBIT_GPU_PARITIES_PASSED',records=records,accounting=s['accounting'],
        prepared_SHA256=sha256((LOCAL/'prepared.json').read_bytes()),new_molecular_energy_evaluations=0,new_model_shots=0,new_QPU_shots=0,
        scope='SMALL_COMPLEX_HOLE_AND_VIRTUAL_SWAP_CUDA_QUALIFICATION_NOT_MOLECULAR_PERFORMANCE',goal_complete=False)
    save(LOCAL/'collected.json',result);print(result['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','status','collect']);a=parser.parse_args();globals()[a.action]()
