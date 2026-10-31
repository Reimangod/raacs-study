"""Preserve bounded TN terminal attempts, never substitute their surrogate for final data."""
import json,subprocess
import numpy as np
from benchmark.gpu_remote import ssh,OPTIONS,HOST
from .io import ROOT,save_json,sha256


def collect():
    for path in ROOT.glob('TN-controls-v1/*/job.json'):
        job=json.loads(path.read_text());target=path.parent/'terminal-evidence-v1'
        if(target/'manifest.json').exists():continue
        state=ssh('sacct -X -n -j '+job['job_id']+' --format=JobID,State,Elapsed,ExitCode -P')
        if not any(s in state for s in('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','COMPLETED')):continue
        size=int(ssh('du -sb '+job['remote']).split()[0])
        if size>100*2**20:raise MemoryError('Inspect oversized TN evidence before copying')
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+job['remote']+'/.',str(target/'remote')],check=True,timeout=120)
        files={str(p.relative_to(target)):sha256(p.read_bytes())for p in(target/'remote').rglob('*')if p.is_file()}
        counts=[]
        for p in(target/'remote/surrogate-evaluations').glob('*-counts.npz'):
            with np.load(p,allow_pickle=False)as a:counts.append(int(a['counts'].sum()))
        records=[json.loads(p.read_text())for p in(target/'remote/surrogate-evaluations').glob('*.json')]
        save_json(target/'manifest.json',{'job':job,'actual_Slurm_accounting':state,'files_SHA256':files,
            'preserved_classical_surrogate_draws_lower_bound':sum(counts),
            'completed_surrogate_objectives':sum(r['status']=='COMPLETED'for r in records),
            'failed_surrogate_objectives':sum(r['status']=='FAILED_PRESERVED'for r in records),
            'unfinished_MPS_or_CI_work_may_not_have_full_records':True,'actual_QPU_shots':0,
            'final_operator_preserved':(target/'remote/optimized-operator.npz').exists(),
            'surrogate_is_not_independent_final_result':True,'no_automatic_unoptimized_substitution':True,
            'initial_operator_is_not_an_optimized_checkpoint':True})
        print('TN_TERMINAL_PRESERVED',path.parent.name,state.strip(),'classical_draws>=',sum(counts),flush=True)


if __name__=='__main__':collect()
