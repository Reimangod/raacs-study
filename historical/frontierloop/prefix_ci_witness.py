"""Retrospective variational witnesses, not new CI optimizations or selection.

For each saved N17 M4 CI basis, project EVERY saved root into that basis and
evaluate its Rayleigh quotient. A lower feasible quotient would falsify the
reported minimum in that basis. Failure to find one is not a global proof.
"""
import argparse
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,space
from .joint_ci_sparse import build_library,matrices

LOCAL=ROOT/'runs/N17-M4-CI-WITNESS-v1'
PARENT=ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-M4-v1'


def project(source_pairs,source_vector,target_pairs):
    lookup={tuple(map(int,p)):float(v) for p,v in zip(source_pairs,source_vector)}
    values=np.array([lookup.get(tuple(map(int,p)),0.) for p in target_pairs])
    norm=float(values@values)
    return (values/np.sqrt(norm) if norm>0 else None),norm


def witness(H,S,core,vector):
    physical=float(vector@(H@vector)+core);s2=float(vector@(S@vector))
    return physical+.1*s2,physical,s2


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing diagnostic; reconcile rather than overwrite')
    space();result=ROOT/'analysis/N17-M4-common-readout-summary.json';summary=load(result)
    if summary['status']!='N17_M4_ALL_FROZEN_LOGICAL_READOUTS_AUDITED':raise RuntimeError('Actual complete parent audit required')
    obs=load(PARENT/'collected/observations.json')
    if len(obs)!=13 or {o['case'] for o in obs}!={'NAPH20-BLOCK-BOYS'}:raise RuntimeError('Fixed all13 Naph roots required')
    physical=load(PARENT/'protocol.json')['physical']['NAPH20-BLOCK-BOYS']
    file=OLD/'data/physical/NAPH20-BLOCK-BOYS/integrals.npz'
    if sha256(file.read_bytes())!=physical['integrals_SHA256']:raise RuntimeError('Local physical integral mismatch')
    LOCAL.mkdir();spec=dict(version='N17_M4_ALL13_CROSS_BASIS_VARIATIONAL_WITNESSES_V1',
        observation_ids=[o['id'] for o in obs],observations_SHA256=sha256((PARENT/'collected/observations.json').read_bytes()),
        parent_audit_SHA256=sha256(result.read_bytes()),code_SHA256=sha256(Path(__file__).read_bytes()),
        math_SHA256={name:sha256((ROOT/'frontierloop'/name).read_bytes()) for name in ['joint_ci_sparse.py','joint_ci_sparse.c']},
        tests_SHA256=sha256((ROOT/'qualification-N17-ci-witness-v1-tests.xml').read_bytes()),
        physical=physical,local_integrals=str(file),planned_classical_Rayleigh_evaluations=169,
        root_optimization_calls=0,per_matrix_wall_seconds=15,global_wall_seconds=120,violation_tolerance_Ha=1e-8,
        new_model_shots=0,new_QPU_shots=0,FCI_calls=0,
        purpose='RETROSPECTIVE_ENGINEERING_DIAGNOSTIC_NOT_ENERGY_GUIDED_POLICY_OR_PERFORMANCE_RETRY',goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N17_WITNESS_FROZEN_ALL13_BY13_NO_NEW_SHOTS_OR_EIGENSOLVES',flush=True)


def run():
    if (LOCAL/'started.json').exists():raise RuntimeError('Existing witness work; no blind retry')
    spec=load(LOCAL/'protocol.json');space()
    if sha256(Path(__file__).read_bytes())!=spec['code_SHA256']:raise RuntimeError('Frozen witness code changed')
    for name,h in spec['math_SHA256'].items():
        if sha256((ROOT/'frontierloop'/name).read_bytes())!=h:raise RuntimeError('Frozen matrix builder changed')
    if sha256(Path(spec['local_integrals']).read_bytes())!=spec['physical']['integrals_SHA256']:
        raise RuntimeError('Physical integrals changed')
    if sha256((PARENT/'collected/observations.json').read_bytes())!=spec['observations_SHA256']:
        raise RuntimeError('Actual saved roots changed')
    save(LOCAL/'started.json',dict(protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),PID=__import__('os').getpid()))
    cpu,tick=process_time(),perf_counter();roots=[];results=[];evals=0
    try:
        library=build_library(LOCAL/'library')
        with np.load(spec['local_integrals'],allow_pickle=False) as a:h=a['h1'];g=a['eri']
        for o in load(PARENT/'collected/observations.json'):
            path=PARENT/'collected'/o['vector_file']
            if sha256(path.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved root vector changed')
            with np.load(path,allow_pickle=False) as a:pairs=a['joint_pairs'];vector=a['coefficients']
            if abs(vector@vector-1)>1e-10:raise RuntimeError('Source vector norm differs')
            roots.append((o,pairs,vector))
        definition=spec['physical'];core=definition['core_energy']
        for target,target_pairs,target_vector in roots:
            if perf_counter()-tick>spec['global_wall_seconds']:raise TimeoutError('Witness diagnostic wall cap')
            H,S,info=matrices(h,g,tuple(definition['nelec']),target_pairs,library,seconds=spec['per_matrix_wall_seconds'])
            self_objective,self_energy,self_s2=witness(H,S,core,target_vector);evals+=1
            if max(abs(self_objective-target['row']['penalized_energy_Ha']),abs(self_energy-target['row']['energy_Ha']),abs(self_s2-target['row']['spin_square']))>1e-8:
                raise RuntimeError('Independent-platform saved-root self-energy parity failed')
            cross=[]
            for source,source_pairs,source_vector in roots:
                if source['id']==target['id']:
                    cross.append(dict(source_id=source['id'],retained_norm=1.,objective_Ha=self_objective,
                        physical_energy_Ha=self_energy,spin_square=self_s2,delta_vs_reported_Ha=self_objective-target['row']['penalized_energy_Ha'],
                        violates_reported_minimum=False,self_parity=True));continue
                projected,norm=project(source_pairs,source_vector,target_pairs)
                record=dict(source_id=source['id'],retained_norm=norm,self_parity=False)
                if projected is None:record.update(objective_Ha=None,violates_reported_minimum=False)
                else:
                    objective,energy,s2=witness(H,S,core,projected);evals+=1;delta=objective-target['row']['penalized_energy_Ha']
                    record.update(objective_Ha=objective,physical_energy_Ha=energy,spin_square=s2,
                        delta_vs_reported_Ha=delta,violates_reported_minimum=delta < -spec['violation_tolerance_Ha'])
                cross.append(record)
            results.append(dict(target_id=target['id'],actual_CI_dimension=len(target_pairs),matrix_work=info,
                self_energy_parity_error_Ha=self_energy-target['row']['energy_Ha'],self_objective_parity_error_Ha=self_objective-target['row']['penalized_energy_Ha'],witnesses=cross))
            save(LOCAL/'progress.json',dict(targets_completed=len(results),planned_targets=13,results=results,actual_classical_Rayleigh_evaluations=evals))
            print('N17_WITNESS_TARGET',len(results),target['id'][:12],flush=True)
        violations=[dict(target_id=r['target_id'],**w) for r in results for w in r['witnesses'] if w['violates_reported_minimum']]
        out=dict(status='ALL13_BY13_WITNESSES_COMPLETE_NO_LOWER_FEASIBLE_WITNESS' if not violations else 'LOWER_FEASIBLE_WITNESS_NUMERICAL_INCIDENT',
            results=results,violations=violations,actual_classical_Rayleigh_evaluations=evals,new_CI_optimizations=0,
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-tick,new_model_shots=0,new_QPU_shots=0,FCI_calls=0,
            global_minimum_proved=False,original_results_overwritten=False,goal_complete=False)
        save(LOCAL/'result.json',out);print(out['status'],len(violations),flush=True)
    except Exception as exc:
        save(LOCAL/'failed.json',dict(status='ENGINEERING_DIAGNOSTIC_FAILED_PRESERVED',error=repr(exc),
            CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-tick,actual_classical_Rayleigh_evaluations=evals,
            new_CI_optimizations=0,new_model_shots=0,new_QPU_shots=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();globals()[a.action]()
