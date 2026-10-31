"""Outcome-blind finite policy and immutable input inventory for N57."""
from math import ceil
from pathlib import Path
import sys
import importlib.metadata as metadata
import xml.etree.ElementTree as ET
from .paths import ROOT,OLD
from .io import load,save,sha256,digest,space
from paperstudy.sqdrift_execution import allocations

LOCAL=ROOT/'runs/N57-FILTERED-SQDRIFT-PAIRED-v1'
CASES=['NAPH20-BLOCK-BOYS','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2']
ARMS=['FULL_N20','FILTERED_SHORT'];SEEDS=[570101,570211,570307]


def length(full,filtered):
    if not 0<filtered<=full:raise ValueError('Positive nonincreasing lambda required')
    return max(1,ceil(20*filtered/full))


def create():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Protocol exists; verify instead')
    space();qual=load(LOCAL/'qualification/result.json')
    if qual['status']!='ALL4_NATIVE_FILTERED_TOY_PROBABILITIES_QUALIFIED':raise ValueError('Small parity required')
    suites=list(ET.parse(ROOT/'qualification-N57-policy-v1.xml').getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=5 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise ValueError('Policy tests required')
    inv=load(ROOT/'analysis/N56-diagonal-filter-inventory.json');cases=[];draws=[]
    for name in CASES:
        row=next(r for r in inv['rows'] if r['case']==name);folder=OLD/'data/physical'/name
        source=OLD/'native-SqDRIFT'/name/'base.json';meta=load(folder/'problem.json')
        if meta['problem_id']!=row['problem_id'] or sha256(source.read_bytes())!=row['source_SHA256']:raise ValueError('Physical source drift')
        short=length(row['original_group_weight_sum'],row['filtered_group_weight_sum'])
        case=dict(case=name,base=str(source),base_SHA256=row['source_SHA256'],short_length=short,
            full_lambda=row['original_group_weight_sum'],filtered_lambda=row['filtered_group_weight_sum'],
            meta=str(folder/'problem.json'),meta_SHA256=sha256((folder/'problem.json').read_bytes()),
            integrals=str(folder/'integrals.npz'),integrals_SHA256=sha256((folder/'integrals.npz').read_bytes()))
        cases.append(case)
        for seed in SEEDS:
            for arm in ARMS:
                members=[]
                for i,(spec,shots) in enumerate(allocations(1000,seed,20 if arm=='FULL_N20' else short)):
                    spec=dict(spec,arm=arm)
                    members.append(dict(index=i,spec=spec,member_id=digest(spec),shots=shots,
                        readout_seed=int(digest(['N57_FINAL',name,seed,i])[:16],16)))
                draw=dict(case=name,arm=arm,seed=seed,problem_id=meta['problem_id'],members=members,model_samples=1000)
                draw['draw_id']=digest(draw);draws.append(draw)
    helpers=['filtered_SqDRIFT_exporter.py','filtered_SqDRIFT_plan.py','filtered_SqDRIFT_source.py',
        'filtered_SqDRIFT_readout.py','strong_bank_prepare.py','strong_projection.py','frozen_projection.py',
        'joint_ci_sparse.py','joint_ci_sparse.c','paired_lowshot.py','mps_batch_worker.py','mps_distribution_check.py']
    external=[OLD/'paperstudy/sqdrift_native_worker.py',OLD/'paperstudy/sqdrift_execution.py',OLD/'paperstudy/lucj.py',OLD/'paperstudy/budget.py']
    p=dict(version='N57_PAIRED_SOURCE_PILOT_NO_ORACLE_INPUTS',cases=cases,draws=draws,seeds=SEEDS,
        code_SHA256={str(ROOT/'frontierloop'/x):sha256((ROOT/'frontierloop'/x).read_bytes()) for x in helpers},
        external_SHA256={str(x):sha256(x.read_bytes()) for x in external},
        qualification_SHA256=sha256((LOCAL/'qualification/result.json').read_bytes()),
        R_caps=[32,128],CI_wall_seconds_per_root=45,CI_total_CPU_seconds=1800,
        source_CPU_seconds=600,source_wall_seconds=1200,new_output_bytes=32*2**20,research_root_bytes=2**30,
        shots_planned=24000,draws_planned=24,members_planned=216,CI_endpoints_planned=48,
        same_readout_seed_pairing_across_arms=True,new_QPU_shots=0,
        library=str(ROOT/'runs/N17-M4-CI-WITNESS-v1/library/libjoint_ci.dylib'),
        library_SHA256=sha256((ROOT/'runs/N17-M4-CI-WITNESS-v1/library/libjoint_ci.dylib').read_bytes()),
        main_packages={k:metadata.version(k) for k in ['qiskit','numpy','scipy','pyscf']},
        old_source_results_not_policy_inputs=True,full_H_used_for_CI=True,
        common_equation='PROJECTED_PHYSICAL_H_PLUS_0p1_S2_NOT_EXACT_S0_NO_OO',
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(LOCAL/'protocol.json',p);print('N57_FROZEN',len(draws),sum(len(x['members']) for x in draws),flush=True)


def verified():
    p=load(LOCAL/'protocol.json')
    if sha256(Path(p['library']).read_bytes())!=p['library_SHA256']:raise ValueError('CI library differs')
    if {k:metadata.version(k) for k in p['main_packages']}!=p['main_packages']:raise ValueError('Main runtime version differs')
    for key in ['code_SHA256','external_SHA256']:
        for file,h in p[key].items():
            if sha256(Path(file).read_bytes())!=h:raise ValueError('Frozen N57 code changed: '+file)
    for c in p['cases']:
        for field in ['base','meta','integrals']:
            if sha256(Path(c[field]).read_bytes())!=c[field+'_SHA256']:raise ValueError('Frozen physical data changed')
    return p


if __name__=='__main__':create()
