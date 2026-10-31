"""N23 readout dispatch. Exact unchanged N17 numerical worker and equations."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .backbone_controls import LOCAL as SAMPLES
from .gaussian_source import checked_counts
from .native_hci_plan import frequency_strings
from .sample_universe_audit import allowed_pool
from .prefix_performance_sampling import capacity

LOCAL=ROOT/'runs/N23-BACKBONE-UNIFORM-CI-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N23-backbone-uniform-ci-v1'


def source_input(draw,definition,terminal_sha):
    path=Path(draw['payload'])
    if draw['status']!='COMPLETED' or draw['actual_samples']!=100000 or sha256(path.read_bytes())!=draw['payload_SHA256']:
        raise ValueError('Actual immutable100k draw required')
    with np.load(path,allow_pickle=False) as a:
        ids,counts=a['blocked_ids'],a['counts']
        if len(ids)!=len(set(map(int,ids))) or any(int(c)<=0 for c in counts) or counts.dtype.kind not in 'iu':raise ValueError('Unique positive integer counts required')
        checked_counts(np.repeat(ids,counts),definition['norb'],tuple(definition['nelec']),100000)
        strings=frequency_strings(ids,counts,definition['norb'],(1<<definition['nelec'][0])-1)
    source=dict(status='AVAILABLE',kind='CLASSICAL_FULL_COUNTS',sample_role=draw['sample_role'],
        expected_problem_id=definition['problem_id'],payload=str(path),payload_SHA256=draw['payload_SHA256'],
        draw_id=draw['draw_id'],replicate=draw['replicate'],source_method=draw['method'],actual_classical_samples=100000,
        actual_QPU_shots=0,sampling_terminal_SHA256=terminal_sha)
    return source,strings,sorted(allowed_pool(source,definition['norb'],tuple(definition['nelec'])))


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N23 CI freeze; do not overwrite')
    capacity();sampling=load(SAMPLES/'protocol.json');actual=load(SAMPLES/'terminal.json')
    if actual['status']!='N23_ALL24_DRAW_STATES_TERMINAL' or len(actual['draws'])!=24 or any(d['status']!='COMPLETED' for d in actual['draws']):raise RuntimeError('All24 actual draw states required; no winner-only queue')
    if actual['protocol_SHA256']!=sha256((SAMPLES/'protocol.json').read_bytes()) or sha256((ROOT/'frontierloop/backbone_controls.py').read_bytes())!=sampling['code_SHA256']:raise RuntimeError('Sampling context changed')
    qualification=ROOT/'qualification-N23-readout-input-v1-tests.xml'
    suites=list(ET.parse(qualification).getroot().iter('testsuite'))
    if sum(int(s.get('tests','0')) for s in suites)!=3 or any(int(s.get('failures','0')) or int(s.get('errors','0')) for s in suites):raise RuntimeError('All readout input tests must pass')
    oldroot=ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-A100-v1';old=load(oldroot/'protocol.json');sources={s['case']:s for s in actual['sources']};plans=[]
    tsha=sha256((SAMPLES/'terminal.json').read_bytes())
    for draw in actual['draws']:
        d=old['physical'][draw['case']];s=sources[draw['case']]
        if s['problem_id']!=d['problem_id'] or s['norb']!=d['norb'] or s['nelec']!=d['nelec']:raise RuntimeError('Wrong physical source')
        source,strings,pool=source_input(draw,d,tsha);hf=(1<<d['nelec'][0])-1
        row=dict(case=draw['case'],method=draw['method']+':'+str(draw['replicate']),source=source,
            selection_mode='SAMPLE_CONSTRAINED',initial_strings=[hf],allowed_strings=pool,frequency_strings=strings,
            HF_string=hf,requested_readouts=['SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'])
        row['task_id']=digest(row);plans.append(row)
    for case,d in sorted(old['physical'].items()):
        hf=(1<<d['nelec'][0])-1
        row=dict(case=case,method='UNRESTRICTED_HF_HCI',source=dict(status='AVAILABLE',kind='CLASSICAL_HF_ONLY',expected_problem_id=d['problem_id']),
            selection_mode='UNRESTRICTED_HF_CONTROL',initial_strings=[hf],allowed_strings=None,frequency_strings=None,HF_string=hf,requested_readouts=['JOINT_TOP_D16384'])
        row['task_id']=digest(row);plans.append(row)
    if len(plans)!=28 or len({p['task_id'] for p in plans})!=28:raise RuntimeError('All24 source tasks plus4strongcontrols required')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in old['pipeline_SHA256'].items():
        path=oldroot/'pipeline'/name
        if sha256(path.read_bytes())!=value:raise RuntimeError('Qualified math changed')
        shutil.copyfile(path,LOCAL/'pipeline'/name)
    worker=(oldroot/'worker.py').read_bytes()
    if sha256(worker)!=old['worker_SHA256']:raise RuntimeError('Qualified worker changed')
    write(LOCAL/'worker.py',worker)
    spec={k:old[k] for k in ['physical','common_runtime','packages','native_binary','native_lock','qualified_external_SHA256','minimal_qualification','pipeline_SHA256','sparse_library']}
    spec.update(version='N23_ALL24_BACKBONE_UNIFORM_DRAWS_PLUS4_HF_V1',remote_root=REMOTE,plans=plans,
        policy=dict(old['policy'],global_wall_seconds=3600),worker_SHA256=sha256(worker),
        preparer_SHA256=sha256(Path(__file__).read_bytes()),sampling_protocol_SHA256=sha256((SAMPLES/'protocol.json').read_bytes()),sampling_terminal_SHA256=tsha,
        qualification_SHA256=sha256(qualification.read_bytes()),planned=28,readout_endpoints=52,
        original_N17_matched_worker_SHA256=old['worker_SHA256'],same_actual_D_not_guaranteed=True,
        source_roles_kept_classical_not_fabricated_QPU_measurements=True,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N23_CI_FROZEN28TASKS52ENDPOINTS_SAME_N17_WORKER',flush=True)


def submit():
    if sha256(Path(__file__).read_bytes())!=load(LOCAL/'protocol.json')['preparer_SHA256']:raise RuntimeError('Frozen dispatcher changed')
    from . import half_source_native_screen as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.submit()


def status():
    from . import nested_ci_study as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.status()


def collect():
    from . import collect_small_pool_bundle as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);args=p.parse_args();globals()[args.action]()
