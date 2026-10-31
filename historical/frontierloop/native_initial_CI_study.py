"""N55 one finite initialization correction; unchanged solver/action/space caps."""
import copy,sys
import xml.etree.ElementTree as ET
import numpy as np
from . import hybrid_CI_gpu_study as base
from .native_CI_initial_guess import initial_guess
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space

OLD=base.LOCAL
LOCAL=ROOT/'runs/N55-NATIVE-GUIDED-HYBRID-CI-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N55-native-guided-hybrid-CI-v1'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N55; reconcile not recreate')
    space();old=base.verified();audit=load(ROOT/'analysis/N54-independent-root-audit.json')
    assert len(audit['coverage'])==2 and not audit['rows']
    assert all(c['result']['physical_H_calls']==299 for c in audit['coverage'])
    tests=ROOT/'qualification-N55-native-initial-v2.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    assert sum(int(x.get('tests','0')) for x in suites)==9
    assert not any(int(x.get('failures','0')) or int(x.get('errors','0')) for x in suites)
    p=copy.deepcopy(old);p['qualifications']=[copy.deepcopy(old['qualifications'][-1])];hashes={};diagnosis=[]
    for c in p['qualifications']+list(p['production'].values()):
        with np.load(OLD/c['file'],allow_pickle=False) as a:data={k:a[k] for k in a.files}
        initial=initial_guess(data);j=int(np.argmin(data['exact_diagonal']))
        diagnosis.append(dict(case=c['id'],minimum_diagonal_Ha=float(data['exact_diagonal'][j]+data['core_shift']),
            old_initial_weight_at_minimum=float(data['initial_vector'][j]**2),
            new_initial_weight_at_minimum=float(initial[j]**2/np.vdot(initial,initial)),
            lowest_pair_same_spin_string=j//c['R']==j%c['R']))
        data['initial_vector']=initial
        blob=arrays(**data);write(LOCAL/c['file'],blob);hashes[c['file']]=sha256(blob)
        c.update(SHA256=sha256(blob),initial_vector_SHA256=sha256(initial.astype('<f8').tobytes()),
            initial_rule='PUBLIC_SELECTEDCI_LOW_DIAGONAL_GUESS_PLUS_0p001_NORMALIZED_REGISTERED_N54_RANDOM_PERTURBATION')
    for file in ['worker.py','hybrid_CI_solver.py','structured_CI_device.py']:
        blob=(OLD/file).read_bytes();write(LOCAL/file,blob);hashes[file]=sha256(blob)
    p.update(version='N55_FINITE_NATIVE_GUIDED_INITIALIZATION_UNCHANGED_FULL_SPACE',input_SHA256=hashes,
        predecessor_N54_protocol_SHA256=sha256((OLD/'protocol.json').read_bytes()),
        predecessor_N54_audit_SHA256=sha256((ROOT/'analysis/N54-independent-root-audit.json').read_bytes()),
        predecessor_N54_measured_worker_CPU_seconds=audit['worker_CPU_seconds'],
        predecessor_N54_accounting=audit['accounting'],initialization_diagnosis=diagnosis,
        same_N51_Davidson_settings_and_initialization=False,same_Davidson_settings_except_initialization=True,
        scientific_space_penalty_caps_tolerances_unchanged=True,full_support_perturbation_required=True,
        cap_extension_authorized=False,automatic_repeat_or_parameter_search_authorized=False,
        missing_energy_if_this_finite_attempt_fails=True,
        qualification_tests_SHA256=sha256(tests.read_bytes()))
    save(LOCAL/'protocol.json',p)
    save(ROOT/'diagnostics/N55-initialization-and-sector-diagnosis-v1.json',dict(
        status='HYPOTHESIS_NOT_YET_PROVEN_INITIALIZATION_CORRECTION',rows=diagnosis,
        prior_CPU_source_roots=0,prior_worker_CPU_seconds=audit['worker_CPU_seconds'],
        actual_failure_cause='299_ACTION_CAP_NOT_GPU_WALL_OR_MEMORY_LIMIT',
        important_counterexample='Pure native symmetric initial vector can miss an antisymmetric lower root despite tiny residual. Keep 0.001 full-support perturbation.',
        initial_failed_test_evidence='qualification-N55-native-initial-v1.xml',
        corrected_tests_evidence='qualification-N55-native-initial-v2.xml',
        scientific_source_energies_still_unknown=True))
    print('N55_NATIVE_GUIDED_TWO_SPACES_FROZEN_NO_CAP_EXTENSION',flush=True)


def main():
    if sys.argv[1]=='prepare':return prepare()
    base.LOCAL=LOCAL;base.REMOTE=REMOTE;base.JOB_TAG='N55'
    if sys.argv[1]=='audit':
        from . import hybrid_CI_audit as auditor
        auditor.LOCAL=LOCAL;auditor.REPORT_TAG='N55'
        return auditor.run()
    return {'submit':base.submit,'status':base.status,'progress':base.progress,'collect':base.collect}[sys.argv[1]]()


if __name__=='__main__':main()
