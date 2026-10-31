"""Retry only the two unqualified cases, preserving the six successful cases."""
import copy,sys
import numpy as np
from . import structured_CI_gpu_study as base
from .structured_CI_device import dense_probes
from .paths import ROOT
from .io import load,save,write,arrays,sha256,space

OLD_LOCAL=base.LOCAL
LOCAL=ROOT/'runs/N53-STRUCTURED-CI-CUDA-QUALIFICATION-v2'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N53-structured-CI-CUDA-qualification-v2'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing successor; reconcile, never recreate')
    space();old=base.verified();failure=load(OLD_LOCAL/'collected/out/failure.json')
    assert failure['error']=="ValueError('CPU generated profiling-vector bytes differ')"
    assert len(failure['rows'])==6
    assert failure['protocol_SHA256']==sha256((OLD_LOCAL/'protocol.json').read_bytes())
    for c,r in zip(old['cases'][:6],failure['rows']):
        assert r['case_id']==c['id'] and r['status']=='QUALIFIED_ACTION_ONLY'
        assert max(r['errors'].values())<=old['action_tolerance']
    p=copy.deepcopy(old);p['cases']=copy.deepcopy(old['cases'][6:]);hashes={}
    for c in p['cases']:
        with np.load(OLD_LOCAL/c['file'],allow_pickle=False) as a:data={k:a[k] for k in a.files}
        x=dense_probes(c['D'],c['dense_seed'])
        assert sha256(x.astype('<f8').tobytes())==c['dense_input_SHA256']
        data['explicit_dense_probes']=x
        blob=arrays(**data);write(LOCAL/c['file'],blob)
        c['original_v1_packet_SHA256']=c['SHA256'];c['SHA256']=sha256(blob)
        hashes[c['file']]=c['SHA256']
    for source,target in [(OLD_LOCAL/'worker.py','worker_base.py'),
            (OLD_LOCAL/'structured_CI_device.py','structured_CI_device.py'),
            (ROOT/'frontierloop/structured_CI_gpu_explicit_worker.py','worker.py')]:
        blob=source.read_bytes();write(LOCAL/target,blob);hashes[target]=sha256(blob)
    p.update(version='N53_EXACT_BYTES_SUCCESSOR_TWO_UNQUALIFIED_CASES_ONLY',input_SHA256=hashes,
        predecessor_failure_SHA256=sha256((OLD_LOCAL/'collected/out/failure.json').read_bytes()),
        predecessor_successful_case_ids=[r['case_id'] for r in failure['rows']],
        identical_CPU_oracle_values_and_dense_probe_bytes=True,
        change='INPUT_TRANSPORT_ONLY_NO_TOLERANCE_OR_SCIENTIFIC_EQUATION_CHANGE',
        successor_preparation_new_H_actions=0,successor_preparation_new_S2_actions=0)
    save(LOCAL/'protocol.json',p)
    save(ROOT/'diagnostics/N53-cross-platform-probe-input-incident-v1.json',dict(
        status='PRESERVED_PARTIAL_SUCCESS_WITH_EXPLICIT_INPUT_SUCCESSOR',job_id='7060',
        successful_cases=6,remaining_cases=2,failed_case=old['cases'][6]['id'],
        failure=failure['error'],cause_scope='Same RNG seed/version did not reproduce byte-identical floating input across Mac and Linux; precise platform arithmetic source not isolated',
        no_mathematical_parity_failure_observed=True,no_new_CI_roots=True,
        remedy='Send original exact float64 probes; retain all oracle values and tolerances; rerun only remaining two cases',
        failure_SHA256=p['predecessor_failure_SHA256']))
    print('N53_V2_TWO_EXPLICIT_INPUT_PACKETS_FROZEN')


def main():
    if sys.argv[1]=='prepare':return prepare()
    base.LOCAL=LOCAL;base.REMOTE=REMOTE
    return {'submit':base.submit,'status':base.status,'collect':base.collect}[sys.argv[1]]()


if __name__=='__main__':main()
