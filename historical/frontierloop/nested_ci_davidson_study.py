"""N21 same-opportunity numerical successor, isolated from frozen N20 work.

Reuses only dispatch/evidence helpers in a fresh process; all remote paths,
intent records and frozen manifests are new. Numerical operator/queue/caps
stay unchanged; the private eigensolver gets diagonal preconditioning.
"""
import argparse
from pathlib import Path
import shutil
from .paths import ROOT
from .io import load,save,write,sha256,digest
from .prefix_performance_sampling import capacity

LOCAL=ROOT/'runs/N21-NESTED-CI-DAVIDSON-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N21-nested-ci-davidson-v1'
WORKER=ROOT/'frontierloop/nested_ci_worker_davidson.py'
ORIGINAL=ROOT/'runs/N20-NESTED-CI-SCALING-v1'


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N21; preserve and inspect, no overwrite')
    capacity(); old=load(ORIGINAL/'protocol.json')
    if load(ROOT/'analysis/N20-nested-CI-summary.json')['status']!='N20_ALL16_POINTS_AND_ALL_ACTUAL_SAVED_ROOTS_AUDITED':
        raise RuntimeError('Close every N20 planned point before successor')
    # No new R, molecule, threshold, energy-selected source or extra budget.
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    write(LOCAL/'worker.py',WORKER.read_bytes())
    write(LOCAL/'pipeline/nested_ci_kernel.py',(ROOT/'frontierloop/nested_ci_davidson.py').read_bytes())
    for name in ('portable_penalized_ci.py','hole_rdm.py','__init__.py'):
        path=ORIGINAL/'pipeline'/name
        if sha256(path.read_bytes())!=old['input_SHA256']['pipeline/'+name]:raise RuntimeError('Old qualified helper changed')
        shutil.copyfile(path,LOCAL/'pipeline'/name)
    spec=dict(old)
    spec.update(version='N21_SAME16_NESTED_CI_DIAGONAL_DAVIDSON_NUMERICAL_SUCCESSOR_V1',remote_root=REMOTE,
        worker_SHA256=sha256(WORKER.read_bytes()),
        # Existing shared submit helper checks its own unchanged source hash.
        preparer_SHA256=sha256((ROOT/'frontierloop/nested_ci_study.py').read_bytes()),
        study_preparer_SHA256=sha256(Path(__file__).read_bytes()),
        original_N20_protocol_SHA256=sha256((ORIGINAL/'protocol.json').read_bytes()),
        original_N20_point_digest=digest(old['points']),
        eigensolver='PYSCF_DAVIDSON_DIAGONAL_PRECONDITIONED',
        solver_change='SAME_PROJECTED_OPERATOR_SINGLE_ROOT_WARM_START_CAPS_SAVED_ROOTS_DIAGONAL_PRECONDITIONING_ONLY',
        acceptance='UNCHANGED_EXPLICIT_RESIDUAL1e-8_PHYSICAL_ENERGY_WITNESS_R128_PARITY',
        internal_convergence_flag_saved_not_imputed=True,
        no_more_automatic_solver_or_time_cap_rescues_after_this_successor=True,
        tests_SHA256=sha256((ROOT/'qualification-N21-preexecution-v3-tests.xml').read_bytes()),
        input_SHA256={'worker.py':sha256((LOCAL/'worker.py').read_bytes()),**{'pipeline/'+p.name:sha256(p.read_bytes()) for p in (LOCAL/'pipeline').iterdir()}})
    if spec['points']!=old['points'] or spec['tasks']!=old['tasks'] or spec['root_wall_seconds']!=75 or spec['child_hard_wall_seconds']!=110:
        raise RuntimeError('Successor changed experiment opportunity or budget')
    save(LOCAL/'protocol.json',spec)
    print('N21_FROZEN_SAME16_POINTS_CAPS_SOURCE_OPERATOR_NO_NEW_SHOTS',flush=True)


def context():
    from . import nested_ci_study as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.WORKER=WORKER
    return shared


def submit():
    spec=load(LOCAL/'protocol.json')
    if sha256(Path(__file__).read_bytes())!=spec['study_preparer_SHA256']:
        raise RuntimeError('N21 freeze/dispatcher changed')
    context().submit()


def status():context().status()
def collect():context().collect()
def analyze():
    from .analyze_nested_ci import run
    run(local=LOCAL,label='N21')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect','analyze']);a=p.parse_args();globals()[a.action]()
