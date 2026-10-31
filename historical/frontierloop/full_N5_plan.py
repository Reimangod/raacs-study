"""N63: unfiltered SqDRIFT5 at the same finite model-sampling budget."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .filtered_SqDRIFT_plan import LOCAL as OLD_LOCAL,verified as old_verified
from .filter_only_plan import verified as N62_verified
from .paths import ROOT
from .io import load,save,digest,sha256,space

LOCAL=ROOT/'runs/N63-FULL-N5-COMMON-BUDGET-v1'


def new_draws(old):
    draws=[]
    for d in old['draws']:
        if d['arm']!='FULL_N20':continue
        row={k:d[k] for k in ['case','seed','problem_id','model_samples']};row['arm']='FULL_N5';members=[]
        for m in d['members']:
            spec=dict(m['spec'],arm='FULL_N5',num_terms=5)
            members.append(dict(m,spec=spec,member_id=digest(spec)))
        row['members']=members;row['draw_id']=digest(row);draws.append(row)
    return draws


def create():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing N63 freeze; do not replace')
    old=old_verified();N62_verified();space()
    suites=list(ET.parse(ROOT/'qualification-N63-policy-v1.xml').getroot().iter('testsuite'))
    if sum(int(x.get('tests',0)) for x in suites)!=4 or any(int(x.get(k,0)) for x in suites for k in ['failures','errors']):raise ValueError('Policy and zero-qubit checker qualification required')
    files=['full_N5_plan.py','full_N5_source.py','full_N5_readout.py','filtered_SqDRIFT_source.py',
        'clifford_targeted_checker.py','stabilizer_target_amplitudes.py','targeted_frame_batch_worker.py','contiguous_CI_audit.py']
    p=dict(version='N63_UNFILTERED_N5_SAME_BUDGET_CONTROL',cases=old['cases'],draws=new_draws(old),seeds=old['seeds'],
        code_SHA256={str(ROOT/'frontierloop'/f):sha256((ROOT/'frontierloop'/f).read_bytes()) for f in files},
        original_protocol_SHA256=sha256((OLD_LOCAL/'protocol.json').read_bytes()),
        exporter_dispatch_arm='FULL_N20',source_spec_arm='FULL_N5',diagonal_terms_retained=True,
        source_policy='ORIGINAL_SDK_UNFILTERED_OPERATOR_N5_NO_NEW_HEURISTIC_COMMON_BUDGET_NOT_FULL_PAPER_REPRODUCTION',
        source_CPU_seconds=300,source_wall_seconds=600,CI_total_CPU_seconds=60,CI_wall_seconds_per_root=15,
        new_output_bytes=6*2**20,research_root_bytes=2**30,R_caps=[32,128],
        shots_planned=12000,draws_planned=12,CI_main_endpoints_planned=24,CI_common_endpoints_planned=72,
        library=old['library'],library_SHA256=old['library_SHA256'],common_equation=old['common_equation'],
        original36_draws_not_repeated=True,energies_and_references_not_source_policy_inputs=True,
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(LOCAL/'protocol.json',p);print('N63_FROZEN',len(p['draws']),flush=True)


def verified():
    old=old_verified();N62_verified();p=load(LOCAL/'protocol.json')
    if p['draws']!=new_draws(old):raise ValueError('N5 draw identity drift')
    for f,h in p['code_SHA256'].items():
        if sha256(Path(f).read_bytes())!=h:raise ValueError('Frozen N63 implementation changed '+f)
    return p


if __name__=='__main__':create()
