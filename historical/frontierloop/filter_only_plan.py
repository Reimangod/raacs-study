"""N62: one new source arm, fixed before additional outcomes."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .paths import ROOT
from .filtered_SqDRIFT_plan import verified as old_verified,LOCAL as OLD_LOCAL
from .io import load,save,sha256,digest,space

LOCAL=ROOT/'runs/N62-FILTER-ONLY-N20-v1'


def new_draws(old):
    draws=[]
    for d in old['draws']:
        if d['arm']!='FULL_N20':continue
        row={k:d[k] for k in ['case','seed','problem_id','model_samples']};row['arm']='FILTERED_N20';members=[]
        for m in d['members']:
            spec=dict(m['spec'],arm='FILTERED_N20')
            members.append(dict(m,spec=spec,member_id=digest(spec)))
        row['members']=members;row['draw_id']=digest(row);draws.append(row)
    return draws


def create():
    if (LOCAL/'protocol.json').exists():raise RuntimeError('Existing freeze; do not replace')
    space();old=old_verified();draws=new_draws(old)
    suites=list(ET.parse(ROOT/'qualification-N62-policy-v1.xml').getroot().iter('testsuite'))
    if sum(int(x.get('tests',0)) for x in suites)!=3 or any(int(x.get(k,0)) for x in suites for k in ['failures','errors']):raise ValueError('Pre-outcome policy tests required')
    qualification=load(OLD_LOCAL/'qualification/Fe-Clifford-frame-v3/terminal.json')
    if not qualification['passed']:raise ValueError('Gate-level qualifier missing')
    files=['filter_only_plan.py','filter_only_source.py','filter_only_readout.py',
        'frame_batch_worker.py','clifford_frame_checker_v3.py','frame_source_successor.py','contiguous_CI_audit.py']
    code={str(ROOT/'frontierloop'/name):sha256((ROOT/'frontierloop'/name).read_bytes()) for name in files}
    p=dict(version='N62_FILTER_ONLY_N20_SOURCE_ABLATION',cases=old['cases'],draws=draws,seeds=old['seeds'],
        original_protocol_SHA256=sha256((OLD_LOCAL/'protocol.json').read_bytes()),
        source_policy='OFFICIAL_DIAGONAL_FILTER_N20_ALL4_SAME_TIMES_SEEDS_SHOTS_NO_NEW_HEURISTIC',
        exporter_dispatch_arm='FILTERED_SHORT',source_spec_arm='FILTERED_N20',
        exporter_dispatch_reason='Existing server explicitly has only FULL_N20/FILTERED_SHORT keys; logical arm spec is independent of operator dispatch key',
        code_SHA256=code,R_caps=[32,128],common_equation=old['common_equation'],
        library=old['library'],library_SHA256=old['library_SHA256'],
        source_CPU_seconds=600,source_wall_seconds=1200,CI_total_CPU_seconds=120,CI_wall_seconds_per_root=15,
        new_output_bytes=12*2**20,research_root_bytes=2**30,shots_planned=12000,draws_planned=12,CI_endpoints_planned=24,
        original24_not_repeated=True,old_result_energies_not_policy_inputs=True,
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(LOCAL/'protocol.json',p);print('N62_FROZEN',len(draws),flush=True)


def verified():
    old=old_verified();p=load(LOCAL/'protocol.json')
    if p['draws']!=new_draws(old):raise ValueError('New arm identity drift')
    if p['original_protocol_SHA256']!=sha256((OLD_LOCAL/'protocol.json').read_bytes()):raise ValueError('Old protocol drift')
    for file,h in p['code_SHA256'].items():
        if sha256(Path(file).read_bytes())!=h:raise ValueError('Frozen N62 code changed '+file)
    return p


if __name__=='__main__':create()
