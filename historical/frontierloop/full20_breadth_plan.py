"""N66: strengthen ordinary FullN20 with the same 30-member budget."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .paths import ROOT
from .io import load,save,sha256,digest,space
from .ensemble_breadth_plan import LOCAL as N65,verified as verified65
from .filtered_SqDRIFT_plan import LOCAL as N57,verified as verified57
LOCAL=ROOT/'runs/N66-FULL20-BREADTH30-v1'

def new_draws(p65,p57):
    rows=[]
    for c in p65['cases']:
        template=next(d for d in p65['draws'] if d['case']==c['case'] and d['arm']=='FULL_N5')
        original=next(d for d in p57['draws'] if d['case']==c['case'] and d['arm']=='FULL_N20' and d['seed']==570101)
        members=[]
        for m in template['members']:
            spec=dict(m['spec'],arm='FULL_N20',num_terms=20)
            members.append(dict(m,spec=spec,member_id=digest(spec)))
        row=dict(case=c['case'],arm='FULL_N20',seed=570101,problem_id=original['problem_id'],members=members,model_samples=1000,ensemble_size=30,
            previous_9_member_draw_id=original['draw_id'])
        row['draw_id']=digest(row);rows.append(row)
    return rows

def create():
    if LOCAL.exists():raise RuntimeError('Existing N66; no replace')
    old=verified57();p65=verified65();space()
    test=ROOT/'qualification-N66-policy-v1.xml';suites=list(ET.parse(test).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=5 or any(int(s.get(k,0)) for s in suites for k in ['failures','errors']):raise ValueError('Five qualification checks required')
    files=['full20_breadth_plan.py','full20_breadth_source.py','full20_breadth_readout.py','filtered_SqDRIFT_source.py',
        'filtered_SqDRIFT_exporter.py','targeted_frame_batch_worker.py','clifford_targeted_checker.py','stabilizer_target_amplitudes.py',
        'strong_bank_prepare.py','strong_projection.py','contiguous_CI_audit.py']
    used=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    if used+6*2**20>2**30:raise OSError('No capacity for frozen N66; no deletion')
    p=dict(version='N66_FULL_N20_WIDTH_MATCHED_CONTROL_NOT_NEW_METHOD',cases=p65['cases'],draws=new_draws(p65,old),R_caps=[32,128],
        code_SHA256={str(ROOT/'frontierloop'/f):sha256((ROOT/'frontierloop'/f).read_bytes()) for f in files},
        old_protocol_SHA256={str(f):sha256(f.read_bytes()) for f in [N65/'protocol.json',N57/'protocol.json']},
        plan_SHA256=sha256((ROOT/'analysis/N65_DECISION_AND_N66_NEXT_JA.md').read_bytes()),qualification_SHA256=sha256(test.read_bytes()),
        source_CPU_seconds=900,source_wall_seconds=1200,CI_total_CPU_seconds=60,CI_wall_seconds_per_root=15,new_output_bytes=6*2**20,
        research_root_bytes=2**30,initial_research_logical_bytes=used,shots_planned=4000,draws_planned=4,members_planned=120,
        library=p65['library'],library_SHA256=p65['library_SHA256'],common_equation=p65['common_equation'],
        old_draws_not_repeated=True,source_choices_use_no_energies_or_references=True,
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(LOCAL/'protocol.json',p);print('N66_FROZEN4_4000',flush=True)

def verified():
    old=verified57();p65=verified65();p=load(LOCAL/'protocol.json');assert p['draws']==new_draws(p65,old)
    for f,h in p['code_SHA256'].items():assert sha256(Path(f).read_bytes())==h,'Frozen N66 changed '+f
    for f,h in p['old_protocol_SHA256'].items():assert sha256(Path(f).read_bytes())==h
    assert sha256(Path(p['library']).read_bytes())==p['library_SHA256']
    return p

if __name__=='__main__':create()
