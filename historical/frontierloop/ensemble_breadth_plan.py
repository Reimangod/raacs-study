"""N65 same-shot ensemble-breadth screen; no energies enter source choices."""
from pathlib import Path
import xml.etree.ElementTree as ET
from .paths import ROOT
from .io import load,save,sha256,digest,space
from .full_N5_plan import LOCAL as N63,verified as verified63
from .filtered_SqDRIFT_plan import LOCAL as N57,verified as verified57
LOCAL=ROOT/'runs/N65-ENSEMBLE-BREADTH30-v1'

def draws_for(parent63,parent57):
    out=[]
    for c in parent63['cases']:
        for arm,pp in [('FULL_N5',parent63),('FILTERED_SHORT',parent57)]:
            original=next(d for d in pp['draws'] if d['case']==c['case'] and d['arm']==arm and d['seed']==570101)
            members=[]
            for k in [1,2,3]:
                shots=1000//3+int(k==1)
                template=next(m['spec'] for m in original['members'] if m['spec']['k']==k)
                for j in range(10):
                    spec=dict(template,seed=int(digest(['paper-common-SqDRIFT-v1',570101,1000,k,j])[:8],16),member_index=j)
                    members.append(dict(index=len(members),spec=spec,member_id=digest(spec),shots=shots//10+int(j<shots%10),
                        readout_seed=int(digest(['N65_FINAL',c['case'],570101,k,j])[:16],16)))
            row=dict(case=c['case'],problem_id=original['problem_id'],arm=arm,seed=570101,members=members,model_samples=1000,
                previous_9_member_draw_id=original['draw_id'],ensemble_size=30)
            row['draw_id']=digest(row);out.append(row)
    return out

def create():
    if LOCAL.exists():raise RuntimeError('Existing N65; no replace')
    old=verified57();n5=verified63();space()
    tests=ROOT/'qualification-N65-policy-v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=5 or any(int(s.get(k,0)) for s in suites for k in ['failures','errors']):raise ValueError('Five policy checks required')
    files=['ensemble_breadth_plan.py','ensemble_breadth_source.py','ensemble_breadth_readout.py',
        'filtered_SqDRIFT_source.py','filtered_SqDRIFT_exporter.py','targeted_frame_batch_worker.py',
        'clifford_targeted_checker.py','stabilizer_target_amplitudes.py','strong_bank_prepare.py','strong_projection.py','contiguous_CI_audit.py']
    p=dict(version='N65_BOTH_ARMS_9_TO30_MEMBERS_FIXED1000_SHOTS',cases=n5['cases'],draws=draws_for(n5,old),R_caps=[32,128],
        code_SHA256={str(ROOT/'frontierloop'/f):sha256((ROOT/'frontierloop'/f).read_bytes()) for f in files},
        old_protocol_SHA256={str(f):sha256(f.read_bytes()) for f in [N63/'protocol.json',N57/'protocol.json']},
        plan_SHA256=sha256((ROOT/'analysis/N65_ENSEMBLE_BREADTH_NEXT_JA.md').read_bytes()),qualification_SHA256=sha256(tests.read_bytes()),
        source_CPU_seconds=900,source_wall_seconds=1200,CI_total_CPU_seconds=60,CI_wall_seconds_per_root=15,
        new_output_bytes=8*2**20,research_root_bytes=2**30,shots_planned=8000,draws_planned=8,members_planned=240,
        library=n5['library'],library_SHA256=n5['library_SHA256'],common_equation=n5['common_equation'],
        old_draws_not_repeated=True,source_choice_uses_no_energies_or_references=True,
        independent_confirmation=False,full_paper_reproduction=False,goal_complete=False)
    save(LOCAL/'protocol.json',p);print('N65_FROZEN8_8000',flush=True)

def verified():
    old=verified57();n5=verified63();p=load(LOCAL/'protocol.json')
    assert p['draws']==draws_for(n5,old)
    for f,h in p['code_SHA256'].items():
        if sha256(Path(f).read_bytes())!=h:raise ValueError('Frozen N65 drift '+f)
    for f,h in p['old_protocol_SHA256'].items():assert sha256(Path(f).read_bytes())==h
    assert sha256(Path(p['library']).read_bytes())==p['library_SHA256']
    return p

if __name__=='__main__':create()
