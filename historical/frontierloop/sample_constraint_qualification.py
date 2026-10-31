"""N12: qualify sample-constrained candidate generation, not molecular outcomes."""
import argparse,shlex,shutil,subprocess
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,space
from .native_dice_toys import definitions,exact_matrices
from .paired_dice_qualification import new_definitions
from .sample_constraint_patch import ORIGINAL_SHA256
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N12-SAMPLE-CONSTRAINT-QUALIFICATION-v1'
REMOTE='/share/reiwatanabe/ceo-sample-constraint-qualification-20261026-v1'
PATCH=ROOT/'frontierloop/sample_constraint_patch.py'


def change(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError('Frozen qualification template drift')
    return text.replace(before,after)


def generated_worker():
    text=(ROOT/'frontierloop/paired_dice_qualification_worker.py').read_text()
    text=change(text,'SHCI/SHCI.cpp','SHCI/SHCIbasics.cpp',5)
    text=change(text,'paired_dice_patch','sample_constraint_patch',2)
    text=change(text,'CEO_DICE_PAIRED_INPUT_V1','CEO_DICE_SAMPLE_POOL_V1',2)
    text=change(text,"['CUSTOM_PAIRED','INVALID_PAIRED']","['CUSTOM_CONSTRAINED','INVALID_CONSTRAINED']")
    text=change(text,"fromfile='original/SHCI.cpp',tofile='paired-adapter/SHCI.cpp'",
                     "fromfile='original/SHCIbasics.cpp',tofile='sample-pool-adapter/SHCIbasics.cpp'")
    text=change(text,"        save(root/'build.json',b)",
        "        b.update(original_dispatcher_SHA256=sha(original/'SHCI/SHCI.cpp'),private_dispatcher_SHA256=sha(source/'SHCI/SHCI.cpp'))\n"
        "        save(root/'build.json',b)")
    return text


def generated_bootstrap():
    text=(ROOT/'frontierloop/paired_dice_bootstrap.py').read_text()
    return change(text,"    prior=Path(spec['prior_failed_root'])\n    for name,value in spec['prior_failed_evidence_SHA256'].items():\n        if sha(prior/name)!=value:raise RuntimeError('Prior failed attempt evidence changed')\n",'')


def constrained_definitions():
    base=definitions()[0];three=new_definitions()[2];hole=definitions()[3]
    common=dict(nelec=[1,1],core_energy=.37,initial_strings=[1],expansion=True,epsilon='1e-12')
    return [dict(common,name='SAMPLE_HF_ONLY',h=base['h'],g=base['g'],allowed=[1],expected_pairs=[(1,1)]),
        dict(common,name='SAMPLE_ALL_TWO',h=base['h'],g=base['g'],allowed=[1,2],expected_pairs=[(a,b) for a in [1,2] for b in [1,2]]),
        dict(common,name='SAMPLE_SUBSET_THREE',h=three['h'],g=three['g'],core_energy=.19,allowed=[1,4],expected_pairs=[(a,b) for a in [1,4] for b in [1,4]]),
        dict(common,name='SAMPLE_HOLES',h=hole['h'],g=hole['g'],nelec=[3,3],initial_strings=[7],allowed=[7,13],expected_pairs=[(a,b) for a in [7,13] for b in [7,13]]),
        dict(common,name='SAMPLE_THRESHOLD',h=base['h'],g=base['g'],allowed=[1,2],epsilon='0.2',expected_pairs=[(1,1)])]


def prepare():
    from pyscf import tools
    if LOCAL.exists():raise RuntimeError('Existing N12; do not overwrite')
    space();p6=load(ROOT/'runs/N6-NATIVE-SOURCE-CONTROLS-v1/protocol.json')
    if load(ROOT/'analysis/N11-sample-universe-membership.json')['numeric_native_exports_with_outside_configurations']!=106:
        raise RuntimeError('Audit all prior sample universe memberships first')
    LOCAL.mkdir();(LOCAL/'inputs').mkdir();fixtures=[]
    for constrained,defs in [(False,definitions()),(True,constrained_definitions())]:
        for d in defs:
            n=len(d['h']);ss,H,S=exact_matrices(d['h'],d['g'],d['nelec'],d['core_energy'])
            variants=['CUSTOM_CONSTRAINED'] if constrained else ['ORIGINAL_DEFAULT','CUSTOM_DEFAULT']
            for variant in variants:
                name=d['name']+'__'+variant;folder=LOCAL/'inputs'/name;folder.mkdir()
                tools.fcidump.from_integrals(folder/'fcidump.txt',d['h'],d['g'],n,tuple(d['nelec']),nuc=d['core_energy'])
                for spin in ['Alpha','Beta']:write(folder/(spin+'Dets.bin'),b''.join(int(x).to_bytes(16,'big') for x in d['initial_strings']))
                expansion=d.get('expansion',False)
                lines=['nroots 1']+([] if d.get('spin_keyword',0) is None else ['spin '+str(d.get('spin_keyword',0))])
                lines+=['orbitals fcidump.txt','schedule','0 '+d.get('epsilon','1e-12' if expansion else '2147483647'),'end',
                    'davidsonTol 1e-10','dE 1e-10','maxiter '+('4' if constrained else ('2' if expansion else '1')),
                    'noio','writeBestDeterminants 2147483647','nPTiter 0','nocc '+str(sum(d['nelec'])),
                    ' '.join(map(str,range(sum(d['nelec'])))),'end']
                write(folder/'input.dat',('\n'.join(lines)+'\n').encode())
                initial=[(a,b) for a in d['initial_strings'] for b in d['initial_strings']]
                expected=d['expected_pairs'] if constrained else ([(a,b) for a in ss for b in ss] if expansion else initial)
                idx=[ss.index(a)*len(ss)+ss.index(b) for a,b in expected]
                if constrained:write(folder/'AllowedHalfDets.bin',b''.join(int(x).to_bytes(16,'big') for x in d['allowed']))
                fixtures.append(dict(name=name,base_name=d['name'],variant=variant,norb=n,nelec=d['nelec'],all_strings=ss,
                    h=d['h'].tolist(),eri=d['g'].tolist(),core_energy=d['core_energy'],H_full=H.tolist(),S2_full=S.tolist(),
                    initial_pairs=initial,allowed_strings=d.get('allowed'),expected_final_pairs=expected,
                    expected_energy_Ha=float(np.linalg.eigvalsh(H[np.ix_(idx,idx)])[0]),expected_returncode=0,
                    input_SHA256={p.name:sha256(p.read_bytes()) for p in folder.iterdir()}))
    parent=next(f for f in fixtures if f['base_name']=='SAMPLE_ALL_TWO')
    for mode in ['EMPTY','DUPLICATE','WRONG_SECTOR','NO_HF','INVALID_MODE']:
        name='INVALID_'+mode;folder=LOCAL/'inputs'/name;shutil.copytree(LOCAL/'inputs'/parent['name'],folder)
        values={'EMPTY':[],'DUPLICATE':[1,1],'WRONG_SECTOR':[1,3],'NO_HF':[2],'INVALID_MODE':[1,2]}[mode]
        write(folder/'AllowedHalfDets.bin',b''.join(int(x).to_bytes(16,'big') for x in values))
        fixtures.append(dict(name=name,variant='INVALID_CONSTRAINED',env_value='true' if mode=='INVALID_MODE' else '1',
            expected_returncode=2,input_SHA256={p.name:sha256(p.read_bytes()) for p in folder.iterdir()}))
    write(LOCAL/'worker.py',generated_worker().encode());write(LOCAL/'bootstrap.py',generated_bootstrap().encode())
    save(LOCAL/'fixtures.json',dict(fixtures=fixtures,synthetic_only=True,new_molecular_evaluations=0))
    spec=dict(version='N12_SAMPLE_CONSTRAINED_HCI_TINY_QUALIFICATION_V1',remote_root=REMOTE,
        worker_SHA256=sha256((LOCAL/'worker.py').read_bytes()),bootstrap_SHA256=sha256((LOCAL/'bootstrap.py').read_bytes()),
        patch_SHA256=sha256(PATCH.read_bytes()),fixtures_SHA256=sha256((LOCAL/'fixtures.json').read_bytes()),
        original_source='/share/reiwatanabe/ceo-native-dice-20261025-v1/runtime/Dice',
        original_main_SHA256=ORIGINAL_SHA256,original_binary=p6['native_binary'],original_binary_SHA256=p6['qualified_external_SHA256'][p6['native_binary']],
        original_dispatcher_SHA256='b11f00a73737624fa14a8e16efbc62790988d6eb362833b0b67620a3c15909f5',
        original_commit='3198db1f8e0066489d173d30a52bc21967969a0f',
        boost_root='/share/reiwatanabe/ceo-native-dice-20261025-v1/runtime/boost_1_85_0',native_lock=p6['native_lock'],
        build_wall_cap_seconds=600,fixture_wall_cap_seconds=20,local_runtime_cap_bytes=128*1024**2,disk_floor_bytes=3*1024**3,
        tests_SHA256=sha256((ROOT/'qualification-sample-constraint-v1-tests.xml').read_bytes()),
        scope='FRESH_SERIAL_SAMPLE_UNIVERSE_COMPONENT_ONLY_NOT_FULL_HCI_HSQD_HARDWARE_OR_ORIGINAL_PAPER_REPRODUCTION',
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,production_molecular_execution_authorized=False,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N12_FROZEN24_TINY_19PHYSICS5NEGATIVE',flush=True)


def submit():
    if (LOCAL/'submission-started.json').exists():raise RuntimeError('Existing N12 intent; reconcile exact job')
    p=load(LOCAL/'protocol.json')
    for file,key in [(LOCAL/'worker.py','worker_SHA256'),(LOCAL/'bootstrap.py','bootstrap_SHA256'),(PATCH,'patch_SHA256')]:
        if sha256(file.read_bytes())!=p[key]:raise RuntimeError('Frozen N12 code changed')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote root exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    for file,name in [(LOCAL/'worker.py','base_worker.py'),(LOCAL/'bootstrap.py','bootstrap.py'),(PATCH,'sample_constraint_patch.py'),
        (LOCAL/'protocol.json','protocol.json'),(LOCAL/'fixtures.json','fixtures.json')]:send(file,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'inputs'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/bootstrap.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=4G','--time=00:15:00',
        '--job-name=ceo-sample-constraint-tiny-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    job=ssh(cmd,timeout=35).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unknown submission; no blind retry')
    save(LOCAL/'job.json',dict(job_id=job,remote=REMOTE,GPU_allocations=0));print('N12_SUBMITTED_CPU_ONLY',job,flush=True)


def status():
    from . import paired_dice_qualification as old
    old.LOCAL=LOCAL;old.REMOTE=REMOTE;old.status(False)


def collect():
    from . import collect_paired_qualification as old
    old.LOCAL=LOCAL;old.REMOTE=REMOTE;old.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args();globals()[a.action]()
