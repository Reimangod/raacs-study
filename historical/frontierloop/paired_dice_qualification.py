"""N9: private opt-in paired input, qualification only (no molecular jobs)."""
import argparse,shlex,subprocess
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,space
from .native_dice_toys import definitions,exact_matrices
from .paired_dice_patch import ORIGINAL_SHA256
from benchmark.gpu_remote import ssh,send,OPTIONS,HOST,REMOTE as BASE

LOCAL=ROOT/'runs/N9-PAIRED-INPUT-QUALIFICATION-v1'
REMOTE='/share/reiwatanabe/ceo-paired-input-qualification-20261026-v1'
WORKER=ROOT/'frontierloop/paired_dice_qualification_worker.py'
PATCH=ROOT/'frontierloop/paired_dice_patch.py'
SOURCE='/share/reiwatanabe/ceo-native-dice-20261025-v1/runtime/Dice'


def new_definitions():
    base=definitions()[0];h,g=base['h'],base['g']
    rng=np.random.default_rng(20261026);v=rng.normal(size=(3,3))*.07
    h3=(v+v.T)/2+np.diag([-1.,-.6,-.1]);f=rng.normal(size=(2,3,3))*.13
    f=(f+f.transpose(0,2,1))/2;g3=np.einsum('lij,lkm->ijkm',f,f)
    hole=definitions()[3]
    return [
        dict(name='PAIRED_DIAGONAL',h=h,g=g,nelec=[1,1],core_energy=.37,pairs=[(1,1),(2,2)]),
        dict(name='PAIRED_REVERSED_LOWEST_SECOND',h=h,g=g,nelec=[1,1],core_energy=.37,pairs=[(2,2),(1,1)]),
        dict(name='PAIRED_SAME_MARGINAL_A',h=h3,g=g3,nelec=[1,1],core_energy=.19,pairs=[(1,1),(2,2),(4,4)]),
        dict(name='PAIRED_SAME_MARGINAL_B',h=h3,g=g3,nelec=[1,1],core_energy=.19,pairs=[(1,1),(2,4),(4,2)]),
        dict(name='PAIRED_HOLES',h=hole['h'],g=hole['g'],nelec=[3,3],core_energy=.37,
             pairs=[(7,7),(7,11),(11,7),(11,11),(13,13),(14,14)]),
        dict(name='PAIRED_NATIVE_EXPANSION',h=h,g=g,nelec=[1,1],core_energy=.37,pairs=[(1,1)],expansion=True)]


def prepare():
    from pyscf import tools
    if LOCAL.exists():raise RuntimeError('Existing N9; no overwrite/retry')
    space();p6=load(ROOT/'runs/N6-NATIVE-SOURCE-CONTROLS-v1/protocol.json')
    if load(ROOT/'analysis/N8-remaining-source-controls-summary.json')['completed_new']!=26:raise RuntimeError('Finish all source controls first')
    LOCAL.mkdir();inp=LOCAL/'inputs';inp.mkdir();fixtures=[]
    for paired,defs in [(False,definitions()),(True,new_definitions())]:
        for d in defs:
            n=len(d['h']);strings,H,S=exact_matrices(d['h'],d['g'],d['nelec'],d['core_energy'])
            pairs=d.get('pairs') if paired else [(a,b) for a in d['initial_strings'] for b in d['initial_strings']]
            variants=['CUSTOM_PAIRED'] if paired else ['ORIGINAL_DEFAULT','CUSTOM_DEFAULT']
            for variant in variants:
                name=d['name']+'__'+variant;folder=inp/name;folder.mkdir()
                tools.fcidump.from_integrals(folder/'fcidump.txt',d['h'],d['g'],n,tuple(d['nelec']),nuc=d['core_energy'])
                aa=[a for a,b in pairs] if paired else d['initial_strings'];bb=[b for a,b in pairs] if paired else d['initial_strings']
                for spin,values in [('Alpha',aa),('Beta',bb)]:write(folder/(spin+'Dets.bin'),b''.join(int(x).to_bytes(16,'big') for x in values))
                expansion=d.get('expansion',False)
                lines=['nroots 1']+([] if d.get('spin_keyword',0) is None else ['spin '+str(d.get('spin_keyword',0))])
                lines+=['orbitals fcidump.txt','schedule','0 '+('1e-12' if expansion else '2147483647'),'end',
                    'davidsonTol 1e-10','dE 1e-10','maxiter '+('2' if expansion else '1'),
                    'noio','writeBestDeterminants 2147483647','nPTiter 0',
                    'nocc '+str(sum(d['nelec'])),' '.join(map(str,range(sum(d['nelec'])))),'end']
                write(folder/'input.dat',('\n'.join(lines)+'\n').encode())
                domain=[(a,b) for a in strings for b in strings] if expansion else pairs
                indices=[strings.index(a)*len(strings)+strings.index(b) for a,b in domain]
                hp=H[np.ix_(indices,indices)];vals=np.linalg.eigvalsh(hp)
                fixtures.append(dict(name=name,base_name=d['name'],variant=variant,norb=n,nelec=d['nelec'],
                    h=d['h'].tolist(),eri=d['g'].tolist(),core_energy=d['core_energy'],H_full=H.tolist(),S2_full=S.tolist(),
                    all_strings=strings,initial_pairs=pairs,expected_final_pairs=domain,expansion=expansion,
                    expected_energy_Ha=float(vals[0]),expected_returncode=0,
                    input_SHA256={p.name:sha256(p.read_bytes()) for p in folder.iterdir()}))
    # Explicit malformed-input checks: no energy/vector is an acceptable result.
    parent=next(x for x in fixtures if x['base_name']=='PAIRED_DIAGONAL')
    for damage in ['DUPLICATE','BETA_LENGTH','ALPHA_TRUNCATED','INVALID_ENV']:
        name='INVALID_'+damage;folder=inp/name;folder.mkdir()
        for p in (inp/parent['name']).iterdir():write(folder/p.name,p.read_bytes())
        if damage=='DUPLICATE':
            for spin in ['Alpha','Beta']:write(folder/(spin+'Dets.bin'),(1).to_bytes(16,'big')*2)
        if damage=='BETA_LENGTH':write(folder/'BetaDets.bin',(1).to_bytes(16,'big'))
        if damage=='ALPHA_TRUNCATED':write(folder/'AlphaDets.bin',b'bad')
        fixtures.append(dict(name=name,variant='INVALID_PAIRED',expected_returncode=2,env_value='true' if damage=='INVALID_ENV' else '1',
            input_SHA256={p.name:sha256(p.read_bytes()) for p in folder.iterdir()}))
    save(LOCAL/'fixtures.json',dict(fixtures=fixtures,synthetic_only=True,new_model_QPU_shots=0))
    spec=dict(version='N9_PAIRED_INPUT_ADAPTER_TINY_QUALIFICATION_V1',remote_root=REMOTE,
        worker_SHA256=sha256(WORKER.read_bytes()),patch_SHA256=sha256(PATCH.read_bytes()),
        tests_SHA256=sha256((ROOT/'qualification-paired-dice-patch-v1-tests.xml').read_bytes()),
        fixtures_SHA256=sha256((LOCAL/'fixtures.json').read_bytes()),original_source=SOURCE,
        original_main_SHA256=ORIGINAL_SHA256,original_binary=p6['native_binary'],
        original_binary_SHA256=p6['qualified_external_SHA256'][p6['native_binary']],
        original_commit='3198db1f8e0066489d173d30a52bc21967969a0f',
        boost_root='/share/reiwatanabe/ceo-native-dice-20261025-v1/runtime/boost_1_85_0',
        native_lock=p6['native_lock'],build_wall_cap_seconds=600,fixture_wall_cap_seconds=20,
        local_runtime_cap_bytes=128*1024**2,disk_floor_bytes=3*1024**3,
        purpose='INPUT_ADAPTER_QUALIFICATION_NOT_MOLECULAR_PERFORMANCE_OR_HIGHEST_PAPER_REPRODUCTION',
        native_HCI_selection_equations_changed=False,original_source_or_binary_modified=False,
        GPL_source_license_preserved=True,production_molecular_execution_authorized=False,
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N9_FROZEN_TINY_FIXTURES',len(fixtures),flush=True)


def submit():
    if (LOCAL/'submission-started.json').exists():raise RuntimeError('Existing N9 intent; reconcile exact job')
    spec=load(LOCAL/'protocol.json')
    for p,key in [(WORKER,'worker_SHA256'),(PATCH,'patch_SHA256'),(LOCAL/'fixtures.json','fixtures_SHA256')]:
        if sha256(p.read_bytes())!=spec[key]:raise RuntimeError('Frozen N9 code/fixture drift')
    if ssh('test ! -e '+shlex.quote(REMOTE)+' && echo NEW',timeout=35).strip()!='NEW':raise RuntimeError('Remote N9 exists')
    ssh('mkdir '+shlex.quote(REMOTE),timeout=35)
    for p,name in [(WORKER,'worker.py'),(PATCH,'paired_dice_patch.py'),(LOCAL/'protocol.json','protocol.json'),(LOCAL/'fixtures.json','fixtures.json')]:send(p,REMOTE+'/'+name)
    subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,str(LOCAL/'inputs'),HOST+':'+REMOTE+'/'],check=True,timeout=50)
    wrap=shlex.join(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',BASE+'/venv/bin/python','-u',REMOTE+'/worker.py',REMOTE+'/protocol.json'])
    cmd=shlex.join(['sbatch','--parsable','--partition=gpu-short','--cpus-per-task=1','--mem=4G','--time=00:15:00',
        '--job-name=ceo-paired-input-tiny-qualification-v1','--output='+REMOTE+'/slurm.out','--error='+REMOTE+'/slurm.err','--wrap='+wrap])
    save(LOCAL/'submission-started.json',dict(command=cmd,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    jid=ssh(cmd,timeout=35).strip().split(';')[0]
    if not jid.isdigit():raise RuntimeError('Uncertain submit; never blind retry')
    save(LOCAL/'job.json',dict(job_id=jid,remote=REMOTE,GPU_allocations=0));print('N9_SUBMITTED_CPU_ONLY',jid,flush=True)


def status(collect=False):
    job=load(LOCAL/'job.json')['job_id'];queue=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=35)
    live='\n'.join(x for x in queue.splitlines() if x.split('|')[0]==job)
    acc=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=35).strip()
    save(LOCAL/'status.json',dict(job_id=job,live=live,accounting=acc));print(live or acc or 'UNRESOLVED_NO_RESTART',flush=True)
    if not collect or live:return
    if not acc:raise RuntimeError('Need authoritative terminal evidence')
    target=LOCAL/'collected';target.mkdir(exist_ok=True)
    # Small fixture files, not the compiled source tree or old research data.
    for name in ['terminal.json','started.json','build.json','build.log','progress.json','slurm.out','slurm.err','outputs','source-diff.patch']:
        if ssh('test -e '+shlex.quote(REMOTE+'/'+name)+' && echo yes || true',timeout=35).strip()!='yes':continue
        subprocess.run(['rtk','proxy','scp','-r','-P','2222',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(target)],check=True,timeout=50)
    print('N9_TINY_EVIDENCE_COLLECTED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='submit':submit()
    else:status(a.action=='collect')
