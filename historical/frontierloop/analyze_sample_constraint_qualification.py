"""Independent audit of every frozen N12 synthetic qualification outcome."""
import struct
from pathlib import Path
import numpy as np
from pyscf import fci
from .paths import ROOT
from .io import load,save,write,sha256
from .sample_constraint_qualification import LOCAL,PATCH,generated_worker,generated_bootstrap
from .native_dice_binary import decode


def run():
    p=load(LOCAL/'protocol.json');fixtures=load(LOCAL/'fixtures.json')['fixtures'];c=LOCAL/'collected'
    t=load(c/'terminal.json');b=load(c/'build.json');started=load(c/'started.json')
    if sha256(generated_worker().encode())!=p['worker_SHA256'] or sha256((LOCAL/'worker.py').read_bytes())!=p['worker_SHA256']:
        raise RuntimeError('Frozen worker changed')
    if sha256(generated_bootstrap().encode())!=p['bootstrap_SHA256'] or sha256(PATCH.read_bytes())!=p['patch_SHA256'] or sha256((LOCAL/'fixtures.json').read_bytes())!=p['fixtures_SHA256']:
        raise RuntimeError('Frozen bootstrap/patch/fixtures changed')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()) or b!=t['build'] or b['returncode']!=0:
        raise RuntimeError('Wrong executed protocol/build')
    if not t['original_source_unchanged'] or not t['original_binary_unchanged']:raise RuntimeError('Original source/binary changed')
    if b['original_dispatcher_SHA256']!=p['original_dispatcher_SHA256'] or b['private_dispatcher_SHA256']!=p['original_dispatcher_SHA256']:
        raise RuntimeError('Dispatcher changed unexpectedly')
    if b['patch_file_SHA256']!=sha256((c/'source-diff.patch').read_bytes()):raise RuntimeError('Source patch proof changed')
    states={r['name']:r for r in t['rows']}
    if len(fixtures)!=24 or len(t['rows'])!=24 or set(states)!={f['name'] for f in fixtures}:raise RuntimeError('All24 required')
    records=[];defaults={}
    for f in fixtures:
        r=states[f['name']];folder=c/'outputs'/f['name']
        if r['status']!='RETURNED_NOT_YET_PHYSICS_CHECKED' or r['returncode']!=f['expected_returncode']:raise RuntimeError('Unexpected execution outcome: '+f['name'])
        expected_bin=p['original_binary_SHA256'] if f['variant']=='ORIGINAL_DEFAULT' else b['binary_SHA256']
        if r['binary_SHA256']!=expected_bin:raise RuntimeError('Wrong binary')
        for name,value in f['input_SHA256'].items():
            if sha256((folder/name).read_bytes())!=value:raise RuntimeError('Tiny input changed')
        for name,value in r['outputs'].items():
            if Path(name).name!=name or sha256((folder/name).read_bytes())!=value:raise RuntimeError('Tiny output changed')
        log=(folder/'native.log').read_text()
        if f['variant']=='INVALID_CONSTRAINED':
            if (folder/'shci.e').exists() or (folder/'dets.bin').exists() or 'CEO_SAMPLE_POOL_INVALID_' not in log:
                raise RuntimeError('Malformed pool not rejected before energy/vector')
            records.append(dict(name=f['name'],status='EXPECTED_INPUT_REJECTION_NO_ENERGY_OR_VECTOR'));continue
        raw=(folder/'shci.e').read_bytes()
        if len(raw)!=8:raise RuntimeError('Exactly one root required')
        energy=struct.unpack('<d',raw)[0];n,pp=decode((folder/'dets.bin').read_bytes());ss=f['all_strings'];ne=tuple(f['nelec'])
        if n!=f['norb'] or len(pp)!=len(f['expected_final_pairs']) or {(a,b) for a,b,_ in pp}!={tuple(x) for x in f['expected_final_pairs']}:
            raise RuntimeError('Actual domain differs: '+f['name'])
        v=np.zeros((len(ss),len(ss)))
        for alpha,beta,value in pp:
            if alpha.bit_count()!=ne[0] or beta.bit_count()!=ne[1]:raise RuntimeError('Wrong electron sector')
            v[ss.index(alpha),ss.index(beta)]=value
        vec=v.ravel();H=np.array(f['H_full']);S=np.array(f['S2_full']);idx=[ss.index(a)*len(ss)+ss.index(b) for a,b in f['expected_final_pairs']]
        residual=float(np.linalg.norm((H@vec-energy*vec)[idx]));norm=float(vec@vec);ray=float(vec@H@vec)
        dm1,dm2=fci.direct_spin1.make_rdm12(v,n,ne)
        erdm=float(np.einsum('ij,ij',f['h'],dm1)+.5*np.einsum('ijkl,ijkl',f['eri'],dm2)+f['core_energy'])
        if abs(norm-1)>1e-8 or abs(energy-f['expected_energy_Ha'])>1e-8 or abs(ray-energy)>1e-8 or residual>1e-8 or abs(erdm-energy)>1e-8:
            raise RuntimeError('Independent H/sector/norm/residual/RDM failed: '+f['name'])
        audit=folder/'CEO-sample-pool-v1.txt'
        if f['variant']=='CUSTOM_CONSTRAINED':
            pool=[int(x) for x in audit.read_text().split()]
            if pool!=f['allowed_strings'] or any(a not in pool or bb not in pool for a,bb,_ in pp) or 'CEO_SAMPLE_POOL_V1 ' not in log:
                raise RuntimeError('Actual selected-space pool/filter mismatch')
        elif audit.exists():raise RuntimeError('Pool mode leaked into default')
        record=dict(name=f['name'],status='INDEPENDENT_PHYSICS_AND_DOMAIN_PASSED',energy_Ha=energy,
            initial_dimension=len(f['initial_pairs']),final_dimension=len(pp),norm_squared=norm,
            projected_residual=residual,RDM_energy_Ha=erdm,spin_square=float(vec@S@vec),allowed_half_pool=f.get('allowed_strings'))
        records.append(record)
        if f['variant'] in ['ORIGINAL_DEFAULT','CUSTOM_DEFAULT']:defaults.setdefault(f['base_name'],{})[f['variant']]=record
    for name,ds in defaults.items():
        if len(ds)!=2 or abs(ds['ORIGINAL_DEFAULT']['energy_Ha']-ds['CUSTOM_DEFAULT']['energy_Ha'])>1e-10:raise RuntimeError('Default native mismatch: '+name)
    result=dict(status='N12_ALL24_AUDITED_19_PHYSICS_5_EXPECTED_REJECTIONS',records=records,
        default_parity_pairs=7,sample_constraint_physics_cases=5,negative_cases=5,binary=b['binary'],binary_SHA256=b['binary_SHA256'],
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
        terminal_SHA256=sha256((c/'terminal.json').read_bytes()),analyzer_SHA256=sha256(Path(__file__).read_bytes()),
        original_source_and_binary_unchanged=True,full_paper_reproduction=False,production_molecular_execution_authorized=False,
        new_molecular_evaluations=0,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,goal_complete=False)
    save(ROOT/'analysis/N12-sample-constraint-qualification-summary.json',result)
    lines=['# N12：観測pool内選抜の小系資格','',result['status'],'',
        f"CPU {t['CPU_seconds']:.3f}秒 / wall {t['wall_seconds']:.3f}秒（build/tiny実行込み）。分子energy/Shot/GPU追加0。",'',
        '元nativeとのdefault7対は一致。pool制限5試験はHFのみ・全pool・subset・hole・有限cutoffを含み、独立H/sector/norm/部分空間residual/RDMおよび実domain所属を確認した。不正pool5試験は非0で停止しenergy/vectorを生成しなかった。',
        '元のSHCI.cpp、元SHCIbasics.cpp、元binaryは不変。別private版で追加候補universeだけを制限した。fresh/serial/closed-shellの登録範囲の資格で、restartや任意のopen-shell、全原著hardware実験の再現を保証しない。','',
        '| 小系 | 初期配置 | 最終配置 | energy Ha |', '|---|---:|---:|---:|']
    for r in records:
        if '__CUSTOM_CONSTRAINED' in r['name']:lines.append(f"| {r['name']} | {r['initial_dimension']} | {r['final_dimension']} | {r['energy_Ha']:.10f} |")
    lines+=['','次の分子評価は別freezeが必要。量子sourceだけでなくCISDにも同じ観測pool制限を与え、unrestricted HF-HCIを強い古典対照として残す。原著Eq.(9)の成分を実装したこと自体を新規成果と呼ばない。']
    write(ROOT/'analysis/N12_SAMPLE_CONSTRAINT_QUALIFICATION_RESULTS_JA.md',('\n'.join(lines)+'\n').encode());print(result['status'],flush=True)


if __name__=='__main__':run()
