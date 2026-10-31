"""Independent PySCF/numpy audit of every N9 tiny output, not molecular data."""
import struct
from pathlib import Path
import numpy as np
from pyscf import fci
from .paths import ROOT
from .io import load,save,sha256,write
from .paired_dice_qualification import LOCAL,WORKER,PATCH
from .native_dice_binary import decode


def run():
    spec=load(LOCAL/'protocol.json');fixtures=load(LOCAL/'fixtures.json')['fixtures'];c=LOCAL/'collected'
    t=load(c/'terminal.json');started=load(c/'started.json');build=load(c/'build.json')
    for p,key in [(WORKER,'worker_SHA256'),(PATCH,'patch_SHA256'),(LOCAL/'fixtures.json','fixtures_SHA256')]:
        if sha256(p.read_bytes())!=spec[key]:raise RuntimeError('Frozen qualifier changed')
    if spec.get('bootstrap_SHA256') and sha256((ROOT/'frontierloop/paired_dice_bootstrap.py').read_bytes())!=spec['bootstrap_SHA256']:
        raise RuntimeError('Registered source-staging bootstrap changed')
    if started['protocol_SHA256']!=sha256((LOCAL/'protocol.json').read_bytes()):raise RuntimeError('Wrong executed protocol')
    if not t['original_source_unchanged'] or not t['original_binary_unchanged'] or build['returncode']!=0:
        raise RuntimeError('Original source/binary changed or adapter not built')
    if build!=t['build'] or build['patch_file_SHA256']!=sha256((c/'source-diff.patch').read_bytes()):raise RuntimeError('Build evidence mismatch')
    states={r['name']:r for r in t['rows']}
    if len(fixtures)!=24 or len(t['rows'])!=24 or set(states)!={f['name'] for f in fixtures}:raise RuntimeError('All24 tiny attempts required')
    records=[];default={}
    for f in fixtures:
        folder=c/'outputs'/f['name'];row=states[f['name']]
        if row['status']!='RETURNED_NOT_YET_PHYSICS_CHECKED' or row['returncode']!=f['expected_returncode']:
            raise RuntimeError('Tiny native returncode mismatch: '+f['name'])
        expected_binary=spec['original_binary_SHA256'] if f['variant']=='ORIGINAL_DEFAULT' else build['binary_SHA256']
        if row['binary_SHA256']!=expected_binary:raise RuntimeError('Wrong binary executed')
        for filename,value in f['input_SHA256'].items():
            if sha256((folder/filename).read_bytes())!=value:raise RuntimeError('Frozen tiny input drift')
        for filename,value in row['outputs'].items():
            if Path(filename).name!=filename or sha256((folder/filename).read_bytes())!=value:raise RuntimeError('Output evidence changed')
        if f['variant']=='INVALID_PAIRED':
            if (folder/'dets.bin').exists() or (folder/'shci.e').exists():raise RuntimeError('Malformed input produced an energy/vector')
            if 'CEO_PAIRED_INPUT_' not in (folder/'native.log').read_text():raise RuntimeError('Not a controlled adapter rejection')
            records.append(dict(name=f['name'],status='EXPECTED_INPUT_REJECTION_NO_ENERGY_OR_VECTOR'));continue
        raw=(folder/'shci.e').read_bytes()
        if len(raw)!=8:raise RuntimeError('One root required')
        energy=struct.unpack('<d',raw)[0];n,pairs=decode((folder/'dets.bin').read_bytes());ne=tuple(f['nelec']);strings=f['all_strings']
        if n!=f['norb'] or len(pairs)!=len(f['expected_final_pairs']) or {(a,b) for a,b,_ in pairs}!={tuple(x) for x in f['expected_final_pairs']}:
            raise RuntimeError('Actual final determinant domain differs: '+f['name'])
        v=np.zeros((len(strings),len(strings)))
        for a,b,value in pairs:
            if a.bit_count()!=ne[0] or b.bit_count()!=ne[1]:raise RuntimeError('Particle sector violation')
            v[strings.index(a),strings.index(b)]=value
        norm=float(np.vdot(v,v));H=np.array(f['H_full']);S=np.array(f['S2_full']);flat=v.ravel();hv=H@flat
        domain=[strings.index(a)*len(strings)+strings.index(b) for a,b in f['expected_final_pairs']]
        # A restricted-space CI vector need not be an eigenvector in full Fock space.
        residual=float(np.linalg.norm((hv-energy*flat)[domain]))
        rayleigh=float(flat@hv);dm1,dm2=fci.direct_spin1.make_rdm12(v,n,ne)
        rdm_energy=float(np.einsum('ij,ij',np.array(f['h']),dm1)+.5*np.einsum('ijkl,ijkl',np.array(f['eri']),dm2)+f['core_energy'])
        s2=float(flat@S@flat)
        if abs(norm-1)>1e-8 or abs(energy-f['expected_energy_Ha'])>1e-8 or abs(rayleigh-energy)>1e-8 or residual>1e-8 or abs(rdm_energy-energy)>1e-8:
            raise RuntimeError('Independent energy/norm/residual/RDM parity failed: '+f['name'])
        if f['variant']=='CUSTOM_PAIRED':
            audit=[list(map(int,line.split())) for line in (folder/'CEO-initial-pairs-v1.txt').read_text().splitlines()]
            if audit!=f['initial_pairs']:raise RuntimeError('Actual constructed initial pairs differ')
        elif (folder/'CEO-initial-pairs-v1.txt').exists():raise RuntimeError('Paired mode leaked into default path')
        r=dict(name=f['name'],status='INDEPENDENT_ENERGY_SECTOR_NORM_RESIDUAL_RDM_PASSED',energy_Ha=energy,
            expected_energy_Ha=f['expected_energy_Ha'],norm_squared=norm,projected_residual=residual,
            RDM_energy_Ha=rdm_energy,spin_square=s2,initial_dimension=len(f['initial_pairs']),final_dimension=len(pairs),
            native_CPU_seconds=row['CPU_seconds'],native_wall_seconds=row['wall_seconds'])
        records.append(r)
        if f['variant'] in ['ORIGINAL_DEFAULT','CUSTOM_DEFAULT']:default.setdefault(f['base_name'],{})[f['variant']]=r
    for name,p in default.items():
        if len(p)!=2 or abs(p['ORIGINAL_DEFAULT']['energy_Ha']-p['CUSTOM_DEFAULT']['energy_Ha'])>1e-10:
            raise RuntimeError('Unchanged default path differs: '+name)
    summary=dict(status='ALL24_TINY_ADAPTER_CASES_AUDITED_20_PHYSICS_4_EXPECTED_REJECTIONS',records=records,
        default_parity_pairs=len(default),paired_physics_cases=6,negative_rejection_cases=4,
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],binary=build['binary'],binary_SHA256=build['binary_SHA256'],
        protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),terminal_SHA256=sha256((c/'terminal.json').read_bytes()),
        analyzer_SHA256=sha256(Path(__file__).read_bytes()),
        new_molecular_evaluations=0,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        molecular_performance_verified=False,production_molecular_execution_authorized=False,goal_complete=False)
    save(ROOT/'analysis/N9-paired-input-qualification-summary.json',summary)
    lines=['# N9：ペア入力の小系資格結果','',summary['status'],'',
        f"CPU {t['CPU_seconds']:.3f}秒 / wall {t['wall_seconds']:.3f}秒。buildと全tiny実行を含む。新分子energy/Shot/GPU割当0。",'',
        '元binaryとのdefault7対は同じenergy。変更版paired6試験は実際の初期ペア・独立Hamiltonian・sector・norm・部分空間residual・RDM energyを確認。不正入力4試験はenergy/vectorを生成せず拒否。元source/binaryのhashは不変。','',
        '| 小系 | 初期配置 | 最終配置 | energy Ha | S² |', '|---|---:|---:|---:|---:|']
    for r in records:
        if '__CUSTOM_PAIRED' in r['name']:lines.append(f"| {r['name']} | {r['initial_dimension']} | {r['final_dimension']} | {r['energy_Ha']:.10f} | {r['spin_square']:.6f} |")
    lines+=['','**これは入力adapterの正しさの資格であり、分子性能や新規性の証明ではない。** spin0はexact singletを保証せず、今回もS²を別に検査した。restricted CIでは全空間residualが0である必要はなく、指定部分空間で固有方程式を検査した。',
        '次の分子実験をする場合は、同じ初期配置数64でmarginal8×8とobserved joint top64を別freezeにし、古典CISDにも同じ入力機会を与える。旧R128結果とは分離する。今回のprotocolは分子実行を許可していない。']
    write(ROOT/'analysis/N9_PAIRED_INPUT_QUALIFICATION_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(summary['status'],flush=True)


if __name__=='__main__':run()
