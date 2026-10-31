"""Read-only error-scale/reference diagnostic, not a new molecular experiment."""
import math
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from .paired_pool_study import LOCAL


def run():
    spec=load(LOCAL/'protocol.json');terminal=load(LOCAL/'collected/terminal.json')
    auditfile=ROOT/'analysis/N10-paired-initial-pool-summary.json';audit=load(auditfile)
    if audit['terminal_SHA256']!=sha256((LOCAL/'collected/terminal.json').read_bytes()):raise RuntimeError('Audited terminal changed')
    references={};anchors=[];rows=[]
    for case,d in spec['physical'].items():
        rf=OLD/'data/evaluation-only'/(case+'.json');r=load(rf);e=r['energies_Ha']
        ref=e['fci_energy'] if e.get('fci_energy') is not None else e['sci_energy']
        if r['problem_id']!=d['problem_id']:raise RuntimeError('Reference problem mismatch')
        integrals=OLD/'data/physical'/case/'integrals.npz'
        if sha256(integrals.read_bytes())!=d['integrals_SHA256']:raise RuntimeError('Physical integrals changed')
        with np.load(integrals,allow_pickle=False) as a:h=a['h1'];g=a['eri']
        no=d['nelec'][0]
        if d['nelec'][0]!=d['nelec'][1]:raise RuntimeError('Closed-shell HF diagnostic only')
        hf=float(d['core_energy']+2*math.fsum(float(h[i,i]) for i in range(no))+
                 math.fsum(float(2*g[i,i,j,j]-g[i,j,j,i]) for i in range(no) for j in range(no)))
        difference=hf-e['hf_energy']
        if abs(difference)>1e-8:raise RuntimeError('Actual Hamiltonian HF anchor differs from reference')
        references[case]=dict(energy_Ha=ref,kind='FCI_ARCHIVE' if e.get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
            sector_and_reference_precision_verified=r.get('sector_and_reference_precision_verified'),reference_SHA256=sha256(rf.read_bytes()))
        anchors.append(dict(case=case,actual_HF_Ha=hf,archived_HF_Ha=e['hf_energy'],difference_Ha=difference,
            integral_SHA256=sha256(integrals.read_bytes()),not_a_full_reference_precision_certificate=True))
    for s in terminal['states']:
        item=s['arms'].get('JOINT_TOP_D16384');native=s.get('native');ref=references[s['case']]
        if not native or not item:continue
        e=native['native_diagnostic_spinfree_energy_Ha'];final=item['row']['energy_Ha']
        rows.append(dict(case=s['case'],method=s['method'],arm=s['initialization_arm'],
            native_variational_diagnostic_Ha=e,common_final_physical_Ha=final,
            native_reference_gap_mHa=1000*(e-ref['energy_Ha']),final_reference_gap_mHa=1000*(final-ref['energy_Ha']),
            final_minus_native_mHa=1000*(final-e),not_a_pure_truncation_error_decomposition=True,
            retained_native_weight=s['selection']['retained_native_weight'],native_dimension=s['selection']['native_joint_dimension'],
            final_dimension=s['selection']['actual_joint_dimension'],reference_kind=ref['kind'],
            reference_precision_verified=ref['sector_and_reference_precision_verified']))
    cases=[]
    for case in spec['physical']:
        good=[r for r in rows if r['case']==case];n=min(good,key=lambda r:r['native_variational_diagnostic_Ha']);f=min(good,key=lambda r:r['common_final_physical_Ha'])
        cases.append(dict(case=case,lowest_native_reference_gap_mHa=n['native_reference_gap_mHa'],
            lowest_final_reference_gap_mHa=f['final_reference_gap_mHa'],reference=references[case],
            lowest_native_method=n['method'],lowest_final_method=f['method']))
    result=dict(status='ALL106_NATIVE_FINAL_ERROR_SCALES_AND4_HF_ANCHORS_CHECKED',rows=rows,cases=cases,HF_anchors=anchors,
        N10_audit_SHA256=sha256(auditfile.read_bytes()),implementation_SHA256=sha256(Path(__file__).read_bytes()),
        native_spinfree_vs_finite_penalty_not_identical_objectives=True,new_CI_solves=0,new_model_shots=0,new_QPU_shots=0,
        reference_sector_precision_not_newly_certified=True,goal_complete=False)
    save(ROOT/'analysis/N10-error-scale-and-reference-diagnostic.json',result)
    lines=['# N10：改善量と、残る精度差の大きさ','',
        'これは保存済みnative/最終結果の再集計とHFの解析式による照合で、新しいCIや量子測定はない。全4条件で実HamiltonianのHF energyがarchive HFと1e-8Ha以内で一致した。ただし、これはreferenceのsectorや精度そのものの証明ではない。','',
        '| 条件 | 最良nativeとreferenceの差 mHa | 最良最終CIとreferenceの差 mHa | reference |','|---|---:|---:|---|']
    for x in cases:lines.append(f"| {x['case']} | {x['lowest_native_reference_gap_mHa']:.6f} | {x['lowest_final_reference_gap_mHa']:.6f} | {x['reference']['kind']} |")
    lines+=['','nativeはspinfree variational diagnostic、最終CIは有限S²罰則で再最適化した状態のphysical energyなので、両者の差を純粋なtruncation errorと呼ばない。referenceのsector/precisionはまだ未検証。',
        'それでも、現在のnative段階自体が保存referenceから大きく離れる条件があることは重要。0.03～0.21mHaの初期化利益を、残る全精度問題の解決と取り違えない。原著と合わせたnative cutoffs/variational-vs-PTの定義、referenceの意味、最終D上限の役割を確認してから追加投資する。']
    write(ROOT/'analysis/N10_ERROR_SCALE_AND_REFERENCE_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],flush=True)


if __name__=='__main__':run()
