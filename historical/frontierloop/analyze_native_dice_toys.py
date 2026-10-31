"""Independent numerical audit of the frozen native DICE synthetic controls."""
import json,struct,traceback
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .native_dice_toys import RUN,decode


def run():
    from pyscf import fci
    fixtures=load(RUN/'fixtures.json');terminal=load(RUN/'terminal.json');spec=load(RUN/'protocol.json')
    if sha256((ROOT/'frontierloop/native_dice_toys.py').read_bytes())!=fixtures['implementation_SHA256']:
        raise RuntimeError('Frozen toy generator/decoder changed')
    if sha256((RUN/'fixtures.json').read_bytes())!=spec['fixtures_SHA256']:
        raise RuntimeError('Frozen tiny Hamiltonians changed')
    records={x['name']:x for x in terminal['rows']};rows=[]
    if set(records)!={x['name'] for x in fixtures['fixtures']}:raise RuntimeError('Seven-control coverage mismatch')
    for case in fixtures['fixtures']:
        name=case['name'];folder=RUN/'collected'/name;record=records[name];row=dict(name=name,native_returncode=record.get('returncode'))
        try:
            if record['status']!='NATIVE_PROCESS_RETURNED_NOT_YET_NUMERICALLY_VERIFIED':
                raise RuntimeError('Native tiny process did not return outputs')
            for filename,value in case['input_SHA256'].items():
                if sha256((folder/filename).read_bytes())!=value:raise RuntimeError('Native collected input mismatch')
            for filename in ['shci.e','dets.bin','spin1RDM.0.0.txt']:
                if sha256((folder/filename).read_bytes())!=record['outputs'][filename]:
                    raise RuntimeError('Native raw output hash mismatch')
            data=(folder/'shci.e').read_bytes()
            if len(data)!=8:raise RuntimeError('Exactly one native energy required')
            reported=struct.unpack('<d',data)[0];n,pairs=decode((folder/'dets.bin').read_bytes());strings=case['all_strings']
            if n!=case['norb']:raise RuntimeError('Native orbital count mismatch')
            allowed={s:i for i,s in enumerate(strings)};v=np.zeros(len(strings)**2);indices=[]
            for a,b,value in pairs:
                if a not in allowed or b not in allowed:raise RuntimeError('Native particle-sector/CI address mismatch')
                idx=allowed[a]*len(strings)+allowed[b];v[idx]=value;indices.append(idx)
            initial={(a,b) for a in case['initial_strings'] for b in case['initial_strings']}
            if not case['expansion'] and {(a,b) for a,b,_ in pairs}!=initial:
                raise RuntimeError('Native fixed-space unexpectedly changed its basis')
            H,S=np.asarray(case['H_full']),np.asarray(case['S2_full']);norm=float(v@v)
            energy=float(v@H@v);s2=float(v@S@v);residual=float(np.linalg.norm((H@v-reported*v)[indices]))
            ci=v.reshape((len(strings),len(strings)));one=fci.direct_spin1.make_rdm1s(ci,n,tuple(case['nelec']))
            native=np.zeros((2*n,2*n));lines=(folder/'spin1RDM.0.0.txt').read_text().splitlines()[1:]
            if not lines:raise RuntimeError('Native spin one-RDM empty')
            for line in lines:
                values=line.split()
                if len(values)!=3:raise RuntimeError('Native one-RDM layout mismatch')
                i,j=int(values[0]),int(values[1])
                if not 0<=i<2*n or not 0<=j<2*n:raise RuntimeError('Native RDM index outside spin orbitals')
                native[i,j]=float(values[2])
            rdm_error=max(float(np.max(abs(native[s::2,s::2]-one[s]))) for s in range(2))
            errors=dict(normalization=abs(norm-1),energy_rayleigh_Ha=abs(energy-reported),
                projected_eigen_residual=residual,one_RDM=rdm_error)
            if max(errors.values())>1e-8:raise RuntimeError('Native independent physical/vector/RDM inconsistency: '+str(errors))
            scopes=[key for key in ['free_minimum_energy_Ha','singlet_minimum_energy_Ha','finite_penalty_physical_energy_Ha']
                if case[key] is not None and abs(reported-case[key])<1e-8]
            if not scopes:raise RuntimeError('Native root matches none of the registered spin-sector controls')
            expanded=len(pairs)>len(initial)
            if case['expansion'] and (not expanded or abs(reported-case['free_minimum_energy_Ha'])>1e-8):
                raise RuntimeError('Native HCI expansion/ground-root control failed')
            row.update(status='INDEPENDENT_PHYSICAL_BASIS_VECTOR_RDM_PASSED',energy_Ha=reported,S2=s2,
                errors=errors,matching_reference_scopes=scopes,native_actual_joint_determinants=len(pairs),
                initial_joint_determinants=len(initial),HCI_actually_expanded=expanded,
                physical_energy_minus_finite_penalty_solution_Ha=reported-case['finite_penalty_physical_energy_Ha'],
                native_exitcode_anomaly=record.get('returncode')!=0)
        except Exception as exc:
            row.update(status='UNQUALIFIED_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
        rows.append(row)
    passed=all(r['status']=='INDEPENDENT_PHYSICAL_BASIS_VECTOR_RDM_PASSED' for r in rows)
    contrast=next(r for r in rows if r['name']=='FERRO_WEAK_SPIN0')
    spin_counterexample=contrast.get('S2',0)>1e-5 and contrast['status']=='INDEPENDENT_PHYSICAL_BASIS_VECTOR_RDM_PASSED'
    finite_binding_mismatch=any(abs(r.get('physical_energy_minus_finite_penalty_solution_Ha',0))>1e-8 for r in rows)
    summary=dict(status='TINY_NATIVE_PHYSICAL_COMPONENT_QUALIFIED_NOT_C3_DROP_IN' if passed else 'TINY_NATIVE_QUALIFICATION_INCOMPLETE',
        controls=7,passed=sum(r['status']=='INDEPENDENT_PHYSICAL_BASIS_VECTOR_RDM_PASSED' for r in rows),rows=rows,
        spin0_not_a_general_exact_S0_guarantee_counterexample=spin_counterexample,
        current_finite_S2_penalty_binding_mismatch=finite_binding_mismatch,
        arbitrary_molecular_HCI_or_paper_performance_reproduced=False,
        original_C3_T3_data_changed=False,new_model_shots=0,new_QPU_shots=0,new_molecular_energy_evaluations=0,
        native_toy_CPU_seconds=terminal['CPU_seconds_including_children'],native_toy_wall_seconds=terminal['wall_seconds'],
        binary_SHA256=terminal['binary_SHA256'],goal_complete=False)
    save(ROOT/'analysis/N2-native-Dice-toy-summary.json',summary)
    text=['# Native DICE：極小物理テストの結果','',f"7条件中{summary['passed']}件で、独立な物理energy・CI vector・one-RDM照合を通過。分子性能実験ではない。",'',
        '| 極小条件 | 状態 | native energy Ha | S² | 初期 / 出力joint determinant数 |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['name']} | {r['status']} | {r.get('energy_Ha','—')} | {r.get('S2','—')} | {r.get('initial_joint_determinants','—')} / {r.get('native_actual_joint_determinants','—')} |")
    text+=['','## 解釈','',
        f"`spin 0`が一般に厳密S=0を保証しない反例を観測：{spin_counterexample}。現在の有限0.1Ha S²罰則との不一致：{finite_binding_mismatch}。",'',
        '反例は事前に登録した人工Hamiltonianで調べたもので、RAACSを勝たせるために分子のスピン規約を変更したものではない。二つのspin問題を混ぜて現在のC3表へnative値を追加しない。', '',
        'HCI expansion条件は初期HF以外のdeterminantが実際に増えたことも要求した。固定basisの対角化だけをnative HCIの探索性能と呼ばない。出力joint determinant数とalpha/beta配列のCartesian積の大きさも別の量である。', '',
        '次はnative HCIをproposal源として使い、必要なら全方式を同一の現在C7有限罰則evaluatorで再評価する設計、または全方式共通の別spin規約での比較を選ぶ。どちらも新しい明示的な共通条件として扱い、現在の結果を上書きしない。native単独の元論文最高性能・40q全般の資格・QPU実証・研究goalの達成は未証明。','']
    write(ROOT/'analysis/N2_NATIVE_DICE_TOYS_JA.md','\n'.join(text).encode())
    print(summary['status'],summary['passed'],'SPIN0_COUNTEREXAMPLE',spin_counterexample,'FINITE_BINDING_MISMATCH',finite_binding_mismatch,flush=True)


if __name__=='__main__':run()
