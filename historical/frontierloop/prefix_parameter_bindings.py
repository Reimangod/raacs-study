"""Declared ansatz coordinates, not numeric transpiler slots or Jacobian rank.

Compare parent/rounded circuit in the SAME Slater-prefix coordinate family.
Known removal of the initial U(n) Slater gauge is reported separately, never
credited to rounding. A shared alpha/beta angle counts once by construction.
"""
import ffsim
import numpy as np
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256
from .slater_prefix_preflight import LOCAL
from .slater_prefix_compression import parse_prefix
from paperstudy.lucj import make_circuit,interaction_pairs


def binding_count(prefix,actions,suffix_coordinates):
    first=prefix.records[0]
    # Native Slater chart: two real coordinates per Givens pair, and any
    # explicitly present shared single-mode phase. No accidental equality test.
    parent=sum(2 if r['name']=='xx_plus_yy' else 1 for r in first)+suffix_coordinates
    removed=0;bindings=[]
    byindex={r['index']:r for r in first}
    for a,b in prefix.pairs:
        if (a in actions)!=(b in actions):raise ValueError('Spin-pair binding broken')
        if a not in actions:continue
        if actions[a]!=actions[b]:raise ValueError('Two spins use different fixed angles')
        target=actions[a];odd=np.isclose(abs(target),np.pi,atol=1e-14)
        if not any(abs(target-t)<1e-14 for t in [0.,np.pi,-np.pi,2*np.pi]):raise ValueError('Unregistered fixed angle')
        removed+=1 if odd else 2
        bindings.append(dict(alpha_gate_index=a,beta_gate_index=b,mixing_angle_fixed=target,
            phase_coordinate='SHARED_BETA_RETAINED' if odd else 'ABSENT_GATE_BETA_IRRELEVANT',
            declared_coordinates_removed=1 if odd else 2,source_beta_numeric_value=byindex[a]['params'][1]))
    return parent,parent-removed,bindings


def run():
    spec=load(LOCAL/'protocol.json');raw=load(LOCAL/'result.json');rows=[]
    for s in spec['sources']:
        if s['status']!='AVAILABLE':continue
        with np.load(s['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
        n=s['norb'];layers=len(op.orbital_rotations);pairs=interaction_pairs(n)
        original_coordinates=len(op.to_parameters(interaction_pairs=pairs));per_J=len(pairs[0])+len(pairs[1])
        if original_coordinates!=(layers+1)*n*n+layers*per_J:raise ValueError('Unrecognized source coordinate family')
        q=ffsim.qiskit.PRE_INIT.run(make_circuit(op,n,tuple(s['nelec'])));prefix=parse_prefix(q.data[0].operation.definition,n)
        suffix_coordinates=layers*(n*n+per_J)
        for r in raw['rows']:
            if (r['case'],r['method'])!=(s['case'],s['method']):continue
            actions=dict(r['actions']) if r['adopted'] else {}
            parent,child,bindings=binding_count(prefix,actions,suffix_coordinates)
            rows.append(dict(case=s['case'],method=s['method'],policy=r['policy'],tier=r['tier'],adopted=r['adopted'],
                original_source_ffsim_declared_coordinates=original_coordinates,
                matched_compiled_parent_declared_coordinates=parent,rounded_declared_coordinates=child,
                rounding_coordinate_reduction=parent-child,rounding_coordinate_reduction_pct=100*(parent-child)/parent,
                known_initial_Slater_chart_change_not_rounding=original_coordinates-parent,
                bindings=bindings,optimized_variables_in_this_study=0,physical_state_Jacobian_rank=None))
    if len(rows)!=60:raise ValueError('All60 numeric prefix family rows required')
    out=dict(status='N15_ALL60_DECLARED_COORDINATE_BINDING_ROWS_CONSTRUCTED',rows=rows,
        definition='REGISTERED_FREE_REAL_COORDINATES_IN_A_MATCHED_PREFIX_SUFFIX_ANSATZ_FAMILY_NOT_JACOBIAN_RANK',
        nominal_counts_can_contain_state_gauges=True,transpiler_numeric_slots_not_used=True,
        no_claim_of_global_functional_independence=True,
        original_result_SHA256=sha256((LOCAL/'result.json').read_bytes()),implementation_SHA256=sha256(Path(__file__).read_bytes()),
        new_CI_solves=0,new_model_shots=0,goal_complete=False)
    save(ROOT/'analysis/N15-declared-coordinate-bindings.json',out)
    lines=['# N15：Parametersを何として数えるか','',
        '本表は、明示したansatz familyに与える自由な実座標の数。実行時に数値が固定されたことを理由に0とはしない。alpha/betaで共有する変数は登録された結合規則に従って1個と数える。偶然同じ数値だった変数を勝手に統合しない。',
        '元ffsim座標と、Slater準備に特化したcompiled座標は別。初期U(n)からSlater chartへ変える既知の表現削減を、新しい丸めの成果へ加算しない。親と子は同じcompiled表現で比較する。',
        'これは最小の物理Jacobian rankではない。通常のansatz座標数を報告することとrank証明は別であり、rank未証明だけを理由に研究を止める必要はない。rank低下を主張する場合に限り別の根拠が必要。','',
        '| 条件 | source | tier | policy | 元ffsim座標 | 同じcompiled親 | 丸め後 | 丸めによる減少 |','|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if r['tier']==.01 and r['policy']=='STATE_OVERLAP_ROUND':lines.append(f"| {r['case']} | {r['method']} | {r['tier']} | {r['policy']} | {r['original_source_ffsim_declared_coordinates']} | {r['matched_compiled_parent_declared_coordinates']} | {r['rounded_declared_coordinates']} | {r['rounding_coordinate_reduction']} |")
    lines+=['','混合角πではθを固定して位相βを残す（alpha/beta共有）。0/2πではβに依存しないためθ/βの両座標を取り除く。変換後local phaseの複数出現を、それぞれ別の独立変数として水増ししない。suffixの座標familyは親子で同じ。今回実行したoptimizer変数は0だが、ansatz資源のParametersが0という意味ではない。']
    write(ROOT/'analysis/N15_PARAMETERS_JA.md',('\n'.join(lines)+'\n').encode());print(out['status'],flush=True)


if __name__=='__main__':run()
