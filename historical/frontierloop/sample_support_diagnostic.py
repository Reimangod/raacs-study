"""Existing-result diagnostic: HCI omission versus source-space limitation.

Only label a Cartesian result as the full observed universe if its actual
single-spin strings exactly match the N13 registered pool. No new CI solve.
"""
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256


def run():
    summary_file=ROOT/'analysis/N13-sample-constrained-summary.json';s=load(summary_file)
    p=load(ROOT/'runs/N13-SAMPLE-CONSTRAINED-ALL-SOURCES-v1/protocol.json');plans={x['task_id']:x for x in p['plans']}
    rows=[]
    for r in s['rows']:
        if not r['method'].startswith('RAACS'):continue
        d=ROOT/'runs/R2-COMMON-CI-v1'/r['case']/r['method'];request=load(d/'request.json');result=load(d/'result.json')
        plan=plans[r['task_id']];pool=plan['allowed_strings'];definition=p['physical'][r['case']]
        if request['source']['payload_SHA256']!=plan['source']['payload_SHA256'] or result['source_payload_SHA256']!=plan['source']['payload_SHA256'] or result['physical_problem_id']!=definition['problem_id']:raise RuntimeError('Different historical source/problem')
        if result['solver']!='SciPy eigsh which=SA k=2 matrix-free physical H plus projected S2' or result['residual_norm']>1e-8 or result['orbital_optimization_performed'] or result['exact_singlet_projection']:raise RuntimeError('Different/incomplete old numerical equation')
        if max(result['independent_RDM_expectation_errors'])>1e-8 or abs(result['penalized_energy_Ha']-result['energy_Ha']-.1*result['spin_square'])>1e-8:raise RuntimeError('Historical physical consistency failed')
        strings=request['CI_strings']
        if result['actual_CI_dimension']!=len(strings)**2 or len(strings)!=len(set(strings)):raise RuntimeError('Historical Cartesian dimension differs')
        full=set(strings)==set(pool)
        rows.append(dict(case=r['case'],method=r['method'],full_observed_pool_equality=full,
            pool_size=len(pool),historical_R128_strings=len(strings),historical_dimension=result['actual_CI_dimension'],
            N13_dimension=r['final_dimension'],historical_energy_Ha=result['energy_Ha'],N13_energy_Ha=r['energy_Ha'],
            physical_difference_mHa=1000*(r['energy_Ha']-result['energy_Ha']),
            penalized_difference_mHa=1000*(r['penalized_energy_Ha']-result['penalized_energy_Ha']),
            historical_spin_square=result['spin_square'],N13_spin_square=r['spin_square'],
            historical_reference_gap_mHa=r['reference_gap_mHa']-1000*(r['energy_Ha']-result['energy_Ha']),
            request_SHA256=sha256((d/'request.json').read_bytes()),result_SHA256=sha256((d/'result.json').read_bytes())))
    if len(rows)!=8 or sum(r['full_observed_pool_equality'] for r in rows)!=7:raise RuntimeError('Audit all8 RAACS cells including incomplete universe')
    full=[r for r in rows if r['full_observed_pool_equality']]
    if any(r['penalized_difference_mHa']< -1e-5 for r in full):raise RuntimeError('Subspace variational ordering inconsistent')
    result=dict(status='ALL8_RAACS_EXISTING_FULL_POOL_DIAGNOSTICS_AUDITED_7_EQUAL_1_TRUNCATED',rows=rows,
        equation='H_PLUS_0p1_S2_NOT_EXACT_SPIN_OR_ABSOLUTE_GROUND_STATE_BOUND',new_CI_solves=0,new_model_shots=0,
        N13_summary_SHA256=sha256(summary_file.read_bytes()),implementation_SHA256=sha256(Path(__file__).read_bytes()),goal_complete=False)
    save(ROOT/'analysis/N13-source-support-diagnostic.json',result)
    lines=['# N13補助診断：RAACSの精度不足は選抜だけの問題か','',
        '新しい計算は行わず、N13と過去R2の同じsource・同じH+0.1S²を照合した。',
        '8セル中7セルは、R2が**観測half pool全体のCartesian空間**を既に解いていた。Naph B4は143 stringsに対してR128の切詰めがあり、全poolの答えとは扱わない。','',
        '| 条件 | RAACS | 全poolか | 全pool/R128 reference差 mHa | N13 reference差 mHa | N13で落とした分 mHa | 罰則目的差 mHa |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['method']} | {'全pool' if r['full_observed_pool_equality'] else 'R128一部のみ'} | {r['historical_reference_gap_mHa']:.6f} | {r['historical_reference_gap_mHa']+r['physical_difference_mHa']:.6f} | {r['physical_difference_mHa']:.6f} | {r['penalized_difference_mHa']:.6f} |")
    lines+=['','## 判断','',
        'RAACSの多くでは、観測poolの全Cartesian空間を使っても大きな誤差が残る。HCIが一部を省いたことによる追加差は最大でもこの7例で0.281mHa程度で、N₂/Feの約100〜331mHaの不足を説明しない。現在の観測データについては、readoutランキングの微修正より、sourceが持つ配置の不足が主要な問題である。',
        'ただし、これは有限0.1S²を含む登録式・保存sourceの診断である。物理Hamiltonian単独の厳密下限、無限Shotでのsupport、RAACS一般の不可能性、各分子の厳密ground stateを証明しない。Naph B4はfull-poolでないという例外も隠さない。',
        '罰則目的の部分空間順序を検査し、physical energy差だけでvariational boundとは呼ばない。旧R2計算費用は既存費用であり今回の新計算へ再加算しない。']
    write(ROOT/'analysis/N13_SOURCE_SUPPORT_DIAGNOSTIC_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],flush=True)


if __name__=='__main__':run()
