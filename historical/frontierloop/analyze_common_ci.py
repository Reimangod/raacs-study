"""Whole R2 comparison, with R1 fixed samplers and all missing/failed cells."""
import csv
from io import StringIO
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256
from .common_ci_cap import DIRECTORY
from .analyze import reference


def main():
    manifest=load(DIRECTORY/'freeze.json');rows=[]
    for cell in manifest['sources']:
        p=DIRECTORY/cell['case']/cell['method_id']/'result.json';r=load(p) if p.exists() else None
        if r:
            assert r['source_payload_SHA256']==cell['payload_SHA256'] and r['physical_problem_id']==cell['expected_problem_id']
            assert sha256(Path(cell['payload']).read_bytes())==cell['payload_SHA256']
        status='COMPLETED' if r else 'MISSING_SAME_PROBLEM_SOURCE' if cell['status']!='AVAILABLE' else 'FAILED_ENGINEERING_PRESERVED' if (p.parent/'failure.json').exists() else 'NOT_COMPLETED'
        q=cell.get('sampler_resources') or {};hf,ref=reference(cell['case'])
        rows.append({'case':cell['case'],'method':cell['method_id'],'source_kind':cell['kind'],'status':status,
            'energy_Ha':r['energy_Ha'] if r else None,'recovery_pct':100*(hf-r['energy_Ha'])/(hf-ref) if r else None,
            'S2':r['spin_square'] if r else None,'CNOT':q.get('cnot_count'),'CNOT_depth':q.get('cnot_depth'),
            'total_depth':q.get('total_depth',q.get('total_unitary_depth')),
            'source_coefficient_slot_proxy_not_rank':cell.get('sampler_coefficient_slots_mean'),
            'primitive_numeric_gate_slots_not_rank':None,'independent_parameter_rank':None,
            'total_model_shot_budget_including_pilot':cell.get('source_total_budget'),
            'final_model_shots':cell.get('source_final_model_shots'),'classical_draws':cell.get('classical_draws'),
            'R_cap':128,'actual_CI_dimension':r['actual_CI_dimension'] if r else None,
            'incremental_CI_CPU_seconds':r['incremental_CPU_seconds'] if r else None,
            'source_construction_CPU_seconds':cell.get('source_construction_CPU_seconds'),
            'source_initialization_reference':cell.get('source_result'),'result_SHA256':sha256(p.read_bytes()) if r else None})
    for case in load(ROOT/'protocol-R1-CI-CAP-SCREEN.json')['cases']:
        hf,ref=reference(case)
        for method,folder in [('F2','F2-POSITIVE-MARGINAL-v1'),('H1','H1-HALF-REMAP-v1'),('D2_EXACT','D2-COMPRESSION-SCREEN')]:
            p=ROOT/'runs/R1-CI-CAP-SCREEN-v1'/case/method/'128/result.json';r=load(p) if p.exists() else None
            src=ROOT/'runs'/folder
            if method=='D2_EXACT':
                d=src/case/'EXACT';draw=load(d/'draw-result.json');req=load(d/'request.json')
                q=req['resources']['Z_sampling_equivalent'];q={**q,'total_depth':q['total_unitary_depth']};slots=None
                initial=req['source_initialization_provenance']
            else:
                d=src/'remote-output/outputs'/case;draw=load(d/'draw-result.json')
                q=draw['sampling_resources_shot_weighted_mean'] if method=='F2' else draw['resources']['sampling']
                slots=q['numeric_gate_coefficient_slots_not_rank'];initial=draw['source_initialization_record']['path']
            if r:assert r['source_counts_SHA256']==draw['counts_SHA256']
            rows.append({'case':case,'method':method+'_FIXED_PARENT','source_kind':'QUANTUM_HALF_COUNTS' if method!='D2_EXACT' else 'QUANTUM_FULL_COUNTS',
                'status':'COMPLETED' if r else 'NOT_COMPLETED','energy_Ha':r['energy_Ha'] if r else None,
                'recovery_pct':100*(hf-r['energy_Ha'])/(hf-ref) if r else None,'S2':r['spin_square'] if r else None,
                'CNOT':q['cnot_count'],'CNOT_depth':q['cnot_depth'],'total_depth':q['total_depth'],
                'source_coefficient_slot_proxy_not_rank':None,'primitive_numeric_gate_slots_not_rank':slots,
                'independent_parameter_rank':None,'total_model_shot_budget_including_pilot':100000,'final_model_shots':100000,
                'classical_draws':0,'R_cap':128,'actual_CI_dimension':r['actual_CI_dimension'] if r else None,
                'incremental_CI_CPU_seconds':r['incremental_CPU_seconds'] if r else None,'source_construction_CPU_seconds':None,
                'source_initialization_reference':initial,'result_SHA256':sha256(p.read_bytes()) if r else None})
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/R2-common-CI-all.csv',stream.getvalue().encode('utf-8-sig'))
    complete=sum(r['status']=='COMPLETED' for r in rows);missing=sum(r['status']=='MISSING_SAME_PROBLEM_SOURCE' for r in rows)
    failed=sum(r['status']=='FAILED_ENGINEERING_PRESERVED' for r in rows);pending=len(rows)-complete-missing-failed
    summary={'completed':complete,'planned_rows_including_R1_reuse':len(rows),'missing_same_problem_sources':missing,
        'failed_engineering':failed,'pending':pending,'new_model_shots':0,'goal_complete':False}
    save(ROOT/'analysis/R2-common-CI-summary.json',summary)
    f=lambda x,d=6:'—' if x is None else f'{x:.{d}f}'
    lines=['# 同じCI上限での強対照・RAACS・古典・半qubit比較','',
        f"全{len(rows)}行中{complete}完了、{missing}同一問題source欠測、{failed}数値計算失敗、{pending}未完了。R2の49 source/7欠測と、R1の12 source/R128結果を統合した。再利用は新しい測定や独立replicateとして数えない。",'',
        '全方式で元の物理軌道、R上限128、OOなし、同じC3有限0.1 Ha罰則・数値backend。値はphysical energyで、厳密S=0ではない。元のC2/C4やR64結果と混ぜて勝利を作らない。','']
    for case in load(ROOT/'protocol-R1-CI-CAP-SCREEN.json')['cases']:
        lines+=['## '+case,'','| 方法 | Energy Ha | Recovery % | CNOT | CNOT Depth | Total Depth | 実CI次元 | CI CPU秒 | 状態 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        selected=[r for r in rows if r['case']==case]
        for r in selected:
            lines.append('| '+' | '.join([r['method'],f(r['energy_Ha'],8),f(r['recovery_pct'],2),f(r['CNOT'],2),f(r['CNOT_depth'],2),f(r['total_depth'],2),str(r['actual_CI_dimension'] or '—'),f(r['incremental_CI_CPU_seconds'],3),r['status']])+' |')
        available=[r for r in selected if r['energy_Ha'] is not None]
        if available:
            best=min(available,key=lambda r:r['energy_Ha'])
            lines+=['',f"現時点の観測最小energyは{best['method']}の{best['energy_Ha']:.8f} Ha。これは事後の観測envelopeで、未完了/欠測や回路・構築費用を無視した「総合勝者」ではない。単一seedの順位は確認試験でもない。",'']
    lines+=['## Parameters・Shot・費用を正しく読む','',
        '- 表の—は未知/対象外/欠測であり、0ではない。古典CISDには量子samplerが無いが、相関状態を0ゲートで量子準備したという意味ではない。',
        '- 独立parameter rankは未認証。CSVではsource係数スロットとprimitiveゲート数値係数を別列にし、異なるproxyを一つのParameters削減率にしない。',
        '- 量子sourceの総予算は100k model Shot。B3/B4はpilot込みで、CIには元のindependent final countsだけを使う。final本数・pilot・構築費用をCSV/元sourceに保持する。F2/H1は同Shotでもfullの半分の片スピン配置本数。',
        '- R上限が同じでも、実CI次元が異なる。小さいsupportのsamplerは128配置に届かず、小さいCIを解いている。原著の大CI/HCI空間への優越ではない。',
        '- CI CPU秒は今回の数値段階だけ。初期化/CCSD/TN/探索の元費用はsourceへの参照へ残し、不明を0にしない。classical drawsとquantum model Shotは別。実QPU Shotは全て0。',
        '- CNOT/Depthは共通論理資源。ensembleはShot重み平均を含む。Rustiqの別合成結果や実機native 2Q数へ無断で置き換えない。',
        '- 数値計算の60秒上限に達した対照は、energyが悪いと判定したものではない。欠測を消して成功例だけで平均せず、追加診断は別記録・全費用込みにする。','',
        '現在の最良論文に総合的に勝ったという目標は未達。まずこの同条件比較を完了し、その結果から配置選択・sampler・回転圧縮のどこを改善するか決める。','',
        '[CI容量24セル試験](R1_CI_CAP_RESULTS_JA.md)、[半qubit比較](H1_F2_COMPARISON_JA.md)、[現在の包括報告](CURRENT_RESULTS_AND_PLAN_JA.md)。']
    write(ROOT/'analysis/R2_COMMON_CI_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(summary,flush=True)


if __name__=='__main__':main()
