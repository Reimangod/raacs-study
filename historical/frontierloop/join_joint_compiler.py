"""Exact-source join of N5 CI results and previously certified G5 sampler QPYs."""
import csv
from io import StringIO
from pathlib import Path
from .paths import ROOT
from .io import load,save,write,sha256
from .joint_ci_dispatch import LOCAL


def run():
    n5=load(ROOT/'analysis/N5-joint-CI-summary.json');spec=load(LOCAL/'protocol.json')
    if n5['status']!='ALL40_PLANNED_READOUTS_AUDITED' or n5['completed_readouts']!=40:raise RuntimeError('Complete N5 comparison required')
    audit=load(ROOT/'analysis/G5-final-report-audit.json')
    for path,value in audit['evidence_SHA256'].items():
        if sha256((ROOT/path).read_bytes())!=value:raise RuntimeError('Prior all-bank proof/report changed: '+path)
    frozen=load(ROOT/'freeze-G5-WHOLE-N20.json');g5=load(ROOT/'analysis/G5-whole-N20-summary.json')
    terminal=load(ROOT/'runs/G5-WHOLE-N20-v1/terminal.json')
    if sha256((ROOT/'runs/G5-WHOLE-N20-v1/compiled-members.sqlite').read_bytes())!=terminal['database_SHA256']:
        raise RuntimeError('Certified all3132 saved sampler circuits changed')
    if terminal['statuses']!={'CERTIFIED_RESOURCE_POINT':3132}:raise RuntimeError('Partial compiler bank cannot be joined')
    rows=[]
    for row in n5['rows']:
        if row['arm']!='JOINT_TOP_D16384':continue
        output=dict(row,resource_variant='ORIGINAL_SAMPLER',certificate_derived_old_counts_reuse=False,
            coefficient_slots_scope='ORIGINAL_SOURCE_SLOTS_NOT_COMPILED_ROTATION_COUNT_OR_INDEPENDENT_RANK')
        if row['method']=='C5_D_SQDRIFT_N20':
            plan=next(p for p in spec['plans'] if p['task_id']==row['task_id']);source=plan['source']
            f=next(f for f in frozen['sources'] if f['case']==row['case'])
            if source['payload_SHA256']!=f['source_counts_SHA256'] or source['source_result_SHA256']!=f['source_result_SHA256']:
                raise RuntimeError('Compiled sampler not from the same N5 observed source')
            if source['expected_problem_id']!=f['problem_id'] or spec['physical'][row['case']]['problem_id']!=f['problem_id']:
                raise RuntimeError('Compiled source physical problem differs from N5')
            for path,key in [('source_counts_path','source_counts_SHA256'),('source_result','source_result_SHA256'),('meta_path','meta_SHA256')]:
                if sha256(Path(f[path]).read_bytes())!=f[key]:raise RuntimeError('Actual compiled source changed')
            result=next(c for c in g5['cases'] if c['case']==row['case']);m=result['mean_resources']
            for key,field in [('CNOT','cnot_count'),('CNOT_Depth','cnot_depth'),('Total_Depth','total_unitary_depth')]:
                if abs(row[key]-m['NATIVE'][field])>1e-8:raise RuntimeError('Resource convention mismatch')
                output[key]=m['SELECTED'][field]
            output.update(resource_variant='G5_WHOLE_BANK_EXACT_IDEAL_SAMPLING_CERTIFIED',certificate_derived_old_counts_reuse=True)
        rows.append(output)
    if len(rows)!=20:raise RuntimeError('All4x5 joint endpoints required')
    save(ROOT/'analysis/N5-G5-bound-resource-comparison.json',dict(rows=rows,new_shots=0,new_energy_evaluations=0,
        new_resource_compilation=False,independent_confirmation=False,goal_complete=False,
        N5_summary_SHA256=sha256((ROOT/'analysis/N5-joint-CI-summary.json').read_bytes()),
        G5_allbank_proof_SHA256=sha256((ROOT/'analysis/G5-final-report-audit.json').read_bytes()),
        shared_G5_production_CPU_seconds=g5['fullbank_CPU_seconds_including_children'],
        shared_G5_audit_CPU_seconds=g5['recount_analysis_CPU_seconds']))
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'analysis/N5-G5-bound-resource-comparison.csv',stream.getvalue().encode('utf-8-sig'))
    names={'CLASSICAL_HF_ONLY':'古典HF-HCI','C5_C_LUCJ_COMPRESSED_L1':'圧縮LUCJ','C5_D_SQDRIFT_N20':'SqDRIFT20＋G5 compiler','RAACS_B4':'RAACS','CLASSICAL_CISD_100K_DRAWS':'古典CISD-HCI'}
    def fmt(v,dec=2):return '該当なし/未確定' if v is None else f'{v:.{dec}f}'
    lines=['# 同じsourceに結合した、現時点の回路・joint CI比較','',
        '新実験を追加せず、N5のjoint CI結果と、同じ保存Shot分布を厳密に保持すると全回路で検証済みのG5 compiler資源を結び付けた。再サンプリングした結果ではない。元source・problem ID・counts hash・全bank証拠を確認した。',
        'CI状態をこの回路が準備したという意味ではない。SQD/QSCI型の「sampling回路の費用と、そのサンプルに古典処理を行った最終精度」の比較である。','']
    for case in spec['physical']:
        lines += ['## '+case,'','| 方法 | Energy Ha | Recovery % | CNOT | CNOT Depth | Total Depth | 元source係数slot平均 | native＋CI CPU秒 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if r['case']==case]:
            lines.append('| '+' | '.join([names[r['method']],fmt(r['energy_Ha'],10),fmt(r['recovery_pct'],3),*[fmt(r[f]) for f in ['CNOT','CNOT_Depth','Total_Depth','coefficient_slots','native_plus_CI_standalone_numeric_CPU']]])+' |')
        interpretations={
            'NAPH20-BLOCK-BOYS':'圧縮LUCJが最も低いenergy。RAACSはLUCJより浅く少ないCNOTだが、energyでは約0.504mHa劣る。compiler後のSqDRIFT20よりRAACSの方が浅く、今回のjoint energyも良い。ただし構築費用や最強TN対照まで含む総合的な勝者ではない。',
            'N2-32-R1p2-P2':'古典CISD-HCIが最小energy。量子source同士ではRAACSが圧縮LUCJより約0.079mHa低く、CNOT/Depthも少ないという有望な開発点がある。一方、SqDRIFT20はRAACSより約0.073mHa低いenergyを持ち、単純な全指標優越にはならない。',
            'N2-32-R2p4-P2':'古典HF-HCIが最小energy。量子sourceでは精度差は小さく、RAACSは浅さで有利、SqDRIFT20はCNOTを減らせてもLUCJより深い。強い古典対照を除いて量子方式の必然性を主張しない。',
            'FE2S2-40-P2':'古典CISD-HCIが最小energy。RAACSは浅いが精度で劣る。SqDRIFT20のcompilerはCNOTを大きく下げる一方、LUCJよりCNOT Depthはまだ大きい。CNOTだけの勝ちを全資源の勝ちにしない。'}
        lines += ['',interpretations[case],'']
    lines += ['## 費用・Parameter・Shotに関する注意','',
        '- native＋CI CPUは各方式を単独で使う際の数値費用proxy。旧source構築・CCSD/TN・RAACS探索・実装・回収は別で、研究全体の総費用ではない。classical行は0 CNOTで最終相関状態を準備した例ではない。',
        '- 全quantum sourceは旧総予算100k model Shots。RAACSはpilotを含むためfinal Shotsは約85k～91k、LUCJ/SqDRIFTはfinal100k。total budgetは同じでもfinalだけでは同数ではない。actual QPU Shot0。classical drawsを量子Shotへ合算しない。',
        '- G5の共通compiler production284.98CPU秒と再計数監査199.19CPU秒等は別の共有campaign費用。20行へ繰り返し足さない。opt1/all-to-allの論理資源であり、実機routing/noiseとは異なる。G6 opt3 prefixだけの数字をwhole bankへ混ぜていない。',
        '- 表の係数slotは独立Parameter数ではない。G5後も独立rankは未証明で、都合よく0や削減済みへ変えていない。CSVの独立Parameter欄は未確定。',
        '- 新しいHCI探索・model/QPU Shot・回路compile・energy solveはこの結合処理では0。N5由来の新しいCI結果と、G5由来の過去compiler証拠を明確に区別した。',
        '- 元論文最高条件の完全再現、全方式の最良版、独立確認、未知分子一般化、総合的な優越性は未達。','']
    write(ROOT/'analysis/N5_G5_CURRENT_COMPARISON_JA.md','\n'.join(lines).encode())
    print('N5_G5_EXACT_SOURCE_JOIN_ALL20_NO_NEW_SHOTS_OR_CI',flush=True)


if __name__=='__main__':run()
