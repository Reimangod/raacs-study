"""All D1 resources with independent saved-circuit recount and full limits."""
from io import BytesIO,StringIO
import csv
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .merged_compression import binding


def main():
    d=ROOT/'runs/D1-MERGED-RESOURCE';r=load(d/'result.json')
    assert r['freeze']==binding() and len(r['rows'])==12
    rows=[]
    for x in r['rows']:
        path=d/x['case']/str(x['max_layers'])
        for name,hash_ in x['QPY_SHA256'].items():assert sha256((path/(name+'.qpy')).read_bytes())==hash_
        q=qpy.load(BytesIO((path/'sampling.qpy').read_bytes()))[0];v=x['resources']['Z_sampling_equivalent']
        assert q.count_ops().get('cx',0)==v['cnot_count']
        assert q.depth(filter_function=lambda i:i.operation.name=='cx')==v['cnot_depth']
        assert q.depth()==v['total_unitary_depth']
        b=x['exact_optimized_source_resources']['Z_sampling_equivalent']
        rows.append({'case':x['case'],'max_layers':x['max_layers'],'original_CNOT':b['cnot_count'],
            'CNOT':v['cnot_count'],'CNOT_reduction_pct':100*(1-v['cnot_count']/b['cnot_count']),
            'original_CNOT_depth':b['cnot_depth'],'CNOT_depth':v['cnot_depth'],
            'original_total_depth':b['total_unitary_depth'],'total_depth':v['total_unitary_depth'],
            'numeric_primitive_coefficient_slots_not_rank':x['compiled_numeric_coefficient_slots_not_rank'],
            'initial_Slater_infidelity_not_final_state':x['part_errors_not_final_correlated_state_fidelity'][0]['infidelity'],
            'remaining_single_particle_unitary_infidelity':x['part_errors_not_final_correlated_state_fidelity'][1]['infidelity'],
            'compiler_CPU_seconds':x['CPU_seconds'],'molecular_energy_evaluations':0,'new_model_shots':0})
    out=StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/D1-all-resource-points.csv',out.getvalue().encode('utf-8-sig'))
    save(ROOT/'CHECKPOINT_MERGED_RESOURCE_AUDIT.json',{'checked_points':len(rows),'QPY_and_CX_depth_recounted':True,
        'original_frozen_sources_unchanged':True,'new_model_shots':0,'energy_verified':False,'goal_complete':False})
    lines=['# 最適化後に圧縮するD1：回路の無駄は減ったが、精度は未確定','',
        'D0で失っていた回転結合・Slater準備の最適化を先に適用した。全4条件×3層上限を固定して実行し、12回路のQPYとCNOT・depthを独立再計数した。native Givens圧縮を使う既知generic対照で、新規samplerではない。','',
        '| 条件 | 層上限 | CNOT 元→D1 | 削減 % | CNOT Depth 元→D1 | Total Depth 元→D1 | 初期Slater infidelity |','|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['case']} | {x['max_layers']} | {x['original_CNOT']}→{x['CNOT']} | {x['CNOT_reduction_pct']:.1f} | {x['original_CNOT_depth']}→{x['CNOT_depth']} | {x['original_total_depth']}→{x['total_depth']} | {x['initial_Slater_infidelity_not_final_state']:.6g} |")
    lines+=['','全12点で回路資源は減った。ただし、強い圧縮では初期Slaterだけでも大きく変わる。初期状態がほぼ直交した点を「精度保持」と呼ばない。逆にSlater忠実度だけでSQD最終精度の失敗とも断定しない。最終相関状態、sampling分布、CI energy、Recoveryはまだ評価していない。','',
        '次は全4条件の無圧縮点と3近似点を同じShot・CI・後処理で比較する。よい層だけを後から独立な提案手法として選ばず、全候補の費用を記録する。GPU実行は実際のprimitive回路を使い、high-level UCJが元状態をsimulateする経路を通さない。','',
        '[既知研究との境界](LITERATURE_BOUNDARY_UPDATE_JA.md)も参照。回路の圧縮率、数値テスト成功、新規性、総合的な論文優越は別の主張である。']
    write(ROOT/'analysis/D1_MERGED_RESOURCE_JA.md',('\n'.join(lines)+'\n').encode())
    print({'D1_points_recounted':12,'goal_complete':False})


if __name__=='__main__':main()
