"""Independently recount saved D0 circuits; do not infer sampling fidelity."""
from io import BytesIO
import csv
from qiskit import qpy
from .paths import ROOT
from .io import load, save, write, sha256
from .native_compression import binding


def main():
    directory = ROOT/'runs/D0-NATIVE-RESOURCE'
    r = load(directory/'result.json')
    assert r['freeze'] == binding()
    assert len(r['rows']) == 12 and r['new_model_shots'] == 0
    out = []
    for row in r['rows']:
        d = directory/row['case']/str(row['max_layers_per_orbital'])
        for name, expected in row['QPY_SHA256'].items():
            assert sha256((d/(name+'.qpy')).read_bytes()) == expected
        qc = qpy.load(BytesIO((d/'sampling.qpy').read_bytes()))[0]
        q = row['resources']['Z_sampling_equivalent']
        assert qc.count_ops().get('cx', 0) == q['cnot_count']
        assert qc.depth(filter_function=lambda x: x.operation.name == 'cx') == q['cnot_depth']
        assert qc.depth() == q['total_unitary_depth']
        b = row['optimized_exact_source_resources']['Z_sampling_equivalent']
        out.append({'case':row['case'], 'layers':row['max_layers_per_orbital'],
            'source_CNOT':b['cnot_count'], 'CNOT':q['cnot_count'],
            'CNOT_reduction_pct':100*(1-q['cnot_count']/b['cnot_count']),
            'source_CNOT_depth':b['cnot_depth'], 'CNOT_depth':q['cnot_depth'],
            'source_total_depth':b['total_unitary_depth'], 'total_depth':q['total_unitary_depth'],
            'numeric_primitive_coefficient_slots_not_rank':row['compiled_numeric_gate_coefficient_slots'],
            'fit_and_compilation_CPU_seconds':row['fit_and_compilation_CPU_seconds'],
            'largest_single_particle_HS_infidelity':max(row['orbital_single_particle_HS_infidelities']),
            'molecular_energy_evaluated':False, 'whole_state_quality_verified':False})
    stream = __import__('io').StringIO(); writer = csv.DictWriter(stream, fieldnames=list(out[0]))
    writer.writeheader(); writer.writerows(out)
    write(ROOT/'analysis/D0-all-resource-points.csv', stream.getvalue().encode('utf-8-sig'))
    save(ROOT/'CHECKPOINT_NATIVE_RESOURCE_AUDIT.json', {'status':'12_SAVED_PHYSICAL_CIRCUITS_RECOUNTED',
        'QPY_SHA256_checked':True, 'source_and_code_binding_checked':True,
        'new_model_shots':0, 'molecular_energy_evaluations':0, 'goal_complete':False})
    lines = ['# native LUCJ圧縮：まず実回路の費用を点検', '',
        '全4条件×3水準を固定して試したcompiler preflight。既知ffsimのGivens圧縮を使ったもので、新しい科学的手法や精度保持の実証ではない。実分子energy評価、CI、OO、新規Shotは0。', '',
        '| 条件 | 回転の層上限 | 元CNOT→近似CNOT | CNOT削減 % | 元CNOT Depth→近似 | 元Total Depth→近似 | 合成CPU秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in out:
        lines.append(f"| {x['case']} | {x['layers']} | {x['source_CNOT']} → {x['CNOT']} | {x['CNOT_reduction_pct']:+.1f} | {x['source_CNOT_depth']} → {x['CNOT_depth']} | {x['source_total_depth']} → {x['total_depth']} | {x['fit_and_compilation_CPU_seconds']:.2f} |")
    lines += ['',
        '削減率が負ならgateが増えた。元回路にはPRE_INITによる回転結合とSlater準備最適化を認めている。一方、この単純近似は各回転を先にprimitiveへ展開するため、その後の回転結合・特殊な準備最適化を使えない。近似すれば自動的に資源が良くなる、という考えは誤り。', '',
        '次版は最適化済みの回転結合・Slater準備を保持してから圧縮する。単一粒子行列の誤差は多電子stateやsampling/CI energyの保証ではない。資源が減った点も、まだ精度が保たれた成功回路とは数えない。', '',
        '高水準UCJ/OrbitalRotationをそのままffsimでsimulateすると近似上限を反映しない経路があることをsmall testで確認した。保存した近似primitive回路をsamplingにも使う。既存の未近似データが不正という意味ではない。', '',
        '[公式native compression](https://qiskit-community.github.io/ffsim/dev/how-to-guides/compress-orbital-rotation.html)に基づく既知対照。全12回路はQPY hashと実CNOT/depthを再計数して検証した。未収束な近似fitterの可能性を残し、maxiter25の出力を最良近似や収束解と呼ばない。']
    write(ROOT/'analysis/D0_NATIVE_RESOURCE_JA.md', ('\n'.join(lines)+'\n').encode())
    print({'resource_points_verified':len(out), 'new_model_shots':0, 'goal_complete':False})


if __name__ == '__main__': main()
