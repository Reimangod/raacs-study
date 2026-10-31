"""N67 audit/report; resource feasibility is not an energy result."""
from io import BytesIO
from pathlib import Path
import gzip
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .pair_group_preflight import LOCAL,metrics

FIELDS=['CNOT','CNOT_Depth','Total_Depth']
def choose(points):return min(points,key=lambda p:tuple(p['resources'][k] for k in FIELDS))

def run():
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json');qual=load(LOCAL/'qualification.json')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h
    assert sha256((LOCAL/'qualification.json').read_bytes())==p['qualifier_SHA256']
    assert len(t['rows'])==12 and len(t['controls'])==12
    assert t['CPU_seconds']<p['CPU_cap'] and t['wall_seconds']<p['wall_cap']
    files=set();controls=[]
    for r in t['rows']:
        for role in ['pair','HF_GROUP','HYBRID']:
            for point in r[role]:
                for kind,expected in [('preparation',point['preparation_resources']),('sampling',point['resources'])]:
                    f=point['files'][kind];blob=Path(f['file']).read_bytes();assert sha256(blob)==f['SHA256']
                    q=qpy.load(BytesIO(gzip.decompress(blob)))[0];assert metrics(q)==expected;files.add(f['file'])
    for r in t['controls']:
        if r['status']=='AVAILABLE':
            for point,actual in zip(r['points'],r['recounted_points']):
                q=qpy.load(BytesIO(Path(point['file']).read_bytes()))[0];assert metrics(q)==actual['resources']
        controls.append(r)
    assert len(files)==112
    rows=[]
    for r in t['rows']:
        candidates=[dict(point,method=c['method']) for c in controls if c['case']==r['case'] and c['status']=='AVAILABLE' for point in c['recounted_points']]
        baseline=choose(candidates);hybrid=choose(r['HYBRID']);pair=choose(r['pair']);single=choose(r['HF_GROUP'])
        rows.append(dict(case=r['case'],time=r['time'],amplitude_kind=r['amplitude_kind'],baseline=baseline,hybrid=hybrid,pair=pair,HF_GROUP=single,
            hybrid_all3_not_increased=all(hybrid['resources'][k]<=baseline['resources'][k] for k in FIELDS),
            independent_parameter_rank=None,energy_Ha=None,Recovery_pct=None,
            reduction_pct={k:100*(baseline['resources'][k]-hybrid['resources'][k])/baseline['resources'][k] for k in FIELDS}))
    s=dict(status='ALL12_HYBRID_RESOURCE_POINTS_AUDITED',rows=rows,controls=controls,actual_new_QPY_recounts=len(files),old_control_QPY_recounts=20,
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],qualifier_CPU_seconds=qual['CPU_seconds'],
        new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,molecular_large_state_verified=False,goal_complete=False,
        partial_resource_promise=any(r['hybrid_all3_not_increased'] for r in rows),not_a_matched_energy_performance_comparison=True)
    save(ROOT/'analysis/N67-pair-group-resource-summary.json',s)
    lines=['# N67：pair相関のある出発点とone-groupの合成検査','',
        '小3fixtureで複素状態・粒子数・2種compiler後の検査を通過し、全4分子条件×3時間の12回路を構築した。新Shot・CI・FCI・optimizer・GPU allocationは0。',
        '元mode順で合成し、HF専用projectionや単純なfermionic mode permutationを流用していない。複素状態の最大誤差は2.23e-15以下。位相を変えると後続groupの測定確率が変わる回帰テストも通過した。',
        '以下のportfolioは全てCNOT→CX Depth→Total Depthの順でopt1/opt3から選択。同じ資源選択規則だが、energy/Shotは未比較であり、資源だけの点検である。','',
        '| 条件 | 時間 | pair CNOT/Depth/Total | HF+group CNOT/Depth/Total | hybrid CNOT/Depth/Total | LUCJ control CNOT/Depth/Total | 3資源とも非増加 |',
        '|---|---:|---|---|---|---|---|']
    show=lambda p:'/'.join(str(p['resources'][k]) for k in FIELDS)
    for r in rows:lines.append(f"| {r['case']} | {r['time']} | {show(r['pair'])} | {show(r['HF_GROUP'])} | {show(r['hybrid'])} | {show(r['baseline'])} | {r['hybrid_all3_not_increased']} |")
    lines+=['','## 判断','',
        'pairだけのCX Depthは11/23/23/31と小さいが、one-groupを素朴なPauli指数の列として分解すると深さが大きくなる。今回の入力集合では、3資源が同時に小さいhybrid候補は限定的だった。深さの問題を残したまま大量Shotへは進めない。',
        'この結果はPauli列という表現での現在のコンパイル結果で、同じunitaryの最小深さ・最小CNOTの下限ではない。次は既存の厳密なRustiq Pauli-network合成で同じunitaryを浅くできるか、同じ12点だけ限定比較する。新しいsourceや角度探索は足さない。',
        '各時間では別のgroupもサンプリングされているため、資源差を時間だけの効果と解釈しない。良い時間/分子だけを今後のsource policyに採用したとはしない。',
        'LUCJは同problem・同generic compilerの10利用可能sourceから20実QPYを再計数し、2欠測を保持。古いコンパイル費用・元の初期化費用は今回の実行費用0と偽らない。全元論文最適化の再現ではない。',
        '資源表のprimitive係数slotは独立parameter rankではない。独立rank、分子energy、Recovery、同Shot/Rでの性能は未証明。',
        f"実費：resource body CPU {t['CPU_seconds']:.6f}秒/wall {t['wall_seconds']:.6f}秒、別の小fixture CPU {qual['CPU_seconds']:.6f}秒。112新QPYと20旧対照を再計数。import前と古い源泉計算の完全cold費用は含まれない。",'']
    write(ROOT/'analysis/N67_PAIR_GROUP_PREFLIGHT_RESULTS_JA.md','\n'.join(lines).encode())
    print('N67_AUDITED112_20',flush=True)

if __name__=='__main__':run()
