"""N68 raw QPY and guard audit; compiler feasibility is not source quality."""
from io import BytesIO
from pathlib import Path
import gzip
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .exact_group_synthesis import LOCAL,choose,FIELDS
from .pair_group_preflight import metrics

def run():
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json');qual=load(LOCAL/'qualification.json')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h
    assert sha256((LOCAL/'qualification.json').read_bytes())==p['qualifier_SHA256']
    assert len(qual['rows'])==6 and max(r['whole_unitary_error'] for r in qual['rows'])<1e-10
    assert len(t['rows'])==12 and t['CPU_seconds']<p['CPU_cap'] and t['wall_seconds']<p['wall_cap']
    controls=load(ROOT/'analysis/N67-pair-group-resource-summary.json')['controls'];rows=[];files=set();improved=0
    for r in t['rows']:
        assert r['selected']==choose(r['baseline'],r['points'])
        for point in r['points']:
            for role,key in [('preparation','preparation_resources'),('sampling','resources')]:
                f=point['files'][role];blob=Path(f['file']).read_bytes();assert sha256(blob)==f['SHA256']
                q=qpy.load(BytesIO(gzip.decompress(blob)))[0];assert metrics(q)==point[key];files.add(f['file'])
        old=r['baseline']['resources'];new=r['selected']['resources']
        assert all(new[k]<=old[k] for k in FIELDS)
        strict=all(new[k]<old[k] for k in FIELDS);improved+=int(strict)
        candidates=[dict(point,method=c['method']) for c in controls if c['case']==r['case'] and c['status']=='AVAILABLE' for point in c['recounted_points']]
        base=min(candidates,key=lambda x:tuple(x['resources'][k] for k in FIELDS))
        rows.append(dict(case=r['case'],time=r['time'],original=old,selected=new,selected_compiler=r['selected']['compiler'],level=r['selected']['level'],
            reference_LUCJ=base['resources'],reference_method=base['method'],strict_all3_improvement_over_original=strict,
            resource_only_below_reference=all(new[k]<base['resources'][k] for k in FIELDS),
            independent_parameter_rank=None,energy_Ha=None,Recovery_pct=None))
    assert len(files)==96
    s=dict(status='ALL12_EXACT_GROUP_SYNTHESIS_RESOURCE_POINTS_AUDITED',rows=rows,new_QPY_recounts=len(files),strict_all3_improved=improved,
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],qualification_CPU_seconds=qual['CPU_seconds'],
        whole_unitary_max_small_error=max(r['whole_unitary_error'] for r in qual['rows']),
        matched_LUCJ_compiler_opportunity=False,molecular_large_state_verified=False,
        new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,independent_confirmation=False,goal_complete=False)
    save(ROOT/'analysis/N68-exact-group-synthesis-summary.json',s)
    lines=['# N68：sourceを変えず、groupの合成方法だけで深さを下げた','',
        '全12点を同じ角度・group・pair初期化で再合成した。小4/6/8qubitの全unitary・hybrid複素状態の6検査を通過し、96実QPYを再計数した。全modeを保持し、HF固定占有のprojectionは使用していない。',
        '3資源が元のgeneric分解より非増加の候補だけを採用する同一guardを使い、実際に全12点で3資源とも厳密に減った。これは既存Rustiqの合成効果で、新しい量子アルゴリズムそのものではない。','',
        '| 条件 | 時間 | 元 CNOT/Depth/Total | 採用 CNOT/Depth/Total | LUCJ参照 CNOT/Depth/Total | 採用compiler |',
        '|---|---:|---|---|---|---|']
    show=lambda d:'/'.join(str(d[k]) for k in FIELDS)
    for r in rows:lines.append(f"| {r['case']} | {r['time']} | {show(r['original'])} | {show(r['selected'])} | {show(r['reference_LUCJ'])} | {r['selected_compiler']}/opt{r['level']} |")
    lines+=['','## 何が効いたのか','',
        'pairだけの深さはもともと小さかった。one-groupを別々のPauli指数として分解した際の直列CNOTが大きかったが、同じunitaryをnetworkとしてまとめると深さが下がった。今回の資源差はsourceの精度向上や新しい化学情報を加えた効果ではない。',
        'これで有限Shotの精度を調べる資源上の見込みは生まれた。しかしLUCJへ同じ追加合成機会をまだ与えていないため、表の参照値に比べて小さいことをmatched-compiler優越とは呼ばない。',
        '次は相関sourceの大きい分子での数値的な状態再構成を、paired encodingとone-groupの小軌道から二経路で確認する。40qubit全Fock statevectorは作らず、新energy/Shotへ進む前に限定検査する。',
        '独立parameter rankは依然不明。primitive係数slotは表現に依存する。分子energy・Recovery・同Shot/Rの性能・noise下の優越は未測定で、最高論文超えは未達。',
        f"実費：resource body CPU {t['CPU_seconds']:.6f}秒/wall {t['wall_seconds']:.6f}秒、別の小unitary qualifier CPU {qual['CPU_seconds']:.6f}秒。新Shot/CI/GPUは0。元N67の費用と不利な結果は保存した。",'']
    write(ROOT/'analysis/N68_EXACT_GROUP_SYNTHESIS_RESULTS_JA.md','\n'.join(lines).encode())
    print('N68_AUDITED96',improved,flush=True)

if __name__=='__main__':run()
