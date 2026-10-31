"""Combine six preserved passes and two explicit-input successor passes."""
import math
from .structured_CI_gpu_successor import LOCAL,OLD_LOCAL
from .paths import ROOT
from .io import load,save,write,sha256


def audit():
    ps=[]
    for root in [OLD_LOCAL,LOCAL]:
        p=load(root/'protocol.json');ps.append(p)
        for file,h in p['input_SHA256'].items():assert sha256((root/file).read_bytes())==h
        collected=load(root/'collection.json')
        for file,h in collected['files'].items():assert sha256((root/'collected'/file).read_bytes())==h
    p1,p2=ps;f=load(OLD_LOCAL/'collected/out/failure.json');t=load(LOCAL/'collected/out/terminal.json')
    assert f['protocol_SHA256']==sha256((OLD_LOCAL/'protocol.json').read_bytes())
    assert t['protocol_SHA256']==sha256((LOCAL/'protocol.json').read_bytes())
    assert t['status']=='ALL_CUDA_ACTION_QUALIFICATIONS_COMPLETED'
    assert f['error']=="ValueError('CPU generated profiling-vector bytes differ')"
    assert sha256((OLD_LOCAL/'collected/out/failure.json').read_bytes())==p2['predecessor_failure_SHA256']
    assert len(f['rows'])==6 and len(t['rows'])==2
    for before,after in zip(p1['cases'][6:],p2['cases']):
        assert {k:v for k,v in before.items() if k!='SHA256'}=={k:v for k,v in after.items() if k not in ['SHA256','original_v1_packet_SHA256']}
    rows=f['rows']+t['rows'];contexts=[f,t]
    for ctx,root in zip(contexts,[OLD_LOCAL,LOCAL]):
        assert str(ctx['job_id'])==str(load(root/'job.json')['job_id'])
        assert 'A100' in ctx['device_name'] and ctx['cupy']==p1['expected_cupy']
        assert ctx['new_model_samples']==ctx['new_CI_eigensolves']==0
    for i,(c,r) in enumerate(zip(p1['cases'],rows)):
        root=OLD_LOCAL if i<6 else LOCAL
        assert r==load(root/'collected/out'/c['id']/'result.json')
        assert c['id']==r['case_id'] and [c['R'],c['D']]==[r['R'],r['D']]
        assert r['source_problem']==c['source_problem'] and r['new_CI_eigensolves']==0
        assert r['status']=='QUALIFIED_ACTION_ONLY' and r['no_dense_D_by_D_H']
        assert r['ERI_factor_cutoff'] is None
        assert r['retained_factor_count']==c['CPU_reference']['math']['retained_factor_count']
        assert all(math.isfinite(v) and 0<=v<=p1['action_tolerance'] for v in r['errors'].values())
    out=dict(status='N53_ALL8_ACTION_CASES_VERIFIED_WITH_PRESERVED_V1_INCIDENT',rows=rows,
        jobs=[x['job_id'] for x in contexts],old_failure_preserved=True,old6_cases_not_repeated=True,
        new_CI_eigensolves=0,new_model_samples=0,new_QPU_shots=0,
        full_pipeline_speedup_certified=False,goal_complete=False,
        worker_wall_seconds_including_failure=f['wall_seconds']+t['whole_wall_seconds'],
        accounting=[load(r/'collection.json')['accounting'] for r in [OLD_LOCAL,LOCAL]])
    save(ROOT/'analysis/N53-CUDA-action-successor-audit.json',out)
    lines=['# N53：実A100の数値照合結果','',
        '8ケースを照合しました。初回6成功は再実行せず、入力のplatform差で止まった2ケースだけを別versionで完了しました。',
        'MacとLinuxで生成した乱数入力のbyte不一致は保全し、元のfloat64値を転送して修正しました。計算式・誤差許容値は不変です。','',
        '| 対象 | CI次元 | 最大照合誤差 | M4 CPU H作用 秒 | A100 H作用 wall 秒 | A100 S²作用wall 秒 | GPU準備wall 秒 |',
        '|---|---:|---:|---|---|---:|---:|']
    for r in rows:
        cpu=r['CPU_reference']['dense_H_CPU_seconds'];gpu=[x['wall_seconds'] for x in r['dense_H_timings']]
        lines.append(f"| {r['case_id']} | {r['D']} | {max(r['errors'].values()):.3g} | {cpu} | {gpu} | {r['dense_S2_timing']['wall_seconds']:.6f} | {r['setup_wall_seconds']:.3f} |")
    lines+=['','小系は全出力、大系は疎入力の全出力と密入力の32成分・ノルム・双線形量を独立CPUと照合。',
        '全signed ERI因子を保持し、同じN51空間・float64・H/S²/定数項を維持。',
        'M4 process CPUとA100共有contextの作用wallは測定範囲が異なり、全solverやcold起動・通信込みの高速化比ではありません。',
        'CI固有値は未計算。新しいenergy、Recovery、CNOT削減は得ておらず、全研究goalは未達です。',
        '次は同じ空間で最低根solverの数値・残差・実費用を別に検証し、2件の欠測energyを埋める段階です。','']
    write(ROOT/'analysis/N53_CUDA_ACTION_RESULTS_JA.md','\n'.join(lines).encode())
    print(out['status'],len(rows))


if __name__=='__main__':audit()
