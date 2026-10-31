"""Derived N53 evidence audit; never submits or reruns numerical work."""
import math
from .structured_CI_gpu_study import LOCAL,verified
from .paths import ROOT
from .io import load,save,write,sha256


def audit():
    p=verified();collection=load(LOCAL/'collection.json');base=LOCAL/'collected'
    for name,digest in collection['files'].items():
        if sha256((base/name).read_bytes())!=digest:raise ValueError('Collected evidence changed')
    t=load(base/'out/terminal.json')
    assert t['status']=='ALL_CUDA_ACTION_QUALIFICATIONS_COMPLETED'
    assert t['protocol_SHA256']==sha256((LOCAL/'protocol.json').read_bytes())
    assert 'A100' in t['device_name'] and t['cupy']==p['expected_cupy']
    assert str(t['job_id'])==str(load(LOCAL/'job.json')['job_id'])
    assert len(t['rows'])==len(p['cases'])==8
    rows=[]
    for c,r in zip(p['cases'],t['rows']):
        assert c['id']==r['case_id'] and [c['R'],c['D']]==[r['R'],r['D']]
        assert r==load(base/'out'/c['id']/'result.json')
        assert r['source_problem']==c['source_problem']
        assert r['status']=='QUALIFIED_ACTION_ONLY' and r['new_CI_eigensolves']==0
        assert all(math.isfinite(v) and 0<=v<=p['action_tolerance'] for v in r['errors'].values())
        factors=c['CPU_reference']['math']['retained_factor_count']
        assert r['retained_factor_count']==factors and r['ERI_factor_cutoff'] is None
        assert r['no_dense_D_by_D_H']
        rows.append(dict(case=c['id'],R=c['R'],D=c['D'],max_error=max(r['errors'].values()),
            CPU_H_seconds=c['CPU_reference']['dense_H_CPU_seconds'],
            GPU_H_wall_seconds=[x['wall_seconds'] for x in r['dense_H_timings']],
            GPU_H_event_seconds=[x['GPU_event_seconds'] for x in r['dense_H_timings']],
            setup_wall_seconds=r['setup_wall_seconds'],
            memory_pool_reserved_bytes=r['peak_observed_memory_pool_reserved_bytes']))
    assert all(t[k]==0 for k in ['new_model_samples','new_QPU_shots','new_CI_eigensolves'])
    result=dict(status='N53_COLLECTED_ACTION_EVIDENCE_VERIFIED',rows=rows,
        job_id=t['job_id'],device=t['device_name'],whole_worker_wall_seconds=t['whole_wall_seconds'],
        whole_worker_CPU_seconds=t['whole_CPU_seconds'],accounting=collection['accounting'],
        full_pipeline_speedup_certified=False,new_energy_results=0,goal_complete=False)
    save(ROOT/'analysis/N53-CUDA-action-audit.json',result)
    lines=['# N53：実A100でのCI行列作用の検証','',
        'これは計算backendの検証であり、Ansatzの精度・CNOT改善や全体目標の達成ではありません。','',
        '| 対象 | R | CI次元 | 最大照合誤差 | CPU H作用 秒 | GPU H作用 wall 秒 | 準備wall 秒 |',
        '|---|---:|---:|---:|---|---|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['R']} | {r['D']} | {r['max_error']:.3g} | {r['CPU_H_seconds']} | {r['GPU_H_wall_seconds']} | {r['setup_wall_seconds']:.3f} |")
    lines+=['','同じN51空間、全符号付きERI因子、float64、HとS²を維持しました。',
        '小系は全ベクトル、大系は疎入力の全出力と密入力の成分・ノルム・双線形量を独立CPU実装と照合しています。',
        '表のCPUはM4 process CPU秒、GPUは共有CUDA context下の作用wall秒です。cold起動・通信・固有値反復全体の高速化比ではありません。',
        '新しいCI固有解・量子Shotは0。次に固有値solverを別に資格試験し、欠測energyを解消する必要があります。','']
    write(ROOT/'analysis/N53_CUDA_ACTION_RESULTS_JA.md','\n'.join(lines).encode())
    print(result['status'],len(rows),flush=True)


if __name__=='__main__':audit()
