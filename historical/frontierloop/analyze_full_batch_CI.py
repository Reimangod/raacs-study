"""Partial-aware N51 report; no scientific kernel execution or root selection."""
from .paths import ROOT
from .io import load,save,write,sha256
from .full_batch_CI_study import LOCAL,verified
from .analyze import reference


def run(scope):
    if scope not in ['M4','ALL']:raise ValueError('Explicit scope')
    p=verified();audit=load(ROOT/'analysis'/f'N51-{scope}-root-audit.json')
    batch_inventory=load(ROOT/'runs/N48-FE-READOUT-CONTRACT-v1/terminal.json')
    rows=[];hf,ref=reference('FE2S2-40-P2')
    for i,task in enumerate(p['tasks']):
        if scope=='M4' and i!=0:continue
        folder=LOCAL/('out' if i==0 else 'collected/out')/task['task_id']
        out=dict(method=task['method'],R=task['R'],D=task['D'],source=task['source'],source_cost=task['source_cost'],
                 reference_kind='SCI_NOT_FCI',reference_precision='UNVERIFIED',
                 total_parent_model_samples_reused=100000,requested_batch_unique_strings=4000,
                 actual_batch_unique_strings=min(4000,next(s['distinct_parent_full_strings'] for s in batch_inventory['rows'] if s['method']==task['method'])),
                 numerical_energy_Ha=None,correlation_energy_Recovery_pct=None,body_CPU_seconds=None)
        resultfile=folder/'result.json';failure=folder/'process-failure.json'
        if not resultfile.exists():
            out['status']=load(failure)['status'] if failure.exists() else 'PENDING_NOT_MISSING'
            if failure.exists():
                incident=load(failure)
                out['censoring_reason']='CPU_LIMIT_SIGXCPU' if incident.get('returncode')==-24 else 'OTHER_PROCESS_OR_WALL_LIMIT'
                out['cap_evidence']=incident
        else:
            r=load(resultfile);out.update(status=r['status'],body_CPU_seconds=r.get('CPU_seconds'),result=r)
            if r['status']=='COMPLETED':
                a=next(a for a in audit['rows'] if a['task_id']==task['task_id'])
                out.update(numerical_energy_Ha=a['energy_Ha'],correlation_energy_Recovery_pct=100*(hf-a['energy_Ha'])/(hf-ref),
                           spin_square=a['spin_square'],saved_root_residual=a['residual_norm'],audit_CPU_seconds=a['CPU_seconds'])
        rows.append(out)
    result=dict(scope=scope,rows=rows,new_model_samples=0,new_QPU_shots=0,
         completed_source_roots=sum(r['status']=='COMPLETED' for r in rows),
         independent_whole_runs_per_source=1,selected_batches_per_parent=1,
         independent_confirmation=False,full_paper_reproduction=False,matched_cold_work=False,
         qualification_roots_are_separate_overhead=True,goal_complete=False,
         root_audit_SHA256=sha256((ROOT/'analysis'/f'N51-{scope}-root-audit.json').read_bytes()))
    partitions=[load(LOCAL/'out/supervisor-M4-terminal.json')]
    if scope=='ALL':
        partitions += [load(LOCAL/'collected/out'/('supervisor-'+name+'-terminal.json')) for name in ['lane0','lane1']]
        result['remote_accounting']=load(LOCAL/'collection.json')['accounting']
    result['measured_children_CPU_seconds_including_caps']=sum(z['children_CPU_seconds'] for z in partitions)
    result['worker_qualification_CPU_seconds']=sum(z['qualification']['CPU_seconds'] for z in partitions)
    result['worker_qualification_small_selected_space_roots']=2*len(partitions)
    result['new_large_source_root_attempts']=len([r for r in rows if r['status']!='PENDING_NOT_MISSING'])
    save(ROOT/'analysis'/f'N51-{scope}-summary.json',result)
    lines=[f'# N51 {scope}：固定full-batchでのFe₂S₂比較','',
           '保存済み100k samples・SDK unique-weightedのbatch0を固定した開発pilot。要求は4000種類だが、SqDRIFT5は全89種類、SqDRIFT20は全803種類が返る。新しい量子Shotなし。各source1 parent・1 batchであり、平均・標準偏差や独立確認ではない。', '',
           '| 方法 | 状態 | energy Ha | 相関energy Recovery % | actual R | actual D | root body CPU秒 |',
           '|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'未取得' if v is None else f'{v:.9f}'
    for r in rows:lines.append(f"| {r['method']} | {r['status']} | {fmt(r['numerical_energy_Ha'])} | {fmt(r['correlation_energy_Recovery_pct'])} | {r['R']} | {r['D']} | {fmt(r['body_CPU_seconds'])} |")
    lines+=['','## 回路と構築費用（旧sourceの記録）','',
            '| 方法 | logical CNOT平均 | CNOT Depth平均 | Total Depth平均 | 係数slot平均 | 既知初期化CPU小計秒 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        s=r['source'];z=s['sampler_resources'];c=r['source_cost']
        lines.append(f"| {r['method']} | {z['cnot_count']:.4f} | {z['cnot_depth']:.4f} | {z['total_unitary_depth']:.4f} | {s['sampler_coefficient_slots_mean']:.4f} | {fmt(c.get('known_disjoint_initializer_CPU_seconds'))} |")
    lines+=['','回路を今回変更した結果ではない。既知CDF初期化はTNと共有しており、表の小計を二重加算しない。SqDRIFTの値はensemble内のShot重み付き平均であり、一個の同じ回路を反復したLUCJと同じ意味の独立parameter数ではない。全cold費用・独立parameter rankは未証明。', '',
            '上表は旧logical資源の記録である。SqDRIFT20には別途Rustiqの改善資源も存在するため、未改善の回路列だけを用いて総合優位を主張しない。CPU秒はM4と遠隔CPUでhardwareが異なる。今回のremote allocationはGPU0である。', '',
            '全方法はH+0.1S²を解き、その根から物理H energyを出した。厳密S=0ではない。FeのRecovery参照はSCIで、FCI・化学精度保証ではない。CIの実R/Dは手法間で違うため、この表だけで同古典費用の優越とはしない。', '',
            'RAACS、完全なauthor-native TN、別CI予算の古典HCI対照はこの4source pilotに含めていない。元の広い研究目標をこの表だけで達成したとは認定しない。上限停止とpendingはenergyの悪化へ置換しない。']
    lines+=['',f"上限停止を含む子processの実測CPU小計 {result['measured_children_CPU_seconds_including_caps']:.6f}秒、worker内の小問題qualification {result['worker_qualification_CPU_seconds']:.6f} CPU秒（別枠）。この他のtest・監査・準備・転送を含む完全cold総量とはしない。失敗cellの個別CPU欄がunknownでも、supervisorのrusageで失敗費用を含む小計は保全している。"]
    write(ROOT/'analysis'/f'N51_{scope}_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print('N51_REPORT',scope,result['completed_source_roots'],[(r['method'],r['status']) for r in rows])


if __name__=='__main__':
    import sys;run(sys.argv[1])
