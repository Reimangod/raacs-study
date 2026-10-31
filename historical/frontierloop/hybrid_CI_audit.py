"""Independent CPU reconstruction of N54 saved GPU roots, no eigensolves."""
from time import process_time,perf_counter
from pathlib import Path
import numpy as np
from pyscf.fci import selected_ci,direct_spin1
from .hybrid_CI_gpu_study import LOCAL,N51,verified
from .energy_cache import hole_integrals
from .hole_rdm import electron_rdms
from .portable_penalized_ci import expectation
from .analyze import reference
from .paths import ROOT
from .io import load,save,write,sha256
REPORT_TAG='N54'


def one(p,c,result,folder):
    cpu,wall=process_time(),perf_counter();old=load(N51/'protocol.json')
    task=next(t for t in old['tasks'] if t['task_id']==c['N51_task_id'])
    for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
        assert sha256(Path(task['source'][name]).read_bytes())==task['source'][key]
    assert result['case_id']==c['id'] and result['protocol_SHA256']==sha256((LOCAL/'protocol.json').read_bytes())
    assert result['root_SHA256']==sha256((folder/'root.npz').read_bytes())
    with np.load(folder/'root.npz',allow_pickle=False) as a:r={k:a[k] for k in a.files}
    s=np.sort(((1<<20)-1)^np.asarray(task['strings'],np.int64));ne=(5,5)
    assert np.array_equal(r['solver_strings'],s) and np.array_equal(r['electron_strings'],task['strings'])
    assert r['solver_nelec'].tolist()==[5,5] and r['electron_nelec'].tolist()==[15,15]
    assert str(r['problem_id'])==old['problem_id'] and str(r['task_id'])==task['task_id']
    assert int(r['norb'])==20 and bool(r['particle_hole_used'])
    v=r['coefficients'].reshape(len(s),len(s));assert abs(float(np.vdot(v,v))-1)<1e-10
    with np.load(N51/'integrals.npz',allow_pickle=False) as a:h,g=a['h1'],a['eri']
    hh,gg,core=hole_integrals(h,g,old['core']);solver=selected_ci.SelectedCI();solver.verbose=0
    ci=selected_ci._as_SCIvector(v,(s,s));links=selected_ci._all_linkstr_index((s,s),20,ne)
    h2=direct_spin1.absorb_h1e(hh,gg,20,ne,.5)
    hv=selected_ci.contract_2e(h2,ci,20,ne,link_index=links);sv=solver.contract_ss(ci,20,ne)
    energy=float(np.vdot(v,hv)+core);spin=float(np.vdot(v,sv));obj=energy+.1*spin
    residual=float(np.linalg.norm(hv+.1*sv-(obj-core)*v))
    d1,d2=solver.make_rdm12(ci,20,ne);e1,e2=electron_rdms(d1,d2)
    independent=expectation(h,g,old['core'],e1,e2)
    errors={k:abs(x-result['row'][k]) for k,x in [('energy_Ha',energy),('penalized_energy_Ha',obj),('spin_square',spin)]}
    errors.update(electron_RDM_energy_Ha=abs(independent-energy),particle_number=abs(float(np.trace(e1))-30),
        pair_number=abs(float(np.einsum('pprr->',e2))-870))
    if max(errors.values())>1e-7 or residual>1e-8:raise ValueError('Independent CPU saved root audit failed')
    assert result['row']['common_equation']==p['equation'] and result['row']['physical_H_calls']<=p['max_H_actions']
    return dict(method=c['id'],task_id=c['N51_task_id'],energy_Ha=energy,spin_square=spin,
        residual_norm=residual,R=c['R'],D=c['D'],errors=errors,CPU_seconds=process_time()-cpu,
        wall_seconds=perf_counter()-wall,root_SHA256=result['root_SHA256'])


def run():
    p=verified();collection=load(LOCAL/'collection.json');base=LOCAL/'collected'
    for file,h in collection['files'].items():assert sha256((base/file).read_bytes())==h
    audits=[];coverage=[];terminals=[]
    for lane,c in p['production'].items():
        terminal=load(base/'out'/lane/'terminal.json');terminals.append(terminal)
        job=load(LOCAL/(lane+'-job.json'))
        assert terminal['job_id']==job['job_id'] and terminal['protocol_SHA256']==sha256((LOCAL/'protocol.json').read_bytes())
        assert 'A100' in terminal['device'] and terminal['versions']==p['numerical_packages']
        for q,r in zip(p['qualifications'],terminal['rows']):
            assert q['id']==r['case_id']
            if r['status']!='COMPLETED':break
            assert max(r['row']['qualification_errors'].values())<=1e-7
        folder=base/'out'/lane/c['id']
        if not (folder/'result.json').exists():
            coverage.append(dict(method=c['id'],status='NOT_RUN_AFTER_QUALIFICATION_FAILURE'));continue
        r=load(folder/'result.json');coverage.append(dict(method=c['id'],status=r['status'],result=r))
        if r['status']=='COMPLETED':audits.append(one(p,c,r,folder))
    out=dict(status=f'{REPORT_TAG}_SAVED_ROOTS_INDEPENDENT_CPU_AUDIT',rows=audits,coverage=coverage,
        accounting=collection['accounting'],worker_CPU_seconds=sum(t['whole_CPU_seconds'] for t in terminals),
        worker_wall_seconds_by_lane=[t['whole_wall_seconds'] for t in terminals],
        qualification_attempts=sum(len(t['rows'])-int(any(r['case_id']==p['production'][lane]['id'] for r in t['rows'])) for lane,t in zip(p['production'],terminals)),
        new_sampling_model_QPU=[0,0],new_audit_eigensolves=0,goal_complete=False)
    if 'predecessor_N54_measured_worker_CPU_seconds' in p:
        out['predecessor_N54_worker_CPU_seconds']=p['predecessor_N54_measured_worker_CPU_seconds']
        out['predecessor_N54_accounting']=p['predecessor_N54_accounting']
    save(ROOT/'analysis'/f'{REPORT_TAG}-independent-root-audit.json',out)
    hf,ref=reference('FE2S2-40-P2');previous=load(ROOT/'analysis/N51-ALL-summary.json');combined=[]
    for row in previous['rows']:
        found=next((a for a in audits if a['method']==row['method']),None)
        if found:
            combined.append(dict(method=row['method'],energy=found['energy_Ha'],Recovery=100*(hf-found['energy_Ha'])/(hf-ref),
                R=row['R'],D=row['D'],source=row['source'],readout=f'{REPORT_TAG}_GPU_INDEPENDENTLY_AUDITED',
                source_cost=row['source_cost']))
        else:
            attempted=next((x for x in coverage if x['method']==row['method']),None)
            combined.append(dict(method=row['method'],energy=row['numerical_energy_Ha'],Recovery=row['correlation_energy_Recovery_pct'],
                R=row['R'],D=row['D'],source=row['source'],readout=REPORT_TAG+'_'+attempted['status'] if attempted else 'N51_CPU_'+row['status'],
                N54_attempt=attempted,source_cost=row['source_cost']))
    save(ROOT/'analysis'/f'{REPORT_TAG}-combined-source-summary.json',dict(rows=combined,source_roots_audit=out,
        new_CI_backend_not_new_Ansatz=True,independent_confirmation=False,full_paper_reproduction=False,goal_complete=False))
    lines=[f'# {REPORT_TAG}：Fe₂S₂のfull-batch比較をGPUで補完','',
        '同じ保存済み100k model samplesの各batch0を使った開発pilot。新しい量子測定はありません。',
        '全てH+0.1S²の最低根を探索し、表にはその根の物理H energyを示します。厳密S=0ではありません。',
        'Recoveryの参照はSCIであり、FCI・参照精度保証ではありません。','',
        '| 方法 | energy Ha | Recovery % | actual R | actual CI次元 | logical CNOT平均 | CNOT Depth平均 | Total Depth平均 | 係数slots平均 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in combined:
        z=r['source']['sampler_resources'];e='欠測' if r['energy'] is None else f"{r['energy']:.9f}"
        rec='欠測' if r['Recovery'] is None else f"{r['Recovery']:.3f}"
        lines.append(f"| {r['method']} | {e} | {rec} | {r['R']} | {r['D']} | {z['cnot_count']:.3f} | {z['cnot_depth']:.3f} | {z['total_unitary_depth']:.3f} | {r['source']['sampler_coefficient_slots_mean']:.3f} |")
    lines+=['','## 既存Rustiq対照（同じsource、別compiler）','',
        '| 方法 | CNOT平均 | CNOT Depth平均 | Total Depth平均 |',
        '|---|---:|---:|---:|']
    for r in combined:
        z=r['source'].get('Rustiq_resources',{})
        if z.get('Rustiq_cnot_count') is not None:
            lines.append(f"| {r['method']} Rustiq | {z['Rustiq_cnot_count']:.3f} | {z['Rustiq_cnot_depth']:.3f} | {z['Rustiq_total_unitary_depth']:.3f} |")
    lines+=['','## 比較の限界と費用','',
        '同Shotでも実CI次元は手法によって異なり、同古典費用の比較ではありません。各source1 parent・1 batchのみです。',
        'SqDRIFT20にはRustiq改善版の資源もあります。上のraw分解値だけに勝っても最強のcompiler対照への勝利にはしません。',
        'slotsは独立parameter rankではありません。ensembleの平均slotsと単一LUCJ回路は意味が違います。',
        'LUCJ TNは旧bounded調整であり、原著native TNの完全再現ではありません。初期化の非収束・元費用・旧600CPU秒停止も保持します。',
        f"今回のworker CPU（qualification込み）小計{out['worker_CPU_seconds']:.6f}秒、laneごとのworker wall{out['worker_wall_seconds_by_lane']}秒。",
        'Slurm実割当は以下。GPU作用だけの時間を総費用とせず、転送・準備・監査・旧失敗も別に保持します。','',
        '```text',collection['accounting'].strip(),'```','',
        f'欠測対照の補完を試みた記録です。今回の大空間sourceの監査済みenergy取得は{len(audits)}/2件。RAACSの総合優越・新Ansatzの改善・全研究目標の達成とはしません。','']
    if 'predecessor_N54_worker_CPU_seconds' in out:
        lines+=['',f"旧N54のworker CPU {out['predecessor_N54_worker_CPU_seconds']:.6f}秒（2件上限停止）は別に保持。新しい成功で失敗費用を消していません。",'']
    write(ROOT/'analysis'/f'{REPORT_TAG}_FULL_BATCH_RESULTS_JA.md','\n'.join(lines).encode())
    print(REPORT_TAG+'_ROOTS_CPU_AUDITED',[(a['method'],a['energy_Ha'],a['residual_norm']) for a in audits],flush=True)


if __name__=='__main__':run()
