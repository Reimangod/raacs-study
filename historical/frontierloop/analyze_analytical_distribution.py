"""N33 descriptive finite-count support diagnostic, no new sampling/energy."""
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from . import analytical_source_sampling as source
from . import analytical_source_readout as readout
from . import finite_source_study as n29


def run():
    cpu,wall=process_time(),perf_counter();p,q,sub=readout.verified();rows=[]
    old=load(n29.LUCJ/'protocol.json')
    for d in p['physical'].values():
        case=next(k for k,v in p['physical'].items() if v==d);n=d['norb'];k=d['nelec'][0];hf=((1<<k)-1)*(1+(1<<n));mask=(1<<n)-1
        for method,folder,sp in [('INITIAL_L1',n29.LUCJ,old),('ANALYTICAL_OPT_L1_8EVAL',source.SAMPLING,sub)]:
            c=next(c for c in sp['circuits'] if c['case']==case and (method!='INITIAL_L1' or c['source'].endswith('L1')))
            for req in c['sampling_requests']:
                f=folder/'decoded'/(req['id']+'.npz');meta=load(f.with_suffix('.json'))
                if sha256(f.read_bytes())!=meta['decoded_SHA256']:raise RuntimeError('Actual decoded evidence drift')
                with np.load(f,allow_pickle=False) as z:ids,counts=z['blocked_ids'],z['counts']
                n29.validate_counts(ids,counts,n,d['nelec']);prob=counts/counts.sum();half={int(x)&mask for x in ids}|{int(x)>>n for x in ids}
                rows.append(dict(case=case,method=method,replicate=req['replicate_label'],draw_id=req['id'],payload_SHA256=meta['decoded_SHA256'],
                    actual_shots=int(counts.sum()),observed_unique_full_configurations=len(ids),observed_unique_half_configurations=len(half),
                    HF_frequency=float(prob[ids==hf].sum()),maximum_full_configuration_frequency=float(prob.max()),
                    empirical_entropy_nats=float(-np.sum(prob*np.log(prob))),empirical_inverse_collision=float(1/np.sum(prob**2)),
                    these_are_finite_empirical_not_true_entropy_or_coverage=True))
    groups=[]
    metrics=('observed_unique_full_configurations','observed_unique_half_configurations','HF_frequency','maximum_full_configuration_frequency','empirical_entropy_nats','empirical_inverse_collision')
    for case,method in sorted({(x['case'],x['method']) for x in rows}):
        local=[x for x in rows if (x['case'],x['method'])==(case,method)]
        groups.append(dict(case=case,method=method,**{name:float(np.mean([x[name] for x in local])) for name in metrics}))
    out=dict(status='N33_ALL24_FIXED_DRAWS_DESCRIPTIVE_SUPPORT_AUDIT',rows=rows,groups=groups,
        source_protocol_SHA256=sha256((source.LOCAL/'protocol.json').read_bytes()),new_model_QPU_shots=0,new_energy_evaluations=0,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,causal_mechanism_proven=False,goal_complete=False)
    save(ROOT/'analysis/N33-distribution-diagnostic.json',out)
    lines=['# N33：VQE調整前後の有限counts分布','',
        '全条件同じ1,000 samples×3反復。新旧は別drawで共通乱数ではない。以下は観測上の指標で、真のentropy・未観測tail・因果機構の証明ではない。新energy/Shotは0。', '',
        '| 条件 | source | 観測full配置数平均 | 観測half配置数平均 | HF頻度平均 | 最大配置頻度平均 | 経験entropy nats |','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['case']} | {g['method']} | {g['observed_unique_full_configurations']:.2f} | {g['observed_unique_half_configurations']:.2f} | {100*g['HF_frequency']:.2f}% | {100*g['maximum_full_configuration_frequency']:.2f}% | {g['empirical_entropy_nats']:.4f} |")
    lines+=['','半配置の観測数が減ると、frequency-based Cartesian CIへ渡せるRも小さくなり得る。ただし配置数やentropyを増やせばenergyが下がるという法則はない。重要な配置の重み、CI相互作用、有限Shotのばらつきも効く。単にentropyを新しい最適化目的として採用しない。',
        'N32のphysical-H VQE改善とN33のSQD精度悪化を混同せず、費用込みのtask-aligned source設計が次の課題。現在の8評価実験から全てのVQE訓練が無意味だとは結論しない。']
    write(ROOT/'analysis/N33_DISTRIBUTION_DIAGNOSTIC_JA.md','\n'.join(lines).encode())
    print(out['status'],groups,flush=True)


if __name__=='__main__':run()
