"""N11 retrospective membership audit; no new quantum/CI computations.

Distinguish unrestricted HCI expansion from selection constrained to observed
single-spin configurations. The former remains a valid classical control.
"""
import csv,gzip,math
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .paired_pool_study import LOCAL
from .native_dice_binary import decode


def allowed_pool(source,n,nelec):
    hf=(1<<nelec[0])-1
    if source['kind']=='CLASSICAL_HF_ONLY':return {hf}
    coefficient_source=source['kind']=='CLASSICAL_COEFFICIENT_MARGINAL'
    if source['kind'] not in ['CLASSICAL_COEFFICIENT_MARGINAL','CLASSICAL_FULL_COUNTS','QUANTUM_FULL_COUNTS']:
        raise ValueError('Unknown observed source')
    mask=(1<<n)-1;pool={hf}
    with np.load(source['payload'],allow_pickle=False) as a:
        ids=a['blocked_ids'];values=a['coefficients'] if coefficient_source else a['counts']
        if len(ids)!=len(values):raise ValueError('Source arrays mismatch')
        for x,v in zip(ids,values):
            if not np.isfinite(v):raise ValueError('Nonfinite source')
            if abs(v)==0:continue
            aa,bb=int(x)&mask,int(x)>>n
            if aa.bit_count()!=nelec[0] or bb.bit_count()!=nelec[1]:raise ValueError('Source sector mismatch')
            pool.update([aa,bb])
    return pool


def outside(pairs,weights,pool):
    if len(pairs)!=len(weights) or not all(math.isfinite(float(x)) and x>=0 for x in weights):raise ValueError('Invalid weights')
    indices=[i for i,(a,b) in enumerate(pairs) if int(a) not in pool or int(b) not in pool]
    return dict(count=len(indices),weight=math.fsum(float(weights[i]) for i in indices))


def run():
    p=load(LOCAL/'protocol.json');c=LOCAL/'collected';t=load(c/'terminal.json')
    auditfile=ROOT/'analysis/N10-paired-initial-pool-summary.json';audit=load(auditfile)
    if audit['terminal_SHA256']!=sha256((c/'terminal.json').read_bytes()):raise RuntimeError('Previously audited outcome drift')
    obs={o['id']:o for o in load(c/'observations.json')};plans={x['task_id']:x for x in p['plans']};rows=[]
    for s in t['states']:
        plan=plans[s['task_id']];source=plan['source'];case=plan['case'];d=p['physical'][case];item=s['arms'].get('JOINT_TOP_D16384')
        row=dict(case=case,method=s['method'],arm=s['initialization_arm'],status=s['status'],
            observed_half_pool_size=None,potential_product_universe_dimension=None,
            native_dimension=None,native_outside_full_sample_pool_count=None,native_outside_full_sample_pool_weight=None,
            final_dimension=None,final_outside_full_sample_pool_count=None,final_outside_full_sample_pool_weight=None)
        if item:
            for name,key in [('payload','payload_SHA256'),('source_result','source_result_SHA256')]:
                if name in source and sha256(Path(source[name]).read_bytes())!=source[key]:raise RuntimeError('Source drift')
            pool=allowed_pool(source,d['norb'],tuple(d['nelec']));folder=c/'cells'/s['task_id'];raw=gzip.decompress((folder/'dets.bin.gz').read_bytes())
            if sha256(raw)!=s['native']['raw_determinants_SHA256']:raise RuntimeError('Native export drift')
            n,pairs=decode(raw)
            if n!=d['norb']:raise RuntimeError('Wrong orbitals')
            native=outside([(a,b) for a,b,_ in pairs],[v*v for _,_,v in pairs],pool)
            o=obs[item['observation_id']];vf=c/o['vector_file']
            if sha256(vf.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved final vector drift')
            with np.load(vf,allow_pickle=False) as a:fp=a['joint_pairs'];v=a['coefficients']
            final=outside(fp,v*v,pool)
            row.update(observed_half_pool_size=len(pool),potential_product_universe_dimension=len(pool)**2,
                native_dimension=len(pairs),native_outside_full_sample_pool_count=native['count'],native_outside_full_sample_pool_weight=native['weight'],
                final_dimension=len(fp),final_outside_full_sample_pool_count=final['count'],final_outside_full_sample_pool_weight=final['weight'])
        rows.append(row)
    if len(rows)!=120 or sum(r['native_dimension'] is not None for r in rows)!=106:raise RuntimeError('All planned/available outcomes required')
    result=dict(status='ALL120_SOURCE_UNIVERSE_MEMBERSHIP_ROWS_AUDITED_106_NUMERIC',rows=rows,
        numeric_native_exports_with_outside_configurations=sum((r['native_outside_full_sample_pool_count'] or 0)>0 for r in rows),
        numeric_final_spaces_with_outside_configurations=sum((r['final_outside_full_sample_pool_count'] or 0)>0 for r in rows),
        N10_summary_SHA256=sha256(auditfile.read_bytes()),implementation_SHA256=sha256(Path(__file__).read_bytes()),
        interpretation='UNRESTRICTED_NATIVE_HCI_EXTENSION_NOT_SAMPLE_CONSTRAINED_HCI_HSQD_EQ9',
        unrestricted_control_valid=True,previous_numeric_energies_invalidated=False,new_CI_solves=0,new_model_shots=0,new_QPU_shots=0,goal_complete=False)
    save(ROOT/'analysis/N11-sample-universe-membership.json',result)
    f=StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);write(ROOT/'analysis/N11-all120-sample-universe.csv',f.getvalue().encode('utf-8-sig'))
    lines=['# N11：古典探索が観測sampleの範囲を越えているか','',
        f"全120予定行/106数値出力を照合。native出力にsample半配置pool外があるもの{result['numeric_native_exports_with_outside_configurations']}件、最終CI空間にも外部配置があるもの{result['numeric_final_spaces_with_outside_configurations']}件。新CI/Shot0。",'',
        'poolは保存観測のα/β union（CISDは非zero係数、HFはHFのみ）にHFを加えたもの。R8/128へ制限する前の全poolを使った。この全poolのCartesian universeからも外れるなら、単なる初期marginal切詰めではなく、source外へのnative配置探索である。','',
        '| 条件・HF control（product arm） | 観測half pool | native配置数 | pool外native重み % | pool外最終CI重み % |','|---|---:|---:|---:|---:|']
    for r in rows:
        if r['method']=='CLASSICAL_HF_ONLY' and r['arm']=='PRODUCT_R8':
            lines.append(f"| {r['case']} | {r['observed_half_pool_size']} | {r['native_dimension']} | {100*r['native_outside_full_sample_pool_weight']:.3f} | {100*r['final_outside_full_sample_pool_weight']:.3f} |")
    lines+=['','これは旧計算が数値的に誤っていたという意味ではない。無制限HCIは強い古典対照として有効。ただし、量子sample内でのHCI-inspired選抜と同一ではないため、HFからの探索が量子sourceと同等だった結果を、そのままsample品質そのものの否定に使わない。',
        'HCI-HSQD Eq.(9)の限定を再現するには、HFから始め、追加候補のα/βが観測half poolに含まれることを毎iterationで強制する必要がある。最後に外部配置を捨てるだけでは、途中の探索経路も異なるため、その実装の再現にならない。',
        '一次資料：[HCI-HSQD](https://arxiv.org/html/2602.01165v1#S2)、[固定DICE選抜実装](https://github.com/caleb-johnson/Dice/blob/3198db1f8e0066489d173d30a52bc21967969a0f/SHCI/SHCIbasics.cpp)。']
    write(ROOT/'analysis/N11_SAMPLE_UNIVERSE_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],result['numeric_native_exports_with_outside_configurations'],result['numeric_final_spaces_with_outside_configurations'],flush=True)


if __name__=='__main__':run()
