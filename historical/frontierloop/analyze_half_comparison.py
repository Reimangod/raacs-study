"""Whole fixed comparison, with H1 raw audit and separate numerical-successor table."""
import csv
from io import BytesIO,StringIO
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .half_remap_remote import DIRECTORY as H1,binding
from .marginal_remote import DIRECTORY as F2


def main():
    manifest=load(H1/'manifest.json')
    if manifest['freeze']!=binding():raise RuntimeError('H1 frozen code drift')
    audits={};rows=[]
    modes=manifest['protocol']['readouts']
    for case,row in manifest['cases'].items():
        d=H1/'remote-output/outputs'/case;f=F2/'remote-output/outputs'/case
        draw=load(d/'draw-result.json') if (d/'draw-result.json').exists() else None
        fd=load(f/'draw-result.json');assert row['physical']['problem_id']==fd['physical_problem_id']
        if draw:
            assert draw['run_id']==row['run_id'] and draw['physical_problem_id']==row['physical']['problem_id']
            assert sha256((d/'half-counts.npz').read_bytes())==draw['counts_SHA256']
            with np.load(d/'half-counts.npz',allow_pickle=False) as a:
                assert int(a['counts'].sum())==draw['known_model_shots']==100000
                assert len(set(map(int,a['half_strings'])))==len(a['half_strings'])
                assert all(int(x).bit_count()==row['physical']['nalpha'] for x in a['half_strings'])
            for kind in ['preparation','sampling']:
                payload=(d/(kind+'.qpy')).read_bytes();assert sha256(payload)==draw['QPY_SHA256'][kind]
                q=qpy.load(BytesIO(payload))[0]
                resources={'n_qubits':q.num_qubits,'cnot_count':int(q.count_ops().get('cx',0)),
                    'cnot_depth':int(q.depth(filter_function=lambda i:i.operation.name=='cx')),
                    'total_depth':int(q.depth()),'numeric_gate_coefficient_slots_not_rank':sum(len(i.operation.params) for i in q)}
                assert resources==draw['resources'][kind]
            assert draw['model_CNOT_applications']==draw['known_model_shots']*draw['resources']['sampling']['cnot_count']
            audits[case]={'raw_counts_and_QPY_resources':'PASSED','model_shots':draw['known_model_shots']}
        else:audits[case]={'status':'NOT_COLLECTED_OR_MISSING'}
        for method,src,sampler in [('H1',d,draw),('F2',f,fd)]:
            for mode in modes:
                p=src/mode/'result.json';r=load(p) if p.exists() else None
                if r:assert r['counts_SHA256']==sampler['counts_SHA256']
                q=(sampler['resources']['sampling'] if method=='H1' else sampler['sampling_resources_shot_weighted_mean']) if sampler else {}
                rows.append({'case':case,'method':method,'readout':mode,'status':'COMPLETED' if r else 'FAILED_ENGINEERING_PRESERVED' if (src/mode/'failure.json').exists() else 'NOT_COMPLETED',
                    'energy_Ha':r['final_energy_Ha'] if r else None,'spin_square':r['history'][-1]['spin_square'] if r else None,
                    'CNOT':q.get('cnot_count'),'CNOT_depth':q.get('cnot_depth'),'total_depth':q.get('total_depth'),
                    'numeric_gate_slots_not_rank':q.get('numeric_gate_coefficient_slots_not_rank'),
                    'independent_parameter_rank':None,'model_shots':sampler['known_model_shots'] if sampler else None,
                    'half_configurations':sampler['half_configurations'] if sampler else None,'r_cap':64,
                    'actual_CI_dimension':r['actual_CI_dimension'] if r else None,
                    'whole_body_readout_CPU_seconds':r['whole_body_CPU_seconds'] if r else None,
                    'source_result_SHA256':sha256(p.read_bytes()) if r else None})
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/H1-F2-all-readouts.csv',stream.getvalue().encode('utf-8-sig'))
    save(ROOT/'analysis/H1-raw-audit.json',audits)
    f=lambda x:'欠測' if x is None else f'{x:.8f}'
    lines=['# 半qubit方式は既存HSQDの移植対照より有利か','',
        'H1はHSQD論文のスピン間ゲート移植を、今回の同一CCSD由来LUCJと共通後処理に適用した対照。原著のHCI選択・実機routing・大CI設定まで再現したものではない。F2は正混合で片スピン分布を保存する別方式。両方式とも100,000 model Shot、同じR64、実QPU Shotは0。','',
        '## 同じ初期実装C3での比較（最低根到達の保証なし）','',
        '| 条件 | H1 energy Ha | F2 energy Ha | F2−H1 mHa | H1 CNOT / CNOT Depth / Total Depth | F2 CNOT / CNOT Depth / Total Depth |',
        '|---|---:|---:|---:|---|---|']
    for case in manifest['protocol']['cases']:
        a=next(r for r in rows if r['case']==case and r['method']=='H1' and r['readout']=='C3-SPIN-OO')
        b=next(r for r in rows if r['case']==case and r['method']=='F2' and r['readout']=='C3-SPIN-OO')
        gap=1000*(b['energy_Ha']-a['energy_Ha']) if a['energy_Ha'] is not None and b['energy_Ha'] is not None else None
        res=lambda r:' / '.join('欠測' if r[k] is None else f'{r[k]:g}' for k in ['CNOT','CNOT_depth','total_depth'])
        lines.append(f"| {case} | {f(a['energy_Ha'])} | {f(b['energy_Ha'])} | {f(gap)} | {res(a)} | {res(b)} |")
    lines+=['','負の差はF2のenergyが低い。単一seedの開発結果で統計的優越/非劣性を認定しない。回路数・合成費用も異なる。両法の元のJ/U情報は残り、独立parameter数削減は未認証。','',
        '## 数値修正版C7：全方式へ同じ最低根ターゲティング','',
        '旧D2の1セルでDavidsonが既知のRayleigh値より約51.54mHa高い根へ収束し、単調性検査が停止した。別の行列作用・SA計算では低い値が再現され、旧検査を緩める問題ではなかった。既存の全結果は残し、新規Shotなしで同じcounts/同じC3目的を別実装で評価する。以下は旧C3と混ぜない。','',
        '| 条件 | full L1 / C7 energy Ha | F2 / C7 energy Ha | H1 / C7 energy Ha |',
        '|---|---:|---:|---:|']
    completed=0
    for case in manifest['protocol']['cases']:
        values=[]
        for method in ['D2_EXACT','F2','H1']:
            p=ROOT/'runs/C7-LOWEST-ROOT-v1'/case/method/'result.json'
            r=load(p) if p.exists() else None
            if r:
                assert r['physical_problem_id']==manifest['cases'][case]['physical']['problem_id']
                assert sha256((p.parent/'optimized-representation.npz').read_bytes())==r['representation_SHA256'];completed+=1
            values.append(r['final_energy_Ha'] if r else None)
        lines.append('| '+case+' | '+' | '.join(f(v) for v in values)+' |')
    lines+=['',f'C7の主比較は{completed}/12完了。元のF25失敗セルの追加回帰診断は別枠で、手法性能平均へ足さない。未完了を旧方式の値で穴埋めしない。修正版でも有限罰則は厳密S=0ではなく、ARPACKの数値的最低根ターゲティングは大域最適化の数学的証明ではない。','',
        'C5の強い量子対照や古典CISDへ同じ数値修正版を適用する前に、それらへの勝利を新旧solver混合表で主張しない。C2/C4もCSVへ残し、勝つspin定義へ取り替えない。最終OO/CI状態をsamplerが量子準備したとは言わない。','',
        '## 研究としての判断','',
        'F2の数学的分布保存と半qubit化は確認できたが、最良の既知手法への実用的優越はまだ確認できていない。H1との資源が同程度なら、F2はensemble生成費用を払ってどの条件で精度・安定性が改善するかを示す必要がある。どちらもR64の制約下のscreenで、原著の大きいHCI/CI空間の精度へ勝ったという意味ではない。','',
        '全4条件を保持し、結果が悪い条件を除かない。F2とH1に優劣がなければ、ensembleを複雑化する前に、配置選択/R制約が主要な精度ボトルネックかを既存countsで点検する。A100待ちを理由に同じdrawをCPUで重複実行しない。','',
        '[F2全結果と費用](F2_POSITIVE_MARGINAL_RESULTS_JA.md)、[D2圧縮結果](D2_COMPRESSION_RESULTS_JA.md)、[HCI-HSQD原著](https://arxiv.org/html/2602.01165v1)。']
    write(ROOT/'analysis/H1_F2_COMPARISON_JA.md',('\n'.join(lines)+'\n').encode())
    result={'H1_raw_audits':audits,'H1_F2_old_readouts_completed':sum(r['status']=='COMPLETED' for r in rows),'C7_primary_readouts_completed':completed,'goal_complete':False}
    save(ROOT/'analysis/H1-F2-summary.json',result);print(result,flush=True)


if __name__=='__main__':main()
