"""Read-only FeS feasibility supplement; no new numerical experiment or solve."""
from __future__ import annotations
import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, file_digest, save_binary, save_json


def report(output=None):
    root=Path(output or ROOT/'results/FeS-feasibility-pilot-v1')
    folder=root/'report-v1';folder.mkdir(parents=True,exist_ok=True)
    source_names=['manifest.json','completion.json','execution-binding.json','public-inputs/provenance.json',
        'physical-problem/problem.json','RAACS-full-cost/result.json','RAACS-full-cost/events.sqlite',
        'RAACS-full-cost/independent-validation.json','same-circuit-bonds/result.json']
    sources={p:file_digest(root/p) for p in source_names}
    saved=folder/'manifest.json'
    if saved.exists():
        old=json.loads(saved.read_text())
        if old['sources']!=sources or old['script_sha256']!=file_digest(__file__):raise RuntimeError('Fe report binding drift')
        if any(file_digest(folder/p)!=sha for p,sha in old['outputs'].items()):raise RuntimeError('Fe report output drift')
        return {'status':'VERIFIED_WITHOUT_RERENDER','new_experiments':0}
    get=lambda p:json.loads((root/p).read_text())
    c=get('completion.json');meta=get('physical-problem/problem.json');r=get('RAACS-full-cost/result.json')
    a=get('RAACS-full-cost/independent-validation.json');b=get('same-circuit-bonds/result.json')
    if c['status']!='QUALIFICATION_EXECUTED_AND_VALIDATED' or a['maximum_energy_difference_Ha']>1e-8:
        raise RuntimeError('Unvalidated source')
    gain=(meta['hf_energy_Ha']-c['final_QSCI_energy_Ha'])*1000
    row={'molecule':'Fe2S2','spatial_orbitals':20,'qubits':40,'nalpha':15,'nbeta':15,
        'initial_reference':'FIRST15_ORBITALS_DOUBLY_OCCUPIED_NOT_VERIFIED_RHF',
        'initial_diagonal_energy_Ha':meta['hf_energy_Ha'],'final_QSCI_energy_Ha':c['final_QSCI_energy_Ha'],
        'gain_vs_occupation_reference_mHa':gain,'ground_state_error_Ha':None,
        'CI_dimension':c['actual_CI_dimension'],'accepted_steps':c['accepted_steps'],
        'stop_reason':r['construction_stop_reason'],
        'pilot_model_samples':r['pilot_raw_model_shots'],'independent_final_model_samples':r['independent_final_raw_model_shots'],
        'total_model_samples':c['actual_model_samples'],
        **{k:c['sampling_resources'][k] for k in ('cnot_count','cnot_depth','parameter_count','total_depth')},
        'given_integrals_pilot_plus_validation_wall_seconds':c['actual_wall_seconds'],
        'process_RSS_hwm_bytes':c['process_RSS_hwm_bytes'],
        'classical_named_H_requests':r['classical_named_H_requests'],
        'S_squared_certified':False,'hardware_shots':0,'single_seed_exploratory_feasibility':True}
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(row));w.writeheader();w.writerow(row)
    save_binary(folder/'result.csv',stream.getvalue().encode())
    figure,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    figure.suptitle('Published Fe2S2 integrals: bounded 40-qubit feasibility pilot',fontsize=13)
    axes[0].bar(['Occupation\nreference','Independent\nfinal QSCI'],[0,gain],color=['#777777','#2563a6'])
    axes[0].set_ylabel('Energy lowering vs occupation reference (mHa)')
    axes[0].set_title('NOT ground-state error / recovery %',fontsize=10)
    axes[0].text(1,gain,f'{gain:.3f}',ha='center',va='bottom')
    axes[1].bar(['Paid\npilots','Independent\nfinal'],[r['pilot_raw_model_shots'],r['independent_final_raw_model_shots']],color=['#a54f27','#2563a6'])
    axes[1].set_ylabel('Model samples (hardware shots = 0)')
    axes[1].set_title('Total 2,000; no free pilot samples',fontsize=10)
    labels=['CNOT','CNOT depth','Parameters','Total depth']
    values=[c['sampling_resources'][f] for f in ('cnot_count','cnot_depth','parameter_count','total_depth')]
    axes[2].barh(labels,values,color='#2563a6')
    axes[2].set_xlabel('Registered logical sampling resource (count / layers)')
    axes[2].set_title('Four resources; no combined score',fontsize=10)
    for i,v in enumerate(values):axes[2].text(v,i,f' {v}',va='center')
    figure.text(.01,-.10,'Initial reference is NOT an optimized RHF. Nalpha/Nbeta checked; total S² not certified.\n'
        'One frozen policy/seed, bounded pilots, CI d=8. No full Fock state/Hamiltonian, FCI, QPU, or ground-accuracy claim.\n'
        'Fe4S4: 72 qubits exceed current qualified <=62-bit representation; not executed.',fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for extension in ('png','svg','pdf'):figure.savefig(folder/f'Fe2S2-feasibility.{extension}',dpi=180,bbox_inches='tight')
    plt.close(figure)
    text=f'''# Fe–S条件付きスケーリング補遺

## 実際に実行できたこと

公開著者積分に対する **[2Fe–2S] / 20軌道・30電子・40qubits** の
上限付きRAACS試験を実行した。既存のH4入力によるquickを先に実行し、
独立検算に通過してからFeへ進んだ。元の研究結果・監査v1/v2は変更していない。

| 指標 | 実測・計算値 |
|---|---:|
| 初期占有配置の対角energy | {meta['hf_energy_Ha']:.12f} Ha |
| 独立final countsのQSCI energy | {c['final_QSCI_energy_Ha']:.12f} Ha |
| 初期占有配置からのenergy低下 | {gain:.6f} mHa |
| 実CI次元 | {c['actual_CI_dimension']} |
| 受理演算子数 | {c['accepted_steps']} |
| 停止理由 | {r['construction_stop_reason']} |
| 探索用 / 独立finalモデルサンプル | {r['pilot_raw_model_shots']} / {r['independent_final_raw_model_shots']} |
| CNOT / CNOT Depth / Parameters / Total Depth | {' / '.join(map(str,values))} |
| 新しい試験・入力取得・検算・bond診断のwall time | {c['actual_wall_seconds']:.3f}秒 |
| process RSS high-water | {c['process_RSS_hwm_bytes']/2**20:.1f} MiB |
| 実ハードウェアshots | 0 |

wallはこのprocess内の計測で、Python起動・事前環境構築を含まず、通信・キャッシュ・
独立検算を含む。既存90セルのcontrolled total-workflow時間と同じscopeではない。
初期占有配置のenergyは積分から計算し、架空のHF測定shotsは加算していない。

## 検証

- 元FCIDUMP: commit固定、959069 bytes、Git blob SHA1とSHA256一致。
- 5配置×5配置の積分kernelを、独立な生成消滅演算子の明示和で照合。
- 最終8配置の64行列要素と最小根を別実装で再計算。
  energy差は {a['maximum_energy_difference_Ha']:.3e} Ha。
- raw counts / logical bit remapping / 合計samples / Nα,Nβ sectorを照合。
  独立finalの有効sector率は {100*a['final_sector_fraction']:.1f}%。
- 全drawのMPS norm²の最小値 {a['minimum_draw_MPS_retained_norm_squared']:.15f}。
  同一最終回路のbond64/128/256はいずれも完成し、最高bondへの正規化overlap²は
  {', '.join(str(t.get('normalized_overlap_squared_to_highest_completed_bond')) for t in b['rows'])}。
  これは今回の浅い回路の収束診断で、一般の大規模回路への厳密保証ではない。

## 言ってはいけないこと

初期状態は「最初の15軌道を二重占有した配置」で、最適化済みRHFとは未確認。
legacy `hf_*` はその互換aliasと明記した。Nα=Nβだけでsinglet ground stateと
認定しない。ground-state referenceの同一積分・offset・sectorとの適合性は未認定。
したがってground-state誤差、化学精度、相関Recovery%、CEO/LUCJへの優位性、
実用的なFe相関問題の解決、量子優位は主張しない。

## [4Fe–4S]

公開inputは存在する。headerは36軌道・54電子・72qubitsで、現在のuint64
経路の認定上限62qubitsを超える。入力不在ではなく**表現上限による条件付き未実施**。
active spaceを無断で切り詰めたり、未認定backendで代用していない。

## 入力と再現

- 著者データ: https://github.com/jrm874/sqd_data_repository
- 固定commit:865bd762d3f96ee79bb362eb3dd46951b48b426e
- input SHA256:{meta['source_sha256']}
- protocol: study-v1/FE_S_BOUNDED_FEASIBILITY.md
- `python -m benchmark.fe_s_feasibility --mode quick`
- `python -m benchmark.fe_s_feasibility --mode pilot`
- `python -m benchmark.fe_s_report`

既存成功結果への再呼出しは再計算しない。partial taskは無償再試行しない。
入力の再配布licenseは未確立で、外部upload/publicationは行っていない。
旧監査はローカルの未導入状況の記録として保持し、この補遺で公開入力の存在と
実際の試行を追加する。本補遺のみで全研究完了・普遍的性能成功とはしない。
'''
    save_binary(folder/'REPORT_JA.md',text.encode())
    outputs={p.name:file_digest(p) for p in folder.iterdir() if p.is_file()}
    save_json(saved,{'schema':'FeS-feasibility-report-v1','sources':sources,'outputs':outputs,
        'script_sha256':file_digest(__file__),'new_experiments':0,'new_eigensolver_calls':0,'hardware_shots':0})
    return {'status':'FEASIBILITY_SUPPLEMENT_WRITTEN','actual_row':row,'outputs':len(outputs)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    print(json.dumps(report(p.parse_args().output)))
