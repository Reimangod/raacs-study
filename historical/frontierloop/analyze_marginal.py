"""Audit compact raw F2 evidence and report every case/readout, not a winner."""
import csv
from io import BytesIO, StringIO
import json
import sqlite3
import numpy as np
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .marginal_remote import DIRECTORY,binding
from .analyze import reference


def audit_case(case,row,manifest):
    d=DIRECTORY/'remote-output/outputs'/case
    if not (d/'draw-result.json').exists():
        return {'status':'MISSING_PRESERVED','draw_verified':False}
    r=load(d/'draw-result.json');members=load(d/'members.json');m=row['physical']
    assert r['run_id']==row['run_id'] and r['physical_problem_id']==m['problem_id']
    assert sha256((d/'members.json').read_bytes())==r['members_SHA256']
    assert sha256((d/'half-counts.npz').read_bytes())==r['counts_SHA256']
    aggregate={};actual_resources=[];total=0;shot_cnot=0
    db=sqlite3.connect('file:'+str(d/'draws.sqlite')+'?mode=ro',uri=True)
    draws={i:(json.loads(request),counts) for i,request,counts in db.execute('SELECT member,request,counts FROM draws')};db.close()
    assert abs(sum(t['probability'] for t in members['members'])-1)<1e-10
    for t in members['members']:
        observed={}
        for kind in ['preparation','sampling']:
            payload=(d/'circuits'/f"{t['member']:03d}-{kind}.qpy").read_bytes()
            assert sha256(payload)==t['QPY_SHA256'][kind]
            q=qpy.load(BytesIO(payload))[0]
            counters={'n_qubits':q.num_qubits,'cnot_count':int(q.count_ops().get('cx',0)),
                'cnot_depth':int(q.depth(filter_function=lambda i:i.operation.name=='cx')),
                'total_depth':int(q.depth()),'numeric_gate_coefficient_slots_not_rank':sum(len(i.operation.params) for i in q)}
            assert counters==t['resources'][kind] and q.num_qubits==m['norb']
            observed[kind]=counters
        actual_resources.append(observed)
        shots=t['model_shots'];total+=shots;shot_cnot+=shots*observed['sampling']['cnot_count']
        if not shots:
            assert t['member'] not in draws
            continue
        request,payload=draws.pop(t['member'])
        assert all(request[k]==v for k,v in t.items())
        with np.load(BytesIO(payload),allow_pickle=False) as a:
            ids,counts=a['half_strings'],a['counts']
            assert int(counts.sum())==shots
            for x,c in zip(ids,counts):
                assert int(x).bit_count()==m['nalpha'] and int(x)<2**m['norb']
                aggregate[int(x)]=aggregate.get(int(x),0)+int(c)
    assert not draws
    with np.load(d/'half-counts.npz',allow_pickle=False) as a:
        assert aggregate=={int(x):int(c) for x,c in zip(a['half_strings'],a['counts'])}
    assert total==manifest['protocol']['model_shots_per_case']==r['known_model_shots']
    assert shot_cnot==r['model_CNOT_applications']
    for k,v in r['sampling_resources_shot_weighted_mean'].items():
        assert abs(sum(t['model_shots']*t['resources']['sampling'][k] for t in members['members'])/total-v)<1e-9
    return {'status':'RAW_MEMBER_COUNTS_AND_QPY_RESOURCE_AUDIT_PASSED',
        'draw_verified':True,'members':len(members['members']),'model_shots':total,
        'model_CNOT_applications':shot_cnot,'energy_not_independently_recomputed':True}


def main():
    manifest=load(DIRECTORY/'manifest.json')
    if manifest['freeze']!=binding():raise RuntimeError('F2 code/freeze changed')
    audits={case:audit_case(case,row,manifest) for case,row in manifest['cases'].items()}
    rows=[]
    for case,row in manifest['cases'].items():
        d=DIRECTORY/'remote-output/outputs'/case
        draw=load(d/'draw-result.json') if (d/'draw-result.json').exists() else None
        bdir=ROOT/'runs/D2-COMPRESSION-SCREEN'/case/'EXACT';b=load(bdir/'request.json')
        assert b['problem_id']==row['physical']['problem_id']
        assert b['source_initialization_provenance']==row['source_initialization_record']['path']
        assert sha256(__import__('pathlib').Path(b['source_initialization_provenance']).read_bytes())==row['source_initialization_record']['SHA256']
        hf,ref=reference(case)
        for mode in manifest['protocol']['readouts']:
            p=d/mode/'result.json';q=bdir/mode/'result.json'
            r=load(p) if p.exists() else None;base=load(q) if q.exists() else None
            if r:assert r['physical_problem_id']==b['problem_id'] and r['counts_SHA256']==draw['counts_SHA256']
            rows.append({'case':case,'readout':mode,'F2_status':'COMPLETED' if r else 'FAILED_ENGINEERING_PRESERVED' if (d/mode/'failure.json').exists() else 'NOT_COMPLETED',
                'F2_energy_Ha':r['final_energy_Ha'] if r else None,'full_L1_energy_Ha':base['final_energy_Ha'] if base else None,
                'delta_energy_mHa':1000*(r['final_energy_Ha']-base['final_energy_Ha']) if r and base else None,
                'F2_recovery_pct':100*(hf-r['final_energy_Ha'])/(hf-ref) if r else None,
                'F2_CNOT':draw['sampling_resources_shot_weighted_mean']['cnot_count'] if draw else None,
                'full_CNOT':b['resources']['Z_sampling_equivalent']['cnot_count'],
                'F2_CNOT_depth':draw['sampling_resources_shot_weighted_mean']['cnot_depth'] if draw else None,
                'full_CNOT_depth':b['resources']['Z_sampling_equivalent']['cnot_depth'],
                'F2_total_depth':draw['sampling_resources_shot_weighted_mean']['total_depth'] if draw else None,
                'full_total_depth':b['resources']['Z_sampling_equivalent']['total_unitary_depth'],
                'F2_model_shots':draw['known_model_shots'] if draw else None,
                'full_model_shots':100000 if (bdir/'draw-result.json').exists() else None,
                'F2_half_configuration_count':draw['half_configurations'] if draw else None,
                'full_half_configuration_count':200000 if (bdir/'draw-result.json').exists() else None,
                'r_cap':64,'actual_F2_CI_dimension':r['actual_CI_dimension'] if r else None,
                'source_model_coordinates_unchanged':True,'independent_parameter_rank':None,
                'F2_pattern_and_compilation_CPU_seconds':draw['pattern_and_all_compilation_CPU_seconds'] if draw else None,
                'F2_sampling_CPU_seconds':draw['sampling_CPU_seconds'] if draw else None,
                'F2_readout_CPU_seconds':r['whole_body_CPU_seconds'] if r else None,
                'F2_result_SHA256':sha256(p.read_bytes()) if r else None,
                'full_result_SHA256':sha256(q.read_bytes()) if base else None})
    stream=StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/F2-all-data.csv',stream.getvalue().encode('utf-8-sig'))
    summary={'raw_audits':audits,'completed_draws':sum(a['draw_verified'] for a in audits.values()),
        'completed_readouts':sum(r['F2_status']=='COMPLETED' for r in rows),
        'planned_draws':4,'planned_readouts':12,'energy_independently_recomputed':False,'goal_complete':False}
    save(ROOT/'analysis/F2-summary.json',summary)
    lines=['# 片スピン正混合F2：全条件の結果','',
        '1層LUCJの片スピン分布を保つ式を使い、半qubitの回路群から測定する開発試験。全量子状態を再現したわけではなく、既存論文に対する新規性・総合優越・統計的同等性を認定する試験でもない。','',
        f"生counts/回路資源の検査を通過したdraw {summary['completed_draws']}/4、読出し完了 {summary['completed_readouts']}/12。古い結果や失敗を上書きしない。",'',
        '## 同Shot・同Rの比較','',
        '各方式100,000 model Shot、R上限64。full回路は1 Shotで2本の片スピン配置、F2は1本。したがって同Shotでも情報量は同じではない。後者を2倍Shotで測る試験は未実施。実QPU Shotは0。','',
        '| 条件 | full→F2 qubit | full→F2 CNOT | full→F2 CNOT Depth | full→F2 Total Depth | C3 ΔE mHa |',
        '|---|---:|---:|---:|---:|---:|']
    f=lambda v:'欠測' if v is None else f'{v:.6f}'
    for case in manifest['protocol']['cases']:
        r=next(x for x in rows if x['case']==case and x['readout']=='C3-SPIN-OO');n=manifest['cases'][case]['physical']['norb']
        lines.append(f"| {case} | {2*n}→{n} | {r['full_CNOT']}→{f(r['F2_CNOT'])} | {r['full_CNOT_depth']}→{f(r['F2_CNOT_depth'])} | {r['full_total_depth']}→{f(r['F2_total_depth'])} | {f(r['delta_energy_mHa'])} |")
    lines+=['','正のΔEはF2のenergy悪化、負は低下。独立な一回ずつのsamplingによる差で、分布を保つ数学的identityが有限Shotの同じ配置poolやCI energyを保証するわけではない。','',
        '## 3種類の読出しを全て報告','',
        '| 条件 | 読出し | F2 energy Ha | full L1 energy Ha | ΔE mHa | F2 Recovery % | 状態 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append('| '+' | '.join([r['case'],r['readout'],f(r['F2_energy_Ha']),f(r['full_L1_energy_Ha']),f(r['delta_energy_mHa']),f(r['F2_recovery_pct']),r['F2_status']])+' |')
    lines+=['','C2は無罰則、C3は有限spin罰則、C4は選択空間内の厳密S=0。同じ意味ではなく、勝つ定義だけを使わない。Recoveryは同一参照に対するenergyの線形変換であり、独立な成功指標ではない。最終の軌道最適化CI状態をsamplerが量子準備したとは呼ばない。','',
        '## 費用・検証・次の判断','',
        '全patternの確率計算・全member回路合成・member別Shot・raw countsを保存し、QPYから全memberのCNOT/depthを数え直した。aggregate countsもSQLiteのmember別結果から再構成した。energyは既存kernelの出力を結合確認しており、別solverでの独立再評価は未実施。','',
        'CSVに合成・sampling・読出しCPU費用、実CI次元、全result SHAを分けて記載。元のCCSD/分解の初期化費用はsource initialization recordを参照し、今回の安いsamplingだけをcold-start総費用と呼ばない。CPUノード実行であり、GPU加速とは報告しない。','',
        '回路群全体は元のJ/U情報を保持する。memberの数値ゲート係数数は独立parameter数ではなく、Parameter半減は未証明。半qubitでもspin内の並列化できない回転が残り、DepthがCNOTと同率で減るとは限らない。','',
        '同じ本数の片スピン配置を得るためにF2のShotを2倍にする場合、総CNOT適用数は概算で上表F2 CNOTの2倍×元Shot数になる。したがって同Shotで約半分という削減を、そのまま同情報量で約半分と主張できない。この2倍Shot条件のenergyは未測定であり、費用換算だけで成功扱いしない。','',
        '全4条件の固定source開発screenで、最強原著TN/SqDRIFT/HCI-HSQD・強い古典SCIへの総合的優越や未知分子への一般化は未証明。固定policyの独立seed比較と費用付き強対照が必要。数値欠測や不利な結果を残し、現状goalは未達。']
    write(ROOT/'analysis/F2_POSITIVE_MARGINAL_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(summary,flush=True)


if __name__=='__main__':main()
