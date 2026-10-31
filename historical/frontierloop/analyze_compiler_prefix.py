"""G6 prefix-only report; no substitution into whole-bank energy rows."""
from collections import Counter
import csv,gzip,json,sqlite3
from io import BytesIO,StringIO
from qiskit import qpy
from .paths import ROOT
from .io import load,save,write,sha256
from .stronger_compiler_prefix import RUN,FREEZE,binding,choose,FIELDS


def run():
    assert binding()==load(FREEZE)
    t=load(RUN/'terminal.json');assert len(t['rows'])==64
    assert sha256((RUN/'compiled-members.sqlite').read_bytes())==t['database_SHA256']
    db=sqlite3.connect('file:'+str(RUN/'compiled-members.sqlite')+'?mode=ro',uri=True)
    assert db.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
    checked=0
    for case,idx,record,*blobs in db.execute('SELECT * FROM members'):
        r=json.loads(record);assert r['status']=='PASSED' and choose(r['resources'])==r['policy']
        for name,blob in zip(['NATIVE','RUSTIQ','PROJECTED'],blobs):
            assert sha256(blob)==r['QPY_gz_SHA256'][name]
            q=qpy.load(BytesIO(gzip.decompress(blob)))[0]
            actual=dict(cnot_count=int(q.count_ops().get('cx',0)),cnot_depth=int(q.depth(lambda x:x.operation.num_qubits==2) or 0),
                total_unitary_depth=int(q.depth() or 0))
            assert actual==r['resources'][name+'_OPT3']
            assert r['independent_opt3_check']['check']['circuits'][name]['status']=='INDEPENDENT_UNTRUNCATED_MPS_PROBABILITY_PASSED'
            checked+=1
    db.close();assert checked==192
    cases=[];flat=[]
    for case in dict.fromkeys(r['case'] for r in t['rows']):
        rows=[r for r in t['rows'] if r['case']==case];total=sum(r['original_shot_weight'] for r in rows)
        assert len(rows)==16 and total==2048
        means={name:{k:sum(r['original_shot_weight']*r['resources'][name][k] for r in rows)/total for k in FIELDS}
            for name in rows[0]['resources']}
        for name,key in [('CONTROL','baseline_resources'),('SELECTED','selected_resources')]:
            means[name]={k:sum(r['original_shot_weight']*r['policy'][key][k] for r in rows)/total for k in FIELDS}
        cases.append(dict(case=case,members=16,prefix_shot_weight=total,mean_resources=means,
            extra_reduction_pct={k:100*(1-means['SELECTED'][k]/means['CONTROL'][k]) for k in FIELDS},
            selection=dict(Counter(r['policy']['selected'] for r in rows)),
            nontrivial_opt3_output_permutations=sum(p!=list(range(len(p))) for r in rows for p in r['input_to_output_permutations'].values())))
        for r in rows:
            f=dict(case=case,index=r['index'],member_id=r['member_id'],selected=r['policy']['selected'],baseline=r['policy']['baseline'])
            for name,res in r['resources'].items(): f.update({name+'_'+k:res[k] for k in FIELDS})
            flat.append(f)
    summary=dict(status='ALL64_OPT3_PREFIX_AND192_QPY_RECOUNTS_PASSED_NOT_WHOLE_BANK',cases=cases,
        CPU_seconds_including_children=t['CPU_seconds_including_children'],wall_seconds=t['wall_seconds'],
        actual_saved_QPY_recounts=checked,new_model_shots=0,new_energy_CI_evaluations=0,
        no_whole_bank_energy_join=True,goal_complete=False)
    save(ROOT/'analysis/G6-opt3-prefix-summary.json',summary)
    out=StringIO();w=csv.DictWriter(out,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write(ROOT/'analysis/G6-opt3-prefix-all.csv',out.getvalue().encode())
    text=['# G6：強いopt3コンパイラにも利益が残るか','',
        '**固定先頭16件×4条件で、全3系列へ同じopt3を与えた後にも、固定占有処理の追加利益が残った。** 64件/192実回路の確率検算と再計数を全数通過。追加Shot/energy/CIは0。','',
        '| 条件 | 強化対照 CNOT→選択 | 追加CNOT削減 | CNOT Depth→選択 | 追加Depth削減 | Total→選択 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        m=c['mean_resources'];p=c['extra_reduction_pct']
        text.append(f"| {c['case']} | {m['CONTROL']['cnot_count']:.2f}→{m['SELECTED']['cnot_count']:.2f} | {p['cnot_count']:.1f}% | {m['CONTROL']['cnot_depth']:.2f}→{m['SELECTED']['cnot_depth']:.2f} | {p['cnot_depth']:.1f}% | {m['CONTROL']['total_unitary_depth']:.2f}→{m['SELECTED']['total_unitary_depth']:.2f} |")
    text+=['','対照にはnative/Rustiq各opt1/opt3の実回路を与え、元native opt1から三資源非増加の中でCNOT優先の最良を採用した。projected opt1/opt3も同じ条件で比較。存在しない各指標別envelopeを混ぜていない。全出力は元nativeよりCNOT・CNOT Depth・Total Depthが非増加。','',
        '## 検算で注意した点','',
        'Qiskit2.6、同basis、seed8137、approximation_degree=1.0、initialzero、単process。同じZ-sampling入力をopt3へ再コンパイルし、測定直前のdiagonalのみ同じルールで除去した。原著の対象hardwareの最高設定そのものを再現したわけではなく、共通all-to-all制御を強くした試験。','',
        'opt3はSWAPをゲートから消して最終qubit permutationへ移せる。独立4qテストで実際に非自明なpermutationを発生させ、並びを戻した確率照合を通した。今回の64実分子prefixでは非自明な追加permutationは0件だったが、未対応のまま放置せずコードへdecoderを登録した。','',
        f"本試験は{t['CPU_seconds_including_children']:.2f}CPU秒/{t['wall_seconds']:.2f}wall秒。全192個の新しいopt3 QPYを保存し、元sparse sourceに対してMPS確率/漏れを検算。旧opt1の証拠はG4を参照。新しい独立Quantum replicateや分子energy観測ではない。",'',
        '## 次の判断','',
        'opt3だけで固定占有処理の利益が全て消えるという心配は、このprefixでは当たらなかった。ただし**各bank783件中の先頭16件・time1・2048 Shot重み**だけ。G5全bankのopt1平均をopt3平均へ置き換えたり、T2の全bank energyとG6のprefix削減率を混ぜたりしない。','',
        'ここでcompiler確認だけを延々増やさない。資源側の小機構は有用な部品として保持し、次はenergy/CI費用のボトルネックへ戻る。固定Global選択を全16source・全4条件へ同じ別subsampling seedsで適用し、旧全頻度順/同union頻度順と全費用を比較する。特にG5ではN₂近距離以外の3条件が既存対照energyに届いていない。','',
        'より強い元source/TN/HCI、独立Quantum確認、独立Parametersは依然未完了。genericな初期状態依存compiler最適化自体の新規性は主張しない。','',
        '原著との違いは[G5 alignment review](G5_PAPER_ALIGNMENT_REVIEW_JA.md)。API根拠：[Qiskit transpile](https://quantum.cloud.ibm.com/docs/en/api/qiskit/compiler)。G5の全bank数値は[G5報告](G5_WHOLE_N20_RESULTS_JA.md)。','']
    write(ROOT/'analysis/G6_OPT3_PREFIX_RESULTS_JA.md','\n'.join(text).encode())
    print('G6_64_MEMBERS_192_QPYS_AUDITED_PREFIX_ONLY')


if __name__=='__main__':run()
