"""Final read-only delivery, integrating existing full-scope evidence and limits.

This is a deliverable review, not a new experiment or a positive-performance
certificate. Old false-at-that-time whole_goal_complete fields stay unchanged.
The human/agent must still make the goal-completion decision from the evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from io import StringIO
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from .current_evidence import ROOT,sha,canonical,csv_text,save_once

REPORT=ROOT/'results/research-synthesis-standard-v3/snapshots/dc823da17be5b76b34b0c1e1cd736b4f9cb514440a04037f9ce069473a4ac9f1'
FIGURES=ROOT/'results/current18figure-index-standard-v1/snapshots/d89264228c0c90ea9eeeae60c3bd75f5306e3dbbbd3505dab1180c17ee0079e3'
OBJECTIVE=Path('/Users/rei/.codex/attachments/b0f3686d-83a1-477c-bc5a-3ec6a43e7830/goal-objective.md')


def test_receipt(path):
    xml=ET.parse(path).getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    fields={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if fields['failures'] or fields['errors'] or fields['skipped']:raise ValueError('Non-green final test receipt')
    return fields


def validate_links(path):
    links=[]
    for link in re.findall(r'\]\(([^)]+)\)',Path(path).read_text()):
        if '://' in link or link.startswith('#'):continue
        target=(Path(path).parent/link.split('#')[0]).resolve()
        if not target.exists():raise ValueError('Broken final delivery link: '+str(target))
        links.append(str(target))
    return links


def build(output=None,test_xml=None):
    from .full_objective_audit_v3 import build as reconcile
    from .current_evidence_review import review
    out=Path(output or ROOT/'results/final-research-delivery-v1');out.mkdir(parents=True,exist_ok=True)
    audit_root=ROOT/'results/full-objective-audit-2026-10-23-v3'
    reconcile() # Verifies the same existing audit and its bound evidence; no solves.
    current=review(REPORT,FIGURES) # Existing all-planned aggregation/link/source review.
    inputs={}
    def load(path):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        inputs[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=sha(path)
        return json.loads(path.read_text())
    def rows(path):
        p=Path(path);inputs[str(p.relative_to(ROOT))]=sha(p)
        with p.open() as f:return list(csv.DictReader(f))
    audit=load(audit_root/'audit.json');items=rows(audit_root/'original-objective-items.csv')
    judgments=rows(audit_root/'requirement-judgments.csv')
    if len(items)!=367 or len(judgments)!=35 or audit['original_objective_sha256']!=sha(OBJECTIVE):
        raise ValueError('Original full objective coverage changed')
    if any(not r['evidence'] or r['group'] not in {j['group'] for j in judgments} for r in items):
        raise ValueError('Original requirement lost evidence/group')
    for path,expected in audit['evidence_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Full-objective evidence drift '+path)
    package=load('results/reproducibility-delivery-v1/manifest.json')
    portable=load('results/portable-delivery-addon-v1/validation.json')
    for base,m in ((ROOT/'results/reproducibility-delivery-v1',package),(ROOT/'results/portable-delivery-addon-v1',portable)):
        for p,expected in m['outputs'].items():
            if sha(base/p)!=expected:raise ValueError('Delivery bytes drift '+p)
    if portable['status']!='RESTORED_REPORT_EXECUTED_WITH_ORIGINAL_ATTACHMENT_FORBIDDEN' or portable['actual_child_exit_code']:
        raise ValueError('Actual restored report execution missing')
    if len(portable['unchanged_scientific_tables'])!=9:raise ValueError('Portable numerical-table parity missing')
    source=load(REPORT/'source-evidence.json');questions=load(REPORT/'questions.json')
    metrics=load(REPORT/'primary-metrics.json');figure_index=load(FIGURES/'index.json')
    fe=load('results/FeS-feasibility-pilot-v1/completion.json')
    fe_physical=load('results/FeS-feasibility-pilot-v1/physical-problem/problem.json')
    fe_verify=load('results/FeS-feasibility-pilot-v1/RAACS-full-cost/independent-validation.json')
    fe_sources=load('results/FeS-feasibility-pilot-v1/public-inputs/provenance.json')
    if [q['number'] for q in questions]!=list(range(1,16)) or [m['number'] for m in metrics]!=list(range(1,11)):
        raise ValueError('Question/metric scope reduced')
    if [r['figure'] for r in figure_index['rows']]!=list(range(1,19)):raise ValueError('Figure scope reduced')
    questions=json.loads(json.dumps(questions))
    questions[11]['answer']+=(' その後、著者公開積分のFe2 CAS(30e,20o)/40qを上限付きで1回実行し、'
        '2000model samples・4演算子・d8・48CNOT/29Depth、独立検算まで完了した。'
        '初期占有配置はoptimizedRHFと未確認で、ground accuracyや他手法優位の証拠ではない。'
        'Fe4の公開入力は72qで、認定62-bit表現を超えるため条件付き未実施。')
    questions[11]['answer']=questions[11]['answer'].replace('Fe–Sは未実施で適用範囲を外挿しない。','Fe–Sに関する後続の追加証拠は次のとおり。')
    questions[11]['evidence']+=['results/FeS-feasibility-pilot-v1/report-v1/REPORT_JA.md',
        'results/full-objective-audit-2026-10-23-v3/audit.json']
    xml=Path(test_xml or ROOT/'results/full-sampler-regression-2026-10-23-v11.xml').resolve()
    tests=test_receipt(xml);inputs[str(xml.relative_to(ROOT))]=sha(xml)
    if tests['tests']<419:raise ValueError('Final suite does not include portability unit tests')
    def link(p,label=None):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        return f'[{label or p.name}]({os.path.relpath(p,out)})'
    outputs={'questions-final.json':canonical(questions),'primary-metrics.json':canonical(metrics),
        'all-367-original-objective-items.csv':csv_text(items),'all-35-requirement-judgments.csv':csv_text(judgments),
        'all-18-figure-relationships.json':canonical(figure_index['rows'])}
    lines=['# QSCI/SQDサンプラー研究：最終成果と限界','',
        '## 結論','',
        '既存データの再解析、強化ベースライン、FCI不要停止法、RAACS、分子スケーリング、'
        'full fermionic SQDと合成gate-noiseの各研究工程を実行し、結果・図・コードを整理した。'
        '**普遍的に優れる新サンプラーを実証できた、という結論ではない。**',
        '研究の実行・納品と、性能仮説の成立を分ける。負の結果・不確実性・欠測・上限停止は保持した。',
        '', '## 読む順番','',
        '1. '+link(ROOT/'dashboard/CEO_LUCJ_Explorer.html','CEO × LUCJ 比較UI（元30条件とB1）'),
        '2. '+link(FIGURES/'INDEX.md','全18図：対象・条件・根拠・注意点'),
        '3. '+link(REPORT/'REPORT_JA.md','主比較の詳細レポート（保存時点v3）'),
        '4. '+link('results/FeS-feasibility-pilot-v1/report-v1/REPORT_JA.md','Fe–S後続補遺'),
        '5. '+link('results/reproducibility-delivery-v1/README.md','コード保存版とrawデータ索引')+' と '+link('results/portable-delivery-addon-v1/README.md','復元レポートの実行方法'),
        '', '## 各工程で実際に終わったこと','',
        '|工程|実施・確認したこと|結論の境界|','|---|---|---|',
        '|A 再解析|元30条件・5236checkpoint、M10–100000、R8–512、全K・配置追跡・複数回路比較|FCIは評価専用。最良途中点は事後選択|',
        '|B 強い対照|LUCJ13構築法390予定/380回路、ADAPT-QSCI/HF分離、TE/SqDRIFT/測定基底/random/SCI|失敗を欠測として保持。最良可能LUCJ/SHCIの保証ではない|',
        '|C 実用停止|LOMO90fold、未使用geometry/活性空間の実成長112件|事前選択主方式のCNOT/Depth節約0%。無料early stoppingを実証せず|',
        '|D RAACS|840構築、5score、角度・浅回路配分対照、独立readout、同初期情報の古典fork|一般的優位・量子測定の必要性は未確立|',
        '|E 大型系|26物理問題8–44q、234終端/146完了/88保存失敗、native routing/bond診断。追加Fe2は40qで完了|大型ground referenceは未認定。Fe4/72qは表現上限による条件付き未実施|',
        '|F SQD/noise|full fermionic selected-ground708/720完了。gate-noise3537/3600完了、60元欠測/3容量打切り|合成target/gate-local noise。実ハードウェアshots0|',
        '|費用|given-integrals90予定/86完了/4CCSD失敗、全pilot/final/CI/保存を含むfresh-process計測|過去CEO訓練の量子測定費用はunknown。QPU速度の測定ではない|',
        '', '## 何が効いたか・何が未解決か','',
        '- **poolだけでは説明できない。** 最適化LUCJ、測定基底、実CI次元、後処理を変えると順位も変わる。'
        '固定CCSD-LUCJへの差をansatz全般への勝利としない。',
        '- **VQEの低energyと良いQSCI samplerは同一ではない。** operator pathを揃えた角度対照と実成長対照を残す。'
        '角度・選択経路・RNGの相互作用は残り、単独の因果効果を認定しない。',
        '- **量子サンプリングの固有の必要性は示せなかった。** 3条件9比較のうち4組は、同初期情報の古典直接追加と同じ配置空間。'
        'N2で古典が低energyでもdが大きく、同次元勝利とはしない。',
        '- **Recoveryによる改善をすべて量子回路へ帰属しない。** 古典生成配置、spin-product、batch/carryoverも区別する。',
        '- **軽い最終回路と、安い構築は別。** 探索pilot・過去の成長費用・古典CI費用を無視しない。'
        '未知の古典→QPU測定換算を捏造しない。',
        '', '## 元の15問への回答','']
    for q in questions:
        evidence=[]
        for p in q['evidence']:
            actual=ROOT/p if p.startswith(('results/','study-v1/')) else REPORT/p
            evidence.append(link(actual))
        lines += [f'### {q["number"]}. {q["title"]}','',q['answer'],'',f'状態：`{q["status"]}`。根拠：'+ '、'.join(evidence),'']
    lines += ['## 10主指標','', '|指標|状態・範囲|','|---|---|']
    lines += [f'|{r["number"]}. {r["name"]}|{r["status"]}：{r["scope"]}|' for r in metrics]
    lines += ['', '## 全18種類の図','', '|番号|関係|代表図|','|---|---|---|']
    lines += [f'|{r["figure"]}|{r["title"]}|'+link(r['representative'],'PNG')+' / '+link(str(Path(r['representative']).with_suffix('.svg')),'SVG')+' / '+link(str(Path(r['representative']).with_suffix('.pdf')),'PDF')+'|' for r in figure_index['rows']]
    lines += ['', '代表18図は軸・単位・対象・制限を目視確認済み。索引1503exportsをSHA結合したが、'
        '全1503枚を一枚ずつ目視・全数値を再計算したとは主張しない。各数値の検査は元の独立監査を参照。',
        '', '## 公平性・保存・再現性','',
        '- 元330セル・5236checkpoint、過去の失敗、MAX_STEPS、古いwhole_goal_complete=falseは保存時点の記録として変更していない。',
        '- 共通M/Rと実d一致を併記。完全な同仕事量と呼ばない。品質比較と自然workflow費用比較を分ける。',
        '- 分子内のgeometry/R/seedの反復を独立した分子数へ水増ししない。分子単位集計とcluster uncertaintyを使う。',
        '- 実測モデルサンプル・実ハードウェアshots・古典秒数・gate×sample proxyは別単位。',
        f'- 統合tests：{tests["tests"]} passed、0 failed、0 skipped。'+link(xml,'実際のXML')+'。テスト件数だけを科学的成功の証明としない。',
        '- 小さなコードarchiveと13GiB級rawデータ索引を保存。rawデータは複製・削除・公開していない。',
        '- 復元ソースから実レポートを生成し、元の添付goalへのopenを禁止しても成功。科学結果9表はbyte-identical。'
        '新しい環境への依存ライブラリ再インストールや全実験の全面再実行をしたとは主張しない。',
        '- Node.jsがdashboard testsに必要。SqDRIFTには別の固定SDK overlayがある。',
        '', '## 条件付き・任意・未成立のもの','',
        '- [4Fe–4S]は公開入力が存在するが72qで、現在認定済み62-bit backendを超える。未認定代替・無断active-space縮小で埋めない。',
        '- 必須ではない追加候補k-UpCCGSD/tUPS/QNP/GQE-QSCI/TrimSQDは未実装。別物で実装済みと偽らない。',
        '- 大型ground accuracy・実機noise耐性・未知分子一般優位・普遍的RAACS/CEO優位・歴史的CEO総測定費用優位は未成立/未知。',
        '- 原著の全figure/全subroutineの厳密な再現、optimizer-independent最適性、唯一の因果分解、投稿・採択保証は主張しない。',
        '', '## 実行研究としての判断','',
        '要求された比較・診断・実用経路・上限付きスケーリング・合成noise評価を実行し、'
        '各結果を保存・可視化・解釈する工程は整理済み。失敗セルはその条件で評価できなかった証拠で、'
        '物理手法の敗北や科学的優越性の成功へ変換しない。',
        '今後、未成立の一般優位を狙って別のheuristic・分子・監査を無計画に増やすべきではない。'
        '今回の研究の納品と、次の研究仮説への追加投資は別の判断である。',
        '本レポートが示すのは、どこまで実行・検証され、何が条件依存で、何が未確定かである。'
        '最終goalの状態変更は、この全要件・証拠照合を読んだ後の判断として別途行う。','']
    outputs['REPORT_JA.md']='\n'.join(lines)+'\n'
    for name,body in outputs.items():save_once(out/name,body)
    links=validate_links(out/'REPORT_JA.md')
    record={'schema':'final-research-delivery-review-v1','status':'FULL_SCOPE_DELIVERABLES_RECONCILED_WITH_NEGATIVE_AND_CONDITIONAL_RESULTS',
        'objective_sha256':sha(OBJECTIVE),'input_sha256':inputs,
        'output_sha256':{p:sha(out/p) for p in outputs},'script_sha256':sha(__file__),
        'original_mapped_items':367,'requirement_judgments':35,'questions':15,'primary_metrics':10,'figure_types':18,
        'checked_report_links':len(links),'current_source_review_status':current['status'],
        'full_local_test_receipt':tests,'portable_report_executed':True,'scientific_tables_portable_byte_identical':9,
        'conditional_not_executed':['Fe4S4 72q: outside currently qualified62-bit route'],
        'optional_not_implemented':['k-UpCCGSD','tUPS/oo-tUPS','QNP','GQE-QSCI','TrimSQD'],
        'scientific_claims_not_established':audit['scientific_claims_not_established'],
        'execution_review_does_not_establish_positive_scientific_superiority':True,
        'new_experiments':0,'new_solver_calls':0,'hardware_shots':0,
        'goal_completion_decision':'SEPARATE_AGENT_REVIEW_REQUIRED_NOT_AUTOMATICALLY_SET'}
    save_once(out/'review.json',canonical(record))
    return {k:record[k] for k in ('status','original_mapped_items','questions','primary_metrics','figure_types','checked_report_links','goal_completion_decision')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--test-xml',type=Path)
    a=p.parse_args();print(json.dumps(build(a.output,a.test_xml),ensure_ascii=False))
