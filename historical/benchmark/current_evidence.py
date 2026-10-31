"""Current-source final synthesis. Read-only with respect to all experiments.

Preserves historical v2 and its archived live inputs. No numerical backend is
imported, no experiment/FCI/CI solve is run, and no old figure is rerendered.
This report distinguishes completed tasks, supported claims and unmet scope.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
import statistics

ROOT=Path(__file__).resolve().parents[1]
OBJECTIVE=Path('/Users/rei/.codex/attachments/b0f3686d-83a1-477c-bc5a-3ec6a43e7830/goal-objective.md')
HISTORICAL='results/research-synthesis-standard-v2/snapshots/061f6d964a47e327fa2ed723e8a70d65ee7fdda4ef067efe16f139eaee9fc86e'
ARCHIVES='results/E-F-resume-2026-10-23-v1/historical-source-map-v1.json'
SOURCES={
 'A':'results/sampler-study-standard-v1/milestone1-audit.json',
 'B1':'results/sampler-baselines-B1-standard-v1/validation.json',
 'B2':'results/sampler-baselines-B2-exactprep-standard-v2/validation.json',
 'B3':'results/nonvariational-B3-standard-v1/validation.json',
 'SqDRIFT':'results/SqDRIFT-ensemble-standard-v2/validation.json',
 'B4':'results/random-B4-standard-v1/validation.json',
 'C_replay':'results/PhaseC-replay-standard-v1/validation.json',
 'C_actual':'results/PhaseC-actual-growth-standard-v1/validation.json',
 'D':'results/RAACS-independent-comparison-standard-sharded-v1/validation.json',
 'D_angles':'results/online-parameter-standard-v1/validation.json',
 'D_bank':'results/fixed-bank-allocation-standard-v1/validation.json',
 'multibasis':'results/multibasis-NOCI-standard-v2/validation.json',
 'SQD':'results/full-fermionic-SQD-projected-standard-v1/selected-ground-validation.json',
 'E':'results/molecular-strong-controls-standard-v2/validation.json',
 'E_physics':'results/molecular-strong-controls-standard-v2/physical-layout-validation-v2.json',
 'E_native':'results/molecular-strong-postprocess-standard-v1-review-v1/validation.json',
 'HCI':'results/integral-threshold-HCI-standard-v1/validation.json',
 'HCI_paired':'results/HCI-paired-diagnostic-standard-v1/result.json',
 'HCI_review':'results/HCI-paired-diagnostic-standard-v1-review-v2/validation.json',
 'F':'results/gate-noise-full-standard-v1/validation.json',
 'F_done':'results/gate-noise-full-standard-v1/completion-sequence-v1.json',
 'cost':'results/total-workflow-standard-v1/validation.json',
 'cost_done':'results/total-workflow-sequence-v1/execution-status.json',
 'direct':'results/RAACS-direct-configuration-fork-v1/report-v2/validation.json',
 'TopK':'results/TopK-discovery-final-checkpoint-figures-v1/independent-value-review-v1.json',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'


def save_once(path,body):
    path=Path(path);body=body.encode() if isinstance(body,str) else body
    if path.exists():
        if path.read_bytes()!=body:raise ValueError('Existing derivative changed; use successor: '+str(path))
    else:path.write_bytes(body)


def csv_text(rows):
    if not rows:raise ValueError('Cannot silently publish an empty planned table')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    f=StringIO();w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows);return f.getvalue()


def terminal_counts(counts):
    nonterminal={'RUNNING','PENDING','NOT_STARTED','STARTED','QUEUED'}
    return sum(n for status,n in counts.items() if status not in nonterminal)


def hci_summary(rows):
    """All eps/R groups retained; R/eps repeats are not independent molecules."""
    groups=defaultdict(list)
    for r in rows:groups[(r['method'],r['R_cap'],r['epsilon1_Ha'])].append(r)
    result=[]
    for (method,rcap,eps),group in sorted(groups.items()):
        available=[r for r in group if r['status']=='PAIRED_DIMENSION_DIAGNOSTIC']
        nontrivial=[r for r in available if int(r['matched_d'])>1]
        differences=[float(r['sampler_minus_truncated_HCI_mHa']) for r in nontrivial]
        result.append(dict(method=method,R_cap=int(rcap),epsilon1_Ha=float(eps),planned=len(group),
            available=len(available),nontrivial=len(nontrivial),trivial_HF=len(available)-len(nontrivial),
            missing=len(group)-len(available),
            median_sampler_minus_truncated_HCI_mHa=statistics.median(differences) if differences else None,
            negative_favors_sampler=sum(x < -1e-8 for x in differences),
            within_numerical_1e_8_mHa=sum(abs(x)<=1e-8 for x in differences),
            positive_favors_truncated_HCI=sum(x > 1e-8 for x in differences),
            ground_state_error=None,work_matched=False,
            scope='Descriptive conditional same-d diagnostic; not native HCI/SHCI or independent molecular replicates'))
    return result


def workflow_summary(data):
    methods=sorted({r['method'] for r in data['rows']});result=[]
    for method in methods:
        rows=[r for r in data['rows'] if r['method']==method]
        good=[r for r in rows if r['status']=='COMPLETED']
        mol=defaultdict(list)
        for r in good:mol[r['molecule']].append(r)
        def equal_molecule(field):
            values=[statistics.mean(r[field] for r in g) for g in mol.values()]
            return statistics.mean(values) if values else None
        result.append(dict(method=method,planned=len(rows),completed=len(good),missing_or_failed=len(rows)-len(good),
            available_molecules=len(mol),
            equal_molecule_observed_total_wall_seconds=equal_molecule('total_workflow_wall_seconds'),
            equal_molecule_all_paid_CNOT_times_model_samples=equal_molecule('all_draw_CNOT_times_shots_proxy'),
            scope='Available geometries only; given-integrals, one realization; not paired speedup',
            historical_CEO_training_shot_cost=None,hardware_shots=0))
    return result


def update_questions(old):
    q=json.loads(json.dumps(old))
    changes={
      10:('SCOPED_ABLATIONS_QUANTUM_NECESSITY_NOT_ESTABLISHED',
          '角度・operator path・固定bankの対照を実施。追加の同初期情報診断では3条件・9既存RAACS軌道中4組が古典直接追加と同じ配置集合・energyになった。量子測定が必要だったとは示せない。N2では古典d64対量子d23–26であり同次元勝利ではない。局所pool/探索/測定/後処理は相互作用し、唯一の加法的因果分解は行わない。',
          [SOURCES['direct'],'study-v1/ATTRIBUTION_AND_COST_INTERPRETATION.md']),
      11:('GIVEN_INTEGRALS_TOTAL_WORKFLOW_MEASURED_NOT_HISTORICAL_CEO_TOTAL',
          '新規の自然workflow比較90予定・86完了・4CCSD初期化失敗を記録。RAACS/fixed-LUCJ/QSCI4-LUCJの全pilot・最終readout・CI・保存を含むfresh-process時間を測定。積分生成/共有変換/外部監査を除き、1実現・host/JIT cacheは非統制。過去CEOの未知の訓練shot費用を補完するものではなく、総費用の一般的優越性は未確立。',
          ['workflow-method-summary.csv','all-planned-workflows.csv','workflow-commonR-and-actuald.csv']),
      12:('MOLECULAR_SCALING_EXECUTED_WITH_PRESERVED_APPROXIMATION_FAILURES',
          '26物理問題、8–44qubitの強対照234セルを終端まで実施（146完了・88保存失敗）。MPS近似不適格・resource cap・CCSD非収束は物理手法の敗北ではない。native routing240回路と同回路bond128/256診断を保存。信頼できる大型基底状態参照がなく、化学精度やgeneric scaling優位は示せない。Fe–Sは未実施で適用範囲を外挿しない。',
          [SOURCES['E'],SOURCES['E_physics'],SOURCES['E_native']]),
      14:('FULL_SQD_AND_GATE_NOISE_EXECUTED_SCOPED_TRADEOFFS',
          'full fermionic SQD selected-groundは708/720完了。追加gate-noiseは3600予定中3537完了、60元CCSD欠測、3容量打切りを保持し、全23370選択空間を独立対角化した。ideal/readout/2Q depolarizing/gate-local T1T2/combined、all-to-all/lineを比較。Recovery・spin-product・batch/carryoverが同時に変化するので、Recovery単独因果効果や実機耐性としない。',
          ['gate-noise-paired-molecule-contrasts.csv',SOURCES['F']]),
      15:('THRESHOLD_HCI_AND_SAME_D_DIAGNOSTICS_EXECUTED_NO_QUANTUM_ADVANTAGE',
          'epsilon-threshold HCI168セル・464選択空間を独立検証。大規模26条件のsame-d診断1872予定中1080組利用可（714非自明・366HF-only）、792欠測。係数切詰HCIとfrequency切詰samplerを再求解した対照であり、native HCI endpoint・最適化SHCI・matched workではない。同初期情報の古典直接追加もあるため、RAACSの量子固有の追加価値はここでは主張しない。',
          ['HCI-all-planned-same-d.csv','HCI-conditional-summary.csv',SOURCES['HCI'],SOURCES['direct']]),
    }
    for row in q:
        if row['number'] in changes:
            row['status'],row['answer'],row['evidence']=changes[row['number']]
        else:
            row['evidence']=[HISTORICAL+'/'+p for p in row['evidence']]
    if [r['number'] for r in q]!=list(range(1,16)):raise ValueError('Missing objective question')
    return q


def metric_rows(old):
    rows=json.loads(json.dumps(old))
    changes={6:('SYNTHETIC_TARGETS_EXECUTED','Original ECR, E240CX compilations, F236 reconstructed native circuits. Not calibrated QPU.'),
             8:('MEASURED_GIVEN_INTEGRALS_THREE_METHOD_SCOPE','Fresh-process total time90planned/86complete includes setup/construction/allpilots/final/CI/persistence after fixed integrals. Historical CEO total remains unknown.'),
             10:('KNOWN_PAID_MODEL_SAMPLES_SCOPED_UNKNOWN_VQE_COST_RETAINED','Cost cohort2000all-paid model samples per completed method; F884475committed. Hardware0, historical VQE/ADAPT measurement cost unknown; no QPU-time conversion.')}
    for row in rows:
        if row['number'] in changes:row['status'],row['scope']=changes[row['number']]
    if len(rows)!=10:raise ValueError('Missing primary metric')
    return rows


def build(output=None,mode='standard'):
    from .historical_synthesis_validation import validate as historical_validate
    if mode not in ('quick','standard'):raise ValueError('Unregistered mode')
    # Both modes synthesize all completed outcomes. Quick is an IO/CLI check,
    # not a new scientifically independent two-case experiment.
    sources={}
    def bind(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        sources[key]=sha(p);return p
    def read(p):return json.loads(bind(p).read_text())
    def table(p):
        with bind(p).open() as f:return list(csv.DictReader(f))
    historical=historical_validate(ROOT/HISTORICAL,ROOT/ARCHIVES)
    bind(ARCHIVES)
    old=read(HISTORICAL+'/source-evidence.json')
    for name,expected in old['table_sha256'].items():
        if sha(bind(HISTORICAL+'/'+name))!=expected:raise ValueError('Historical table drift')
    read(HISTORICAL+'/review-v1/validation.json')
    data={key:read(path) for key,path in SOURCES.items()}
    if data['F_done']['status']!='GATE_NOISE_EXECUTED_AUDITED_REPORTED':raise ValueError('F not finished')
    if data['cost_done']['status']!='WORKFLOW_TOTAL_COST_COHORTS_AUDITED_REPORTED':raise ValueError('Cost not finished')
    if data['HCI_review']['status']!='ALL_PLANNED_PAIRS_AND_NAMESPACED_PHYSICAL_IDENTITIES_REVIEWED':raise ValueError('Paired review missing')
    if data['cost']['terminal']!=90 or data['cost']['unexpected_failures']:raise ValueError('Cost terminal coverage incomplete')
    db=bind('results/gate-noise-full-standard-v1/events.sqlite')
    if sha(db)!=data['F']['database_sha256']:raise ValueError('F database drift')
    with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as connection:
        counts=dict(connection.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
        chunks,shots=connection.execute('SELECT count(*),sum(shots) FROM chunks').fetchone()
    if terminal_counts(counts)!=data['F']['planned'] or shots!=data['F']['known_committed_new_model_shots']:
        raise ValueError('Actual F counts inconsistent')
    # Review inputs are live scientific artifacts, not just a green status label.
    for path,expected in data['HCI_review']['dependencies'].items():
        if sha(bind(path))!=expected:raise ValueError('Paired review input drift')
    for path,expected in data['cost']['input_hashes'].items():
        if sha(bind(ROOT/'results/total-workflow-standard-v1'/path))!=expected:raise ValueError('Workflow input drift')
    paired=table('results/HCI-paired-diagnostic-standard-v1-review-v2/all-planned-pairs.csv')
    if len(paired)!=data['HCI_review']['planned_pairs']:raise ValueError('Paired row coverage')
    questions=update_questions(old['questions']);metrics=metric_rows(old['primary_metrics'])
    outputs={
      'questions.json':canonical(questions),'primary-metrics.json':canonical(metrics),
      'HCI-all-planned-same-d.csv':csv_text(paired),
      'HCI-conditional-summary.csv':csv_text(hci_summary(paired)),
      'workflow-method-summary.csv':csv_text(workflow_summary(data['cost'])),
      'workflow-commonR-and-actuald.csv':csv_text(data['cost']['comparisons']),
      'all-planned-workflows.csv':bind('results/total-workflow-standard-v1/report-v1/all-planned-workflows.csv').read_text(),
      'gate-noise-paired-molecule-contrasts.csv':bind('results/gate-noise-full-standard-v1/report-v1/paired-molecule-contrasts.csv').read_text(),
      'direct-configuration-comparisons.csv':bind('results/RAACS-direct-configuration-fork-v1/report-v2/comparison-corrected.csv').read_text(),
      'historical-validation.json':canonical(historical),
    }
    # Evidence titles come from the full current owner objective, not a shortened plan.
    objective=bind(OBJECTIVE).read_text()
    from .research_synthesis_v2 import question_titles
    titles=question_titles(objective)
    if any(q['title']!=titles[q['number']] for q in questions):raise ValueError('Objective question drift')
    bind('benchmark/historical_synthesis_validation.py');bind('benchmark/current_evidence.py')
    bind('study-v1/ATTRIBUTION_AND_COST_INTERPRETATION.md');bind('study-v1/HCI_PAIRED_REVIEW_NAMESPACE_FIX.md')
    record={'schema':'current-source-scalable-sampler-synthesis-v3','mode':mode,
      'historical_tables_are_immutable_reused_evidence_not_current_runtime':True,
      'runtime':{'E':dict(planned=data['E']['planned'],terminal=data['E']['records'],
                         statuses=dict(Counter(r['status'] for r in data['E']['rows']))),
                 'F':dict(planned=data['F']['planned'],terminal=terminal_counts(counts),statuses=counts,
                          committed_chunks=chunks,known_model_samples=shots),
                 'cost':dict(planned=90,terminal=90,completed=data['cost']['completed']),
                 'HCI_same_d':{k:v for k,v in data['HCI_review'].items() if k not in ('dependencies','physical_identity_records')}},
      'sources':SOURCES,'source_sha256':sources,
      'output_sha256':{p:hashlib.sha256(b.encode()).hexdigest() for p,b in outputs.items()},
      'questions':15,'primary_metrics':10,'new_experiments':0,'new_solver_calls':0,
      'new_FCI_solves':0,'hardware_shots':0,'whole_goal_complete':False,
      'remaining_scope':['Unified18-figure semantic/source inventory and final CLI verification',
                         'Requirement-by-requirement full-objective and conditional LevelC applicability audit',
                         'Historical VQE end-to-end quantum cost is unknown, not measured by given-integrals cohort']}
    identity=hashlib.sha256(canonical(record).encode()).hexdigest()
    out=Path(output or ROOT/'results/research-synthesis-standard-v3')/'snapshots'/identity
    out.mkdir(parents=True,exist_ok=True)
    lines=['# QSCI/SQDサンプラー研究：最新結果の統合 v3','',
      '新しい分子実験は追加せず、完了済み結果と独立検証を統合した。全目的達成の証明ではない。',
      '過去v2の検証済み17表はそのまま再利用し、当時のE/F未完了状態だけを現在の状態へ更新した。','',
      '## 要点','',
      '- CEOが固定LUCJより良い場合はあるが、最適化方法・実CI次元・構築費用によって順位は変わる。',
      '- 事後の早いcheckpointと、実際に早く止める方法は別。主FCI-free停止則の節約は0%。',
      '- RAACSの広範な優越性、量子測定の必要性は示していない。同初期情報の古典直接追加が4/9組で同じ空間を得た。',
      '- 大型条件はground reference未知。MPS不適格・CCSD失敗は物理的劣位ではなく欠測として残す。',
      '- all-pilot費用付きの3手法比較は完了。ただし過去CEOの総測定費用やQPU速度の実証ではない。','',
      '## 終端状態（成功数と区別）','',
      f'- E：{data["E"]["records"]}/{data["E"]["planned"]}終端。状態別：{record["runtime"]["E"]["statuses"]}。',
      f'- F：{terminal_counts(counts)}/{data["F"]["planned"]}終端。状態別：{counts}。既知committed model samples={shots:,}。',
      '- total-workflow：90/90終端、86完了、4CCSD初期化失敗。',
      f'- HCI same-d：1872予定、{data["HCI_review"]["nontrivial_available_pairs"]}非自明、'
      f'{data["HCI_review"]["trivial_HF_pairs"]}HF-only、792欠測。R/epsilon違いは独立分子ではない。','',
      '## 比較の二つの目的','',
      '**品質比較**：同一問題、共通M/R/後処理と実d一致対照を併記。実dを合わせても仕事量は同じにならない。',
      '**費用対効果**：各手法の自然な構築・sampling・CIを許し、何に払ったかを示す。'
      '時間・shots・Depthを無理に全部同時一致させない。未知の費用は0にしない。','',
      'total-workflow時間は保存済み積分を受け取ったfresh processのPopen直前から終了まで。'
      'import、private CCSD初期化、構築、全pilot/final sampling、CI、保存を含む。'
      '事前RHF/積分生成、共有export、外部検証、作図を除外。host/JIT cacheは非統制、1実現。'
      'モデルsample2000/cellを実機shotやQPU秒と同一視しない。','',
      '## 15の問いへの回答','']
    for q in questions:
        lines += [f'### {q["number"]}. {q["title"]}','',q['answer'],'',f'判定：`{q["status"]}`。',
                  '根拠：'+'、'.join(f'[{Path(p).name}]({os.path.relpath(ROOT/p if p.startswith(("results/","study-v1/")) else out/p,out)})' for p in q['evidence']),'']
    lines += ['## 10主指標の範囲','', '|指標|状態|範囲|','|---|---|---|']
    lines += [f'|{m["number"]}. {m["name"]}|{m["status"]}|{m["scope"]}|' for m in metrics]
    lines += ['','## 効果の帰属と限界','',
      'pool/ansatz、角度とoperator path、測定基底とanchor、Recovery/spin-product/carryover、'
      '古典候補探索、停止方法の六層を区別する。古典後処理で説明できる改善を量子固有効果としない。',
      '多基底は元の完全prepを基底回転の前に実行。入力ベクトル数と保持rank、数値cutoffを区別する。'
      '限定外部残差は全空間収束の証明ではない。',
      '古いstatevector小系・事後FCI評価と、実用経路のmatrix-free/MPS・counts-only処理を区別する。',
      '実機shotsは全体で0。Synthetic readout/depolarizing/gate-local thermal/routingは実機較正ではない。',
      '3seedのノイズ平均を90%成功保証としない。既存shots-to90はtested-grid・信頼区間・未達を保持。',
      '独立分子は6種類で、同一分子の距離・R・epsilonを独立nとして推測しない。'
      '追加HCI same-d統計は全群を出す記述的対照で、性能を見たepsilon選択をしない。',
      'Fe–Sは未実施。large共役系/拡張N2の存在をFe–Sへの適用証明として代用しない。',
      '原作者実装の全再現・optimizer非依存のLUCJ最適性・最適化SHCIとの優越性・未知分子一般化は未確立。','',
      '## 再利用した比較表と現在の図','',
      f'[全1320 CEO-MVPペア]({os.path.relpath(ROOT/HISTORICAL/"all-case-primary-pairs.csv",out)})、'
      f'[分子等重み比較]({os.path.relpath(ROOT/HISTORICAL/"equal-molecule-primary-comparisons.csv",out)})。',
      '最新表はこのディレクトリのCSVとsource-evidence.jsonに結合。元実験のraw counts/checkpointsは複製・変更していない。',
      '18図の統一semantic/source indexは別作業。個別結果のwhole_goal_complete=falseを総合完了へ勝手に書き換えない。','']
    outputs['REPORT_JA.md']='\n'.join(lines)+'\n'
    # The report itself is bound too; identity above depends on all its evidence.
    record['output_sha256']['REPORT_JA.md']=hashlib.sha256(outputs['REPORT_JA.md'].encode()).hexdigest()
    for p,value in sources.items():
        if sha(ROOT/p if not Path(p).is_absolute() else Path(p))!=value:raise ValueError('Source changed during synthesis')
    for name,body in outputs.items():save_once(out/name,body)
    save_once(out/'source-evidence.json',canonical(record))
    pointer=Path(output or ROOT/'results/research-synthesis-standard-v3')/'LATEST.md'
    pointer.write_text(f'# Current-source synthesis\n\n[Report](snapshots/{identity}/REPORT_JA.md)\n\nFull-objective completion remains unproven.\n')
    return {'snapshot':str(out),'status':'CURRENT_SOURCES_INTEGRATED','questions':15,'metrics':10,
            'new_experiments':0,'whole_goal_complete':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','standard'],default='standard')
    p.add_argument('--output',type=Path);a=p.parse_args();print(json.dumps(build(a.output,a.mode),sort_keys=True))


if __name__=='__main__':main()
