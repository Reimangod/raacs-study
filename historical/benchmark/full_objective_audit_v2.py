"""Additive semantic correction of auditv1, preserving all numerical evidence.

V1 mistakenly grouped the four introductory performance aspirations with source
preservation. V2 explicitly classifies them as evaluated/unestablished targets.
It does not rerun45,831sealed-artifact hashes or any scientific computation;
the freshly completed v1 preservation audit is retained as evidence.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .current_evidence import ROOT,OBJECTIVE,canonical,csv_text,save_once,sha


def correct_items(rows,text,judgments):
    original=text.splitlines()
    start=next(i for i,line in enumerate(original,1) if line.startswith('# 0.'))
    target=next(j for j in judgments if j['group']=='12target')
    fixed=[]
    for row in rows:
        r=dict(row);line=int(r['line'])
        if line < start:
            if r['kind']=='list_item':
                r.update(group='12target',assessment=target['status'],evidence=target['evidence'])
            else:r['assessment']='CONTEXT_TITLE_NOT_A_SCIENTIFIC_RESULT'
        fixed.append(r)
    return fixed


def build(output=None):
    old=ROOT/'results/full-objective-audit-2026-10-23-v1'
    out=Path(output or ROOT/'results/full-objective-audit-2026-10-23-v2');out.mkdir(exist_ok=True)
    source=json.loads((old/'audit.json').read_text())
    if sha(ROOT/'benchmark/full_objective_audit.py')!=source['script_sha256']:raise ValueError('V1 auditor code drift')
    if sha(OBJECTIVE)!=source['original_objective_sha256']:raise ValueError('Objective changed')
    with (old/'original-objective-items.csv').open() as f:original=list(csv.DictReader(f))
    with (old/'requirement-judgments.csv').open() as f:judgments=list(csv.DictReader(f))
    corrected=correct_items(original,OBJECTIVE.read_text(),judgments)
    changed=[int(a['line']) for a,b in zip(original,corrected) if a!=b]
    if len(corrected)!=367 or len(changed)!=5:raise ValueError('Unexpected amendment scope')
    for r in corrected:
        if int(r['line']) in (11,12,13,14) and r['assessment']!='EVALUATED_NOT_UNIVERSALLY_ESTABLISHED':
            raise ValueError('Intro target relabelled as achieved')
    # E1AB also explicitly binds the original small-molecule cohort; extended E
    # cases do not by themselves prove LiH/BeH2 were actually executed.
    for row in judgments:
        if row['group']=='E1AB':
            row['evidence']+=';results/standard-v1/execution-manifest-final.json;results/sampler-study-standard-v1/milestone1-audit.json'
    evidence=dict(source['evidence_sha256'])
    for row in judgments:
        for path in row['evidence'].split(';'):evidence[path]=sha(ROOT/path)
    record=dict(schema='full-objective-evidence-audit-v2',
      status='FULL_OBJECTIVE_AUDIT_SEMANTIC_TARGET_CLASSIFICATION_CORRECTED',
      original_objective_sha256=sha(OBJECTIVE),
      parent_audit_sha256=sha(old/'audit.json'),parent_judgments_sha256=sha(old/'requirement-judgments.csv'),
      parent_items_sha256=sha(old/'original-objective-items.csv'),
      corrected_lines=changed,original_items=len(corrected),unmapped_items=0,
      sealed_preservation_audit_reused_not_reexecuted=source['sealed_cohorts'],
      evidence_sha256=evidence,conditional_applicability=source['conditional_applicability'],
      scientific_claims_not_established=source['scientific_claims_not_established'],
      script_sha256=sha(__file__),new_experiments=0,new_eigensolver_calls=0,hardware_shots=0,
      whole_goal_complete=False,
      scope='Corrected final item/evidence audit, not proof of universal scientific success. Original v1 and all scientific records retained.')
    save_once(out/'audit.json',canonical(record))
    save_once(out/'original-objective-items.csv',csv_text(corrected))
    save_once(out/'requirement-judgments.csv',csv_text(judgments))
    save_once(out/'READ_FIRST_JA.md',
      '# 全目的監査v2：科学的目標と実行確認の分離\n\n'
      'v1の冒頭4目標（費用・ゲート・Depth・shots削減）が保存確認と同じ分類になっていたため訂正。'
      '4目標は`EVALUATED_NOT_UNIVERSALLY_ESTABLISHED`、最初の表題は文脈であって結果ではない。'
      '元の367項目・line番号・v1・実験結果は保持し、5行だけを訂正した。'
      'E1ABの根拠には元の小分子cohortも明示的に結合した。\n\n'
      'v1で実際に再照合した元40,944・A1,345・B1 3,542 artifacts（計45,831件、7,593,306,504 bytes）'
      'の全hash/size一致はそのまま根拠として保持。再計算や再照合を新しく行ったとは数えない。\n\n'
      '[v1詳細監査](../full-objective-audit-2026-10-23-v1/AUDIT.md)と、本版2CSV/JSONを併読する。'
      '条件付きFe–S未実施、未知の過去測定費用、大型ground誤差未知を成功へ変更しない。'
      '普遍的な性能優位は未達で、データの生成・検証・可視化の実施とは別。\n')
    return dict(status=record['status'],original_items=367,corrected_lines=changed,
                new_experiments=0,whole_goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    print(json.dumps(build(p.parse_args().output),sort_keys=True))
