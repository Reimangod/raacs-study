"""Finish standard audit/report without rerunning completed acquisitions."""
import argparse
import json
from pathlib import Path
from .common import ROOT,file_digest,save_json


def execute(root):
    root=Path(root)
    files=['multibasis_standard_audit.py','multibasis_standard_report.py','multibasis_completion.py']
    binding={'scope':'POST_PRODUCTION_PRE_ANALYSIS_CODE_BINDING; not a pre-outcome science freeze',
             'implementation':{p:file_digest(ROOT/'benchmark'/p) for p in files}}
    path=root/'analysis-code-binding-v1.json'
    if path.exists() and json.loads(path.read_text())!=binding:raise RuntimeError('Frozen analysis source changed; use successor')
    save_json(path,binding)
    final=root/'completion-v1.json'
    if final.exists():return json.loads(final.read_text())
    from .multibasis_standard_audit import audit
    from .multibasis_standard_report import report
    validation=audit(root);output=report(root)
    result={'status':'MULTIBASIS_ALL30_EXECUTED_AUDITED_REPORTED','tasks':validation['planned_tasks'],
        'completed':validation['tasks_by_status'].get('COMPLETED',0),'missing':validation['preserved_missing_source_tasks'],
        'report':str(root/'report-v1'),'report_manifest_sha256':file_digest(root/'report-v1/manifest.json'),
        'linked_model_shots':validation['linked_unique_quantum_model_shots'],
        'new_model_shots':validation['new_standard_quantum_model_shots'],'hardware_shots':0,'whole_goal_complete':False}
    save_json(final,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/multibasis-NOCI-standard-v2'))
    print(json.dumps(execute(p.parse_args().output)))
