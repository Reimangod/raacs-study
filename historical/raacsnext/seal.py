"""Final delivery checks; no new scientific outcomes or protocol mutation."""
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
from time import time
from .paths import ROOT
from .io import load,save,sha256


def main():
    root=ET.parse(ROOT/'report/qualification-tests.xml').getroot()
    suites=root.findall('testsuite')if root.tag=='testsuites'else[root]
    counts={k:sum(int(s.get(k,0))for s in suites)for k in('tests','failures','errors','skipped')}
    if counts!={'tests':13,'failures':0,'errors':0,'skipped':0}:raise RuntimeError('Qualification suite not green')
    audit=load(ROOT/'report/final-independent-evidence-audit.json')
    if audit['runs_checked']!=57 or audit['unknown_draws']or audit['shared_model_shots']!=5700000:
        raise RuntimeError('Final audit incomplete')
    with sqlite3.connect('file:'+str(ROOT/'queue.sqlite')+'?mode=ro',uri=True)as db:
        states=dict(db.execute('SELECT status,COUNT(*)FROM cells GROUP BY status'))
    if states!={'COMPLETED':57,'NOT_EXECUTED_STOP_RULE':111}:raise RuntimeError('Queue has unresolved states')
    completion=load(ROOT/'FINAL_COMPLETION.json')
    for name,h in completion['evidence_SHA256'].items():
        if sha256((ROOT/name).read_bytes())!=h:raise RuntimeError('Evidence changed: '+name)
    files=[p for p in ROOT.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.suffix not in('.lock','.sqlite','.sqlite-wal','.sqlite-shm')]
    record={'status':'PASS_BOUNDED_PILOT_DELIVERY','finished_unix':time(),'tests':counts,'planned_pilot_states':states,
        'completed_scope':'Approved pilot with registered early stop, not 600-run full confirmation or superiority',
        'scientific_superiority_achieved':False,'conditional_confirmation_executed':False,
        'CPU_cap_overshoots_disclosed':True,'D0_ID_aliases_resolved_non_destructively':True,
        'source_results_or_old_scientific_code_changed':False,
        'evidence_SHA256':{str(p.relative_to(ROOT)):sha256(p.read_bytes())for p in files if p.name!='FINAL_DELIVERY_QA.json'}}
    save(ROOT/'FINAL_DELIVERY_QA.json',record)
    print({'status':record['status'],'tests':counts,'states':states,'bound_files':len(record['evidence_SHA256'])},flush=True)


if __name__=='__main__':main()
