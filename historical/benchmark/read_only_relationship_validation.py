"""Verify existing complete relationship evidence WITHOUT rendering/writing.

The historical followup.run also exports presentation files on every call.
Never invoke that renderer as a cached validation operation.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    for path,s in m['source_sha256'].items():assert digest(ROOT/path)==s
    for path,s in m['output_sha256'].items():assert digest(root/path)==s
    p=root/'presentation-v2/validation-and-style-v2.json';r=json.loads(p.read_text())
    assert r['status']=='ALL_PLOTTED_TABLES_RECONSTRUCTED_PRESENTATION_V2_EXPORTED'
    assert r['source_manifest_sha256']==digest(root/'manifest.json')
    assert r['auditor_sha256']==digest(ROOT/'benchmark/final_relationship_followup.py')
    for path,s in r['presentation_output_sha256'].items():assert digest(p.parent/path)==s
    return {'status':'EXISTING_RELATIONSHIP_MANIFEST_AND_PRESENTATION_HASHES_VERIFIED_READ_ONLY',
        'root':str(root),'new_renderings':0,'new_draws':0,'existing_numerical_validation':r['status'],
        'source_manifest_sha256':r['source_manifest_sha256'],'whole_goal_complete':False}
