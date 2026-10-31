"""Read-only validation of an old synthesis after its live runtime advances.

Only explicitly mapped byte-identical runtime archives may replace a changed
live input. This never treats an old runtime snapshot as the current status.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def resolve_input(source,expected,entries,root):
    source=Path(source);p=source if source.is_absolute() else root/source
    if p.exists() and digest(p)==expected:return p,False
    if str(source) not in entries:raise ValueError('Unmapped changed historical source: '+str(source))
    record=entries[str(source)]
    if record['sha256']!=expected:raise ValueError('Archive does not match expected scientific identity')
    archived=(root/record['archive']).resolve()
    if not archived.is_relative_to((root/'results').resolve()):raise ValueError('Archive must remain within project results')
    if digest(archived)!=expected:raise ValueError('Changed archive bytes')
    return archived,True


def validate(snapshot,archive_map,root=ROOT):
    snapshot=Path(snapshot);mapping=json.loads(Path(archive_map).read_text())
    if mapping['schema']!='explicit-pre-resume-runtime-source-map-v1':raise ValueError('Unknown archive schema')
    parent=json.loads((snapshot/'source-evidence.json').read_text())
    review=json.loads((snapshot/'review-v1/validation.json').read_text())
    assert review['status']=='SYNTHESIS_PAIRED_MEANS_AND_ASSOCIATIONS_RECONSTRUCTED_DIRECT_RAACS_SUPPLEMENT'
    expected=dict(parent['source_sha256']);expected.update(review['input_sha256']);resolved={};fallback=[]
    for source,sha in expected.items():
        p,archived=resolve_input(source,sha,mapping['entries'],root)
        resolved[source]={'resolved_path':str(p),'sha256':sha,'archived':archived}
        if archived:fallback.append(source)
    for name,sha in parent['table_sha256'].items():assert digest(snapshot/name)==sha
    for name,sha in review['output_sha256'].items():assert digest(snapshot/'review-v1'/name)==sha
    producer='research_synthesis_v2.py' if parent['schema'].endswith('-v2') else 'research_synthesis.py'
    assert digest(root/'benchmark'/producer)==parent['script_sha256']
    assert digest(root/'benchmark/research_synthesis_review.py')==review['reviewer_sha256']
    return {'status':'HISTORICAL_REVIEW_INPUTS_AND_OUTPUTS_VALIDATED_VIA_EXACT_ARCHIVE',
        'snapshot':str(snapshot),'archive_map_sha256':digest(archive_map),'validator_sha256':digest(__file__),
        'input_files_checked':len(resolved),'archived_input_origins_used':fallback,
        'parent_table_count':len(parent['table_sha256']),'review_output_count':len(review['output_sha256']),
        'historical_status_is_not_current_runtime':True,'new_statistical_reanalysis':False,
        'new_draws':0,'new_solver_calls':0,'whole_goal_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--archive-map',required=True);a=p.parse_args()
    print(json.dumps(validate(a.snapshot,a.archive_map),sort_keys=True))
