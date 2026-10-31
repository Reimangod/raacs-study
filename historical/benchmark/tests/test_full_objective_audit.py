import json
import pytest
from benchmark.full_objective_audit import extract_items, verify_sealed
from benchmark.current_evidence import sha


def test_extraction_preserves_original_lines_and_conditional_fe():
    text='# 5. Phase E\n## E1. Molecules\n- H8\n- [2Fe–2S]\n## E2. Rules\n- FCI禁止\n# 10. CLI\npython -m benchmark.run --phase scaling --mode standard\n'
    rows=extract_items(text)
    assert next(r for r in rows if r['original_text']=='- H8')['group']=='E1AB'
    assert next(r for r in rows if 'Fe–' in r['original_text'])['group']=='E1C'
    assert next(r for r in rows if r['kind']=='named_command')['line']==8


def test_preservation_checks_bytes_not_merely_manifest_pointer(tmp_path):
    (tmp_path/'raw').write_bytes(b'original');(tmp_path/'code.zip').write_bytes(b'capsule')
    manifest=dict(artifacts={'raw':{'bytes':8,'sha256':sha(tmp_path/'raw')}},code_capsule='code.zip',code_capsule_sha256=sha(tmp_path/'code.zip'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    pointer=dict(dataset_id='fixture',manifest='manifest.json',manifest_sha256=sha(tmp_path/'manifest.json'))
    (tmp_path/'REPRODUCIBILITY.json').write_text(json.dumps(pointer))
    assert verify_sealed(tmp_path)['artifacts']==1
    (tmp_path/'raw').write_bytes(b'tampered')
    with pytest.raises(ValueError):verify_sealed(tmp_path)


def test_full_real_objective_has_no_unmapped_items_or_hidden_figures():
    from benchmark.full_objective_audit import OBJECTIVE,JUDGMENTS
    rows=extract_items(OBJECTIVE.read_text());groups={j[0] for j in JUDGMENTS}
    assert all(r['group'] in groups for r in rows)
    assert len([r for r in rows if r['kind']=='named_figure'])==18
    assert len([r for r in rows if r['group']=='B3optional'])==4
    assert any(r['group']=='Foptional' for r in rows)
    assert any(r['group']=='12target' for r in rows)
