import json
import pytest
from benchmark.workflow_cost_presentation_v2 import valid_row, save_once, validate_cached, sha


def record():
    return dict(status='COMPLETED', unknown_inflight=False,
                total_workflow_wall_seconds=4., all_draw_CNOT_times_shots_proxy=0,
                readouts=[dict(R=64, ground_error_mHa=1.)])


def test_zero_gate_cost_is_valid_but_missing_is_not_zero():
    assert valid_row(record())
    r=record();r['all_draw_CNOT_times_shots_proxy']=None
    with pytest.raises(ValueError):valid_row(r)


def test_missing_or_partial_work_does_not_enter_frontier():
    r=record();r['status']='FAILED_ENGINEERING'
    assert not valid_row(r)
    r=record();r['unknown_inflight']=True
    assert not valid_row(r)


@pytest.mark.parametrize('key,value', [('total_workflow_wall_seconds',0),
                                     ('all_draw_CNOT_times_shots_proxy',-1)])
def test_invalid_cost_rejected(key,value):
    r=record();r[key]=value
    with pytest.raises(ValueError):valid_row(r)


def test_save_once_never_overwrites_existing_bytes(tmp_path):
    p=tmp_path/'record';save_once(p,b'one');t=p.stat().st_mtime_ns
    save_once(p,b'one');assert p.stat().st_mtime_ns==t
    with pytest.raises(ValueError):save_once(p,b'two')
    assert p.read_bytes()==b'one'


def test_cached_export_checks_source_and_output_without_render(tmp_path):
    p=tmp_path/'image.png';p.write_bytes(b'fixture')
    m={'input_sha256':{'source':'hash'},'output_sha256':{'image.png':sha(p)}}
    (tmp_path/'manifest.json').write_text(json.dumps(m));before=p.stat().st_mtime_ns
    assert validate_cached(tmp_path,{'source':'hash'})==m
    assert p.stat().st_mtime_ns==before
    with pytest.raises(ValueError):validate_cached(tmp_path,{'source':'other'})
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError):validate_cached(tmp_path,{'source':'hash'})
