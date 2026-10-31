import pytest
from benchmark.final_research_delivery import test_receipt as receipt,validate_links


def test_final_receipt_does_not_convert_failures_or_skips_to_success(tmp_path):
    p=tmp_path/'report.xml'
    p.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    assert receipt(p)['tests']==3
    for name in ('failures','errors','skipped'):
        p.write_text(f'<testsuites><testsuite tests="3" {name}="1"/></testsuites>')
        with pytest.raises(ValueError):receipt(p)


def test_final_links_must_point_to_real_artifacts(tmp_path):
    p=tmp_path/'REPORT.md';p.write_text('[missing](missing.csv)')
    with pytest.raises(ValueError):validate_links(p)
    (tmp_path/'real.csv').write_text('energy\n1\n');p.write_text('[data](real.csv)')
    assert len(validate_links(p))==1
