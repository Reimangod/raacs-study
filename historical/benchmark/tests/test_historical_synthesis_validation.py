import pytest
from benchmark.historical_synthesis_validation import digest,resolve_input


def test_changed_live_input_uses_only_exact_declared_archive(tmp_path):
    live=tmp_path/'live.json';live.write_text('old runtime')
    expected=digest(live);archive=tmp_path/'results/archive.json';archive.parent.mkdir();archive.write_text('old runtime')
    live.write_text('new runtime')
    entries={'live.json':{'sha256':expected,'archive':'results/archive.json'}}
    path,used=resolve_input('live.json',expected,entries,tmp_path)
    assert path==archive and used
    archive.write_text('wrong archived runtime')
    with pytest.raises(ValueError):resolve_input('live.json',expected,entries,tmp_path)


def test_changed_unmapped_source_never_rescued_or_zeroed(tmp_path):
    p=tmp_path/'source.csv';p.write_text('real data');old=digest(p);p.write_text('changed data')
    with pytest.raises(ValueError):resolve_input('source.csv',old,{},tmp_path)
    path,used=resolve_input('source.csv',digest(p),{},tmp_path)
    assert path==p and not used
