from io import BytesIO
import tarfile
import pytest
from benchmark.reproducibility_delivery import make_archive,verify_archive,streaming_sha


def test_archive_exact_code_bytes_and_no_implicit_data_copy(tmp_path):
    root=tmp_path/'src';root.mkdir();(root/'code.py').write_text('print("hi")\n')
    archive=tmp_path/'code.tar.gz';make_archive(archive,[root/'code.py'],root)
    destination=tmp_path/'restored';destination.mkdir()
    verify_archive(archive,{'code.py':streaming_sha(root/'code.py')},destination)
    assert (destination/'code.py').read_bytes()==(root/'code.py').read_bytes()
    assert not (destination/'results').exists()
    with pytest.raises(ValueError):verify_archive(archive,{'code.py':'0'*64},destination)


def test_archive_rejects_traversal_even_with_matching_content_hash(tmp_path):
    archive=tmp_path/'unsafe.tar.gz';raw=b'x'
    with tarfile.open(archive,'w:gz') as tar:
        member=tarfile.TarInfo('../escape.py');member.size=1;tar.addfile(member,BytesIO(raw))
    with pytest.raises(ValueError):verify_archive(archive,{'../escape.py':'irrelevant'},tmp_path/'dest')
    assert not (tmp_path/'escape.py').exists()
