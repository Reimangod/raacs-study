import io,json,hashlib,zipfile
from pathlib import Path
from raacs_study.archives import JoinedParts,verify_archives


def test_joined_parts_seek_and_read(tmp_path):
    payload=b"abcdefghij0123456789"
    paths=[]
    for i,raw in enumerate((payload[:7],payload[7:15],payload[15:])):
        p=tmp_path/str(i);p.write_bytes(raw);paths.append(p)
    with JoinedParts(paths)as f:
        assert f.read(10)==payload[:10]
        assert f.seek(-4,2)==len(payload)-4 and f.read()==payload[-4:]
        f.seek(0);assert hashlib.file_digest(f,"sha256").hexdigest()==hashlib.sha256(payload).hexdigest()


def test_zip_can_read_across_part_boundaries(tmp_path):
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w")as z:z.writestr("a.txt",b"test payload")
    raw=data.getvalue();paths=[]
    for i in range(0,len(raw),13):
        p=tmp_path/str(i);p.write_bytes(raw[i:i+13]);paths.append(p)
    with JoinedParts(paths)as f:
        with zipfile.ZipFile(f)as z:assert z.read("a.txt")==b"test payload"
