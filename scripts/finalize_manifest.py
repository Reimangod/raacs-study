"""Seal only the derived data, source snapshots and policy bindings, not itself."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import sha256_file,write_json

files=[]
for folder in("data","configs","historical","results"):
    for p in sorted((ROOT/folder).rglob("*")):
        if p.is_file()and p.name!="derived-manifest.json"and not p.name.startswith("upload-receipts")and "downloads"not in p.parts and "__pycache__"not in p.parts:
            files.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
write_json(ROOT/"data/derived-manifest.json",{"schema":"derived-files-1","files":files})
print("Sealed",len(files),"derived/source/configuration files")
