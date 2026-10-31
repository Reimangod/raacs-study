"""Read-only post-capture comparison against every indexed original source."""
import argparse,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import sha256_file,write_json

p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True)
p.add_argument("--output",type=Path,default=ROOT/"verification/source-preservation.json")
a=p.parse_args();workspace=a.workspace.resolve();errors=[];checked=0;total=0;tick=time.monotonic()
with gzip.open(ROOT/"data/source-inventory.jsonl.gz","rt")as stream:
    for line in stream:
        row=json.loads(line);path=workspace/row["source_relative_path"]
        if not path.is_file()or path.stat().st_size!=row["bytes"]or sha256_file(path)!=row["sha256"]:
            errors.append(row["source_relative_path"])
        checked+=1;total+=row["bytes"]
        if checked%25000==0:print(json.dumps({"original_files_checked":checked}),flush=True)
for row in json.loads((ROOT/"data/copied-source-manifest.json").read_text()):
    path=workspace/row["source_relative_path"]
    if not path.is_file()or sha256_file(path)!=row["sha256"]:errors.append(row["source_relative_path"])
result={"passed":not errors,"archived_original_files_checked":checked,"copied_originals_checked":len(json.loads((ROOT/"data/copied-source-manifest.json").read_text())),
        "original_bytes_checked":total,"mismatches":errors,"read_only":True,"wall_seconds":time.monotonic()-tick}
write_json(a.output,result);print(json.dumps(result),flush=True)
if errors:raise SystemExit(1)
