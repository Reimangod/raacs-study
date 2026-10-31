"""Report recorded cohort bindings vs the captured source bytes; no experiments."""
from pathlib import Path
import sys,json,gzip,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import sha256_file,write_json
p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);a=p.parse_args()
workspace=a.workspace.resolve();inventory={}
with gzip.open(ROOT/"data/source-inventory.jsonl.gz","rt")as f:
    for line in f:
        row=json.loads(line);inventory[row["source_relative_path"]]=row
copied={r["source_relative_path"]:r for r in json.loads((ROOT/"data/copied-source-manifest.json").read_text())}
rows=[]
for file in sorted((ROOT/"configs").glob("*-original-protocol.json")):
    d=json.loads(file.read_text())
    for key in("source_hashes","bindings"):
        for raw,expected in d.get(key,{}).items():
            src=Path(raw)
            try:relative=src.relative_to(workspace).as_posix()
            except ValueError:relative=None
            row={"protocol":file.relative_to(ROOT).as_posix(),"binding":raw,"expected_sha256":expected}
            record=(inventory.get(relative)or copied.get(relative))if relative else None
            if record:
                row.update(captured_sha256=record["sha256"],captured=(record.get("archive")or record.get("destination")),
                           status="MATCH"if record["sha256"]==expected else"RECORDED_HASH_DIFFERS_FROM_CAPTURE")
            elif src.is_file():
                current=sha256_file(src)
                row.update(current_source_sha256=current,status="SOURCE_PRESENT_BUT_NOT_CAPTURED",current_matches_recorded=current==expected)
            else:row["status"]="RECORDED_SOURCE_MISSING"
            rows.append(row)
from collections import Counter
write_json(ROOT/"verification/protocol-bindings.json",{"scope":"top-level scientific cohort source_hashes/bindings; not a full runtime or causal audit",
           "summary":dict(Counter(r["status"]for r in rows)),"bindings":rows})
print(dict(Counter(r["status"]for r in rows)))
