"""Index every captured protocol/freeze/binding-shaped JSON without relabelling it."""
from pathlib import Path
import sys,json,gzip,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import write_csv
rows=[]
with gzip.open(ROOT/"data/source-inventory.jsonl.gz","rt")as f:
    for line in f:
        item=json.loads(line);name=Path(item["source_relative_path"]).name
        if name.endswith(".json")and re.search(r"(^|[-_])(protocol|freeze|binding|contract)([._-]|$)",name,re.I):
            rows.append({**item,"classification":"potential configuration/execution binding; not individually certified as an independent policy"})
write_csv(ROOT/"data/policy-records.csv",rows)
registry=ROOT/"configs/policy-registry.json";value=json.loads(registry.read_text())
value["all_captured_configuration_records"]="data/policy-records.csv"
value["configuration_record_count"]=len(rows)
registry.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
print("Indexed",len(rows),"saved configuration/binding records")
