"""Read-only server-side digest reconciliation of the private data delivery."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import write_json

def api(path):return json.loads(subprocess.run(["gh","api",path],check=True,capture_output=True,text=True).stdout)
repo="Reimangod/raacs-study";tag="data-v0.1.0"
details=api("repos/"+repo)
release=next(r for r in api("repos/"+repo+"/releases?per_page=100")if r["tag_name"]==tag)
assets=api(f"repos/{repo}/releases/{release['id']}/assets?per_page=100")
found={r["name"]:r for r in assets}
manifest=json.loads((ROOT/"data/archive-manifest.json").read_text())
rows=[];receipts=[]
for archive in manifest["archives"]:
    for expected in archive["parts"]:
        actual=found.get(expected["name"])
        passed=bool(actual and actual["state"]=="uploaded"and actual["size"]==expected["bytes"]and actual.get("digest")=="sha256:"+expected["sha256"])
        rows.append({"asset":expected["name"],"passed":passed,"bytes":expected["bytes"],"expected_sha256":expected["sha256"],
                     "server_state":actual["state"]if actual else"MISSING","server_digest":actual.get("digest")if actual else None})
        if passed:receipts.append({"name":actual["name"],"bytes":actual["size"],"server_digest":actual["digest"],
                                   "asset_id":actual["id"],"url":actual["browser_download_url"]})
result={"passed":bool(details["private"]and all(r["passed"]for r in rows)),"repository":repo,"private":details["private"],
        "release_id":release["id"],"release_tag":tag,"expected_assets":len(rows),"matching_uploaded_assets":len(receipts),
        "scope":"GitHub server size/digest reconciliation; not a full remote re-download or scientific validation","assets":rows}
write_json(ROOT/"verification/remote-assets.json",result)
if result["passed"]:write_json(ROOT/"data/upload-receipts.json",{"repository":repo,"private":True,"release_tag":tag,"assets":receipts})
print(json.dumps({k:v for k,v in result.items()if k!="assets"},indent=2))
if not result["passed"]:raise SystemExit(1)
