"""Download one private asset through gh and verify an original protocol blob."""
import json,subprocess,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import sha256_file,write_json
from raacs_study.archives import extract_original
repo="Reimangod/raacs-study";tag="data-v0.1.0";name="b4-small-molecule-comparison-2026-10-28-v1.zip"
def api(path):return json.loads(subprocess.run(["gh","api",path],check=True,capture_output=True,text=True).stdout)
release=next(r for r in api("repos/"+repo+"/releases?per_page=100")if r["tag_name"]==tag)
asset=next(r for r in api(f"repos/{repo}/releases/{release['id']}/assets?per_page=100")if r["name"]==name)
entry=next(r for r in json.loads((ROOT/"data/archive-manifest.json").read_text())["archives"]if r["name"]==name)
if asset["state"]!="uploaded":raise RuntimeError("Asset is not complete")
with tempfile.TemporaryDirectory(prefix="raacs-remote-smoke-")as folder:
    directory=Path(folder);download=directory/name
    with download.open("xb")as dest:
        subprocess.run(["gh","api",f"repos/{repo}/releases/assets/{asset['id']}","-H","Accept: application/octet-stream"],stdout=dest,check=True)
    digest=sha256_file(download)
    if download.stat().st_size!=entry["bytes"]or digest!=entry["sha256"]:raise RuntimeError("Downloaded ZIP mismatch")
    original="outputs/b4-small-molecule-comparison-2026-10-28-v1/protocol.json"
    extracted=directory/"original-protocol.json"
    result=extract_original(ROOT,directory,original,extracted)
    if extracted.read_bytes()!=(ROOT/"configs/small-original-protocol.json").read_bytes():raise RuntimeError("Remote protocol differs from committed original")
    report={"passed":True,"scope":"one real private-asset download plus original-blob extraction, not re-download of every archive",
            "asset":name,"asset_bytes":entry["bytes"],"asset_sha256":digest,
            "original_path":original,"original_sha256":result["sha256"],"original_bytes_match_committed":True,
            "new_model_samples":0,"new_CI_solves":0}
write_json(ROOT/"verification/remote-download-smoke.json",report);print(json.dumps(report,indent=2))
