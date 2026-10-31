"""Local reproducibility capsule + content-addressed dataset manifest, no upload."""
import argparse
import json
from pathlib import Path
import zipfile
from .common import ROOT,digest,file_digest,save_binary,save_json


def package(root):
    root=Path(root).resolve();audit=json.loads((root/"requirements-audit.json").read_text())
    if audit["status"]!="COMPLETE_WITH_EXPLICIT_LIMITS" or audit["issues"] or audit["pending"]:
        raise RuntimeError("Complete the existing requirements, not a new experiment, before packaging")
    sources={}
    for directory in (ROOT/"benchmark",ROOT/"gpu",ROOT/"vendor/ceo"):
        for p in directory.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and (p.suffix in (".py",".txt",".md") or p.name=="LICENSE"):
                sources[str(p.relative_to(ROOT))]=p
    for name in ("README.md","AGENTS.md","STUDY_PROTOCOL.md","IMPLEMENTATION_STATUS.md","THIRD_PARTY_NOTICES.md","pyproject.toml","uv.lock"):
        sources[name]=ROOT/name
    frozen=ROOT/"results/engine-v2-qualification/runtime-source"
    for p in frozen.rglob("*"):
        if p.is_file():sources["frozen-a100-runtime/"+str(p.relative_to(frozen))]=p
    hashes={k:file_digest(v) for k,v in sorted(sources.items())};source_id=digest(hashes)
    destination=root/"provenance";destination.mkdir(exist_ok=True)
    capsule=destination/("code-"+source_id+".zip")
    if not capsule.exists():
        with zipfile.ZipFile(capsule,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,path in sorted(sources.items()):
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
            info=zipfile.ZipInfo("SOURCE_MANIFEST.json",date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,json.dumps({"source_id":source_id,"sha256":hashes,"zip_time":"fixed metadata for reproducible packaging, not experiment time"},sort_keys=True,indent=2)+"\n")
    artifacts={}
    for p in sorted(root.rglob("*")):
        if not p.is_file():continue
        relative=p.relative_to(root)
        if relative.parts[0] in ("provenance","locks") or p.name=="REPRODUCIBILITY.json" or p.suffix==".tmp":continue
        artifacts[str(relative)]={"bytes":p.stat().st_size,"sha256":file_digest(p)}
    payload={"schema":"local-reproducibility-1","artifact_count":len(artifacts),"total_artifact_bytes":sum(x["bytes"] for x in artifacts.values()),
             "artifacts":artifacts,"code_capsule":str(capsule.relative_to(root)),"code_capsule_sha256":file_digest(capsule),
             "source_id":source_id,"hardware_quantum_shots":0,
             "excluded":"ephemeral locks, this pointer, and content-addressed provenance files; capsule is explicitly hashed above",
             "scope":"local frozen benchmark with initialization failures/caps; not publication or hardware-performance certification"}
    identity=digest(payload);manifest=destination/("dataset-"+identity+".json");save_json(manifest,payload)
    failures=[]
    for relative,item in artifacts.items():
        if file_digest(root/relative)!=item["sha256"]:failures.append(relative)
    with zipfile.ZipFile(capsule) as z:
        for name,sha in hashes.items():
            import hashlib
            if hashlib.sha256(z.read(name)).hexdigest()!=sha:failures.append("capsule:"+name)
    if failures:raise RuntimeError("Pack-time verification failed: "+repr(failures))
    pointer={"dataset_id":identity,"manifest":str(manifest.relative_to(root)),"manifest_sha256":file_digest(manifest),
             "code_capsule":str(capsule.relative_to(root)),"code_capsule_sha256":file_digest(capsule),
             "artifact_count":len(artifacts),"bytes":payload["total_artifact_bytes"],"packtime_all_hashes_verified":True,
             "external_upload":False}
    save_json(root/"REPRODUCIBILITY.json",pointer);print(pointer);return pointer


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);package(p.parse_args().output)
