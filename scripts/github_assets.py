"""Plan and upload verified <=1GiB parts without duplicating large local ZIPs.

Authentication is read from gh in memory. It is never printed or written.
Only the explicitly named private repository and draft release are written.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote,urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import sha256_file,write_json


class Slice:
    def __init__(self,path,offset,size):
        self.stream=path.open("rb");self.stream.seek(offset);self.remaining=size
    def read(self,size=-1):
        if size is None or size<0:size=self.remaining
        data=self.stream.read(min(size,self.remaining));self.remaining-=len(data);return data
    def close(self):self.stream.close()


def gh_json(*args):
    p=subprocess.run(["gh",*args],check=True,capture_output=True,text=True)
    return json.loads(p.stdout)


def part_plan(entry,path,limit):
    if path.stat().st_size!=entry["bytes"]or sha256_file(path)!=entry["sha256"]:raise ValueError("Archive changed before upload planning")
    if entry["bytes"]<=limit:
        return [{"name":entry["name"],"bytes":entry["bytes"],"sha256":entry["sha256"],"offset":0}]
    result=[]
    with path.open("rb")as stream:
        offset=0;index=0
        while offset<entry["bytes"]:
            size=min(limit,entry["bytes"]-offset);left=size;hasher=hashlib.sha256()
            while left:
                raw=stream.read(min(left,1024**2))
                if not raw:raise OSError("Truncated archive")
                hasher.update(raw);left-=len(raw)
            result.append({"name":entry["name"]+f".part{index:03d}","bytes":size,"sha256":hasher.hexdigest(),"offset":offset})
            offset+=size;index+=1
    return result


def upload(repo,tag,path,part,release,token):
    listed=gh_json("api",f"repos/{repo}/releases/{release['id']}/assets?per_page=100")
    existing=next((x for x in listed if x["name"]==part["name"]),None)
    if existing:
        if existing["size"]==part["bytes"]and existing.get("digest")=="sha256:"+part["sha256"]:
            print(json.dumps({"reused_verified_asset":part["name"]}),flush=True);return existing
        raise RuntimeError("Existing asset mismatch; never clobber: "+part["name"])
    url=release["upload_url"].split("{")[0]+"?name="+quote(part["name"])
    parsed=urlsplit(url)
    connection=http.client.HTTPSConnection(parsed.hostname,timeout=600,blocksize=1024*1024)
    stream=Slice(path,part["offset"],part["bytes"])
    try:
        connection.request("POST",parsed.path+"?"+parsed.query,body=stream,headers={
            "Authorization":"Bearer "+token,"Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28","Content-Type":"application/octet-stream",
            "Content-Length":str(part["bytes"]),"User-Agent":"raacs-research-archive"})
        response=connection.getresponse();raw=response.read()
        if response.status!=201:
            raise RuntimeError("Asset upload HTTP "+str(response.status)+"; inspect release before retry")
        result=json.loads(raw)
    finally:
        stream.close();connection.close()
    if result["size"]!=part["bytes"]or result.get("digest")!="sha256:"+part["sha256"]:
        raise RuntimeError("GitHub reported a different asset digest")
    print(json.dumps({"uploaded":part["name"],"bytes":part["bytes"],"server_digest_verified":True}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["plan","upload"])
    p.add_argument("--repo",default="Reimangod/raacs-study");p.add_argument("--tag",default="data-v0.1.0")
    p.add_argument("--directory",type=Path,default=ROOT/"artifacts")
    p.add_argument("--limit-bytes",type=int,default=1024**3)
    p.add_argument("--only-asset",action="append",default=[])
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--receipts",type=Path,default=ROOT/"data/upload-receipts.json")
    a=p.parse_args();manifest_path=ROOT/"data/archive-manifest.json"
    manifest=json.loads(manifest_path.read_text())
    if a.mode=="plan":
        for entry in manifest["archives"]:
            entry["parts"]=part_plan(entry,a.directory/entry["name"],a.limit_bytes)
            entry["release_tag"]=a.tag;entry["repository"]=a.repo
        write_json(manifest_path,manifest)
        print(json.dumps({"planned_assets":sum(len(x["parts"])for x in manifest["archives"]),"max_asset_bytes":a.limit_bytes}),flush=True)
        return
    details=gh_json("api",f"repos/{a.repo}")
    if not details["private"]:raise RuntimeError("Uploader is restricted to private delivery")
    releases=gh_json("api",f"repos/{a.repo}/releases?per_page=100")
    release=next((r for r in releases if r["tag_name"]==a.tag),None)
    if release is None:raise RuntimeError("Draft release not found; do not create duplicates")
    if not release["draft"]:raise RuntimeError("Use a mutable draft during upload, not an immutable final release")
    token=subprocess.run(["gh","auth","token"],check=True,capture_output=True,text=True).stdout.strip()
    receipts=[];tasks=[]
    for entry in manifest["archives"]:
        if not entry.get("parts"):raise ValueError("Plan parts before uploading")
        wanted=[part for part in entry["parts"]if not a.only_asset or part["name"]in a.only_asset]
        if not wanted:continue
        path=a.directory/entry["name"]
        if sha256_file(path)!=entry["sha256"]:raise ValueError("Changed archive")
        tasks.extend((path,part)for part in wanted)
    if len({part["name"]for _,part in tasks})!=len(tasks):raise ValueError("Duplicate upload assignment")
    with ThreadPoolExecutor(max_workers=max(1,min(3,a.workers)))as pool:
        futures=[pool.submit(upload,a.repo,a.tag,path,part,release,token)for path,part in tasks]
        for future in as_completed(futures):
            asset=future.result()
            receipts.append({"name":asset["name"],"bytes":asset["size"],"server_digest":asset.get("digest"),"asset_id":asset["id"],"url":asset["browser_download_url"]})
            write_json(a.receipts,{"repository":a.repo,"release_tag":a.tag,"private":True,"assets":receipts})


if __name__=="__main__":main()
