"""Content-addressed evidence archives. Never execute serialized objects."""
import hashlib
import json
import zipfile
import io
import bisect
from pathlib import Path
from .evidence import safe_member_name, sha256_file


class JoinedParts(io.RawIOBase):
    """Seekable virtual concatenation: no second multi-GiB ZIP is materialized."""
    def __init__(self,paths):
        self.paths=list(paths);self.bounds=[0]
        for p in self.paths:self.bounds.append(self.bounds[-1]+p.stat().st_size)
        self.position=0;self.handle=None;self.current=-1
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.bounds[-1]+offset
        if position<0:raise ValueError("Negative seek")
        self.position=position;return position
    def read(self,size=-1):
        if size is None or size<0:size=max(0,self.bounds[-1]-self.position)
        blocks=[]
        while size>0 and self.position<self.bounds[-1]:
            part=bisect.bisect_right(self.bounds,self.position)-1
            if part!=self.current:
                if self.handle:self.handle.close()
                self.handle=self.paths[part].open("rb");self.current=part
            local=self.position-self.bounds[part]
            self.handle.seek(local)
            raw=self.handle.read(min(size,self.bounds[part+1]-self.position))
            if not raw:raise OSError("Truncated archive part")
            blocks.append(raw);self.position+=len(raw);size-=len(raw)
        return b"".join(blocks)
    def readinto(self,buffer):
        raw=self.read(len(buffer));buffer[:len(raw)]=raw;return len(raw)
    def close(self):
        if self.handle:self.handle.close()
        super().close()


def locate(repo:Path,name:str):
    import gzip
    manifest=json.loads((repo/"data/archive-manifest.json").read_text())
    if name.startswith("/"):
        prefix=manifest.get("original_workspace_prefix","")
        if not prefix or not name.startswith(prefix+"/"):raise ValueError("Unknown absolute-path origin")
        name=name[len(prefix)+1:]
    if not safe_member_name(name):raise ValueError("Unsafe original path")
    with gzip.open(repo/"data/source-inventory.jsonl.gz","rt")as stream:
        for line in stream:
            row=json.loads(line)
            if row["source_relative_path"]==name:return row
    raise FileNotFoundError("Original path not captured: "+name)


def extract_original(repo:Path,directory:Path,name:str,output:Path):
    """Read one verified blob; never execute it or overwrite an existing file."""
    row=locate(repo,name)
    manifest=json.loads((repo/"data/archive-manifest.json").read_text())
    entry=next(x for x in manifest["archives"]if x["name"]==row["archive"])
    file=directory/entry["name"];virtual=None
    if file.is_file():source=file
    elif entry.get("parts"):
        paths=[directory/p["name"]for p in entry["parts"]]
        if any(not p.is_file()for p in paths):raise FileNotFoundError("Download all parts first")
        virtual=JoinedParts(paths);source=virtual
    else:raise FileNotFoundError(file)
    if output.exists():raise FileExistsError("Never overwrite: "+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_name(output.name+".pending")
    try:
        with zipfile.ZipFile(source)as archive,archive.open(row["blob"])as src,pending.open("xb")as dst:
            hasher=hashlib.sha256();size=0
            while raw:=src.read(1024**2):
                hasher.update(raw);size+=len(raw);dst.write(raw)
        if size!=row["bytes"]or hasher.hexdigest()!=row["sha256"]:raise ValueError("Original-blob digest mismatch")
        # Link+unlink is no-clobber even if another process creates the target.
        import os
        os.link(pending,output);pending.unlink()
    finally:
        if virtual:virtual.close()
    return {**row,"output":str(output),"verified":True}


def verify_archives(repo:Path,directory:Path,full=False):
    manifest=json.loads((repo/"data/archive-manifest.json").read_text())
    errors=[];checked=[]
    for entry in manifest["archives"]:
        file=directory/entry["name"]
        virtual=None
        if file.is_file():
            size=file.stat().st_size;digest=sha256_file(file)
            source=file
        elif entry.get("parts"):
            paths=[directory/x["name"]for x in entry["parts"]]
            if any(not p.is_file()for p in paths):errors.append("missing archive parts: "+entry["name"]);continue
            bad=[p.name for p,part in zip(paths,entry["parts"])if p.stat().st_size!=part["bytes"]or sha256_file(p)!=part["sha256"]]
            if bad:errors.append("part mismatch: "+str(bad));continue
            virtual=JoinedParts(paths)
            digest=hashlib.file_digest(virtual,"sha256").hexdigest();virtual.seek(0)
            size=virtual.bounds[-1];source=virtual
        else:errors.append("missing: "+entry["name"]);continue
        if size!=entry["bytes"]or digest!=entry["sha256"]:
            if virtual:virtual.close()
            errors.append("archive mismatch: "+entry["name"]);continue
        n=0
        if full:
            with zipfile.ZipFile(source)as archive:
                if any(not safe_member_name(x.filename)for x in archive.infolist()):
                    errors.append("unsafe archive member: "+entry["name"]);continue
                if "inventory.jsonl"in archive.namelist():
                    inventory=[json.loads(x)for x in archive.read("inventory.jsonl").decode().splitlines()]
                    blob_hashes={}
                    for row in inventory:
                        if not safe_member_name(row["source_relative_path"]):
                            errors.append("unsafe logical path");continue
                        blob=row["blob"]
                        if blob not in blob_hashes:
                            with archive.open(blob)as stream:
                                blob_hashes[blob]=hashlib.file_digest(stream,"sha256").hexdigest()
                        if archive.getinfo(blob).file_size!=row["bytes"]or blob_hashes[blob]!=row["sha256"]:
                            errors.append("blob mismatch: "+row["source_relative_path"])
                        n+=1
                else:
                    # Preserved pre-existing ZIP; outer SHA anchors its exact bytes.
                    bad=archive.testzip()
                    if bad:errors.append("CRC mismatch: "+bad)
        checked.append({"archive":entry["name"],"original_files_verified":n})
        if virtual:virtual.close()
    return {"passed":not errors,"full":full,"archives":checked,"errors":errors}
