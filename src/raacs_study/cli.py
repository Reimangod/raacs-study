from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="Safe saved-data operations for the RAACS research archive")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("verify","reproduce","policy","archives","ci-check","locate","extract"):
        p=sub.add_parser(name)
        p.add_argument("--repo",type=Path,default=Path.cwd())
        if name=="reproduce":p.add_argument("--output",type=Path,default=Path("reproduced"))
        if name=="archives":
            p.add_argument("--directory",type=Path,required=True)
            p.add_argument("--full",action="store_true",help="Verify every mapped original blob in each archive")
        if name=="ci-check":p.add_argument("--output",type=Path,required=True)
        if name in("locate","extract"):p.add_argument("--path",required=True)
        if name=="extract":
            p.add_argument("--directory",type=Path,required=True)
            p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="verify":
        from .evidence import verify_repository
        result=verify_repository(args.repo)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result["passed"]:raise SystemExit(1)
    elif args.command=="reproduce":
        from .evidence import reproduce
        print(json.dumps(reproduce(args.repo,args.output),ensure_ascii=False,indent=2))
    elif args.command=="policy":
        print((args.repo/"configs/policy-registry.json").read_text())
    elif args.command=="archives":
        from .archives import verify_archives
        result=verify_archives(args.repo,args.directory,args.full)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result["passed"]:raise SystemExit(1)
    elif args.command=="ci-check":
        from .ci_check import run
        print(json.dumps(run(args.repo,args.output),indent=2))
    elif args.command=="locate":
        from .archives import locate
        print(json.dumps(locate(args.repo,args.path),indent=2))
    elif args.command=="extract":
        from .archives import extract_original
        print(json.dumps(extract_original(args.repo,args.directory,args.path,args.output),indent=2))


if __name__=="__main__":main()
