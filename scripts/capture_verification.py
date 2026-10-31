"""Persist measured verification outputs; no new molecular experiments."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import verify_repository,write_json
from raacs_study.archives import verify_archives
from raacs_study.ci_check import run

out=ROOT/"verification";out.mkdir(exist_ok=True)
catalog=verify_repository(ROOT);archives=verify_archives(ROOT,ROOT/"artifacts",True)
write_json(out/"catalog-integrity.json",catalog)
write_json(out/"archive-integrity.json",archives)
if not catalog["passed"]or not archives["passed"]:raise SystemExit("Integrity verification failed")
run(ROOT,out/"saved-ci-check.json")
result=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True)
write_json(out/"local-tests.json",{"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr,
                                  "scope":"new archive package tests only, not every historical study test"})
if result.returncode:raise SystemExit(result.returncode)
print(result.stdout,flush=True)
