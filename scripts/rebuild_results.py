"""Rebuild explicitly derived tables/figures without touching original evidence."""
from pathlib import Path
import sys,shutil,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from raacs_study.evidence import reproduce,sha256_file,write_csv
with tempfile.TemporaryDirectory(prefix="raacs-results-")as folder:
    source=Path(folder);report=reproduce(ROOT,source)
    target=ROOT/"results";target.mkdir(exist_ok=True)
    for p in sorted(source.rglob("*")):
        if p.is_file():
            dest=target/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write_csv(target/"figure-sources.csv",[
        {"figure":name,"source":"data/catalog.csv","source_sha256":sha256_file(ROOT/"data/catalog.csv"),
         "selection":"separate cohort/case/context/final_shots; >=3 labels; all recorded rows actual-D matched",
         "claim_scope":"saved-data derivative; not a fresh experiment or all-method superiority claim"}
        for name in report["figures"]])
    print("Rebuilt",report["summary_rows"],"summary rows and",len(report["figures"]),"SVG figures")
