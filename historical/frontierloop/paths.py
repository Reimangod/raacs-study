from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
V2=ROOT.parent/'raacs-practical-frontier-v2-2026-10-25'
OLD=ROOT.parent/'raacs-lucj-sqdrift-paper-aligned-plan-2026-10-25'/'execution-v1'
PROJECT=WORKSPACE/'ceo-sampler-benchmark'
for p in(V2,OLD,PROJECT):
    if str(p)not in sys.path:sys.path.insert(0,str(p))
