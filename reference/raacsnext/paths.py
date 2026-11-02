from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE / 'ceo-sampler-benchmark'
OLD = ROOT.parent / 'raacs-lucj-sqdrift-paper-aligned-plan-2026-10-25' / 'execution-v1'
for folder in (PROJECT, OLD):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
