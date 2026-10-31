"""Read-only reuse of qualified fixed-sampler engine for an explicit new cohort."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .common import ROOT, file_digest, save_json
from .strong_baselines.evaluate import evaluate


def run(root):
    root = Path(root)
    bound = {"cohort_manifest_sha256": file_digest(root / "manifest.json"),
             "readout_engine_sha256": file_digest(ROOT / "benchmark/strong_baselines/evaluate.py"),
             "adapter_sha256": file_digest(__file__), "view": "common HF inserted, not full SQD or native HF-free QSCI"}
    path = root / "fixed-evaluation-binding.json"
    if path.exists() and json.loads(path.read_text()) != bound:
        raise RuntimeError("Fixed-cohort evaluation code/config drift")
    save_json(path, bound)
    with threadpool_limits(limits=1):
        return evaluate(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(json.dumps(run(p.parse_args().output)))
