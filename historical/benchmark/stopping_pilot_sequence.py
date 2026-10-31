"""Hash source, generate historical pilot bank, then audit every saved frame."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import ROOT, file_digest, save_json
from .study.reanalysis import frozen_inventory
from .stopping_pilot_bank import build
from .stopping_pilot_audit import audit


def run(root, source, mode):
    root, source = Path(root).resolve(), Path(source).resolve(); root.mkdir(exist_ok=True, parents=True)
    binding = {"driver_sha256": file_digest(__file__), "generator_sha256": file_digest(ROOT / "benchmark/stopping_pilot_bank.py"),
               "auditor_sha256": file_digest(ROOT / "benchmark/stopping_pilot_audit.py"), "mode": mode, "source": str(source)}
    path = root / "sequence-binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise RuntimeError("Pilot driver binding drift")
    save_json(path, binding)
    save_json(root / "sequence-status.json", {"stage": "VERIFYING_ALL_SOURCE_HASHES", "whole_study_complete": False})
    frozen_inventory(source, root)
    save_json(root / "sequence-status.json", {"stage": "GENERATING_FRESH_HISTORICAL_PILOTS", "whole_study_complete": False})
    generation = build(root, source, mode)
    save_json(root / "sequence-status.json", {"stage": "AUDITING_ALL_OBSERVATION_FRAMES", "generation": generation, "whole_study_complete": False})
    validation = audit(root)
    save_json(root / "sequence-status.json", {"stage": "HISTORICAL_PILOT_BANK_AUDITED", "validation": validation,
              "actual_online_growth_not_yet_executed": True, "whole_study_complete": False})
    return validation


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--output"); p.add_argument("--source", default=str(ROOT / "results/standard-v1")); args = p.parse_args()
    print(json.dumps(run(args.output or ROOT / f"results/PhaseC-pilot-bank-{args.mode}-v1", args.source, args.mode)))
