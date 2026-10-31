import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .ci import solve
from .io import ROOT, save_json
from .lucj import small_parity


def run():
    records = [small_parity(seed=s) for s in (101, 211, 307)]
    save_json(ROOT / "qualification/LUCJ-small-parity.json", {"fixtures": records})
    case = "N2-32-R1p2-P2"
    physical = ROOT / "data/physical" / case
    meta = json.loads((physical / "problem.json").read_text())
    with np.load(physical / "integrals.npz") as a:
        h1, eri = a["h1"].copy(), a["eri"].copy()
    with np.load(ROOT / "qualification" / (case+"-naive") / "timing-counts.npz") as a:
        ids, counts = a["blocked_ids"].copy(), a["counts"].copy()
    outputs = []
    for r in (8, 16, 32):
        outputs.append(solve(h1, eri, meta["core_energy"], (meta["nalpha"], meta["nbeta"]), ids, counts, r))
    report = {"phase": "S1_FIXED_COUNTS_CI_QUALIFICATION_NOT_PRIMARY", "case_id": case,
              "problem_id": meta["problem_id"], "readouts": outputs,
              "counts_same_across_R": True, "new_model_shots": 0,
              "new_actual_QPU_shots": 0, "whole_study_complete": False}
    save_json(ROOT / "qualification/common-CI-N2-fixed-counts.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
