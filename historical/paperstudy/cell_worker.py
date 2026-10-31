"""One isolated common-budget cell. A parent owns the durable queue claim."""
import argparse
import json
from pathlib import Path

from .common_execution import run_cell
from .io import ROOT, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--budget", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--lane", default="C_PRIMARY")
    a = p.parse_args()
    if a.method == "D_SQDRIFT_N20":
        from .sqdrift_execution import run
        run(a.case, a.budget, a.seed, a.lane)
    else:
        run_cell(a.case, a.method, a.budget, a.seed, a.lane)


if __name__ == "__main__":
    main()
