"""Additive PhaseC entrypoint around the immutable already-qualified runner."""
import argparse
import json
from .common import ROOT
from .actual_growth_cohort import run


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["quick", "standard"], default="quick")
    p.add_argument("--output"); p.add_argument("--workers", type=int, default=2); a = p.parse_args()
    print(json.dumps(run(a.output or ROOT/f"results/PhaseC-actual-growth-{a.mode}-v1", a.mode, a.workers)))


if __name__ == "__main__": main()
