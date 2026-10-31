"""Explicit portable goal-input adapter for the frozen read-only report.

Does not rewrite the old source or any experiment. A relocated objective is
permitted only if its exact byte digest matches the owner's frozen objective.
The derivative gets its own provenance/snapshot, not the historical identity.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from unittest.mock import patch

OBJECTIVE_SHA256='a8d1dc79339645ccecf7078fe09db5a372972e1c6d383d1e38d8a16b60a65960'


def run(objective,output,mode='standard'):
    from . import current_evidence as report
    objective=Path(objective).resolve()
    if report.sha(objective)!=OBJECTIVE_SHA256:
        raise ValueError('Portable goal input differs from the frozen full owner objective')
    # This provider changes only the location of exactly identical input bytes.
    # Its path in the source map is deliberately different, yielding a separate
    # report identity; no fake claim of byte-identical historical provenance.
    with patch.object(report,'OBJECTIVE',objective):
        return report.build(output,mode)


def main():
    import json
    p=argparse.ArgumentParser();p.add_argument('--objective',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['quick','standard'],default='standard')
    a=p.parse_args();print(json.dumps(run(a.objective,a.output,a.mode),ensure_ascii=False))


if __name__=='__main__':main()
