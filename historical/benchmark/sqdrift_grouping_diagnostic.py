"""Diagnostic only: do both SDK grouping conventions preserve coefficient ratios?"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def diagnose(request):
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.terms.ordering import canonical_order
    from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
    terms = [(tuple((bool(a), int(q)) for q, a in row["term"]), complex(*row["coefficient"])) for row in request["fermion_terms"]]
    reports = []
    for physicist in (False, True):
        operator = canonical_order(FermionOperator.from_terms(terms).normal_ordered().simplify(atol=1e-12))
        group_terms_by_electronic_structure(operator, request["n_qubits"], two_body_physicist_order=physicist)
        bad = []; maximum = 0.
        for index, (weight, group) in enumerate(zip(operator.group_weights(), operator.split_out_groups())):
            error = max((abs(weight*np.sign(c)-c) for _, c in group.iter_terms()), default=0.)
            maximum = max(maximum, error)
            if error > 1e-10:
                bad.append({"group": index, "mean_magnitude": weight, "first_moment_error": error,
                            "terms": [(repr(a), [c.real, c.imag]) for a, c in group.iter_terms()]})
        reports.append({"two_body_physicist_order": physicist, "max_first_moment_error": maximum,
                        "bad_groups": len(bad), "example_bad_groups": bad[:3]})
    return {"problem_id": request["problem_id"], "reports": reports,
            "scope": "diagnostic; not a result for QSCI energy and not a global claim of an SDK bug"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    args = p.parse_args(); result = diagnose(json.loads(Path(args.input).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result))
