import json
import numpy as np
from benchmark.random_structure_baselines import method_specs, construct
from benchmark.common import ROOT
from benchmark.chemistry import load_reference


def test_full_seed_cohort_and_classical_proposal_is_not_a_free_quantum_circuit(tmp_path):
    specs = method_specs("standard")
    assert len(specs) == 13 and len({x["name"] for x in specs}) == 13
    case = "H2_R0.7"; source = ROOT / "results/standard-v1"
    ref = load_reference(source / "references" / case)
    path = sorted((source / "checkpoints" / case / "ceo_ovp").glob("step-*.json"))[-1]
    cp = json.loads(path.read_text())
    result = construct(case, ref, cp, path, specs[0], tmp_path)
    assert result["sampler_type"] == "CLASSICAL_PROPOSAL"
    assert result["resources"]["cnot_count"] is None
    with np.load(tmp_path / "cells" / case / specs[0]["name"] / "final.npz") as z:
        assert np.allclose(z["probability"][ref["ids"]], 1/len(ref["ids"]))
