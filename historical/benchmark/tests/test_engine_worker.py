import json
import numpy as np
from benchmark.config import Case
from benchmark.chemistry import build_reference
from benchmark.adapt import run_adapt
from benchmark.engine_bundle import create_bundle
from benchmark.engine_worker import execute


def test_worker_full_selection_bfgs_matches_native(tmp_path):
    ref=build_reference(Case("H2",.7),tmp_path)
    run_adapt(ref,"ceo_ovp",tmp_path,max_steps=3)
    bundle=create_bundle(tmp_path,"H2_R0.7","ceo_ovp",tmp_path/"bundle",3)
    terminal=execute(bundle,tmp_path/"worker","numpy")
    assert terminal["backend"]=="numpy" and terminal["device"]=="CPU"
    assert "fci_energy" not in json.dumps(json.load(open(bundle/"request.json")))
    for path in sorted((tmp_path/"checkpoints/H2_R0.7/ceo_ovp").glob("step-*.json")):
        a=json.load(open(path));b=json.load(open(tmp_path/"worker"/path.name))
        assert a["selected_operators"]==b["selected_operators"]
        assert abs(a["vqe_energy"]-b["energy"])<1e-10
        assert np.allclose(a["optimized_parameters"],b["optimized_parameters"],atol=1e-9)
