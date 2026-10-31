import json
import numpy as np
from benchmark.config import Case
from benchmark.chemistry import build_reference
from benchmark.adapt import run_adapt
from benchmark.common import file_digest


def test_explicit_extension_preserves_prefix_and_matches_uninterrupted(tmp_path):
    r=build_reference(Case("H4",1.1),tmp_path/"reference")
    root=tmp_path/"extended"
    run_adapt(r,"ceo_ovp",root,max_steps=1)
    cp=root/"checkpoints/H4_R1.1/ceo_ovp/step-0001.json"
    before=file_digest(cp)
    run_adapt(r,"ceo_ovp",root,max_steps=2,allow_extension=True)
    assert file_digest(cp)==before
    run_adapt(r,"ceo_ovp",tmp_path/"direct",max_steps=2)
    a=json.loads(cp.with_name("step-0002.json").read_text())
    b=json.loads((tmp_path/"direct/checkpoints/H4_R1.1/ceo_ovp/step-0002.json").read_text())
    assert a["selected_operators"]==b["selected_operators"]
    assert np.allclose(a["optimized_parameters"],b["optimized_parameters"],atol=1e-12)
    assert abs(a["vqe_energy"]-b["vqe_energy"])<1e-12
