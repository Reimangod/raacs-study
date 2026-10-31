import json
from benchmark.common import ROOT
from benchmark.raacs_standard_v2 import configuration


def test_standard_configuration_is_JSON_stable_and_full_cohort():
    cfg = configuration(ROOT/"results/not-executed-launcher-test", ROOT/"results/standard-v1")
    assert json.loads(json.dumps(cfg)) == cfg
    assert len(cfg["case_ids"]) == 30 and len(cfg["variants"]) == 28
    assert cfg["exact_small_cells"] == 810 and cfg["MPS_cells"] == 30
    assert len(cfg["case_ids"])*len(cfg["variants"]) == 840
