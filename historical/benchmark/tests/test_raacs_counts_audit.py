import json
from pathlib import Path
import pytest
from benchmark.raacs_counts_audit import audit_cell


def test_independent_counts_audit_reconstructs_saved_quick_cells():
    from benchmark.common import ROOT
    root = ROOT/"results/RAACS-quick-v1"
    if not (root/"manifest.json").exists(): pytest.skip("Saved qualification dataset unavailable")
    cfg = json.loads((root/"manifest.json").read_text())
    checked = 0
    for path in sorted((root/"cells").glob("*/*/result.json")):
        case = path.parent.parent.name
        a = audit_cell(Path(cfg["source"])/"references"/case, path.parent)
        assert a["pilot_shots"]+a["final_shots"] == 4000
        assert a["final_circuits_symbolically_rebuilt"] > 0
        assert a["no_all_draw_reexecution_claim"]
        checked += 1
    assert checked == 12
