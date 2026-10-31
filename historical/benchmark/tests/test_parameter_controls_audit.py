"""Integration and adversarial checks on separately qualified quick evidence."""
import json
from pathlib import Path
import shutil
import sqlite3
import pytest
from benchmark.common import ROOT, file_digest
from benchmark.local_sampler_primitives import pack_counts64, unpack_counts64
from benchmark.parameter_controls_audit import audit_cell

QUICK = ROOT/"results/parameter-controls-quick-v1"
pytestmark = pytest.mark.skipif(not (QUICK/"validation.json").exists(), reason="Actual quick qualification not available")


@pytest.mark.parametrize("mode", ["source_qsci_grid", "fixed_pi4", "one_parameter_vqe", "blockwise_vqe", "full_vqe", "qsci_coordinate"])
def test_real_count_draws_parameter_scope_and_actual_quantum_circuit(mode):
    cfg = json.loads((QUICK/"manifest.json").read_text())
    item = next(r for r in cfg["queue"] if r["case"] == "H4_R1.1")
    result = audit_cell(QUICK, cfg, item, mode)
    assert result["physical_QPY_vectors_checked"] == 2 and result["draws_rebuilt"] >= 4


@pytest.fixture
def copied_cell(tmp_path):
    cfg = json.loads((QUICK/"manifest.json").read_text())
    item = next(r for r in cfg["queue"] if r["case"] == "H4_R1.1")
    shutil.copy2(QUICK/"manifest.json", tmp_path/"manifest.json")
    relative = Path("cells")/item["case"]/"s101"/"full_vqe"
    folder = tmp_path/relative
    shutil.copytree(QUICK/relative, folder)
    return tmp_path, folder, cfg, item


def test_rejects_cost_double_counting(copied_cell):
    root, folder, cfg, item = copied_cell
    path = folder/"result.json"; result = json.loads(path.read_text())
    result["new_actual_model_shots"] += result["inherited_pilot_shots"]
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError): audit_cell(root, cfg, item, "full_vqe")


def test_rejects_false_stationarity_even_if_artifact_hash_updated(copied_cell):
    root, folder, cfg, item = copied_cell
    path = folder/"optimization.json"; control = json.loads(path.read_text())
    control["full_stationary"] = not control["full_stationary"]
    path.write_text(json.dumps(control))
    path = folder/"result.json"; result = json.loads(path.read_text())
    result["artifact_sha256"]["optimization.json"] = file_digest(folder/"optimization.json")
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError): audit_cell(root, cfg, item, "full_vqe")


def test_rejects_modified_counts_even_with_updated_sqlite_hash(copied_cell):
    root, folder, cfg, item = copied_cell
    db = sqlite3.connect(folder/"events.sqlite")
    did, blob = db.execute("SELECT id,counts FROM draws LIMIT 1").fetchone()
    ids, counts = unpack_counts64(blob)
    # The second bin is a singleton in this actual draw; take from the largest
    # bin so the mutation preserves positive counts and the total shot count.
    donor = int(counts.argmax()); recipient = (donor+1) % len(counts)
    assert len(counts) >= 2 and counts[donor] > 1
    counts[recipient] += 1; counts[donor] -= 1
    db.execute("UPDATE draws SET counts=? WHERE id=?", (pack_counts64(ids, counts), did)); db.commit(); db.close()
    path = folder/"result.json"; result = json.loads(path.read_text())
    result["events_sha256"] = file_digest(folder/"events.sqlite")
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError): audit_cell(root, cfg, item, "full_vqe")
