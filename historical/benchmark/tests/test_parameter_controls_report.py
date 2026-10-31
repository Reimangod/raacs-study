import json
import pandas as pd
import pytest
from benchmark.common import ROOT, file_digest

OUTPUT = ROOT/"results/parameter-controls-standard-v1"
REPORT = OUTPUT/"report-v1"
pytestmark = pytest.mark.skipif(not (REPORT/"report-manifest.json").exists(), reason="Actual all-30-case report not available")


def test_report_coverage_no_best_seed_and_actual_new_shots():
    manifest = json.loads((REPORT/"report-manifest.json").read_text())
    audit = json.loads((OUTPUT/"validation.json").read_text())
    frame = pd.read_csv(REPORT/"all-cells.csv")
    assert len(frame) == manifest["cells"] == 540
    assert frame[["case", "source_seed", "mode"]].drop_duplicates().shape[0] == 540
    assert set(frame.source_seed) == {101, 211, 307} and frame.case.nunique() == 30
    assert manifest["totals"]["unique_source_constructions"] == 90
    assert manifest["totals"]["new_raw_model_shots"] == frame.new_actual_model_shots.sum()
    assert manifest["totals"]["new_raw_model_shots"] == audit["totals"]["new_pilot_shots"]+audit["totals"]["new_final_shots"]
    assert not manifest["whole_study_complete"] and not manifest["D5_online_requirement_completed"]
    assert manifest["totals"]["hardware_shots"] == 0 and manifest["totals"]["VQE_QPU_objective_cost"] is None


def test_cluster_inference_and_unknown_cost_scope_and_artifact_integrity():
    manifest = json.loads((REPORT/"report-manifest.json").read_text())
    frame = pd.read_csv(REPORT/"all-cells.csv")
    for r in manifest["summaries"]:
        assert r["molecule_clusters"] == 6
        expected = frame[frame["mode"] == r["mode"]].groupby("molecule").delta_QSCI_mean_error_mHa_vs_source.mean().mean()
        assert r["mean_QSCI_delta_vs_source_mHa"] == pytest.approx(expected, abs=1e-10)
        assert r["QPU_objective_cost_known"] == (not r["mode"].endswith("_vqe"))
    for name, sha in manifest["artifacts"].items(): assert file_digest(REPORT/name) == sha
    assert len(list(REPORT.glob("*.png"))) == len(list(REPORT.glob("*.svg"))) == len(list(REPORT.glob("*.pdf"))) == 6
