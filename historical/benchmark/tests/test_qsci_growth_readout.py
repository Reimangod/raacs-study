import json
import sqlite3
import numpy as np
import pandas as pd
from benchmark.common import ROOT, save_json
from benchmark.chemistry import load_reference
from benchmark import qsci_growth_readout as reader


def test_all_invalid_native_draws_are_missing_not_omitted(tmp_path, monkeypatch):
    source = ROOT / "results/standard-v1"
    ref = load_reference(source / "references/H2_R0.7")
    p = np.zeros(1 << ref["meta"]["n_qubits"]); p[-1] = 1.
    record = {"case_id": "H2_R0.7", "method": "TEST_INVALID_SECTOR", "task_id": "TEST_ONLY_NOT_PRODUCTION",
              "source_step": 0, "probability": p, "native_hf_insertion": False}
    save_json(tmp_path / "manifest.json", {"source": str(source), "mode": "quick", "cases": ["H2_R0.7"]})
    monkeypatch.setattr(reader, "samplers", lambda *_: (ref, [record], []))
    result = reader.evaluate(tmp_path)
    summary = pd.read_csv(tmp_path / "independent-readouts-v2/summary.csv")
    native = summary[summary.view == "native"]
    common = summary[summary.view == "common_HF_inserted"]
    assert len(native) == len(common) == 21
    assert (native.missing_energy_repeats == 4).all()
    assert native.conditional_mean_error_mHa.isna().all()
    assert (native.unconditional_success_probability == 0.).all()
    assert (common.valid_energy_repeats == 4).all()
    assert result["draws"] == 12
    assert result["simulated_readout_raw_shots"] == 4*(1000+10000+100000)
    again = reader.evaluate(tmp_path)
    assert again == result  # Exact same saved draws, not new measurement events.
    with sqlite3.connect(tmp_path / "independent-readouts-v2/H2_R0.7/readouts.sqlite") as db:
        assert db.execute("SELECT count(*) FROM draws").fetchone()[0] == 12
