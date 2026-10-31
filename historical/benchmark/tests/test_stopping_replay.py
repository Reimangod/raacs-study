from copy import deepcopy
import pytest
from benchmark.stopping_replay import POLICIES, select_prefix
from benchmark.stopping_calibration import WHITELIST, choose_training_rule


def episode():
    frames = []
    for i in range(6):
        frames.append(dict(problem_id="problem", trajectory_id="trajectory", step=i+1,
            qsci_energy_Ha=-1., actual_dimension=2, new_observed_valid_nonHF=0,
            cumulative_pilot_raw_shots=64*(i+1), hf_raw_fraction=.5,
            cnot_count=10*(i+1), cnot_depth=8*(i+1), cumulative_classical_seconds=.1*(i+1),
            current_pool_gradient_norm=None, external_residual_norm_Ha=.001,
            pt2_proxy_Ha=-1e-6, residual_scope="test proxy"))
    return dict(frames=frames, evidence=[{"checkpoint_id": f"cp{i+1}"} for i in range(6)])


def training():
    records = []
    for mol in ("H2", "H4"):
        for p in POLICIES:
            for rep in range(2):
                records.append(dict(id=f"{mol}-{p}-{rep}", case_id=mol+"_R1.0", molecule=mol,
                    pool="ceo_mvp", policy=p, total_M=1000, rep=rep, cap=64,
                    energy_Ha=-1. if p in ("native_terminal", "gradient") else -.9998,
                    native_energy_Ha=-1., cnot_count=100 if p in ("native_terminal", "gradient") else 50,
                    cnot_depth=80 if p in ("native_terminal", "gradient") else 40,
                    native_cnot_count=100, native_cnot_depth=80,
                    pilot_shots=0 if p in ("native_terminal", "gradient") else 192))
    assert all(set(r) == set(WHITELIST) for r in records)
    return records


def test_paid_prefix_no_future_inspection():
    ep = episode()
    for name in POLICIES[2:]:
        answer = select_prefix(ep, name)
        assert answer["stop"] and answer["selected_checkpoint_id"] == "cp3"
        assert answer["pilot_shots"] == 192 and answer["observed_probes"] == 3
        changed = deepcopy(ep); changed["frames"][3:] = [{"invalid_future_frame": True}]
        assert select_prefix(changed, name) == answer


def test_native_and_gradient_no_unneeded_pilots_unknown_not_zero():
    ep = episode()
    assert select_prefix(ep, "native_terminal")["pilot_shots"] == 0
    answer = select_prefix(ep, "gradient")
    assert not answer["stop"] and answer["pilot_shots"] == 0
    ep["frames"][-1]["current_pool_gradient_norm"] = 1e-5
    answer = select_prefix(ep, "gradient")
    assert answer["stop"] and answer["pilot_shots"] == 0
    assert answer["selected_checkpoint_id"] == "cp6"


def test_training_rejects_fci_extra_fields_and_heldout():
    rows = training()
    assert choose_training_rule(rows, "N2")["selected_policy"] not in ("native_terminal", "gradient")
    bad = deepcopy(rows); bad[0]["fci_energy"] = -2.
    with pytest.raises(ValueError, match="whitelist"):
        choose_training_rule(bad, "N2")
    with pytest.raises(ValueError, match="Heldout"):
        choose_training_rule(rows, "H2")


def test_all_bad_rules_use_native_equivalent_fallback():
    rows = training()
    for r in rows:
        if r["policy"] not in ("native_terminal", "gradient"):
            r["energy_Ha"] = -.99
    result = choose_training_rule(rows, "N2")
    assert result["selected_policy"] in ("native_terminal", "gradient")
    assert not result["FCI_used"]


def test_training_weights_molecules_not_rows():
    rows = training()
    # One thousand duplicates cannot make a bad molecular cluster disappear.
    for r in rows:
        if r["policy"] not in ("native_terminal", "gradient"):
            r["energy_Ha"] = -1. if r["molecule"] == "H2" else -.998
    rows += [{**r, "id": r["id"]+f"-dup{i}"} for r in rows if r["molecule"] == "H2" for i in range(30)]
    result = choose_training_rule(rows, "N2")
    assert result["selected_policy"] in ("native_terminal", "gradient")
    scores = {s["policy"]: s for s in result["training_scores"]}
    assert scores["energy_plateau"]["mean_molecular_regret_mHa"] == pytest.approx(1.)
    assert scores["energy_plateau"]["worst_molecular_mean_regret_mHa"] == pytest.approx(2.)
