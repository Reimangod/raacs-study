from benchmark.fixed_cohort_report import group_name


def test_only_seed_suffix_grouped_not_trotter_steps():
    assert group_name("TE_t1_steps1") == "TE_t1_steps1"
    assert group_name("TE_t1_steps4") != group_name("TE_t1_steps1")
    assert group_name("Pauli_qDRIFT_t1_steps64_s211") == "Pauli_qDRIFT_t1_steps64"
    assert group_name("M1000-R128-s307") == "HF-separated M1000-R128"
