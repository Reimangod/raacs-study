import numpy as np
import pandas as pd
from benchmark.b2_report import correlation_recovery, partition_coverage


def test_recovery_not_configuration_recovery_and_unknown_reference():
    assert np.allclose(correlation_recovery([0, 50, 100], -1, -1.1), [100, 50, 0])
    assert np.isnan(correlation_recovery(0, -1, -1))


def test_missing_source_control_is_not_a_planned_growth_attempt():
    growth, source = partition_coverage(pd.DataFrame({"method": ["adapt_qsci_original/M1000-R32-s101", "source_final_lucj_2"]}))
    assert len(growth) == len(source) == 1
