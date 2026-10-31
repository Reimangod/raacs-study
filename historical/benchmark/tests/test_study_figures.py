import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.study_figures import energy_limits


def test_energy_limits_pad_in_correct_transform_units():
    fig, ax = plt.subplots()
    energy_limits(ax, [20., 40., 500.])
    lo, hi = ax.get_ylim()
    assert .1 < lo < 1.59360143764
    assert 500 < hi < 2000
    plt.close(fig)


def test_energy_limits_do_not_hide_negative_confidence_tail():
    fig, ax = plt.subplots()
    energy_limits(ax, [-.01, .02, 100.])
    lo, hi = ax.get_ylim()
    assert -.1 < lo < -.01
    assert 100 < hi < 1000
    plt.close(fig)
