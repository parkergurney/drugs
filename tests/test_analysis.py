import numpy as np
import pytest
from precision_md.analysis import gate1_pass, speed_ratio_ci


def test_paired_bootstrap_speedup():
    ratio, ci = speed_ratio_ci(np.ones(20) * 2, np.ones(20), iterations=100)
    assert ratio == pytest.approx(2) and ci == pytest.approx((2, 2))


def test_gate1_conditions():
    summary = {"finite_count":300, "total_count":300, "speedup":1.21, "speedup_ci_low":1.1,
               "batch1_slowdown":1.0, "discrepancy_above_noise":True}
    assert gate1_pass(summary)
    assert not gate1_pass(summary | {"finite_count":299})
