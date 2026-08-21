import pytest
from precision_md.oracle import checkpoint_bootstrap, fit_threshold, oracle_cost, oracle_speedup


def row(cid, unsafe=False, signal=.1):
    return {"checkpoint_id":cid, "unsafe":unsafe, "fast_seconds":2., "audit_seconds":1.,
            "checkpoint_seconds":.5, "restore_seconds":.5, "fp32_seconds":10.,
            "relative_force_disagreement":signal}


def test_hand_calculated_economics():
    rows = [row(0), row(1, True)]
    assert oracle_cost(rows[0]) == pytest.approx(3.5)
    assert oracle_cost(rows[1]) == pytest.approx(14.)
    assert oracle_speedup(rows) == pytest.approx(20/17.5)


def test_checkpoint_bootstrap_and_threshold():
    rows = [row(i, i >= 6, i / 10) for i in range(10)]
    low, high = checkpoint_bootstrap(rows, iterations=50)
    assert low < high
    fit = fit_threshold(rows, "relative_force_disagreement")
    assert fit["recall"] >= .8 and fit["precision"] >= .5
