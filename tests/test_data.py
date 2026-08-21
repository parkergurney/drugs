import numpy as np
import pytest
from precision_md.data import Frame, close_contact, ordinary_indices, select_high_force


def test_selection_is_deterministic():
    lengths = {"a": 50, "b": 50, "c": 50}
    assert all(np.array_equal(ordinary_indices(lengths, (4, 3, 3))[k], ordinary_indices(lengths, (4, 3, 3))[k]) for k in lengths)


def test_high_force_cap_and_order():
    rows = [{"frame_id": str(i), "molecule": "a" if i < 8 else "b", "max_force": 20-i} for i in range(12)]
    selected = select_high_force(rows, count=6, cap=3)
    assert sum(r["molecule"] == "a" for r in selected) == 3
    assert len(selected) == 6


def test_high_force_selection_can_exclude_ordinary_ids():
    rows = [
        {"frame_id": str(i), "molecule": "a" if i % 2 else "b", "max_force": 20-i}
        for i in range(12)
    ]
    ordinary_ids = {"0", "1"}
    selected = select_high_force(
        [row for row in rows if row["frame_id"] not in ordinary_ids],
        count=4,
        cap=2,
    )

    assert not ordinary_ids.intersection(row["frame_id"] for row in selected)


def test_close_contact_exact_distance():
    frame = Frame("x", "mol", np.ones(4), np.array([[0,0,0],[1,0,0],[2,0,0],[4,0,0]], float))
    got = close_contact(frame, .8)
    assert np.linalg.norm(got.positions[3] - got.positions[0]) == pytest.approx(.8)
