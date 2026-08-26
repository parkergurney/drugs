import numpy as np
import pytest
from precision_md.data import (
    Frame, close_contact, ensure_disjoint_sources, load_excluded_sources,
    ordinary_indices, select_high_force, source_frame_id,
)


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


def test_ordinary_selection_excludes_frozen_sources():
    excluded = {"a-0", "a-1", "b-2"}
    selected = ordinary_indices({"a": 10, "b": 10}, (4, 4), seed=7,
                                excluded_sources=excluded)

    selected_ids = {
        f"{molecule}-{index}"
        for molecule, indices in selected.items()
        for index in indices
    }
    assert not selected_ids & excluded


def test_selection_exclusion_includes_close_contact_sources(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(
        '{"frames":["ethanol-7", "aspirin-9-cc-0.8-3"]}\n'
    )

    excluded, checksum = load_excluded_sources(path)

    assert excluded == {"ethanol-7", "aspirin-9"}
    assert len(checksum) == 64
    assert source_frame_id("aspirin-9-cc-1.0-4") == "aspirin-9"


def test_overlap_validation_fails_closed():
    with pytest.raises(ValueError, match="overlaps excluded sources"):
        ensure_disjoint_sources({"ethanol-1", "aspirin-2"}, {"ethanol-1"})
