import numpy as np

from precision_md.benchmark import _add_discrepancies, _representative_frames


def test_representative_frames_span_full_dataset():
    frames = [{"frame_id": str(index)} for index in range(300)]
    selected = _representative_frames(frames, 3)

    assert [frame["frame_id"] for frame in selected] == ["0", "149", "299"]


def test_discrepancies_are_paired_by_frame():
    rows = [
        {"frame_id": "a", "policy": "fp32", "energy_ev": 2.0,
         "finite": True, "atom_count": 2},
        {"frame_id": "a", "policy": "tf32", "energy_ev": 2.2,
         "finite": True, "atom_count": 2},
    ]
    forces = {
        ("a", "fp32"): np.array([[1., 0., 0.], [0., 1., 0.]]),
        ("a", "tf32"): np.array([[1.1, 0., 0.], [0., 1.2, 0.]]),
    }

    updated = _add_discrepancies(rows, forces)[1]

    assert np.isclose(updated["energy_error_ev"], .2)
    assert np.isclose(updated["max_force_error_ev_per_a"], .2)
    assert np.isclose(updated["mean_force_error_ev_per_a"], .15)


def test_discrepancies_skip_nonfinite_fast_rows():
    rows = [
        {"frame_id": "a", "policy": "fp32", "energy_ev": 2.0,
         "finite": True, "atom_count": 1},
        {"frame_id": "a", "policy": "bf16_amp", "energy_ev": np.nan,
         "finite": False, "atom_count": 1},
    ]

    assert "energy_error_ev" not in _add_discrepancies(rows, {})[1]
