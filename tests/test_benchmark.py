import numpy as np
import pytest

from precision_md.benchmark import (
    _add_discrepancies, _materialize_trial_frames, _refuse_overwrite,
    _representative_frames, _trial_paths,
)


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


def test_trial_paths_isolate_run_and_accept_frozen_frames(tmp_path):
    class Config:
        output_dir = tmp_path / "trials"

    frozen = tmp_path / "frames.npz"
    output, frames = _trial_paths(Config(), frozen, "run-01")

    assert output == tmp_path / "trials" / "run-01"
    assert frames == frozen


def test_benchmark_refuses_to_overwrite_trial(tmp_path):
    (tmp_path / "timings.parquet").touch()

    with pytest.raises(FileExistsError, match="choose a new --run-id"):
        _refuse_overwrite(tmp_path)


def test_trial_id_cannot_escape_output_directory(tmp_path):
    class Config:
        output_dir = tmp_path / "trials"

    with pytest.raises(ValueError, match="one directory name"):
        _trial_paths(Config(), tmp_path / "frames.npz", "../overwritten")


def test_trial_materializes_verified_frame_copy(tmp_path):
    source = tmp_path / "canonical.npz"
    source.write_bytes(b"frozen")
    trial = tmp_path / "trial"
    trial.mkdir()

    local, canonical, digest = _materialize_trial_frames(trial, source)

    assert local == (trial / "frames.npz").resolve()
    assert canonical == source.resolve()
    assert local.read_bytes() == b"frozen"
    assert len(digest) == 64
