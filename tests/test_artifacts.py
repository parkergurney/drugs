import json

import numpy as np
import pandas as pd
import pytest

from precision_md.artifacts import freeze_dataset, validate_frozen_dataset


def _source_dataset(path):
    path.mkdir()
    frames = []
    for stratum in ("ordinary", "high_force", "close_contact"):
        for index in range(100):
            frames.append({
                "frame_id": f"{stratum}-{index}", "stratum": stratum,
                "molecule": "ethanol", "atomic_numbers": np.array([1]),
                "positions": np.zeros((1, 3)),
            })
    np.savez_compressed(path / "frames.npz", frames=np.array(frames, dtype=object))
    (path / "selection.json").write_text(
        json.dumps({"frames": [frame["frame_id"] for frame in frames]}) + "\n"
    )
    pd.DataFrame({"frame_id": [f"candidate-{i}" for i in range(3000)]}).to_parquet(
        path / "candidate_scores.parquet", index=False
    )


def test_freeze_dataset_is_portable_and_idempotent(tmp_path):
    source, output = tmp_path / "source", tmp_path / "frozen" / "p1"
    _source_dataset(source)

    first = freeze_dataset(source, output, "p1")
    second = freeze_dataset(source, output, "p1")

    assert first == second
    assert validate_frozen_dataset(output, "p1")["frame_count"] == 300
    assert set((output / "SHA256SUMS").read_text().splitlines()[0].split())


def test_freeze_dataset_refuses_different_existing_payload(tmp_path):
    source, changed, output = tmp_path / "source", tmp_path / "changed", tmp_path / "p1"
    _source_dataset(source)
    _source_dataset(changed)
    freeze_dataset(source, output, "p1")
    selection = json.loads((changed / "selection.json").read_text())
    selection["extra"] = True
    (changed / "selection.json").write_text(json.dumps(selection) + "\n")

    with pytest.raises(FileExistsError, match="different frozen dataset or provenance"):
        freeze_dataset(changed, output, "p1")


def test_validate_dataset_detects_checksum_change(tmp_path):
    source, output = tmp_path / "source", tmp_path / "p1"
    _source_dataset(source)
    freeze_dataset(source, output, "p1")
    (output / "selection.json").write_text('{}\n')

    with pytest.raises((ValueError, KeyError)):
        validate_frozen_dataset(output, "p1")
