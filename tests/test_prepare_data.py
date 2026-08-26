import json
from types import SimpleNamespace

import numpy as np

from precision_md.cli import prepare_data
from precision_md.config import Gate1Config


class FakeEvaluator:
    model_hash = "fake-model-hash"

    def __init__(self, model, device):
        pass

    def evaluate(self, batch, policy):
        scale = float(batch.positions[0, 0]) + 1.0
        return SimpleNamespace(
            finite=True,
            error=None,
            forces=np.ones((len(batch.atomic_numbers), 3)) * scale,
        )


def test_prepare_data_records_provenance_and_enforces_exclusion(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "rmd17"
    dataset_dir.mkdir()
    molecules = ["a", "b", "c"]
    for molecule_index, molecule in enumerate(molecules):
        positions = np.zeros((500, 4, 3), dtype=float)
        positions[:, 0, 0] = np.arange(500) + molecule_index * 1000
        np.savez_compressed(
            dataset_dir / f"{molecule}.npz",
            R=positions,
            nuclear_charges=np.array([6, 1, 1, 1]),
        )
    exclusion = tmp_path / "p1-selection.json"
    exclusion.write_text(
        json.dumps({"frames": ["a-0", "b-1-cc-0.8-2", "c-2"]}) + "\n"
    )
    output_dir = tmp_path / "c1"
    config = Gate1Config(
        dataset_id="c1-test",
        seed=17,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        molecules=molecules,
        ordinary_counts=[34, 33, 33],
        candidate_pool=300,
        exclude_selection=exclusion,
        device="cpu",
    )
    monkeypatch.setattr("precision_md.cli.MaceEvaluator", FakeEvaluator)

    prepare_data(config)

    selection = json.loads((output_dir / "selection.json").read_text())
    manifest = json.loads((output_dir / "dataset-manifest.json").read_text())
    assert len(selection["frames"]) == 300
    assert not {"a-0", "b-1", "c-2"} & set(selection["source_frames"])
    assert manifest["dataset_id"] == "c1-test"
    assert manifest["model_hash"] == "fake-model-hash"
    assert manifest["excluded_source_count"] == 3
    assert manifest["overlap_count"] == 0
    assert manifest["stratum_counts"] == {
        "ordinary": 100,
        "high_force": 100,
        "close_contact": 100,
    }
    assert len(manifest["frames_sha256"]) == 64
