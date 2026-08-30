import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precision_md.config import D1Config
from precision_md.d1_analysis import (
    _center_energies, _first_divergences, _timing_summary, analyze_d1, validate_d1,
)
from precision_md.d1_diagnostics import DIAGNOSTIC_FILES
from precision_md.d1_probe import TraceCollector, TraceSetupError
from precision_md.d1_selection import select_d1_frames, write_d1_selection
from precision_md.data import sha256_file


HASH = "a" * 64


def _config(tmp_path, **changes):
    values = {
        "dataset": tmp_path / "dataset",
        "c1_analysis": tmp_path / "analysis",
        "output_dir": tmp_path / "d1",
        "expected_frames_sha256": HASH,
        "expected_evaluations_sha256": HASH,
        "expected_model_hash": HASH,
    } | changes
    return D1Config(**values)


def test_d1_config_freezes_timing_policy(tmp_path):
    with pytest.raises(ValueError, match="timing policies"):
        _config(tmp_path, timing_policies=["fp32", "tf32"])


def test_d1_selection_is_deterministic_and_includes_nonfinite(tmp_path):
    dataset, analysis = tmp_path / "dataset", tmp_path / "analysis"
    dataset.mkdir(); analysis.mkdir()
    frames, rows = [], []
    for stratum in ("ordinary", "high_force", "close_contact"):
        for index in range(7):
            frame_id = f"{stratum}-{index}"
            frames.append({
                "frame_id": frame_id, "molecule": "ethanol", "stratum": stratum,
                "atomic_numbers": np.array([1]), "positions": np.zeros((1, 3)),
            })
            for process in ("run-1", "run-2"):
                for policy, scale in (("tf32", 1.0), ("bf16_amp", 2.0)):
                    failed = policy == "bf16_amp" and stratum == "close_contact" and index == 3
                    rows.append({
                        "frame_id": frame_id, "policy": policy, "stratum": stratum,
                        "finite": not failed, "max_force_error_ev_per_a": (
                            np.nan if failed else scale * index
                        ), "process_id": process,
                    })
    np.savez_compressed(dataset / "frames.npz", frames=np.array(frames, dtype=object))
    pd.DataFrame(rows).to_parquet(analysis / "evaluations.parquet", index=False)
    config = _config(
        tmp_path,
        expected_frames_sha256=sha256_file(dataset / "frames.npz"),
        expected_evaluations_sha256=sha256_file(analysis / "evaluations.parquet"),
    )
    first = select_d1_frames(config)
    second = select_d1_frames(config)
    assert first == second
    selected = {row["frame_id"]: row for row in first["records"]}
    assert "close_contact-3" in selected
    assert any(
        reason["kind"] == "nonfinite"
        for reason in selected["close_contact-3"]["reasons"]
    )
    assert first["frame_count"] == len(selected)


def test_d1_selection_reuses_original_commit_when_scientific_fields_match(
    tmp_path, monkeypatch
):
    output = tmp_path / "d1"
    output.mkdir()
    original = {
        "schema_version": 1,
        "selection_script_commit": "original",
        "records": [{"frame_id": "f"}],
    }
    (output / "d1-selection.json").write_text(json.dumps(original, indent=2) + "\n")
    monkeypatch.setattr(
        "precision_md.d1_selection.select_d1_frames",
        lambda config: original | {"selection_script_commit": "repair"},
    )

    reused = write_d1_selection(_config(tmp_path, output_dir=output))

    assert reused["selection_script_commit"] == "original"
    assert json.loads((output / "d1-selection.json").read_text()) == original


@pytest.mark.skipif(
    not Path("data/frozen/p1/frames.npz").is_file()
    or not Path("artifacts/analysis/c1-reproduction/evaluations.parquet").is_file(),
    reason="local immutable P1/C1 artifacts are not installed",
)
def test_frozen_p1_c1_selection_has_expected_27_cases(tmp_path):
    config = _config(
        tmp_path,
        dataset=Path("data/frozen/p1"),
        c1_analysis=Path("artifacts/analysis/c1-reproduction"),
        expected_frames_sha256=(
            "f7e759b6f0050b82eae88ff99416a2d43f50eac9e2e944a7524e80eaff40a28d"
        ),
        expected_evaluations_sha256=(
            "da741dc69df6052f83c2036282af52bb1b403eb85c4ee4fb3aa8f1ac55cedaa8"
        ),
    )
    selection = select_d1_frames(config)
    assert selection["frame_count"] == 27
    selected = {row["frame_id"] for row in selection["records"]}
    assert {
        "aspirin-8953-cc-1.0-67", "aspirin-95030-cc-1.2-98",
        "ethanol-46298-cc-1.0-16", "malonaldehyde-27944-cc-1.2-41",
        "malonaldehyde-44567-cc-0.8-48", "malonaldehyde-83937-cc-1.2-59",
    } <= selected


def test_first_divergence_uses_frozen_policy_thresholds(tmp_path):
    config = _config(tmp_path)
    traces = pd.DataFrame([
        {"frame_id": "f", "policy": "tf32", "repeat": 0,
         "boundary_order": 0, "boundary": "radial", "module_class": "R",
         "operation_group": "radial_and_cutoff",
         "nonfinite_count": 0, "relative_rms_from_fp32": 0.005},
        {"frame_id": "f", "policy": "tf32", "repeat": 0,
         "boundary_order": 1, "boundary": "tensor_product", "module_class": "TP",
         "operation_group": "equivariant_contraction",
         "nonfinite_count": 0, "relative_rms_from_fp32": 0.02},
        {"frame_id": "f", "policy": "tf32", "repeat": 0,
         "boundary_order": 2, "boundary": "forces", "module_class": "M",
         "operation_group": "force_gradient",
         "nonfinite_count": 1, "relative_rms_from_fp32": np.nan},
    ])
    output = _first_divergences(config, traces).iloc[0]
    assert output.first_threshold_boundary == "tensor_product"
    assert output.first_nonfinite_boundary == "forces"


def test_first_divergence_rejects_empty_trace_schema(tmp_path):
    with pytest.raises(ValueError, match="operator trace is empty"):
        _first_divergences(_config(tmp_path), pd.DataFrame())


def test_trace_collector_skips_script_modules_but_observes_eager_parent():
    torch = pytest.importorskip("torch")

    class ScaleShiftMACE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_energies_fn = torch.nn.Identity()
            self.node_embedding = torch.jit.trace(
                torch.nn.Linear(2, 2), torch.ones(1, 2)
            )
            self.spherical_harmonics = torch.nn.Identity()
            self.radial_embedding = torch.nn.Identity()
            self.interactions = torch.nn.ModuleList()
            self.products = torch.nn.ModuleList()
            self.readouts = torch.nn.Sequential(torch.nn.Identity())
            self.scale_shift = torch.nn.Identity()

        def forward(self, value):
            return self.readouts(self.node_embedding(value))

    model = ScaleShiftMACE()
    collector = TraceCollector(model, "fp32", "frame")
    with collector:
        model(torch.ones(1, 2))

    assert "node_embedding" in collector.skipped_script_modules
    assert any(row["boundary"].startswith("readouts") for row in collector.rows)


def test_trace_collector_hook_setup_is_transactional(monkeypatch):
    torch = pytest.importorskip("torch")

    class ScaleShiftMACE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_energies_fn = torch.nn.Identity()
            self.node_embedding = torch.nn.Identity()
            self.spherical_harmonics = torch.nn.Identity()
            self.radial_embedding = torch.nn.Identity()
            self.interactions = torch.nn.ModuleList()
            self.products = torch.nn.ModuleList()
            self.readouts = torch.nn.ModuleList()
            self.scale_shift = torch.nn.Identity()

    model = ScaleShiftMACE()
    monkeypatch.setattr(
        model.node_embedding,
        "register_forward_hook",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken hook")),
    )
    collector = TraceCollector(model, "fp32", "frame")
    with pytest.raises(TraceSetupError, match="node_embedding"):
        collector.__enter__()
    assert not collector._handles
    assert not model.node_embedding._forward_pre_hooks


def test_energy_centering_is_within_policy_and_composition():
    table = pd.DataFrame([
        {"policy": "tf32", "molecule": "ethanol", "raw_energy_error_from_fp32_ev": 1.0},
        {"policy": "tf32", "molecule": "ethanol", "raw_energy_error_from_fp32_ev": 3.0},
        {"policy": "bf16_amp", "molecule": "ethanol", "raw_energy_error_from_fp32_ev": 10.0},
    ])
    centered = _center_energies(table)
    assert centered.composition_centered_energy_error_ev.tolist() == [-1.0, 1.0, 0.0]


def test_timing_summary_pairs_process_then_iteration(tmp_path):
    config = _config(tmp_path, timed_iterations=2)
    rows = []
    for process_index in range(3):
        for batch in (1, 8, 32):
            for iteration in range(2):
                for policy, seconds in (("fp32", 2.0), ("tf32", 1.0), ("bf16_amp", 4.0)):
                    rows.append({
                        "process_id": f"p{process_index}", "policy": policy,
                        "batch_size": batch, "iteration": iteration,
                        "phase": "steady", "component": "energy_forward",
                        "wall_seconds": seconds,
                    })
    summary = _timing_summary(config, pd.DataFrame(rows))
    tf32 = summary[(summary.policy == "tf32") & (summary.batch_size == 8)].iloc[0]
    bf16 = summary[(summary.policy == "bf16_amp") & (summary.batch_size == 8)].iloc[0]
    assert tf32.fp32_over_policy_ratio == pytest.approx(2.0)
    assert bf16.fp32_over_policy_ratio == pytest.approx(0.5)
    assert tf32.process_count == 3


def test_d1_runpod_script_uses_uv_and_unique_processes():
    script = (Path(__file__).parents[1] / "scripts" / "run-a40-d1.sh").read_text()
    assert "uv run precision-md diagnose-d1" in script
    assert "d1-time-01 d1-time-02 d1-time-03" in script
    assert "nvidia-smi" in script


def test_d1_analysis_and_validation_complete_artifact(tmp_path):
    dataset, c1, output = tmp_path / "dataset", tmp_path / "c1", tmp_path / "d1"
    dataset.mkdir(); c1.mkdir(); output.mkdir()
    frame = {
        "frame_id": "f", "molecule": "ethanol", "stratum": "ordinary",
        "atomic_numbers": np.array([1]), "positions": np.zeros((1, 3)),
    }
    np.savez_compressed(dataset / "frames.npz", frames=np.array([frame], dtype=object))
    pd.DataFrame([{
        "frame_id": "f", "policy": "tf32", "stratum": "ordinary",
        "finite": True, "max_force_error_ev_per_a": 1.0, "process_id": "c1",
    }]).to_parquet(c1 / "evaluations.parquet", index=False)
    config = _config(
        tmp_path, dataset=dataset, c1_analysis=c1, output_dir=output,
        expected_frames_sha256=sha256_file(dataset / "frames.npz"),
        expected_evaluations_sha256=sha256_file(c1 / "evaluations.parquet"),
        timed_iterations=2,
    )
    (output / "d1-selection.json").write_text(json.dumps({
        "schema_version": 1, "frame_count": 1,
        "input_sha256": {"frames.npz": config.expected_frames_sha256},
        "records": [{"frame_id": "f", "molecule": "ethanol",
                     "stratum": "ordinary", "reasons": []}],
    }))

    trace_run = output / "runs" / "d1-trace-01"
    trace_run.mkdir(parents=True)
    trace_rows = []
    for policy in config.diagnostic_policies:
        trace_rows.append({
            "frame_id": "f", "policy": policy, "repeat": 0,
            "boundary_order": 0, "boundary": "model.total_energy",
            "module_class": "ScaleShiftMACE", "operation_group": "energy_accumulation",
            "nonfinite_count": 0,
            "relative_rms_from_fp32": 0.02 if policy == "tf32" else 0.0,
        })
    energy_rows = [{
        "frame_id": "f", "molecule": "ethanol", "stratum": "ordinary",
        "policy": policy, "raw_energy_error_from_fp32_ev": 0.0,
    } for policy in config.diagnostic_policies]
    for filename in DIAGNOSTIC_FILES:
        if filename == "operator-traces.parquet":
            table = pd.DataFrame(trace_rows)
        elif filename == "energy-decomposition.parquet":
            table = pd.DataFrame(energy_rows)
        else:
            table = pd.DataFrame([{"frame_id": "f", "policy": "fp32", "finite": True}])
        table.to_parquet(trace_run / filename, index=False)
    (trace_run / "manifest.json").write_text(json.dumps({
        "run_type": "instrumented_diagnostics", "experiment_id": config.experiment_id,
    }))

    for process_index in range(3):
        run = output / "runs" / f"d1-time-{process_index + 1:02d}"
        run.mkdir()
        rows = []
        for batch in config.batch_sizes:
            for iteration in range(config.timed_iterations):
                for policy, seconds in (("fp32", 2.0), ("tf32", 1.0), ("bf16_amp", 4.0)):
                    rows.append({
                        "process_id": run.name, "policy": policy, "batch_size": batch,
                        "iteration": iteration, "phase": "steady",
                        "component": "prepared_model_total", "wall_seconds": seconds,
                    })
        pd.DataFrame(rows).to_parquet(run / "timing-components.parquet", index=False)
        (run / "manifest.json").write_text(json.dumps({
            "run_type": "uninstrumented_component_timing",
            "experiment_id": config.experiment_id,
        }))
    for run in [trace_run, *(output / "runs").glob("d1-time-*")]:
        pd.DataFrame([{
            "timestamp": "now", "name": "NVIDIA A40",
            "temperature.gpu": 40, "utilization.gpu": 50,
        }]).to_csv(run / "gpu-before.csv", index=False)

    result = analyze_d1(config)
    validated = validate_d1(config)
    assert result["evidence_status"] == "evidence_complete"
    assert validated["timing_process_count"] == 3
    assert (output / "SHA256SUMS").is_file()
