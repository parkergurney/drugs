import numpy as np
import pandas as pd
import pytest
from precision_md.analysis import (
    analyze_trials, gate1_pass, hierarchical_speed_ratio_ci, speed_ratio_ci,
)


def test_paired_bootstrap_speedup():
    ratio, ci = speed_ratio_ci(np.ones(20) * 2, np.ones(20), iterations=100)
    assert ratio == pytest.approx(2) and ci == pytest.approx((2, 2))


def test_gate1_conditions():
    summary = {"finite_count":300, "total_count":300, "speedup":1.21, "speedup_ci_low":1.1,
               "batch1_slowdown":1.0, "discrepancy_above_noise":True}
    assert gate1_pass(summary)
    assert not gate1_pass(summary | {"finite_count":299})


def test_hierarchical_bootstrap_uses_process_pairs():
    pairs = [
        (np.ones(10) * 2, np.ones(10)),
        (np.ones(10) * 4, np.ones(10) * 2),
    ]
    ratio, ci = hierarchical_speed_ratio_ci(pairs, iterations=100)
    assert ratio == pytest.approx(2)
    assert ci == pytest.approx((2, 2))


def test_analyze_trials_combines_isolated_processes(tmp_path):
    trials = tmp_path / "trials"
    for process_number in (1, 2):
        trial = trials / f"run-{process_number:02d}"
        trial.mkdir(parents=True)
        (trial / "manifest.json").write_text(
            '{"frames_sha256":"frames", "model_hash":"model", '
            '"config_sha256":"config", "dataset_id":"p1", '
            '"experiment_id":"c1"}\n'
        )
        evaluations = []
        for frame_id in ("a", "b"):
            evaluations += [
                {"frame_id": frame_id, "policy": "fp32", "finite": True,
                 "molecule": "ethanol", "stratum": "ordinary",
                 "max_force_error_ev_per_a": np.nan, "energy_error_ev": np.nan},
                {"frame_id": frame_id, "policy": "tf32", "finite": True,
                 "molecule": "ethanol", "stratum": "ordinary",
                 "max_force_error_ev_per_a": 0.1, "energy_error_ev": 0.2},
            ]
        pd.DataFrame(evaluations).to_parquet(trial / "evaluations.parquet")
        timings = []
        for batch_size in (1, 8):
            for iteration in range(4):
                timings += [
                    {"policy": "fp32", "batch_size": batch_size,
                     "iteration": iteration, "wall_seconds": 2.0, "finite": True},
                    {"policy": "tf32", "batch_size": batch_size,
                     "iteration": iteration, "wall_seconds": 1.0, "finite": True},
                ]
        pd.DataFrame(timings).to_parquet(trial / "timings.parquet")
        (trial / "gpu-telemetry.csv").write_text(
            "name, temperature.gpu, utilization.gpu, power.draw\n"
            "NVIDIA A40, 55 C, 80 %, 250 W\n"
        )

    result = analyze_trials(trials, tmp_path / "analysis", iterations=100)

    assert result["process_count"] == 2
    assert result["policies"]["tf32"]["speedup"] == pytest.approx(2)
    assert (tmp_path / "analysis" / "analysis.json").exists()
    assert (tmp_path / "analysis" / "process-speedups.parquet").exists()
    assert (tmp_path / "analysis" / "finite-counts.parquet").exists()
    combined = pd.read_parquet(tmp_path / "analysis" / "timings.parquet")
    assert set(combined.process_id) == {"run-01", "run-02"}
    assert result["gpu_conditions"]["gpu_names"] == ["NVIDIA A40"]
    assert result["gpu_conditions"]["deviations"] == []


def test_hierarchical_point_estimate_weights_processes_equally():
    pairs = [
        (np.array([2.0]), np.array([1.0])),
        (np.ones(100) * 3.0, np.ones(100) * 2.0),
    ]
    ratio, _ = hierarchical_speed_ratio_ci(pairs, iterations=50)
    assert ratio == pytest.approx(2.5 / 1.5)


def test_analysis_times_completed_nonfinite_outputs_without_hiding_failure(tmp_path):
    trials = tmp_path / "trials"
    for process_number in (1, 2):
        trial = trials / f"run-{process_number:02d}"
        trial.mkdir(parents=True)
        (trial / "manifest.json").write_text(
            '{"frames_sha256":"frames", "model_hash":"model", '
            '"config_sha256":"config", "dataset_id":"p1", '
            '"experiment_id":"c1"}\n'
        )
        evaluations = []
        for frame_id in ("a", "b"):
            evaluations += [
                {"frame_id": frame_id, "policy": "fp32", "finite": True,
                 "molecule": "ethanol", "stratum": "ordinary",
                 "max_force_error_ev_per_a": np.nan, "energy_error_ev": np.nan},
                {"frame_id": frame_id, "policy": "bf16_amp", "finite": False,
                 "molecule": "ethanol", "stratum": "ordinary",
                 "max_force_error_ev_per_a": np.nan, "energy_error_ev": np.nan},
            ]
        pd.DataFrame(evaluations).to_parquet(trial / "evaluations.parquet")
        timings = []
        for batch_size in (1, 8, 32):
            for iteration in range(4):
                timings += [
                    {"policy": "fp32", "batch_size": batch_size,
                     "iteration": iteration, "wall_seconds": 2.0, "finite": True},
                    {"policy": "bf16_amp", "batch_size": batch_size,
                     "iteration": iteration, "wall_seconds": 1.0,
                     "finite": batch_size != 32},
                ]
        pd.DataFrame(timings).to_parquet(trial / "timings.parquet")

    result = analyze_trials(trials, tmp_path / "analysis", iterations=100)
    batch32 = next(
        row for row in result["performance"]
        if row["policy"] == "bf16_amp" and row["batch_size"] == 32
    )

    assert batch32["speedup"] == pytest.approx(2)
    assert batch32["output_finite_count"] == 0
    assert batch32["output_total_count"] == 8
    assert not batch32["all_timed_outputs_finite"]
    assert not result["policies"]["bf16_amp"]["passes"]
