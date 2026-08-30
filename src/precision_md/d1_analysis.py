from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import _telemetry_rows, hierarchical_speed_ratio_ci
from .config import D1Config
from .data import sha256_file
from .d1_diagnostics import DIAGNOSTIC_FILES, _load_inputs
from .d1_selection import d1_config_sha256


FINAL_FILES = (
    *DIAGNOSTIC_FILES,
    "first-divergence.parquet",
    "timing-components.parquet",
    "timing-summary.parquet",
    "telemetry-summary.parquet",
    "telemetry-metrics.parquet",
    "analysis.json",
    "report.md",
    "manifest.json",
)


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic_run(config: D1Config) -> Path:
    candidates = []
    for path in sorted((config.output_dir / "runs").glob("*/manifest.json")):
        if _manifest(path).get("run_type") == "instrumented_diagnostics":
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise ValueError(f"D1 requires exactly one diagnostic run, found {len(candidates)}")
    return candidates[0]


def _timing_runs(config: D1Config) -> list[Path]:
    candidates = []
    for path in sorted((config.output_dir / "runs").glob("*/manifest.json")):
        if _manifest(path).get("run_type") == "uninstrumented_component_timing":
            candidates.append(path.parent)
    if len(candidates) != 3:
        raise ValueError(f"D1 requires exactly three timing processes, found {len(candidates)}")
    return candidates


def _first_divergences(config: D1Config, traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reduced = traces[traces["policy"].isin(config.diagnostic_thresholds)]
    for (frame_id, policy, repeat), group in reduced.groupby(
        ["frame_id", "policy", "repeat"], sort=True
    ):
        if "trace_level" in group and (group.trace_level == "targeted").any():
            group = group[group.trace_level == "targeted"]
        group = group.sort_values("boundary_order")
        nonfinite = group[group["nonfinite_count"] > 0]
        threshold = config.diagnostic_thresholds[policy]
        crossed = group[group["relative_rms_from_fp32"] > threshold]
        nonfinite_row = nonfinite.iloc[0] if len(nonfinite) else None
        crossed_row = crossed.iloc[0] if len(crossed) else None
        rows.append({
            "frame_id": frame_id, "policy": policy, "repeat": int(repeat),
            "diagnostic_threshold": threshold,
            "first_nonfinite_order": (
                int(nonfinite_row.boundary_order) if nonfinite_row is not None else np.nan
            ),
            "first_nonfinite_boundary": (
                str(nonfinite_row.boundary) if nonfinite_row is not None else None
            ),
            "first_nonfinite_module_class": (
                str(nonfinite_row.module_class) if nonfinite_row is not None else None
            ),
            "first_nonfinite_operation_group": (
                str(nonfinite_row.operation_group) if nonfinite_row is not None else None
            ),
            "first_threshold_order": (
                int(crossed_row.boundary_order) if crossed_row is not None else np.nan
            ),
            "first_threshold_boundary": (
                str(crossed_row.boundary) if crossed_row is not None else None
            ),
            "first_threshold_operation_group": (
                str(crossed_row.operation_group) if crossed_row is not None else None
            ),
            "first_threshold_relative_rms": (
                float(crossed_row.relative_rms_from_fp32) if crossed_row is not None else np.nan
            ),
            "localized_nonfinite": nonfinite_row is not None,
            "localized_threshold_crossing": crossed_row is not None,
        })
    return pd.DataFrame(rows)


def _center_energies(energies: pd.DataFrame) -> pd.DataFrame:
    energies = energies.copy()
    means = energies.groupby(["policy", "molecule"])[
        "raw_energy_error_from_fp32_ev"
    ].transform("mean")
    energies["composition_centered_energy_error_ev"] = (
        energies["raw_energy_error_from_fp32_ev"] - means
    )
    return energies


def _timing_summary(config: D1Config, timings: pd.DataFrame):
    steady = timings[timings["phase"] == "steady"].copy()
    rows = []
    for component in sorted(steady.component.unique()):
        component_table = steady[steady.component == component]
        for batch_size in config.batch_sizes:
            for policy in ("tf32", "bf16_amp"):
                pairs = []
                process_ratios = []
                for process_id, process in component_table.groupby("process_id", sort=True):
                    reference = process[
                        (process.policy == "fp32") & (process.batch_size == batch_size)
                    ][["iteration", "wall_seconds"]]
                    candidate = process[
                        (process.policy == policy) & (process.batch_size == batch_size)
                    ][["iteration", "wall_seconds"]]
                    paired = reference.merge(candidate, on="iteration", suffixes=("_ref", "_fast"))
                    if len(paired) != config.timed_iterations:
                        raise ValueError(
                            f"unpaired D1 timings for {process_id}/{component}/{policy}/{batch_size}"
                        )
                    values = paired[["wall_seconds_ref", "wall_seconds_fast"]].to_numpy(float)
                    if not np.isfinite(values).all() or (values <= 0).any():
                        raise ValueError("D1 timing contains invalid wall-clock measurements")
                    pair = (paired.wall_seconds_ref.to_numpy(), paired.wall_seconds_fast.to_numpy())
                    pairs.append(pair)
                    process_ratios.append(float(pair[0].mean() / pair[1].mean()))
                ratio, interval = hierarchical_speed_ratio_ci(
                    pairs, seed=config.seed + batch_size + len(rows)
                )
                rows.append({
                    "component": component, "policy": policy,
                    "batch_size": batch_size, "process_count": len(pairs),
                    "fp32_over_policy_ratio": ratio,
                    "ratio_ci_low": interval[0], "ratio_ci_high": interval[1],
                    "process_ratio_std": float(np.std(process_ratios, ddof=1)),
                })
    return pd.DataFrame(rows)


def _render_report(analysis: dict, first: pd.DataFrame, timing: pd.DataFrame) -> str:
    nonfinite = first[first.localized_nonfinite]
    crossings = first[first.localized_threshold_crossing]
    lines = [
        "# D1 P1 failure-localization report", "",
        "D1 is exploratory. This report summarizes direct measurements and does not change C1.", "",
        "## Evidence status", "",
        f"- Selected frames: {analysis['selected_frame_count']}",
        f"- Instrumented diagnostic processes: {analysis['diagnostic_process_count']}",
        f"- Independent timing processes: {analysis['timing_process_count']}",
        f"- Cases with a localized nonfinite boundary: {len(nonfinite)}",
        f"- Cases crossing a frozen diagnostic threshold: {len(crossings)}", "",
        "## First-boundary counts", "",
    ]
    if len(nonfinite):
        for boundary, count in nonfinite.first_nonfinite_boundary.value_counts().items():
            lines.append(f"- Nonfinite: `{boundary}` — {count}")
    else:
        lines.append("- No nonfinite boundary was observed in isolated traces.")
    if len(crossings):
        for boundary, count in crossings.first_threshold_boundary.value_counts().items():
            lines.append(f"- Threshold: `{boundary}` — {count}")
    lines.extend(["", "## Timing ratios", ""])
    totals = timing[timing.component.isin([
        "prepared_model_total", "coordinate_to_result_total"
    ])]
    for row in totals.itertuples(index=False):
        lines.append(
            f"- {row.component}, batch {row.batch_size}, {row.policy}: "
            f"FP32/policy {row.fp32_over_policy_ratio:.3f} "
            f"(95% CI {row.ratio_ci_low:.3f}–{row.ratio_ci_high:.3f})"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "Use `first-divergence.parquet` together with the forward/backward and batching "
        "tables to support the D1 scientific conclusion. Mixed or unlocalized mechanisms "
        "must remain reported as such; artifact completeness is not a causal verdict.", "",
    ])
    return "\n".join(lines)


def analyze_d1(config: D1Config) -> dict:
    _, selected = _load_inputs(config)
    diagnostic_run = _diagnostic_run(config)
    timing_runs = _timing_runs(config)
    if (config.output_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to replace completed D1 analysis: {config.output_dir}")

    diagnostic_tables = {}
    for filename in DIAGNOSTIC_FILES:
        path = diagnostic_run / filename
        if not path.is_file():
            raise FileNotFoundError(f"incomplete D1 diagnostic run: {path}")
        diagnostic_tables[filename] = pd.read_parquet(path)
    diagnostic_tables["energy-decomposition.parquet"] = _center_energies(
        diagnostic_tables["energy-decomposition.parquet"]
    )
    for filename, table in diagnostic_tables.items():
        table.to_parquet(config.output_dir / filename, index=False)

    traces = diagnostic_tables["operator-traces.parquet"]
    first = _first_divergences(config, traces)
    first.to_parquet(config.output_dir / "first-divergence.parquet", index=False)
    timing_tables = []
    for run in timing_runs:
        timing_tables.append(pd.read_parquet(run / "timing-components.parquet"))
    timings = pd.concat(timing_tables, ignore_index=True)
    timings.to_parquet(config.output_dir / "timing-components.parquet", index=False)
    timing_summary = _timing_summary(config, timings)
    timing_summary.to_parquet(config.output_dir / "timing-summary.parquet", index=False)

    telemetry_summary, telemetry_metrics, gpu_names = [], [], set()
    all_runs = [diagnostic_run, *timing_runs]
    for run in all_runs:
        paths = sorted(run.glob("gpu-*.csv"))
        metrics, names = _telemetry_rows(run.name, paths)
        telemetry_metrics.extend(metrics)
        gpu_names.update(names)
        telemetry_summary.append({
            "run_id": run.name, "telemetry_file_count": len(paths),
            "telemetry_present": bool(paths),
            "telemetry_files": json.dumps([path.name for path in paths]),
        })
    pd.DataFrame(telemetry_summary).to_parquet(
        config.output_dir / "telemetry-summary.parquet", index=False
    )
    pd.DataFrame(
        telemetry_metrics,
        columns=("process_id", "source_file", "metric", "sample_count",
                 "minimum", "mean", "maximum"),
    ).to_parquet(config.output_dir / "telemetry-metrics.parquet", index=False)
    missing_telemetry = [row["run_id"] for row in telemetry_summary
                         if not row["telemetry_present"]]
    gpu_deviations = []
    if missing_telemetry:
        gpu_deviations.append(f"missing telemetry: {', '.join(missing_telemetry)}")
    if len(gpu_names) > 1:
        gpu_deviations.append("multiple GPU names recorded")
    if gpu_names and any("A40" not in name for name in gpu_names):
        gpu_deviations.append("telemetry includes a non-A40 GPU")

    analysis = {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "analysis_type": "exploratory_failure_localization",
        "evidence_status": "evidence_complete",
        "scientific_gate_decision": "review_required",
        "selected_frame_count": len(selected),
        "diagnostic_process_count": 1, "timing_process_count": len(timing_runs),
        "localized_nonfinite_case_count": int(first.localized_nonfinite.sum()),
        "threshold_crossing_case_count": int(first.localized_threshold_crossing.sum()),
        "diagnostic_thresholds": config.diagnostic_thresholds,
        "relative_rms_floor": config.relative_rms_floor,
        "frames_sha256": config.expected_frames_sha256,
        "evaluations_sha256": config.expected_evaluations_sha256,
        "model_hash": config.expected_model_hash,
        "config_sha256": d1_config_sha256(config),
        "timing_process_ids": [path.name for path in timing_runs],
        "gpu_conditions": {
            "gpu_names": sorted(gpu_names),
            "all_runs_have_telemetry": not missing_telemetry,
            "deviations": gpu_deviations,
        },
    }
    (config.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (config.output_dir / "report.md").write_text(
        _render_report(analysis, first, timing_summary), encoding="utf-8"
    )
    manifest = analysis | {
        "artifact_schema_version": 1,
        "diagnostic_run_id": diagnostic_run.name,
        "files_sha256": {
            name: sha256_file(config.output_dir / name)
            for name in FINAL_FILES if name != "manifest.json"
            and (config.output_dir / name).is_file()
        },
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_sums(config.output_dir)
    return analysis


def _write_sums(root: Path):
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in paths
    ), encoding="utf-8")


def validate_d1(config: D1Config) -> dict:
    _, selected = _load_inputs(config)
    root = config.output_dir
    missing = [name for name in (*FINAL_FILES, "SHA256SUMS") if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete D1 artifact: {', '.join(missing)}")
    manifest = _manifest(root / "manifest.json")
    for field, expected in (
        ("experiment_id", config.experiment_id),
        ("frames_sha256", config.expected_frames_sha256),
        ("evaluations_sha256", config.expected_evaluations_sha256),
        ("model_hash", config.expected_model_hash),
        ("selected_frame_count", len(selected)),
        ("timing_process_count", 3),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"D1 manifest {field} mismatch")
    sums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe D1 checksum path: {name}")
        sums[name] = digest
    for name, expected in sums.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"D1 checksum mismatch: {name}")
    selection_ids = {row["frame_id"] for row in json.loads(
        (root / "d1-selection.json").read_text()
    )["records"]}
    traces = pd.read_parquet(root / "operator-traces.parquet")
    energies = pd.read_parquet(root / "energy-decomposition.parquet")
    if set(traces.frame_id) != selection_ids or set(energies.frame_id) != selection_ids:
        raise ValueError("D1 final tables do not cover the frozen selection")
    expected_policies = set(config.diagnostic_policies)
    for frame_id, group in energies.groupby("frame_id"):
        if set(group.policy) != expected_policies:
            raise ValueError(f"D1 energy policies incomplete for {frame_id}")
    timing = pd.read_parquet(root / "timing-components.parquet")
    steady = timing[timing.phase == "steady"]
    for keys, group in steady.groupby(["process_id", "policy", "batch_size", "component"]):
        if len(group) != config.timed_iterations:
            raise ValueError(f"D1 timing count mismatch for {keys}")
    for run in [_diagnostic_run(config), *_timing_runs(config)]:
        if not list(run.glob("gpu-*.csv")):
            raise FileNotFoundError(f"D1 run is missing GPU telemetry: {run}")
    return {
        "experiment_id": config.experiment_id,
        "evidence_status": manifest.get("evidence_status"),
        "selected_frame_count": len(selected),
        "timing_process_count": 3,
        "checksummed_file_count": len(sums),
    }
