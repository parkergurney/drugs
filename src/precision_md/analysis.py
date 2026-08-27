from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .oracle import checkpoint_bootstrap, evaluate_threshold, fit_threshold, oracle_speedup


def speed_ratio_ci(reference, fast, iterations=2000, seed=20260819):
    reference, fast = np.asarray(reference), np.asarray(fast)
    if len(reference) != len(fast): raise ValueError("paired timing arrays must match")
    rng, ratios = np.random.default_rng(seed), []
    for _ in range(iterations):
        idx = rng.integers(0, len(reference), len(reference))
        ratios.append(reference[idx].mean() / fast[idx].mean())
    return float(np.mean(reference) / np.mean(fast)), tuple(np.quantile(ratios, [.025, .975]))


def hierarchical_speed_ratio_ci(pairs, iterations=2000, seed=20260819):
    """Bootstrap paired timings by process, then by iteration within process."""
    normalized = []
    for reference, fast in pairs:
        reference, fast = np.asarray(reference), np.asarray(fast)
        if not len(reference) or len(reference) != len(fast):
            raise ValueError("each process must contain matched paired timings")
        normalized.append((reference, fast))
    if not normalized:
        raise ValueError("at least one process is required")
    point = np.mean([pair[0].mean() for pair in normalized]) / np.mean(
        [pair[1].mean() for pair in normalized]
    )
    rng, ratios = np.random.default_rng(seed), []
    for _ in range(iterations):
        sampled_processes = rng.integers(0, len(normalized), len(normalized))
        references, fast_values = [], []
        for process_index in sampled_processes:
            reference, fast = normalized[process_index]
            sample = rng.integers(0, len(reference), len(reference))
            references.append(reference[sample].mean())
            fast_values.append(fast[sample].mean())
        ratios.append(np.mean(references) / np.mean(fast_values))
    return float(point), tuple(np.quantile(ratios, [.025, .975]))


def _paired_process_timings(timings, policy, batch_size):
    pairs = []
    for _, process in timings.groupby("process_id", sort=True):
        reference = process[
            (process.policy == "fp32") & (process.batch_size == batch_size)
        ][["iteration", "wall_seconds"]]
        fast = process[
            (process.policy == policy) & (process.batch_size == batch_size)
        ][["iteration", "wall_seconds"]]
        paired = reference.merge(fast, on="iteration", suffixes=("_ref", "_fast"))
        if len(paired) != len(reference) or len(paired) != len(fast):
            raise ValueError(
                f"structurally unpaired timings for {policy}, batch {batch_size}"
            )
        wall_values = paired[["wall_seconds_ref", "wall_seconds_fast"]].to_numpy(
            dtype=float
        )
        if not np.isfinite(wall_values).all() or (wall_values <= 0).any():
            raise ValueError(
                f"invalid wall-clock timings for {policy}, batch {batch_size}"
            )
        pairs.append((paired.wall_seconds_ref.to_numpy(),
                      paired.wall_seconds_fast.to_numpy()))
    return pairs


def _telemetry_rows(process_id, paths):
    rows, gpu_names = [], set()
    for path in paths:
        try:
            table = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        columns = {str(column).strip().lower(): column for column in table.columns}
        for normalized, original in columns.items():
            if normalized.startswith("name"):
                gpu_names.update(table[original].dropna().astype(str).str.strip())
        metrics = {
            "temperature_c": "temperature.gpu",
            "utilization_percent": "utilization.gpu",
            "power_w": "power.draw",
            "sm_clock_mhz": "clocks.sm",
            "memory_clock_mhz": "clocks.mem",
            "memory_used_mib": "memory.used",
        }
        for metric, prefix in metrics.items():
            original = next(
                (column for normalized, column in columns.items()
                 if normalized.startswith(prefix)), None
            )
            if original is None:
                continue
            numeric = pd.to_numeric(
                table[original].astype(str).str.extract(r"([-+]?\d+(?:\.\d+)?)")[0],
                errors="coerce",
            ).dropna()
            if len(numeric):
                rows.append({
                    "process_id": process_id, "source_file": path.name,
                    "metric": metric, "sample_count": len(numeric),
                    "minimum": float(numeric.min()), "mean": float(numeric.mean()),
                    "maximum": float(numeric.max()),
                })
    return rows, sorted(gpu_names)


def analyze_trials(trials_root, output_root, iterations=2000, seed=20260819):
    """Combine isolated benchmark processes and analyze process-level replication."""
    trials_root, output_root = Path(trials_root), Path(output_root)
    trial_dirs = sorted(path.parent for path in trials_root.glob("*/manifest.json"))
    if not trial_dirs:
        raise FileNotFoundError(f"no trial manifests found under {trials_root}")
    evaluations, timings, manifests = [], [], []
    for trial_dir in trial_dirs:
        process_id = trial_dir.name
        required = [trial_dir / "evaluations.parquet", trial_dir / "timings.parquet"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete trial {process_id}: {', '.join(missing)}")
        manifest = json.loads((trial_dir / "manifest.json").read_text())
        manifests.append((process_id, manifest))
        evaluation = pd.read_parquet(required[0]).assign(process_id=process_id)
        timing = pd.read_parquet(required[1]).assign(process_id=process_id)
        evaluations.append(evaluation); timings.append(timing)
    invariant_fields = {
        "frames_sha256": "all trials must declare the same frames_sha256",
        "model_hash": "all trials must use the same model_hash",
        "config_sha256": "all trials must use the same config_sha256",
        "dataset_id": "all trials must use the same dataset_id",
        "experiment_id": "all trials must use the same experiment_id",
    }
    invariants = {}
    for field, message in invariant_fields.items():
        values = {manifest.get(field) for _, manifest in manifests}
        if None in values or len(values) != 1:
            raise ValueError(message)
        invariants[field] = next(iter(values))
    evaluations, timings = pd.concat(evaluations, ignore_index=True), pd.concat(
        timings, ignore_index=True
    )
    policies = sorted(set(evaluations.policy) - {"fp32"})
    summaries, performance_rows, process_rows = {}, [], []
    for policy in policies:
        choices = []
        for batch_size in sorted(set(timings.batch_size)):
            pairs = _paired_process_timings(timings, policy, batch_size)
            ratio, interval = hierarchical_speed_ratio_ci(
                pairs, iterations=iterations, seed=seed + int(batch_size)
            )
            per_process = []
            for (process_id, _), (reference, fast) in zip(manifests, pairs, strict=True):
                process_timing = timings[
                    (timings.process_id == process_id)
                    & (timings.policy == policy)
                    & (timings.batch_size == batch_size)
                ]
                process_speedup = float(reference.mean() / fast.mean())
                per_process.append(process_speedup)
                process_rows.append({
                    "process_id": process_id, "policy": policy,
                    "batch_size": int(batch_size), "speedup": process_speedup,
                    "reference_mean_seconds": float(reference.mean()),
                    "policy_mean_seconds": float(fast.mean()),
                    "paired_iterations": len(reference),
                    "output_finite_count": int(process_timing.finite.sum()),
                    "output_total_count": len(process_timing),
                })
            policy_timing = timings[
                (timings.policy == policy) & (timings.batch_size == batch_size)
            ]
            performance_rows.append({
                "policy": policy, "batch_size": int(batch_size),
                "process_count": len(pairs), "speedup": ratio,
                "speedup_ci_low": interval[0], "speedup_ci_high": interval[1],
                "process_speedup_std": float(np.std(per_process, ddof=1))
                if len(per_process) > 1 else 0.0,
                "process_speedup_min": float(np.min(per_process)),
                "process_speedup_max": float(np.max(per_process)),
                "output_finite_count": int(policy_timing.finite.sum()),
                "output_total_count": len(policy_timing),
                "all_timed_outputs_finite": bool(policy_timing.finite.all()),
            })
            if batch_size != 1:
                choices.append((ratio, interval, int(batch_size)))
        ratio, interval, batch_size = max(choices, default=(0, (0, 0), None))
        batch1 = _paired_process_timings(timings, policy, 1)
        batch1_reference = np.array([pair[0].mean() for pair in batch1])
        batch1_fast = np.array([pair[1].mean() for pair in batch1])
        policy_evaluations = evaluations[evaluations.policy == policy]
        force_errors = (
            policy_evaluations["max_force_error_ev_per_a"].to_numpy(dtype=float)
            if "max_force_error_ev_per_a" in policy_evaluations else np.array([])
        )
        finite_force_errors = force_errors[np.isfinite(force_errors)]
        item = {
            "process_count": len(trial_dirs),
            "finite_count": int(policy_evaluations.finite.sum()),
            "total_count": int(len(policy_evaluations)),
            "all_processes_finite": bool(policy_evaluations.finite.all()),
            "speedup": ratio,
            "speedup_ci_low": interval[0],
            "speedup_ci_high": interval[1],
            "winning_batch_size": batch_size,
            "batch1_slowdown": float(batch1_fast.mean() / batch1_reference.mean()),
            "max_force_error_ev_per_a": float(np.nanmax(force_errors, initial=0)),
            "mean_max_force_error_ev_per_a": (
                float(finite_force_errors.mean()) if len(finite_force_errors) else np.nan
            ),
        }
        item["passes"] = (
            item["all_processes_finite"] and item["speedup"] >= 1.2
            and item["speedup_ci_low"] > 1.0 and item["batch1_slowdown"] <= 1.05
        )
        summaries[policy] = item
    output = {
        "analysis_type": "hierarchical_process_bootstrap",
        "timing_validity_rule": (
            "pair completed measurements by process/batch/iteration; require positive "
            "finite wall time; report model-output finiteness separately"
        ),
        "analysis_amendment": (
            "studies/confirmatory-c1/amendments/"
            "2026-08-27-timing-output-validity.md"
        ),
        "process_ids": [path.name for path in trial_dirs],
        "process_count": len(trial_dirs),
        **invariants,
        "policies": summaries,
        "performance": performance_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    evaluations.to_parquet(output_root / "evaluations.parquet", index=False)
    timings.to_parquet(output_root / "timings.parquet", index=False)
    pd.DataFrame(performance_rows).to_parquet(
        output_root / "performance.parquet", index=False
    )
    pd.DataFrame(process_rows).to_parquet(
        output_root / "process-speedups.parquet", index=False
    )
    finite_counts = evaluations.groupby(
        ["process_id", "policy", "stratum"], dropna=False
    ).agg(total_count=("finite", "size"), finite_count=("finite", "sum")).reset_index()
    finite_counts["nonfinite_count"] = (
        finite_counts.total_count - finite_counts.finite_count
    )
    finite_counts.to_parquet(output_root / "finite-counts.parquet", index=False)
    error_rows = evaluations[evaluations.policy != "fp32"].copy()
    error_summaries = []
    for keys, group in error_rows.groupby(
        ["process_id", "policy", "molecule", "stratum"], dropna=False
    ):
        energy = group.get("energy_error_ev", pd.Series(dtype=float)).to_numpy(dtype=float)
        force = group.get(
            "max_force_error_ev_per_a", pd.Series(dtype=float)
        ).to_numpy(dtype=float)
        finite_energy, finite_force = energy[np.isfinite(energy)], force[np.isfinite(force)]
        error_summaries.append({
            **dict(zip(("process_id", "policy", "molecule", "stratum"), keys)),
            "count": len(group),
            "finite_count": int(group.finite.sum()),
            "energy_abs_mean_ev": float(np.mean(np.abs(finite_energy)))
            if len(finite_energy) else np.nan,
            "energy_abs_max_ev": float(np.max(np.abs(finite_energy)))
            if len(finite_energy) else np.nan,
            "force_mean_ev_per_a": float(np.mean(finite_force))
            if len(finite_force) else np.nan,
            "force_p95_ev_per_a": float(np.quantile(finite_force, .95))
            if len(finite_force) else np.nan,
            "force_max_ev_per_a": float(np.max(finite_force))
            if len(finite_force) else np.nan,
        })
    pd.DataFrame(error_summaries).to_parquet(
        output_root / "error-summary.parquet", index=False
    )
    telemetry, telemetry_metrics, all_gpu_names = [], [], set()
    for trial_dir in trial_dirs:
        paths = sorted(trial_dir.glob("gpu-*.csv"))
        metric_rows, gpu_names = _telemetry_rows(trial_dir.name, paths)
        telemetry_metrics.extend(metric_rows)
        all_gpu_names.update(gpu_names)
        telemetry.append({
            "process_id": trial_dir.name,
            "telemetry_file_count": len(paths),
            "telemetry_present": bool(paths),
            "telemetry_files": json.dumps([path.name for path in paths]),
        })
    pd.DataFrame(telemetry).to_parquet(
        output_root / "telemetry-summary.parquet", index=False
    )
    pd.DataFrame(
        telemetry_metrics,
        columns=("process_id", "source_file", "metric", "sample_count",
                 "minimum", "mean", "maximum"),
    ).to_parquet(output_root / "telemetry-metrics.parquet", index=False)
    missing_processes = [
        row["process_id"] for row in telemetry if not row["telemetry_present"]
    ]
    deviations = []
    if missing_processes:
        deviations.append(f"missing telemetry: {', '.join(missing_processes)}")
    if len(all_gpu_names) > 1:
        deviations.append("multiple GPU names were recorded")
    if all_gpu_names and any("A40" not in name for name in all_gpu_names):
        deviations.append("telemetry includes a non-A40 GPU")
    output["gpu_conditions"] = {
        "all_processes_have_telemetry": all(row["telemetry_present"] for row in telemetry),
        "missing_processes": missing_processes,
        "gpu_names": sorted(all_gpu_names),
        "mixed_gpu_names": len(all_gpu_names) > 1,
        "metric_row_count": len(telemetry_metrics),
        "deviations": deviations,
    }
    (output_root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def gate1_pass(summary):
    return (summary["finite_count"] == 300 and summary["total_count"] == 300
            and summary["speedup"] >= 1.2 and summary["speedup_ci_low"] > 1.0
            and summary["batch1_slowdown"] <= 1.05 and summary["discrepancy_above_noise"])


def gate2_verdict(rows, diagnostics):
    passing_oracle, passing_signals = False, False
    for length, group in pd.DataFrame(rows).groupby("block_length"):
        held = group[group.checkpoint_id >= 60].to_dict("records")
        unsafe_rate = np.mean([r["unsafe"] for r in held])
        speedup = oracle_speedup(held); ci = checkpoint_bootstrap(held)
        passing_oracle |= .05 <= unsafe_rate <= .30 and speedup >= 1.1 and ci[0] > 1
        passing_signals |= diagnostics.get(str(length), {}).get("precision", 0) >= .5 and diagnostics.get(str(length), {}).get("recall", 0) >= .8
    overall_rate = np.mean([r["unsafe"] for r in rows]) if rows else np.nan
    if passing_oracle and passing_signals: return "PROCEED"
    if passing_oracle: return "SIGNALS_INADEQUATE"
    if np.isfinite(overall_rate) and overall_rate <= .05: return "STATIC_ONLY"
    return "STOP"


def analyze_results(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    output = {"verdict": "STOP", "gate1": {}, "gate2": {}}
    gate1 = root / "gate1" / "summary.json"
    evaluations, timings = root / "gate1" / "evaluations.parquet", root / "gate1" / "timings.parquet"
    if evaluations.exists() and timings.exists():
        ev, tm = pd.read_parquet(evaluations), pd.read_parquet(timings)
        summaries = {}
        for policy in (p for p in ev.policy.unique() if p != "fp32"):
            choices = []
            for batch in (8, 32):
                ref = tm[(tm.policy == "fp32") & (tm.batch_size == batch) & tm.finite].sort_values("iteration").wall_seconds.to_numpy()
                fast = tm[(tm.policy == policy) & (tm.batch_size == batch) & tm.finite].sort_values("iteration").wall_seconds.to_numpy()
                if len(ref) and len(ref) == len(fast):
                    ratio, ci = speed_ratio_ci(ref, fast); choices.append((ratio, ci, batch))
            ratio, ci, batch = max(choices, default=(0, (0, 0), None))
            b1ref = tm[(tm.policy == "fp32") & (tm.batch_size == 1) & tm.finite].wall_seconds.mean()
            b1fast = tm[(tm.policy == policy) & (tm.batch_size == 1) & tm.finite].wall_seconds.mean()
            paired = ev[ev.policy == policy].merge(ev[ev.policy == "fp32"], on="frame_id", suffixes=("_fast", "_ref"))
            force_errors = paired.max_force_error_ev_per_a_fast.to_numpy(dtype=float)
            energy_errors = paired.energy_error_ev_fast.to_numpy(dtype=float)
            energy_by_molecule = {}
            for molecule, group in paired.groupby("molecule_fast"):
                values = group.energy_error_ev_fast.to_numpy(dtype=float)
                energy_by_molecule[molecule] = {
                    "mean_ev": float(np.nanmean(values)),
                    "std_ev": float(np.nanstd(values)),
                    "min_ev": float(np.nanmin(values)),
                    "max_ev": float(np.nanmax(values)),
                    "range_ev": float(np.nanmax(values) - np.nanmin(values)),
                }
            force_noise_floor = 10 * np.finfo(np.float32).eps
            item = {"finite_count": int(ev[(ev.policy == policy) & ev.finite].frame_id.nunique()),
                    "total_count": int(ev[ev.policy == policy].frame_id.nunique()), "speedup": ratio,
                    "speedup_ci_low": ci[0], "speedup_ci_high": ci[1], "winning_batch_size": batch,
                    "batch1_slowdown": float(b1fast / b1ref),
                    "discrepancy_above_noise": bool(np.nanmax(force_errors, initial=0) > force_noise_floor),
                    "force_noise_floor_ev_per_a": float(force_noise_floor),
                    "max_force_error_ev_per_a": float(np.nanmax(force_errors, initial=0)),
                    "mean_max_force_error_ev_per_a": float(np.nanmean(force_errors)),
                    "p95_max_force_error_ev_per_a": float(np.nanquantile(force_errors, .95)),
                    "mean_energy_error_ev": float(np.nanmean(energy_errors)),
                    "std_energy_error_ev": float(np.nanstd(energy_errors)),
                    "energy_offset_by_molecule": energy_by_molecule,
                    "unsupported_count": int(ev[(ev.policy == policy) & ev.unsupported].frame_id.nunique())}
            item["passes"] = gate1_pass(item); summaries[policy] = item
        passing = [(name, item) for name, item in summaries.items() if item["passes"]]
        selected = None
        if passing:
            passing.sort(key=lambda pair: pair[1]["speedup"], reverse=True)
            selected = passing[0][0]
            if len(passing) > 1:
                best_name, best = passing[0]; second_name, second = passing[1]
                intervals_overlap = not (
                    best["speedup_ci_low"] > second["speedup_ci_high"]
                    or second["speedup_ci_low"] > best["speedup_ci_high"]
                )
                if intervals_overlap and "bf16_amp" in (best_name, second_name):
                    selected = "bf16_amp"
        output["gate1"] = {
            "policies": summaries, "selected_policy": selected,
            "gate_passed": selected is not None,
            "timing_uses_genuine_graph_batches": bool(tm.genuine_graph_batch.all()),
        }
        gate1.write_text(json.dumps(output["gate1"], indent=2) + "\n")
    if gate1.exists(): output["gate1"] = json.loads(gate1.read_text())
    forks = root / "gate2" / "forks.parquet"
    if forks.exists():
        rows = pd.read_parquet(forks).to_dict("records"); diagnostics = {}
        for length in sorted({r["block_length"] for r in rows}):
            group = [r for r in rows if r["block_length"] == length]
            train = [r for r in group if r["checkpoint_id"] < 60]; held = [r for r in group if r["checkpoint_id"] >= 60]
            best = fit_threshold(train, "relative_force_disagreement")
            diagnostics[str(length)] = evaluate_threshold(held, "relative_force_disagreement", best["threshold"]) if best else {}
        output["gate2"] = {"diagnostics": diagnostics}; output["verdict"] = gate2_verdict(rows, diagnostics)
    (root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    return output
