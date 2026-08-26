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
    point = np.concatenate([pair[0] for pair in normalized]).mean() / np.concatenate(
        [pair[1] for pair in normalized]
    ).mean()
    rng, ratios = np.random.default_rng(seed), []
    for _ in range(iterations):
        sampled_processes = rng.integers(0, len(normalized), len(normalized))
        references, fast_values = [], []
        for process_index in sampled_processes:
            reference, fast = normalized[process_index]
            sample = rng.integers(0, len(reference), len(reference))
            references.append(reference[sample])
            fast_values.append(fast[sample])
        ratios.append(
            np.concatenate(references).mean() / np.concatenate(fast_values).mean()
        )
    return float(point), tuple(np.quantile(ratios, [.025, .975]))


def _paired_process_timings(timings, policy, batch_size):
    pairs = []
    for _, process in timings.groupby("process_id", sort=True):
        reference = process[
            (process.policy == "fp32") & (process.batch_size == batch_size)
            & process.finite
        ][["iteration", "wall_seconds"]]
        fast = process[
            (process.policy == policy) & (process.batch_size == batch_size)
            & process.finite
        ][["iteration", "wall_seconds"]]
        paired = reference.merge(fast, on="iteration", suffixes=("_ref", "_fast"))
        if len(paired) != len(reference) or len(paired) != len(fast):
            raise ValueError(
                f"unpaired finite timings for {policy}, batch {batch_size}"
            )
        pairs.append((paired.wall_seconds_ref.to_numpy(),
                      paired.wall_seconds_fast.to_numpy()))
    return pairs


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
    frame_hashes = {manifest.get("frames_sha256") for _, manifest in manifests}
    if None in frame_hashes or len(frame_hashes) != 1:
        raise ValueError("all trials must declare the same frames_sha256")
    model_hashes = {manifest.get("model_hash") for _, manifest in manifests}
    if None in model_hashes or len(model_hashes) != 1:
        raise ValueError("all trials must use the same model_hash")
    evaluations, timings = pd.concat(evaluations, ignore_index=True), pd.concat(
        timings, ignore_index=True
    )
    policies = sorted(set(evaluations.policy) - {"fp32"})
    summaries = {}
    for policy in policies:
        choices = []
        for batch_size in sorted(set(timings.batch_size) - {1}):
            pairs = _paired_process_timings(timings, policy, batch_size)
            ratio, interval = hierarchical_speed_ratio_ci(
                pairs, iterations=iterations, seed=seed + int(batch_size)
            )
            choices.append((ratio, interval, int(batch_size)))
        ratio, interval, batch_size = max(choices, default=(0, (0, 0), None))
        batch1 = _paired_process_timings(timings, policy, 1)
        batch1_reference = np.concatenate([pair[0] for pair in batch1])
        batch1_fast = np.concatenate([pair[1] for pair in batch1])
        policy_evaluations = evaluations[evaluations.policy == policy]
        force_errors = (
            policy_evaluations["max_force_error_ev_per_a"].to_numpy(dtype=float)
            if "max_force_error_ev_per_a" in policy_evaluations else np.array([])
        )
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
                float(np.nanmean(force_errors)) if len(force_errors) else np.nan
            ),
        }
        item["passes"] = (
            item["all_processes_finite"] and item["speedup"] >= 1.2
            and item["speedup_ci_low"] > 1.0 and item["batch1_slowdown"] <= 1.05
        )
        summaries[policy] = item
    output = {
        "analysis_type": "hierarchical_process_bootstrap",
        "process_ids": [path.name for path in trial_dirs],
        "process_count": len(trial_dirs),
        "frames_sha256": next(iter(frame_hashes)),
        "model_hash": next(iter(model_hashes)),
        "policies": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    evaluations.to_parquet(output_root / "evaluations.parquet", index=False)
    timings.to_parquet(output_root / "timings.parquet", index=False)
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
