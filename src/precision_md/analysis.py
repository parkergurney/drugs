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
