from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .model import MaceEvaluator
from .types import AtomicBatch


def _atomic_batch(frame):
    return AtomicBatch(
        np.asarray(frame["atomic_numbers"]), np.asarray(frame["positions"]),
        (frame["frame_id"],),
    )


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _representative_frames(frames, count):
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[index] for index in indices]


def _add_discrepancies(rows, forces):
    table = pd.DataFrame(rows)
    references = table[table.policy == "fp32"].set_index("frame_id")
    for row in rows:
        if row["policy"] == "fp32" or not row["finite"]:
            continue
        reference = references.loc[row["frame_id"]]
        reference_force = forces[(row["frame_id"], "fp32")]
        fast_force = forces[(row["frame_id"], row["policy"])]
        force_error = np.linalg.norm(fast_force - reference_force, axis=1)
        force_norm = np.linalg.norm(reference_force, axis=1)
        row.update({
            "energy_error_ev": row["energy_ev"] - float(reference.energy_ev),
            "energy_error_per_atom_ev": (
                row["energy_ev"] - float(reference.energy_ev)
            ) / row["atom_count"],
            "max_force_error_ev_per_a": float(force_error.max()),
            "mean_force_error_ev_per_a": float(force_error.mean()),
            "p95_force_error_ev_per_a": float(np.quantile(force_error, 0.95)),
            "relative_max_force_error": float(
                force_error.max() / max(force_norm.max(), 1e-12)
            ),
        })
    return rows


def run_benchmark(config, allow_gpu=False):
    if config.device.startswith("cuda") and not allow_gpu:
        raise RuntimeError("GPU benchmark requires explicit --allow-gpu-benchmark")
    frames_file = config.output_dir / "frames.npz"
    if not frames_file.exists():
        raise FileNotFoundError(f"run prepare-data first: {frames_file}")
    with np.load(frames_file, allow_pickle=True) as data:
        frames = data["frames"].tolist()
    if len(frames) != 300:
        raise ValueError(f"Gate 1 requires exactly 300 frames, found {len(frames)}")
    frame_ids = [frame["frame_id"] for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Gate 1 frame IDs must be unique across strata")

    evaluator = MaceEvaluator(config.model, config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, timings, forces = [], [], {}

    # Numerical characterization uses real disconnected graph batches and
    # stores one result row per frame and policy.
    accuracy_batch_size = max(config.batch_sizes)
    for policy in config.policies:
        completed = 0
        for chunk in _chunks(frames, accuracy_batch_size):
            prepared = evaluator.prepare_batch([_atomic_batch(frame) for frame in chunk])
            result = evaluator.evaluate_prepared_batch(prepared, policy)
            metadata_json = json.dumps(result.metadata, sort_keys=True)
            for index, frame in enumerate(chunk):
                energy = result.energies[index] if result.energies is not None else np.nan
                finite = bool(
                    result.forces is not None and np.isfinite(energy)
                    and np.isfinite(result.forces[index]).all()
                )
                frame_error = result.error
                if result.error is None and not finite:
                    frame_error = "nonfinite energy or force output"
                rows.append({
                    "frame_id": frame["frame_id"], "molecule": frame["molecule"],
                    "stratum": frame["stratum"], "policy": policy,
                    "energy_ev": energy, "finite": finite,
                    "unsupported": result.unsupported, "error": frame_error,
                    "atom_count": len(frame["atomic_numbers"]),
                    "batch_edge_count": result.edge_count,
                    "model_hash": result.metadata.get("model_hash"),
                    "gpu": result.metadata.get("gpu"),
                    "metadata_json": metadata_json,
                })
                if result.forces is not None:
                    forces[(frame["frame_id"], policy)] = result.forces[index]
            completed += len(chunk)
            print(f"Accuracy {policy}: {completed}/300 frames", flush=True)

    rows = _add_discrepancies(rows, forces)

    # Timing uses a fixed representative disconnected graph batch for each
    # workload size. Policies are interleaved per iteration to reduce clock and
    # temperature drift. Graph construction and warm-up are outside samples.
    rng = np.random.default_rng(config.seed)
    for batch_size in config.batch_sizes:
        sample = _representative_frames(frames, batch_size)
        prepared = evaluator.prepare_batch([_atomic_batch(frame) for frame in sample])
        for policy in config.policies:
            for _ in range(config.warmups):
                evaluator.evaluate_prepared_batch(prepared, policy)
        for iteration in range(config.timed_iterations):
            policy_order = list(config.policies)
            rng.shuffle(policy_order)
            for policy in policy_order:
                result = evaluator.evaluate_prepared_batch(prepared, policy)
                atom_count = sum(result.atom_counts)
                wall = result.wall_seconds
                timings.append({
                    "policy": policy, "batch_size": batch_size,
                    "iteration": iteration, "policy_order": policy_order.index(policy),
                    "wall_seconds": wall, "cuda_seconds": result.cuda_seconds,
                    "frames_per_second": batch_size / wall if wall else np.nan,
                    "atom_steps_per_second": atom_count / wall if wall else np.nan,
                    "allocated_bytes": result.allocated_bytes,
                    "peak_bytes": result.peak_bytes,
                    "edge_count": result.edge_count,
                    "finite": result.finite, "unsupported": result.unsupported,
                    "error": result.error,
                    "genuine_graph_batch": True,
                })
            if (iteration + 1) % 10 == 0:
                print(
                    f"Timing batch={batch_size}: {iteration + 1}/{config.timed_iterations}",
                    flush=True,
                )

    pd.DataFrame(rows).to_parquet(config.output_dir / "evaluations.parquet", index=False)
    pd.DataFrame(timings).to_parquet(config.output_dir / "timings.parquet", index=False)
    import zarr
    store = zarr.open_group(str(config.output_dir / "forces.zarr"), mode="w")
    for (frame_id, policy), value in forces.items():
        group = store.require_group(policy)
        group.create_dataset(frame_id, data=value, shape=value.shape, dtype="f8")
    manifest = evaluator.manifest() | {
        "seed": config.seed, "platform": platform.platform(), "model": config.model,
        "accuracy_batch_size": accuracy_batch_size,
        "timing_scope": "prepared disconnected graph batch; model call plus output transfer",
        "genuine_graph_batch": True,
        "warmups": config.warmups, "timed_iterations": config.timed_iterations,
    }
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Gate 1 benchmark complete in {config.output_dir}", flush=True)
