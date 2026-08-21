#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from precision_md.config import Gate1Config, load_config
from precision_md.model import MaceEvaluator
from precision_md.types import AtomicBatch


def atomic_batch(frame):
    return AtomicBatch(
        np.asarray(frame["atomic_numbers"]), np.asarray(frame["positions"]),
        (frame["frame_id"],),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate1.yaml")
    parser.add_argument("--output", default="results/preflight/batch-invariance.json")
    args = parser.parse_args()
    config = load_config(args.config, Gate1Config)
    with np.load(config.output_dir / "frames.npz", allow_pickle=True) as data:
        frames = data["frames"].tolist()

    # Three deterministic examples from each stratum.
    selected = []
    for stratum in ("ordinary", "high_force", "close_contact"):
        group = [frame for frame in frames if frame["stratum"] == stratum]
        selected.extend(group[index] for index in (0, len(group) // 2, len(group) - 1))

    evaluator = MaceEvaluator(config.model, config.device)
    combined = evaluator.evaluate_prepared_batch(
        evaluator.prepare_batch([atomic_batch(frame) for frame in selected]), "fp32"
    )
    if not combined.finite:
        raise RuntimeError(f"combined FP32 batch failed: {combined.error}")

    rows = []
    for index, frame in enumerate(selected):
        individual = evaluator.evaluate_prepared_batch(
            evaluator.prepare_batch([atomic_batch(frame)]), "fp32"
        )
        if not individual.finite:
            raise RuntimeError(f"individual FP32 graph failed: {frame['frame_id']}")
        energy_error = abs(combined.energies[index] - individual.energies[0])
        force_error = np.abs(combined.forces[index] - individual.forces[0])
        force_scale = max(float(np.abs(individual.forces[0]).max()), 1e-12)
        rows.append({
            "frame_id": frame["frame_id"], "stratum": frame["stratum"],
            "energy_abs_error_ev": float(energy_error),
            "force_max_abs_error_ev_per_a": float(force_error.max()),
            "force_max_relative_error": float(force_error.max() / force_scale),
        })

    summary = {
        "frames": rows,
        "max_energy_abs_error_ev": max(row["energy_abs_error_ev"] for row in rows),
        "max_force_abs_error_ev_per_a": max(row["force_max_abs_error_ev_per_a"] for row in rows),
        "max_force_relative_error": max(row["force_max_relative_error"] for row in rows),
        "passes": all(
            row["energy_abs_error_ev"] <= 1e-3
            and row["force_max_relative_error"] <= 1e-5
            for row in rows
        ),
        "criteria": {
            "energy_abs_error_ev": 1e-3,
            "force_max_relative_error": 1e-5,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passes"]:
        raise SystemExit("FP32 batching invariance validation failed")


if __name__ == "__main__":
    main()
