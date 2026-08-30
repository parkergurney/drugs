from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import _atomic_batch, _representative_frames
from .config import D1Config
from .d1_diagnostics import _load_inputs, _policy_context
from .d1_selection import d1_config_sha256
from .model import MaceEvaluator


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cuda_timed(torch, function):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    start = time.perf_counter()
    value = function()
    end_event.record()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start, start_event.elapsed_time(end_event) / 1000


def _cpu_prepared(evaluator, frames):
    return evaluator.prepare_batch(
        [_atomic_batch(frame) for frame in frames], target_device="cpu"
    )


def _gpu_graph(evaluator, cpu_prepared):
    graph = cpu_prepared.graph.clone().to(evaluator.device)
    torch = evaluator.torch
    dtype = next(evaluator.calculator.models[0].parameters()).dtype
    for key in graph.keys:
        if torch.is_tensor(graph[key]) and torch.is_floating_point(graph[key]):
            graph[key] = graph[key].to(dtype=dtype)
    return graph


def _component_sample(evaluator, frames, policy, iteration, phase, process_id):
    torch = evaluator.torch
    rows = []
    start = time.perf_counter()
    cpu_prepared = _cpu_prepared(evaluator, frames)
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "graph_construction_cpu",
        "wall_seconds": time.perf_counter() - start, "cuda_seconds": np.nan,
    })

    graph, wall, cuda = _cuda_timed(torch, lambda: _gpu_graph(evaluator, cpu_prepared))
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "host_to_device",
        "wall_seconds": wall, "cuda_seconds": cuda,
    })
    model = evaluator.calculator.models[0]
    batch = graph.to_dict()

    def forward():
        with _policy_context(policy, evaluator.device):
            return model(
                batch, compute_force=False, compute_stress=False, training=False,
                compute_edge_forces=False, compute_atomic_stresses=False,
            )
    out, wall, cuda = _cuda_timed(torch, forward)
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "energy_forward",
        "wall_seconds": wall, "cuda_seconds": cuda,
    })

    def backward():
        gradient = torch.autograd.grad(
            outputs=[out["interaction_energy"]], inputs=[batch["positions"]],
            grad_outputs=[torch.ones_like(out["interaction_energy"])],
            retain_graph=False, create_graph=False, allow_unused=False,
        )[0]
        return -gradient
    forces, wall, cuda = _cuda_timed(torch, backward)
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "force_gradient",
        "wall_seconds": wall, "cuda_seconds": cuda,
    })

    converted, wall, cuda = _cuda_timed(
        torch, lambda: (out["energy"].detach().float(), forces.detach().float())
    )
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "output_conversion",
        "wall_seconds": wall, "cuda_seconds": cuda,
    })
    arrays, wall, cuda = _cuda_timed(
        torch, lambda: (converted[0].cpu().numpy(), converted[1].cpu().numpy())
    )
    finite = bool(np.isfinite(arrays[0]).all() and np.isfinite(arrays[1]).all())
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "device_to_host",
        "wall_seconds": wall, "cuda_seconds": cuda, "output_finite": finite,
    })
    pipeline = sum(row["wall_seconds"] for row in rows)
    rows.append({
        "process_id": process_id, "policy": policy, "batch_size": len(frames),
        "iteration": iteration, "phase": phase, "component": "component_pipeline_sum",
        "wall_seconds": pipeline, "cuda_seconds": np.nan, "output_finite": finite,
    })
    return rows, cpu_prepared


def _complete_rows(evaluator, frames, policy, cpu_prepared, iteration, phase, process_id):
    torch = evaluator.torch
    prepared_gpu = cpu_prepared
    prepared_gpu.graph = cpu_prepared.graph.clone().to(evaluator.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = evaluator.evaluate_prepared_batch(prepared_gpu, policy)
    prepared_wall = time.perf_counter() - start

    torch.cuda.synchronize()
    start = time.perf_counter()
    end_to_end_prepared = evaluator.prepare_batch([_atomic_batch(frame) for frame in frames])
    end_result = evaluator.evaluate_prepared_batch(end_to_end_prepared, policy)
    end_to_end_wall = time.perf_counter() - start
    return [
        {
            "process_id": process_id, "policy": policy, "batch_size": len(frames),
            "iteration": iteration, "phase": phase, "component": "prepared_model_total",
            "wall_seconds": prepared_wall, "cuda_seconds": result.cuda_seconds,
            "output_finite": result.finite, "error": result.error,
        },
        {
            "process_id": process_id, "policy": policy, "batch_size": len(frames),
            "iteration": iteration, "phase": phase, "component": "coordinate_to_result_total",
            "wall_seconds": end_to_end_wall, "cuda_seconds": end_result.cuda_seconds,
            "output_finite": end_result.finite, "error": end_result.error,
        },
    ]


def run_d1_timing(config: D1Config, run_id: str, timing_seed: int, allow_gpu=False):
    if config.device.startswith("cuda") and not allow_gpu:
        raise RuntimeError("GPU timing requires --allow-gpu-benchmark")
    if Path(run_id).name != run_id:
        raise ValueError("--run-id must be one directory name")
    all_frames, _ = _load_inputs(config)
    run_dir = config.output_dir / "runs" / run_id
    if run_dir.exists() and any(
        path.name not in {"gpu-before.csv", "gpu-telemetry.csv"}
        for path in run_dir.iterdir()
    ):
        raise FileExistsError(f"refusing to overwrite D1 timing process: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = MaceEvaluator(config.model, config.device, "float32")
    if evaluator.model_hash != config.expected_model_hash:
        raise ValueError("D1 timing model hash mismatch")
    rng = np.random.default_rng(timing_seed)
    rows = []
    for batch_size in config.batch_sizes:
        frames = _representative_frames(all_frames, batch_size)
        # Preserve one explicitly labeled cold observation per policy.
        for policy in config.timing_policies:
            component_rows, cpu_prepared = _component_sample(
                evaluator, frames, policy, -1, "cold", run_id
            )
            rows.extend(component_rows)
            rows.extend(_complete_rows(
                evaluator, frames, policy, cpu_prepared, -1, "cold", run_id
            ))
        for _ in range(config.warmups):
            for policy in config.timing_policies:
                _, cpu_prepared = _component_sample(
                    evaluator, frames, policy, -2, "warmup", run_id
                )
                _complete_rows(
                    evaluator, frames, policy, cpu_prepared, -2, "warmup", run_id
                )
        for iteration in range(config.timed_iterations):
            order = list(config.timing_policies)
            rng.shuffle(order)
            for policy_order, policy in enumerate(order):
                component_rows, cpu_prepared = _component_sample(
                    evaluator, frames, policy, iteration, "steady", run_id
                )
                for row in component_rows:
                    row["policy_order"] = policy_order
                rows.extend(component_rows)
                complete = _complete_rows(
                    evaluator, frames, policy, cpu_prepared, iteration, "steady", run_id
                )
                for row in complete:
                    row["policy_order"] = policy_order
                rows.extend(complete)
            if (iteration + 1) % 10 == 0:
                print(f"D1 timing batch={batch_size}: {iteration + 1}/{config.timed_iterations}", flush=True)
    pd.DataFrame(rows).to_parquet(run_dir / "timing-components.parquet", index=False)
    manifest = evaluator.manifest() | {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "run_id": run_id, "run_type": "uninstrumented_component_timing",
        "git_commit": _git_commit(), "platform": platform.platform(),
        "frames_sha256": config.expected_frames_sha256,
        "evaluations_sha256": config.expected_evaluations_sha256,
        "model_hash": config.expected_model_hash, "timing_seed": timing_seed,
        "config_sha256": d1_config_sha256(config),
        "policies": config.timing_policies, "batch_sizes": config.batch_sizes,
        "warmups": config.warmups, "timed_iterations": config.timed_iterations,
        "instrumented": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
