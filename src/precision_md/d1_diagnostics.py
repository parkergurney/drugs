from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import _atomic_batch
from .config import D1Config
from .d1_probe import (
    TraceCollector, TraceSetupError, add_geometry_trace, compare_trace_to_fp32,
    targeted_operation_mode,
)
from .d1_selection import _verify_inputs, d1_config_sha256
from .model import MaceEvaluator
from .policies import precision_context


DIAGNOSTIC_FILES = (
    "operator-traces.parquet",
    "energy-decomposition.parquet",
    "finite-difference.parquet",
    "equivariance.parquet",
    "batching-invariance.parquet",
)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_inputs(config: D1Config):
    frames_path, _ = _verify_inputs(config)
    selection_path = config.output_dir / "d1-selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("run select-d1 before diagnose-d1")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("input_sha256", {}).get("frames.npz") != config.expected_frames_sha256:
        raise ValueError("D1 selection does not match configured P1 frames")
    with np.load(frames_path, allow_pickle=True) as archive:
        all_frames = archive["frames"].tolist()
    by_id = {frame["frame_id"]: frame for frame in all_frames}
    selected = [by_id[row["frame_id"]] for row in selection["records"]]
    return all_frames, selected


def _policy_context(policy: str, device: str):
    if policy == "fp64":
        return nullcontext()
    return precision_context(policy, device.split(":")[0])


def _raw_call(evaluator, prepared, policy, trace=False, repeat=0, targeted=False):
    torch = evaluator.torch
    model = evaluator.calculator.models[0]
    graph = prepared.graph.clone()
    model_dtype = next(model.parameters()).dtype
    for key in graph.keys:
        value = graph[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            graph[key] = value.to(dtype=model_dtype)
    batch = graph.to_dict()
    collector = TraceCollector(
        model, policy, prepared.frame_ids[0], repeat,
        trace_level="targeted" if targeted else "coarse",
    ) if trace else None
    manager = collector if collector is not None else nullcontext()
    operation_manager = (
        targeted_operation_mode(collector) if targeted and collector is not None
        else nullcontext()
    )
    error = None
    out = None
    try:
        with _policy_context(policy, evaluator.device), manager, operation_manager:
            if collector is not None:
                add_geometry_trace(collector, graph)
            out = model(
                batch, compute_stress=False, training=False,
                compute_edge_forces=False, compute_atomic_stresses=False,
            )
            if collector is not None:
                collector._record("model.total_energy", model, "output", out["energy"])
                collector._record(
                    "model.interaction_energy", model, "output", out["interaction_energy"]
                )
                collector._record("model.forces", model, "output", out["forces"])
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        if isinstance(exc, TraceSetupError):
            raise
        error = f"{type(exc).__name__}: {exc}"
    return out, collector, error


def _numpy(tensor):
    return tensor.detach().double().cpu().numpy()


def _base_result(evaluator, frame, policy, trace, repeat, targeted=False):
    prepared = evaluator.prepare_batch([_atomic_batch(frame)])
    out, collector, error = _raw_call(
        evaluator, prepared, policy, trace, repeat, targeted
    )
    if out is None:
        return {
            "energy": np.nan, "interaction_energy": np.nan, "forces": None,
            "finite": False, "error": error, "collector": collector,
        }
    energy_factor = evaluator.calculator.energy_units_to_eV
    force_factor = energy_factor / evaluator.calculator.length_units_to_A
    if collector is not None:
        collector._record(
            "result_conversion.fp32_cast", evaluator.calculator.models[0], "output",
            (out["energy"].detach().float(), out["forces"].detach().float()),
        )
        collector._record(
            "result_transfer.cpu", evaluator.calculator.models[0], "output",
            (out["energy"].detach().float().cpu(), out["forces"].detach().float().cpu()),
        )
    energy = float(_numpy(out["energy"]).reshape(-1)[0] * energy_factor)
    interaction = float(
        _numpy(out["interaction_energy"]).reshape(-1)[0] * energy_factor
    )
    forces = _numpy(out["forces"]) * force_factor
    return {
        "energy": energy,
        "interaction_energy": interaction,
        "forces": forces,
        "finite": bool(np.isfinite(energy) and np.isfinite(forces).all()),
        "error": error,
        "collector": collector,
    }


def _energy_only(evaluator, frame, policy):
    prepared = evaluator.prepare_batch([_atomic_batch(frame)])
    torch = evaluator.torch
    model = evaluator.calculator.models[0]
    graph = prepared.graph.clone()
    dtype = next(model.parameters()).dtype
    for key in graph.keys:
        if torch.is_tensor(graph[key]) and torch.is_floating_point(graph[key]):
            graph[key] = graph[key].to(dtype=dtype)
    try:
        with _policy_context(policy, evaluator.device):
            out = model(
                graph.to_dict(), compute_force=False, compute_stress=False,
                training=False, compute_edge_forces=False,
                compute_atomic_stresses=False,
            )
        energy = (
            float(_numpy(out["energy"]).reshape(-1)[0])
            * evaluator.calculator.energy_units_to_eV
        )
        return energy, prepared.edge_count
    except (RuntimeError, NotImplementedError, TypeError):
        return np.nan, prepared.edge_count


def _control_component(frame_id: str, atom_count: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{frame_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (atom_count * 3)


def _discrepancy_component(reference, candidate) -> int:
    if candidate is None:
        return 0
    flat = candidate.reshape(-1)
    nonfinite = np.flatnonzero(~np.isfinite(flat))
    if len(nonfinite):
        return int(nonfinite[0])
    return int(np.argmax(np.abs(flat - reference.reshape(-1))))


def _finite_difference_rows(config, evaluator, frame, policy, base, reference):
    atom_count = len(frame["atomic_numbers"])
    components = {
        "maximum_discrepancy": _discrepancy_component(reference["forces"], base["forces"]),
        "deterministic_control": _control_component(frame["frame_id"], atom_count, config.seed),
    }
    rows = []
    for kind, flat_component in components.items():
        atom, component = divmod(flat_component, 3)
        autograd_force = (
            float(base["forces"][atom, component]) if base["forces"] is not None else np.nan
        )
        for step in config.finite_difference_steps_a:
            plus = dict(frame)
            minus = dict(frame)
            plus["positions"] = np.asarray(frame["positions"], dtype=float).copy()
            minus["positions"] = np.asarray(frame["positions"], dtype=float).copy()
            plus["positions"][atom, component] += step
            minus["positions"][atom, component] -= step
            energy_plus, edge_count_plus = _energy_only(evaluator, plus, policy)
            energy_minus, edge_count_minus = _energy_only(evaluator, minus, policy)
            finite_difference = -(energy_plus - energy_minus) / (2 * step)
            rows.append({
                "frame_id": frame["frame_id"], "molecule": frame["molecule"],
                "stratum": frame["stratum"], "policy": policy,
                "component_kind": kind, "atom_index": atom,
                "cartesian_component": component, "step_a": step,
                "energy_plus_ev": energy_plus, "energy_minus_ev": energy_minus,
                "edge_count_plus": edge_count_plus,
                "edge_count_minus": edge_count_minus,
                "edge_count_changed": edge_count_plus != edge_count_minus,
                "finite_difference_force_ev_per_a": finite_difference,
                "autograd_force_ev_per_a": autograd_force,
                "absolute_difference_ev_per_a": abs(finite_difference - autograd_force),
                "finite": bool(np.isfinite([energy_plus, energy_minus, finite_difference, autograd_force]).all()),
            })
    return rows


def _orthogonal_transforms(frame_id: str, seed: int):
    local_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:rotation:{frame_id}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(local_seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    reflection = np.diag([-1.0, 1.0, 1.0])
    return (("rotation", q), ("reflection", reflection))


def _equivariance_rows(config, evaluator, frame, policy, base):
    rows = []
    positions = np.asarray(frame["positions"], dtype=float)
    center = positions.mean(axis=0)
    for transform_name, transform in _orthogonal_transforms(frame["frame_id"], config.seed):
        changed = dict(frame)
        changed["positions"] = (positions - center) @ transform.T + center
        result = _base_result(evaluator, changed, policy, False, 0)
        expected_forces = (
            base["forces"] @ transform.T if base["forces"] is not None else None
        )
        force_residual = (
            float(np.max(np.linalg.norm(result["forces"] - expected_forces, axis=1)))
            if result["forces"] is not None and expected_forces is not None
            else np.nan
        )
        rows.append({
            "frame_id": frame["frame_id"], "molecule": frame["molecule"],
            "stratum": frame["stratum"], "policy": policy,
            "transform": transform_name,
            "energy_residual_ev": result["energy"] - base["energy"],
            "max_force_equivariance_residual_ev_per_a": force_residual,
            "finite": bool(result["finite"] and base["finite"] and np.isfinite(force_residual)),
            "error": result["error"],
        })
    return rows


def _batch_rows(config, evaluator, frame, policy, base, all_frames):
    frame_index = next(i for i, value in enumerate(all_frames) if value["frame_id"] == frame["frame_id"])
    original = all_frames[(frame_index // 32) * 32:(frame_index // 32 + 1) * 32]
    fillers = [value for value in sorted(all_frames, key=lambda x: x["frame_id"])
               if value["frame_id"] != frame["frame_id"]]
    workloads = [("original_c1_chunk", original)]
    for size in (8, 32):
        workloads.append((f"controlled_{size}", [frame] + fillers[:size - 1]))
    rows = []
    for context, frames in workloads:
        prepared = evaluator.prepare_batch([_atomic_batch(value) for value in frames])
        out, _, error = _raw_call(evaluator, prepared, policy, False, 0)
        target_index = next(i for i, value in enumerate(frames) if value["frame_id"] == frame["frame_id"])
        energy = np.nan
        forces = None
        if out is not None:
            energy = float(_numpy(out["energy"])[target_index]) * evaluator.calculator.energy_units_to_eV
            offsets = np.cumsum([0] + [len(value["atomic_numbers"]) for value in frames])
            forces = _numpy(out["forces"])[offsets[target_index]:offsets[target_index + 1]]
            forces *= evaluator.calculator.energy_units_to_eV / evaluator.calculator.length_units_to_A
        force_delta = (
            float(np.max(np.linalg.norm(forces - base["forces"], axis=1)))
            if forces is not None and base["forces"] is not None else np.nan
        )
        rows.append({
            "frame_id": frame["frame_id"], "molecule": frame["molecule"],
            "stratum": frame["stratum"], "policy": policy,
            "batch_context": context, "batch_size": len(frames),
            "energy_difference_ev": energy - base["energy"],
            "max_force_difference_ev_per_a": force_delta,
            "finite": bool(np.isfinite(energy) and forces is not None and np.isfinite(forces).all()),
            "error": error,
        })
    return rows


def _write_case(case_dir: Path, tables: dict[str, list[dict]], facts: dict):
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{case_dir.name}-", dir=case_dir.parent))
    try:
        for name, rows in tables.items():
            pd.DataFrame(rows).to_parquet(staging / name, index=False)
        (staging / "case.json").write_text(json.dumps(facts, indent=2) + "\n")
        os.replace(staging, case_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _case_complete(case_dir: Path, config: D1Config, frame_id: str, repeat: int) -> bool:
    required = (*DIAGNOSTIC_FILES, "case.json")
    if not all((case_dir / name).is_file() for name in required):
        return False
    facts = json.loads((case_dir / "case.json").read_text())
    return (
        facts.get("frame_id") == frame_id
        and facts.get("repeat") == repeat
        and facts.get("frames_sha256") == config.expected_frames_sha256
        and facts.get("model_hash") == config.expected_model_hash
    )


def run_d1_diagnostics(config: D1Config, run_id: str, allow_gpu=False):
    if config.device.startswith("cuda") and not allow_gpu:
        raise RuntimeError("GPU diagnostics require --allow-gpu-diagnostic")
    if Path(run_id).name != run_id:
        raise ValueError("--run-id must be one directory name")
    all_frames, selected = _load_inputs(config)
    run_dir = config.output_dir / "runs" / run_id
    cases_dir = run_dir / "cases"
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to replace completed diagnostic run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    fp32_evaluator = MaceEvaluator(config.model, config.device, "float32")
    if fp32_evaluator.model_hash != config.expected_model_hash:
        raise ValueError(
            f"D1 model hash mismatch: {fp32_evaluator.model_hash} != {config.expected_model_hash}"
        )
    fp64_evaluator = MaceEvaluator(config.model, config.device, "float64")
    evaluators = {policy: fp64_evaluator if policy == "fp64" else fp32_evaluator
                  for policy in config.diagnostic_policies}

    for repeat in range(config.trace_repeats):
        for frame in selected:
            case_dir = cases_dir / f"repeat-{repeat:02d}" / frame["frame_id"]
            if _case_complete(case_dir, config, frame["frame_id"], repeat):
                continue
            if case_dir.exists():
                raise FileExistsError(f"refusing to overwrite incomplete D1 case: {case_dir}")
            base_results = {}
            trace_rows = []
            reference_trace = None
            execution_order = ["fp32", "fp64", "tf32", "bf16_amp"]
            for policy in execution_order:
                if policy not in config.diagnostic_policies:
                    continue
                result = _base_result(evaluators[policy], frame, policy, True, repeat)
                base_results[policy] = result
                collector = result["collector"]
                if collector is None:
                    continue
                if policy == "fp32":
                    reference_trace = (collector.rows, collector.tensors)
                elif reference_trace is not None:
                    compare_trace_to_fp32(
                        collector.rows, collector.tensors, reference_trace,
                        config.relative_rms_floor,
                    )
                trace_rows.extend(collector.rows)

            reference = base_results["fp32"]
            implicated = []
            for policy in ("tf32", "bf16_amp"):
                if policy not in base_results:
                    continue
                collector = base_results[policy]["collector"]
                threshold = config.diagnostic_thresholds[policy]
                crossed = collector is not None and any(
                    np.isfinite(row.get("relative_rms_from_fp32", np.nan))
                    and row.get("relative_rms_from_fp32", 0.0) > threshold
                    for row in collector.rows
                )
                if not base_results[policy]["finite"] or crossed:
                    implicated.append(policy)
            if implicated:
                targeted_reference_result = _base_result(
                    fp32_evaluator, frame, "fp32", True, repeat, targeted=True
                )
                targeted_reference = (
                    targeted_reference_result["collector"].rows,
                    targeted_reference_result["collector"].tensors,
                )
                trace_rows.extend(targeted_reference_result["collector"].rows)
                for policy in implicated:
                    targeted_result = _base_result(
                        evaluators[policy], frame, policy, True, repeat, targeted=True
                    )
                    targeted_collector = targeted_result["collector"]
                    compare_trace_to_fp32(
                        targeted_collector.rows, targeted_collector.tensors,
                        targeted_reference, config.relative_rms_floor,
                    )
                    trace_rows.extend(targeted_collector.rows)
            energy_rows, fd_rows, equivariance_rows, batch_rows = [], [], [], []
            for policy in config.diagnostic_policies:
                result = base_results[policy]
                energy_rows.append({
                    "frame_id": frame["frame_id"], "molecule": frame["molecule"],
                    "stratum": frame["stratum"], "policy": policy,
                    "atom_count": len(frame["atomic_numbers"]),
                    "total_energy_ev": result["energy"],
                    "interaction_energy_ev": result["interaction_energy"],
                    "atomic_reference_energy_ev": result["energy"] - result["interaction_energy"],
                    "raw_energy_error_from_fp32_ev": result["energy"] - reference["energy"],
                    "energy_error_per_atom_ev": (
                        result["energy"] - reference["energy"]
                    ) / len(frame["atomic_numbers"]),
                    "finite": result["finite"], "error": result["error"],
                })
                fd_rows.extend(_finite_difference_rows(
                    config, evaluators[policy], frame, policy, result, reference
                ))
                equivariance_rows.extend(_equivariance_rows(
                    config, evaluators[policy], frame, policy, result
                ))
                batch_rows.extend(_batch_rows(
                    config, evaluators[policy], frame, policy, result, all_frames
                ))
            tables = {
                "operator-traces.parquet": trace_rows,
                "energy-decomposition.parquet": energy_rows,
                "finite-difference.parquet": fd_rows,
                "equivariance.parquet": equivariance_rows,
                "batching-invariance.parquet": batch_rows,
            }
            required_trace_columns = {
                "frame_id", "policy", "repeat", "boundary_order", "boundary",
                "nonfinite_count",
            }
            trace_columns = set(pd.DataFrame(trace_rows).columns)
            if not trace_rows or not required_trace_columns <= trace_columns:
                missing = sorted(required_trace_columns - trace_columns)
                raise RuntimeError(
                    "D1 instrumentation produced an unusable operator trace "
                    f"for {frame['frame_id']} repeat={repeat}; "
                    f"rows={len(trace_rows)}, missing_columns={missing}"
                )
            _write_case(case_dir, tables, {
                "schema_version": 1, "frame_id": frame["frame_id"], "repeat": repeat,
                "experiment_id": config.experiment_id,
                "frames_sha256": config.expected_frames_sha256,
                "model_hash": config.expected_model_hash,
                "script_module_hook_policy": (
                    "skip unsupported ScriptModule hooks; observe nearest eager "
                    "boundaries and targeted TorchDispatch operations"
                ),
                "skipped_script_modules": sorted({
                    name
                    for result in base_results.values()
                    if result["collector"] is not None
                    for name in result["collector"].skipped_script_modules
                }),
            })
            print(f"D1 diagnostics {frame['frame_id']} repeat={repeat} complete", flush=True)

    for filename in DIAGNOSTIC_FILES:
        paths = sorted(cases_dir.glob(f"repeat-*/*/{filename}"))
        if not paths:
            raise RuntimeError(f"D1 produced no {filename}")
        pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).to_parquet(
            run_dir / filename, index=False
        )
    manifest = fp32_evaluator.manifest() | {
        "schema_version": 1, "experiment_id": config.experiment_id,
        "run_id": run_id, "run_type": "instrumented_diagnostics",
        "git_commit": _git_commit(), "platform": platform.platform(),
        "frames_sha256": config.expected_frames_sha256,
        "evaluations_sha256": config.expected_evaluations_sha256,
        "model_hash": config.expected_model_hash,
        "config_sha256": d1_config_sha256(config),
        "fp64_model_hash": fp64_evaluator.model_hash,
        "selected_frame_count": len(selected), "trace_repeats": config.trace_repeats,
        "policies": config.diagnostic_policies,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace completed diagnostic run: {run_dir}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
