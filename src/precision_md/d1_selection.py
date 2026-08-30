from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .config import D1Config
from .data import sha256_file


REDUCED_POLICIES = ("tf32", "bf16_amp")
STRATA = ("ordinary", "high_force", "close_contact")
QUANTILE_LABELS = {0.0: "minimum", 0.5: "median", 0.95: "p95", 1.0: "maximum"}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def d1_config_sha256(config: D1Config) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _verify_inputs(config: D1Config) -> tuple[Path, Path]:
    frames = config.dataset / "frames.npz"
    evaluations = config.c1_analysis / "evaluations.parquet"
    for path, expected in (
        (frames, config.expected_frames_sha256),
        (evaluations, config.expected_evaluations_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing D1 input: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"D1 input checksum mismatch for {path}: {observed} != {expected}")
    return frames, evaluations


def select_d1_frames(config: D1Config) -> dict:
    """Select and freeze D1 cases solely from the immutable C1/P1 outputs."""
    frames_path, evaluations_path = _verify_inputs(config)
    table = pd.read_parquet(evaluations_path)
    required = {
        "frame_id", "policy", "stratum", "finite", "max_force_error_ev_per_a",
        "process_id",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"C1 evaluations are missing D1 columns: {sorted(missing)}")
    table = table[table["policy"].isin(REDUCED_POLICIES)].copy()
    grouped = table.groupby(["policy", "stratum", "frame_id"], as_index=False).agg(
        finite=("finite", "all"),
        max_force_error_ev_per_a=("max_force_error_ev_per_a", "max"),
        process_count=("process_id", "nunique"),
    )

    reasons: dict[str, list[dict]] = {}
    nonfinite = grouped[~grouped["finite"]].sort_values(
        ["policy", "stratum", "frame_id"]
    )
    for row in nonfinite.itertuples(index=False):
        reasons.setdefault(row.frame_id, []).append({
            "kind": "nonfinite",
            "policy": row.policy,
            "stratum": row.stratum,
            "process_count": int(row.process_count),
        })

    for policy in REDUCED_POLICIES:
        for stratum in STRATA:
            candidates = grouped[
                (grouped["policy"] == policy)
                & (grouped["stratum"] == stratum)
                & grouped["finite"]
            ].copy()
            if candidates.empty:
                raise ValueError(f"no finite D1 candidates for {policy}/{stratum}")
            values = candidates["max_force_error_ev_per_a"].to_numpy(dtype=float)
            for quantile in config.selection_quantiles:
                target = float(np.quantile(values, quantile))
                candidates["distance"] = (
                    candidates["max_force_error_ev_per_a"] - target
                ).abs()
                row = candidates.sort_values(["distance", "frame_id"]).iloc[0]
                reasons.setdefault(str(row.frame_id), []).append({
                    "kind": "ranked_control",
                    "policy": policy,
                    "stratum": stratum,
                    "quantile": quantile,
                    "label": QUANTILE_LABELS[quantile],
                    "target_error_ev_per_a": target,
                    "selected_error_ev_per_a": float(row.max_force_error_ev_per_a),
                    "process_count": int(row.process_count),
                })

    with np.load(frames_path, allow_pickle=True) as archive:
        frame_records = {frame["frame_id"]: frame for frame in archive["frames"].tolist()}
    missing_frames = set(reasons) - set(frame_records)
    if missing_frames:
        raise ValueError(f"selected frame IDs absent from P1: {sorted(missing_frames)}")
    records = []
    for frame_id in sorted(reasons):
        frame = frame_records[frame_id]
        records.append({
            "frame_id": frame_id,
            "molecule": frame["molecule"],
            "stratum": frame["stratum"],
            "reasons": reasons[frame_id],
        })
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "dataset_id": config.dataset_id,
        "selection_rule": (
            "all reduced-policy nonfinite frames plus nearest minimum, median, "
            "p95, and maximum finite maximum-force-error frame per policy/stratum; "
            "lexicographic frame_id tie-break; deduplicate without replacement"
        ),
        "frame_count": len(records),
        "records": records,
        "input_sha256": {
            "frames.npz": config.expected_frames_sha256,
            "evaluations.parquet": config.expected_evaluations_sha256,
        },
        "selection_script_commit": _git_commit(),
        "config_sha256": d1_config_sha256(config),
    }


def write_d1_selection(config: D1Config) -> dict:
    selection = select_d1_frames(config)
    output = config.output_dir / "d1-selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(selection, indent=2) + "\n"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        # A code-only diagnostic repair must retain the prospectively frozen
        # selection and its original selecting commit. Permit reuse only when
        # every scientific field is identical.
        comparable_existing = existing | {"selection_script_commit": None}
        comparable_selection = selection | {"selection_script_commit": None}
        if comparable_existing != comparable_selection:
            raise FileExistsError(f"refusing to replace a different D1 selection: {output}")
        return existing
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    return selection
