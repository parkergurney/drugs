from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256_file


DATASET_PAYLOADS = ("frames.npz", "selection.json", "candidate_scores.parquet")


def _dataset_facts(root: Path) -> dict:
    missing = [name for name in DATASET_PAYLOADS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset is missing: {', '.join(missing)}")
    with np.load(root / "frames.npz", allow_pickle=True) as archive:
        if "frames" not in archive:
            raise ValueError("frames.npz does not contain a frames array")
        frames = archive["frames"].tolist()
    if len(frames) != 300:
        raise ValueError(f"frozen dataset must contain 300 frames, found {len(frames)}")
    frame_ids = [frame.get("frame_id") for frame in frames]
    if None in frame_ids or len(set(frame_ids)) != 300:
        raise ValueError("frozen dataset frame IDs must be present and unique")
    strata = {
        name: sum(frame.get("stratum") == name for frame in frames)
        for name in ("ordinary", "high_force", "close_contact")
    }
    if strata != {"ordinary": 100, "high_force": 100, "close_contact": 100}:
        raise ValueError(f"unexpected frozen dataset strata: {strata}")
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    selection_ids = selection.get("frames")
    if (
        not isinstance(selection_ids, list)
        or len(selection_ids) != 300
        or len(set(selection_ids)) != 300
        or set(selection_ids) != set(frame_ids)
    ):
        raise ValueError("selection.json does not identify the frozen frames")
    candidate_count = len(pd.read_parquet(root / "candidate_scores.parquet"))
    if candidate_count != 3000:
        raise ValueError(
            f"candidate_scores.parquet must contain 3000 rows, found {candidate_count}"
        )
    return {
        "frame_count": len(frames),
        "stratum_counts": strata,
        "candidate_count": candidate_count,
    }


def _read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksum line in {path}: {line!r}")
        digest, name = parts
        name = name.lstrip("*")
        if Path(name).name != name or len(digest) != 64:
            raise ValueError(f"unsafe or invalid checksum entry in {path}: {line!r}")
        sums[name] = digest
    return sums


def validate_frozen_dataset(root: str | Path, expected_dataset_id: str | None = None) -> dict:
    root = Path(root)
    facts = _dataset_facts(root)
    manifest_path, sums_path = root / "manifest.json", root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise FileNotFoundError("frozen dataset requires manifest.json and SHA256SUMS")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_dataset_id is not None and manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError(
            f"expected dataset_id {expected_dataset_id!r}, found {manifest.get('dataset_id')!r}"
        )
    sums = _read_sums(sums_path)
    expected_names = {*DATASET_PAYLOADS, "manifest.json"}
    if set(sums) != expected_names:
        raise ValueError(f"SHA256SUMS entries must be exactly {sorted(expected_names)}")
    for name, expected in sums.items():
        observed = sha256_file(root / name)
        if observed != expected:
            raise ValueError(f"checksum mismatch for {root / name}: {observed} != {expected}")
    payload_hashes = manifest.get("payload_sha256", {})
    for name in DATASET_PAYLOADS:
        if payload_hashes.get(name) != sums[name]:
            raise ValueError(f"manifest payload hash does not match {name}")
    if manifest.get("frame_count") != facts["frame_count"]:
        raise ValueError("manifest frame_count does not match frames.npz")
    if manifest.get("stratum_counts") != facts["stratum_counts"]:
        raise ValueError("manifest stratum_counts do not match frames.npz")
    if manifest.get("candidate_count") != facts["candidate_count"]:
        raise ValueError("manifest candidate_count does not match candidate scores")
    return manifest


def freeze_dataset(
    source: str | Path,
    output: str | Path,
    dataset_id: str,
    provenance: str | Path | None = None,
) -> dict:
    """Atomically create or verify one portable immutable dataset bundle."""
    source, output = Path(source), Path(output)
    facts = _dataset_facts(source)
    source_hashes = {name: sha256_file(source / name) for name in DATASET_PAYLOADS}
    preparation_manifest = None
    for name in ("dataset-manifest.json", "manifest.json"):
        candidate = source / name
        if candidate.is_file():
            preparation_manifest = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if preparation_manifest is not None:
        prepared_id = preparation_manifest.get("dataset_id")
        if prepared_id is not None and prepared_id != dataset_id:
            raise ValueError(
                f"prepared dataset_id {prepared_id!r} does not match {dataset_id!r}"
            )
        if preparation_manifest.get("exclude_selection") is not None:
            if preparation_manifest.get("overlap_count") != 0:
                raise ValueError("excluded-source dataset must report zero overlap")
    external_provenance = None
    if provenance is not None:
        external_provenance = json.loads(Path(provenance).read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        **facts,
        "payload_sha256": source_hashes,
        "preparation_manifest": preparation_manifest,
        "external_provenance": external_provenance,
    }
    if output.exists():
        existing = validate_frozen_dataset(output, dataset_id)
        if existing != manifest:
            raise FileExistsError(
                f"refusing to replace different frozen dataset or provenance: {output}"
            )
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name in DATASET_PAYLOADS:
            shutil.copy2(source / name, staging / name)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        sum_names = (*DATASET_PAYLOADS, "manifest.json")
        (staging / "SHA256SUMS").write_text(
            "".join(f"{sha256_file(staging / name)}  {name}\n" for name in sum_names),
            encoding="utf-8",
        )
        validate_frozen_dataset(staging, dataset_id)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_frozen_dataset(output, dataset_id)


def validate_trial(
    trial: str | Path,
    dataset: str | Path,
    experiment_id: str | None = None,
) -> dict:
    trial, dataset = Path(trial), Path(dataset)
    required_files = ("frames.npz", "evaluations.parquet", "timings.parquet", "manifest.json")
    missing = [name for name in required_files if not (trial / name).is_file()]
    if not (trial / "forces.zarr").is_dir():
        missing.append("forces.zarr/")
    if not list(trial.glob("gpu-*.csv")):
        missing.append("gpu-*.csv")
    if missing:
        raise FileNotFoundError(f"incomplete trial {trial}: {', '.join(missing)}")
    dataset_manifest = validate_frozen_dataset(dataset)
    manifest = json.loads((trial / "manifest.json").read_text(encoding="utf-8"))
    expected_hash = dataset_manifest["payload_sha256"]["frames.npz"]
    if manifest.get("frames_sha256") != expected_hash:
        raise ValueError(f"trial manifest frame hash differs from dataset: {trial}")
    if sha256_file(trial / "frames.npz") != expected_hash:
        raise ValueError(f"trial frame copy differs from dataset: {trial}")
    if manifest.get("dataset_id") != dataset_manifest.get("dataset_id"):
        raise ValueError(f"trial dataset_id differs from dataset bundle: {trial}")
    if experiment_id is not None and manifest.get("experiment_id") != experiment_id:
        raise ValueError(f"trial experiment_id differs from {experiment_id}: {trial}")
    evaluations = pd.read_parquet(trial / "evaluations.parquet")
    timings = pd.read_parquet(trial / "timings.parquet")
    policies = manifest.get("policies")
    batch_sizes = manifest.get("batch_sizes")
    if not isinstance(policies, list) or not policies:
        raise ValueError(f"trial manifest does not declare policies: {trial}")
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise ValueError(f"trial manifest does not declare batch_sizes: {trial}")
    expected_evaluations = 300 * len(policies)
    if len(evaluations) != expected_evaluations or set(evaluations.policy) != set(policies):
        raise ValueError(
            f"trial evaluation rows/policies do not match manifest: {trial}"
        )
    if set(timings.policy) != set(policies) or set(timings.batch_size) != set(batch_sizes):
        raise ValueError(f"trial timing policies/batches do not match manifest: {trial}")
    expected_timings = (
        len(policies) * len(batch_sizes)
        * int(manifest.get("timed_iterations", 0))
    )
    if len(timings) != expected_timings:
        raise ValueError(
            f"trial timing row count {len(timings)} != {expected_timings}: {trial}"
        )
    return manifest
