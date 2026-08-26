from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Frame:
    frame_id: str
    molecule: str
    atomic_numbers: np.ndarray
    positions: np.ndarray
    stratum: str = "ordinary"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frame_id(frame_id: str) -> str:
    return frame_id.split("-cc-", 1)[0]


def load_excluded_sources(path: str | Path | None) -> tuple[set[str], str | None]:
    if path is None:
        return set(), None
    path = Path(path)
    selection = json.loads(path.read_text(encoding="utf-8"))
    frame_ids = selection.get("frames")
    if not isinstance(frame_ids, list) or not all(isinstance(item, str) for item in frame_ids):
        raise ValueError(f"invalid selection file: {path}")
    return {source_frame_id(item) for item in frame_ids}, sha256_file(path)


def ensure_disjoint_sources(selected_sources, excluded_sources) -> None:
    overlap = sorted(set(selected_sources) & set(excluded_sources))
    if overlap:
        raise ValueError(f"prepared dataset overlaps excluded sources: {overlap[:5]}")


def ordinary_indices(lengths: dict[str, int], counts=(34, 33, 33), seed=20260819,
                     excluded_sources=frozenset()):
    rng = np.random.default_rng(seed)
    selected = {}
    for mol, count in zip(lengths, counts, strict=True):
        eligible = np.array([
            index for index in range(lengths[mol])
            if f"{mol}-{index}" not in excluded_sources
        ])
        if len(eligible) < count:
            raise ValueError(f"insufficient non-excluded ordinary frames for {mol}")
        selected[mol] = np.sort(rng.choice(eligible, count, replace=False))
    return selected


def select_high_force(candidates, count=100, cap=40):
    """Candidates are mappings with molecule and max_force; deterministic ties."""
    chosen, per_mol = [], {}
    for row in sorted(candidates, key=lambda x: (-x["max_force"], x["molecule"], x["frame_id"])):
        mol = row["molecule"]
        if per_mol.get(mol, 0) < cap:
            chosen.append(row); per_mol[mol] = per_mol.get(mol, 0) + 1
        if len(chosen) == count: break
    if len(chosen) != count:
        raise ValueError("insufficient candidates under per-molecule cap")
    return chosen


def close_contact(frame: Frame, target: float, variant: int = 0) -> Frame:
    """Move the deterministically selected nonbonded (index separation >2) pair."""
    xyz = np.array(frame.positions, dtype=float, copy=True)
    pairs = [(i, j) for i in range(len(xyz)) for j in range(i + 3, len(xyz))]
    if not pairs: raise ValueError("frame has no eligible nonbonded pair")
    i, j = pairs[variant % len(pairs)]
    direction = xyz[j] - xyz[i]
    norm = np.linalg.norm(direction)
    if norm == 0: direction, norm = np.array([1., 0., 0.]), 1.
    xyz[j] = xyz[i] + target * direction / norm
    return Frame(f"{frame.frame_id}-cc-{target:.1f}-{variant}", frame.molecule,
                 frame.atomic_numbers.copy(), xyz, "close_contact")
