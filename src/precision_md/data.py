from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Frame:
    frame_id: str
    molecule: str
    atomic_numbers: np.ndarray
    positions: np.ndarray
    stratum: str = "ordinary"


def ordinary_indices(lengths: dict[str, int], counts=(34, 33, 33), seed=20260819):
    rng = np.random.default_rng(seed)
    return {mol: np.sort(rng.choice(lengths[mol], n, replace=False))
            for (mol, n) in zip(lengths, counts, strict=True)}


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
