from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Checkpoint:
    checkpoint_id: int
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    time_fs: float
    metadata: dict

    def clone(self):
        return Checkpoint(self.checkpoint_id, self.positions.copy(), self.velocities.copy(),
                          self.masses.copy(), self.time_fs, dict(self.metadata))


def serialize_checkpoint(checkpoint: Checkpoint, path):
    start = time.perf_counter()
    np.savez_compressed(path, positions=checkpoint.positions, velocities=checkpoint.velocities,
                        masses=checkpoint.masses, time_fs=checkpoint.time_fs,
                        checkpoint_id=checkpoint.checkpoint_id,
                        metadata=np.array([checkpoint.metadata], dtype=object))
    return time.perf_counter() - start


def restore_checkpoint(path):
    start = time.perf_counter()
    with np.load(path, allow_pickle=True) as f:
        cp = Checkpoint(int(f["checkpoint_id"]), f["positions"], f["velocities"],
                        f["masses"], float(f["time_fs"]), f["metadata"][0])
    return cp, time.perf_counter() - start


def wrapped_angle_difference(a_deg, b_deg):
    return np.abs((np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0 - 180.0)


def pair_distance_rmsd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape: raise ValueError("position arrays must have the same shape")
    iu = np.triu_indices(len(a), 1)
    da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)[iu]
    db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)[iu]
    return float(np.sqrt(np.mean((da - db) ** 2)))


def conformer_state(dihedral_deg):
    x = float(dihedral_deg) % 360
    if 120 <= x < 240: return "trans"
    return "gauche+" if x < 120 else "gauche-"


def unsafe_label(reference, fast):
    finite = all(np.isfinite(np.asarray(v)).all() for d in (reference, fast)
                 for v in d.values() if isinstance(v, (int, float, list, tuple, np.ndarray)))
    reasons = []
    if not finite: reasons.append("nonfinite")
    if reference["conformer"] != fast["conformer"]: reasons.append("conformer")
    if wrapped_angle_difference(reference["dihedral_deg"], fast["dihedral_deg"]) > 5: reasons.append("dihedral")
    if pair_distance_rmsd(reference["positions"], fast["positions"]) > 0.02: reasons.append("pair_distance_rmsd")
    ref_delta = reference["endpoint_energy_ev"] - reference["initial_energy_ev"]
    fast_delta = fast["endpoint_energy_ev"] - fast["initial_energy_ev"]
    if abs(ref_delta - fast_delta) * 23.0605478306 > 0.1: reasons.append("delta_energy")
    return bool(reasons), reasons


def basin_boundary_distance(dihedral_deg):
    x = float(dihedral_deg) % 360
    return min(abs((x - boundary + 180) % 360 - 180) for boundary in (120, 240))
