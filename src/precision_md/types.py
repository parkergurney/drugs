from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class AtomicBatch:
    atomic_numbers: np.ndarray
    positions: np.ndarray
    frame_ids: tuple[str, ...] = ()
    cells: np.ndarray | None = None
    pbc: np.ndarray | None = None

    @property
    def atom_count(self) -> int:
        return int(np.asarray(self.atomic_numbers).size)


@dataclass
class EvaluationResult:
    policy: str
    energy: np.ndarray | None = None
    forces: np.ndarray | None = None
    wall_seconds: float | None = None
    cuda_seconds: float | None = None
    allocated_bytes: int | None = None
    peak_bytes: int | None = None
    atom_count: int = 0
    edge_count: int | None = None
    finite: bool = False
    unsupported: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchedEvaluationResult:
    policy: str
    frame_ids: tuple[str, ...]
    energies: np.ndarray | None = None
    forces: tuple[np.ndarray, ...] | None = None
    wall_seconds: float | None = None
    cuda_seconds: float | None = None
    allocated_bytes: int | None = None
    peak_bytes: int | None = None
    atom_counts: tuple[int, ...] = ()
    edge_count: int | None = None
    finite: bool = False
    unsupported: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedMaceBatch:
    graph: Any
    frame_ids: tuple[str, ...]
    atom_counts: tuple[int, ...]
    edge_count: int
